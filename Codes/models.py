import os 
import numpy as np
import math
import random

import torch
import torch.nn as nn

from torch_geometric.nn import HGTConv, MessagePassing
from EdgeAwareHGTConv import EdgeAwareHGTConv

import utils

class Encoder(nn.Module):
    def __init__(self, node_types, edge_types, edge_attr_sizes, in_channels_dict, hidden_channels, dropout=0.5, num_layers=2):
        """
        异构图编码器
        
        Args:
            node_types: 节点类型列表
            edge_types: 边类型列表
            in_channels_dict: 每种节点类型的输入特征维度
            hidden_channels: 隐藏层维度
            dropout: dropout率
            num_layers: HGT层数
        """
        super(Encoder, self).__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_channels_dict = in_channels_dict
        self.edge_attr_sizes = edge_attr_sizes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 为每种节点类型创建独立的线性投影层
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(in_channels_dict[node_type], hidden_channels)
        
        # 多层HGT卷积
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()  # 添加Dropout层列表
        for _ in range(num_layers):
            self.convs.append(EdgeAwareHGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=(node_types, edge_types),
                heads=8,
                edge_attr_sizes=edge_attr_sizes
            ))
            self.dropouts.append(nn.Dropout(dropout))  # 添加Dropout层
        
        self.prelu = nn.PReLU(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. 特征投影
        x_proj_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.lin_dict:  # 防止未知节点类型
                x_proj_dict[node_type] = self.lin_dict[node_type](x)
        
        # 2. 多层HGT卷积
        out_dict = x_proj_dict
        for i, conv in enumerate(self.convs):
            out_dict = conv(out_dict, edge_index_dict, edge_attr_dict)
            # 每层后应用激活和dropout
            for node_type in self.node_types:
                if node_type in out_dict:
                    out_dict[node_type] = self.dropouts[i](self.prelu(out_dict[node_type]))
        out_dict = {k: self.dropout(v) for k, v in out_dict.items()}
        
        return out_dict

class Summary(MessagePassing):
    """
    异构图节点级Summary - 按边类型聚合邻域信息
    
    支持多种聚合策略和注意力机制
    """
    def __init__(self, hidden_channels, aggr='mean', use_attention=False):
        """
        Args:
            hidden_channels: 特征维度
            aggr: 聚合函数类型 ('max', 'mean', 'sum', 'attention')
            use_attention: 是否使用注意力机制
        """
        super().__init__(aggr=aggr if aggr != 'attention' else 'add')
        self.hidden_channels = hidden_channels
        self.use_attention = use_attention or aggr == 'attention'
        
        if self.use_attention:
            self.att_linear = nn.Linear(hidden_channels, 1)
            self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x_dict, edge_index_dict):
        """
        前向传播：计算所有节点的邻域Summary
        """
        # 确保输入数据在同一设备
        device = next(iter(x_dict.values())).device
        edge_index_dict = {
            k: v.to(device) for k, v in edge_index_dict.items()
        }
        
        # 使用零初始化,避免梯度问题
        out_dict = {node_type: torch.zeros_like(x) for node_type, x in x_dict.items()}
        edge_count_dict = {node_type: torch.zeros(x.size(0), device=device) 
                          for node_type, x in x_dict.items()}
        
        for edge_type, edge_index in edge_index_dict.items():
            if edge_index.size(1) == 0:  # 跳过空边
                continue
                
            src_type, rel_type, dst_type = edge_type
            if src_type not in x_dict or dst_type not in x_dict:
                continue
            
            max_idx = edge_index[1].max()
            if max_idx >= x_dict[dst_type].size(0):
                raise ValueError(
                    f"Invalid edge indices for {edge_type}: "
                    f"Max index {max_idx} >= {x_dict[dst_type].size(0)}"
                )
                
            src_features = x_dict[src_type]
            dst_num_nodes = x_dict[dst_type].size(0)
            
            # 执行消息传递
            out = self.propagate(edge_index=edge_index,x=src_features,size=(src_features.size(0), dst_num_nodes))
            
            # 累加不同边类型的聚合结果
            if out.size(0) == out_dict[dst_type].size(0):
                out_dict[dst_type] = out_dict[dst_type] + out
                # 记录边的数量用于后续归一化
                edge_count_dict[dst_type] += torch.bincount(
                    edge_index[1], minlength=out_dict[dst_type].size(0)
                ).float()

            # 检查目标索引是否越界
            if edge_index[1].max() >= dst_num_nodes:
                print(f"Warning: Invalid edge indices for {edge_type}. Max index {edge_index[1].max()} >= {dst_num_nodes}")
                continue
        
        # 对没有邻居的节点，使用自身特征
        for node_type in out_dict:
            mask = edge_count_dict[node_type] == 0
            out_dict[node_type][mask] = x_dict[node_type][mask]
            
        return out_dict
    
    def message(self, x_j, edge_index_i=None):
        """消息函数"""
        if self.use_attention:
            # 计算注意力权重
            att_weights = self.leaky_relu(self.att_linear(x_j))
            return x_j * att_weights
        return x_j

def corruption(x_dict, edge_index_dict, corruption_rate=1.0):
    """
    改进的负样本生成策略
    
    Args:
        x_dict: 节点特征字典
        edge_index_dict: 边索引字典
        corruption_rate: 打乱比例(0-1)
    """
    corrupted_x_dict = {}
    
    for node_type, x in x_dict.items():
        if corruption_rate == 1.0:
            # 完全打乱
            corrupted_x_dict[node_type] = x[torch.randperm(x.size(0))]
        else:
            # 部分打乱
            num_corrupt = int(x.size(0) * corruption_rate)
            indices = torch.randperm(x.size(0))
            corrupt_indices = indices[:num_corrupt]
            
            corrupted_x = x.clone()
            corrupted_x[corrupt_indices] = x[torch.randperm(x.size(0))[:num_corrupt]]
            corrupted_x_dict[node_type] = corrupted_x
    
    return corrupted_x_dict, edge_index_dict

def uniform(size, tensor):
    """权重初始化"""
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def reset(nn):
    """模型参数重置"""
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class GraphLocalInfomax(torch.nn.Module):
    def __init__(self, hidden_channels, encoder, summary, corruption, 
                 dropout=0.5, loss_type='bce', temperature=1.0):
        """
        异构图局部互信息最大化模型
        
        Args:
            hidden_channels: 隐藏层维度
            encoder: 编码器
            summary: 摘要器
            corruption: 负样本生成函数
            dropout: dropout率
            loss_type: 损失函数类型 ('bce', 'infonce')
            temperature: InfoNCE温度参数
        """
        super(GraphLocalInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        self.loss_type = loss_type
        self.temperature = temperature
        
        # 为每种节点类型创建独立的权重矩阵
        self.weight_dict = nn.ModuleDict()
        for node_type in encoder.node_types:
            self.weight_dict[node_type] = nn.Linear(
                hidden_channels, hidden_channels, bias=False
            )
        
        # 可选的非线性变换
        self.transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
            
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        for node_type in self.weight_dict:
            uniform(self.hidden_channels, self.weight_dict[node_type].weight)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 编码原始图得到正样本表示
        pos_z_dict = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        
        # 生成负样本
        corrupted_x_dict, corrupted_edge_index_dict = self.corruption(x_dict, edge_index_dict)
        neg_z_dict = self.encoder(corrupted_x_dict, edge_index_dict, edge_attr_dict)
        
        # 生成图摘要
        summary_dict = self.summary(pos_z_dict, edge_index_dict)
        for node_type in summary_dict:
            summary_dict[node_type] = self.dropout(summary_dict[node_type])
            
        return pos_z_dict, neg_z_dict, summary_dict

    def discriminate(self, z_dict, summary_dict):
        """判别器：计算节点-摘要相似度"""
        scores_dict = {}
        for node_type in z_dict:
            if node_type not in summary_dict or node_type not in self.weight_dict:
                continue
                
            z = z_dict[node_type]
            summary = summary_dict[node_type]
            weight = self.weight_dict[node_type]
            
            # 可选的非线性变换
            # z = self.transform(z)
            
            # 计算相似度得分
            if self.loss_type == 'infonce':
                # 使用余弦相似度
                z_norm = torch.nn.functional.normalize(z, dim=1)
                summary_transformed = torch.nn.functional.normalize(
                    torch.matmul(summary, weight.weight), dim=1
                )
                value = torch.sum(z_norm * summary_transformed, dim=1) / self.temperature
            else:
                # 原始内积
                value = torch.sum(torch.mul(z, torch.matmul(summary, weight.weight)), dim=1)
            
            scores_dict[node_type] = value
            
        return scores_dict

    def loss(self, pos_z_dict, neg_z_dict, summary_dict):
        """计算损失函数"""
        total_loss = 0
        num_types = 0
        
        for node_type in pos_z_dict:
            if node_type not in summary_dict:
                continue
                
            pos_score = self.discriminate(
                {node_type: pos_z_dict[node_type]},
                {node_type: summary_dict[node_type]}
            ).get(node_type)
            
            neg_score = self.discriminate(
                {node_type: neg_z_dict[node_type]},
                {node_type: summary_dict[node_type]}
            ).get(node_type)
            
            if pos_score is None or neg_score is None:
                continue
            
            if self.loss_type == 'infonce':
                # InfoNCE损失
                logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)
                labels = torch.zeros(pos_score.size(0), dtype=torch.long, device=pos_score.device)
                node_loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
            else:
                # 二元交叉熵损失（原始DGI）
                pos_loss = torch.nn.functional.logsigmoid(pos_score)
                neg_loss = torch.nn.functional.logsigmoid(-neg_score)
                node_loss = -(pos_loss + neg_loss).mean()
            
            total_loss += node_loss
            num_types += 1
        
        return total_loss / max(num_types, 1)  # 防止除零

    def get_embeddings(self, x_dict, edge_index_dict):
        """获取节点嵌入表示"""
        with torch.no_grad():
            embeddings = self.encoder(x_dict, edge_index_dict)
        return embeddings