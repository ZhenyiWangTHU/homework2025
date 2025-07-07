import os 
import numpy as np
import math
import random

import torch
import torch.nn as nn

from torch_geometric.nn import HGTConv, MessagePassing

import utils

# class Encoder(nn.Module):
#         def __init__(self, in_channels, hidden_channels, dropout=0.5):
#             super(Encoder, self).__init__()
#             self.conv = GATConv(in_channels, hidden_channels, heads=1, concat=False)
#             self.prelu = nn.PReLU(hidden_channels)
#             self.dropout = nn.Dropout(dropout)

#         def forward(self, x, edge_index):
#             out = self.conv(x, edge_index)
#             out = self.prelu(out)
#             out = self.dropout(out)
#             return out
class Encoder(nn.Module):
    def __init__(self, node_types, in_channels_dict, hidden_channels, dropout=0.5):
        # node_types：所有节点类型的列表（如 ['author', 'paper', 'institution']）
        # in_channels_dict：字典，记录每种节点类型的输入特征维度（如 {'author': 128, 'paper': 256, 'institution': 64}）
        super(Encoder, self).__init__()
        self.node_types = node_types
        self.in_channels_dict = in_channels_dict
        self.hidden_channels = hidden_channels
        
        # 为每种节点类型创建独立的线性投影层
        self.lin_dict = nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = nn.Linear(in_channels_dict[node_type], hidden_channels)
        
        # HGT卷积层，处理异构图
        self.conv = HGTConv(
            hidden_channels, # 输入特征维度
            hidden_channels, # 输出特征维度
            metadata=(node_types, utils.get_edge_types()),  # 元数据：节点类型和边类型
            heads=8, 
            dropout=dropout
        )
        self.prelu = nn.PReLU(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        # 1. 特征投影：将不同节点类型的特征映射到统一维度
        x_proj_dict = {
            node_type: self.lin_dict[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # 2. 异构图卷积：通过HGTConv更新节点表示
        out_dict = self.conv(x_proj_dict, edge_index_dict)
        
        # 3. 后处理：应用激活函数和Dropout
        for node_type in self.node_types:
            out_dict[node_type] = self.dropout(self.prelu(out_dict[node_type]))
        
        return out_dict

# class Summary(MessagePassing):
#       # aggregation type: 1.mean, 2.max, 3.sum
#       def __init__(self, aggr='max'):
#           super().__init__(aggr=aggr)

#       def forward(self, x, edge_index):
#           return self.propagate(edge_index, x=x)

#       def message(self, x_j):
#           return x_j

class Summary(MessagePassing):
    """
    异构图节点级Summary - 按边类型聚合邻域信息
    
    功能：
    - 对每种边类型分别进行消息传递
    - 将不同类型邻居的信息汇总到目标节点
    - 支持多种聚合方式（max、mean、sum）
    """
    def __init__(self, aggr='mean'):
        """
        初始化摘要器
        
        参数：
            aggr: 聚合函数类型，可选 'max', 'mean', 'sum'
        """
        super().__init__(aggr=aggr)  # 调用父类初始化，设置聚合方式
        
    def forward(self, x_dict, edge_index_dict):
        """
        前向传播：计算所有节点的邻域Summary
        
        参数：
            x_dict: 节点特征字典 {节点类型: 特征张量}
            edge_index_dict: 边索引字典 {(源类型, 关系, 目标类型): 边索引张量}
        
        返回：
            out_dict: 节点摘要字典 {节点类型: 摘要张量}
        """
        # 初始化输出字典，初始值为节点原始特征的副本
        out_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
        
        # 遍历每种边类型，分别进行消息传递
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type  # 边类型三元组：(源类型, 关系, 目标类型)
            
            # 获取源节点和目标节点的特征
            src_features = x_dict[src_type]
            dst_features = x_dict[dst_type]
            
            # 执行消息传递：
            # 1. 根据edge_index找到源节点和目标节点
            # 2. 从源节点提取特征并生成消息
            # 3. 将消息传递给目标节点并聚合
            out = self.propagate(
                edge_index=edge_index,
                x=(src_features, dst_features)  # 传递源和目标节点特征
            )
            
            # 将本次边类型的聚合结果累加到目标节点的摘要中
            out_dict[dst_type] += out
            
        return out_dict
    
    def message(self, x_j, x_i):
        """
        消息函数：定义如何从源节点生成消息
        
        参数：
            x_j: 源节点特征张量（自动根据edge_index提取）
            
        返回：
            直接返回源节点特征作为消息
        """
        return x_j  # 简单方案：直接传递源节点特征

# def corruption(x, edge_index):
#     return x[torch.randperm(x.size(0))], edge_index

def corruption(x_dict, edge_index_dict):
    # 对每种节点类型进行打乱
    corrupted_x_dict = {
        node_type: x[torch.randperm(x.size(0))]
        for node_type, x in x_dict.items()
    }
    return corrupted_x_dict, edge_index_dict

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

# class GraphLocalInfomax(torch.nn.Module):
#     def __init__(self, hidden_channels, encoder, summary, corruption, dropout=0.5):
#         super(GraphLocalInfomax, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.encoder = encoder
#         self.summary = summary
#         self.corruption = corruption
#         self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
#         self.dropout = nn.Dropout(dropout)
#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.encoder)
#         reset(self.summary)
#         uniform(self.hidden_channels, self.weight)

#     def forward(self, x, edge_index):
#         pos_z = self.encoder(x, edge_index)
#         cor = self.corruption(x, edge_index)
#         cor = cor if isinstance(cor, tuple) else (cor, )
#         neg_z = self.encoder(*cor)
#         summary = self.summary(pos_z, edge_index)
#         summary = self.dropout(summary)
#         return pos_z, neg_z, summary

#     def discriminate(self, z, summary, sigmoid=True):
#         value = torch.sum(torch.mul(z, torch.matmul(summary, self.weight)), dim=1)
#         return value

#     def loss(self, pos_z, neg_z, summary):
#         pos_loss = self.discriminate(pos_z, summary)
#         neg_loss = self.discriminate(neg_z, summary)
#         return -torch.log(1/(1 + torch.exp(torch.clamp(neg_loss-pos_loss, max=10)))).mean()

class GraphLocalInfomax(torch.nn.Module):
    def __init__(self, hidden_channels, encoder, summary, corruption, dropout=0.5):
        super(GraphLocalInfomax, self).__init__()
        self.hidden_channels = hidden_channels
        self.encoder = encoder
        self.summary = summary
        self.corruption = corruption
        
        # 为每种节点类型创建独立的权重矩阵
        self.weight_dict = nn.ModuleDict()
        for node_type in encoder.node_types:
            self.weight_dict[node_type] = nn.Parameter(
                torch.Tensor(hidden_channels, hidden_channels)
            )
            
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.summary)
        for node_type in self.weight_dict:
            uniform(self.hidden_channels, self.weight_dict[node_type])

    def forward(self, x_dict, edge_index_dict):
        # 编码原始图得到正样本表示
        pos_z_dict = self.encoder(x_dict, edge_index_dict)
        
        # 生成负样本（打乱节点特征）
        corrupted_x_dict, corrupted_edge_index_dict = self.corruption(x_dict, edge_index_dict)
        # 使用相同的边结构，但特征被打乱
        neg_z_dict = self.encoder(corrupted_x_dict, edge_index_dict)
        
        # 生成图摘要（节点级表示聚合）
        summary_dict = self.summary(pos_z_dict, edge_index_dict)
        for node_type in summary_dict:
            summary_dict[node_type] = self.dropout(summary_dict[node_type])
            
        return pos_z_dict, neg_z_dict, summary_dict

    def discriminate(self, z_dict, summary_dict):
        # 为每种节点类型计算互信息得分
        scores_dict = {}
        for node_type in z_dict:
            # 获取当前节点类型的表示和摘要
            z = z_dict[node_type]
            summary = summary_dict[node_type]
            weight = self.weight_dict[node_type]
            
            # 计算内积得分: z · (W · summary)
            value = torch.sum(torch.mul(z, torch.matmul(summary, weight)), dim=1)
            scores_dict[node_type] = value
            
        return scores_dict

    def loss(self, pos_z_dict, neg_z_dict, summary_dict):
        # 计算所有节点类型的互信息损失
        total_loss = 0
        for node_type in pos_z_dict:
            pos_score = self.discriminate(
                {node_type: pos_z_dict[node_type]},
                {node_type: summary_dict[node_type]}
            )[node_type]
            
            neg_score = self.discriminate(
                {node_type: neg_z_dict[node_type]},
                {node_type: summary_dict[node_type]}
            )[node_type]
            
            # 使用稳定的sigmoid交叉熵公式
            node_loss = -torch.log(
                1 / (1 + torch.exp(torch.clamp(neg_score - pos_score, max=10)))
            ).mean()
            
            total_loss += node_loss
            
        # 对所有节点类型的损失取平均
        return total_loss / len(pos_z_dict)