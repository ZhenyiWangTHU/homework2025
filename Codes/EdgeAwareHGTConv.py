import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index


class EdgeAwareHGTConv(MessagePassing):
    r"""The Edge Aware Heterogeneous Graph Transformer (HGT) operator based on the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using EdgeAwareHGTConv, see ``_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        edge_attr_sizes: Dict[EdgeType, int] = None,  # 新增：边属性维度字典
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i
            for i, edge_type in enumerate(metadata[1])
        }
        
        ### process edge attr ###
        self.edge_attr_sizes = edge_attr_sizes or {}  # 边类型到属性维度的映射
        self.edge_attr_mlp = nn.ModuleDict()  # 边属性MLP变换层
        # 为每个有属性的边类型创建线性层，将边属性映射到注意力空间
        for edge_type in self.edge_types:
            if edge_type in self.edge_attr_sizes and self.edge_attr_sizes[edge_type] > 0:
                # 映射维度为 heads（与多头注意力维度匹配）
                edge_key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
                input_dim = self.edge_attr_sizes[edge_type]
                
                # 使用两层MLP，中间层维度设为输入维度和输出维度的平均值，可根据需要调整
                hidden_dim = (input_dim + heads) // 2
                
                self.edge_attr_mlp[edge_key] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),  # 使用ReLU激活函数
                    nn.Linear(hidden_dim, heads)
                )
        ###
        
        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels,
                                        self.out_channels * 3)

        self.out_lin = HeteroDictLinear(self.out_channels, self.out_channels,
                                        types=self.node_types)

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False,
                                  is_sorted=True)

        self.skip = ParameterDict({
            node_type: Parameter(torch.empty(1))
            for node_type in self.node_types
        })

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type = '__'.join(edge_type)
            self.p_rel[edge_type] = Parameter(torch.empty(1, heads))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        """Concatenates a dictionary of features."""
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self, k_dict: Dict[str, Tensor], v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        """Constructs the source node representations."""
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        # Flatten into a single tensor with shape [num_edge_types * heads, D]:
        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            # construct type_vec for curr edge_type with shape [H, D]
            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = torch.arange(H, dtype=torch.long).view(-1, 1).repeat(
                1, N) * num_edge_types + edge_type_offset

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],  # Support both.
        edge_attr_dict: Optional[Dict[EdgeType, Tensor]] = None  # 新增：边属性字典
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, torch.Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[Tuple[str, str, str], torch.Tensor]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :class:`torch.Tensor` of
                shape :obj:`[2, num_edges]` or a
                :class:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[torch.Tensor]]` - The output node
            embeddings for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        
        F = self.out_channels
        H = self.heads
        D = F // H

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

        # Compute K, Q, V over node types:
        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict)

        ### 处理边属性：将边属性转换为注意力偏置 ###
        edge_attr_processed = {}

        for edge_type, attr in edge_attr_dict.items():
            # if edge_type not in self.edge_attr_sizes or self.edge_attr_sizes[edge_type] == 0:
            #     continue  # 跳过无属性的边类型
            # 用线性层映射边属性到多头维度（[num_edges, heads]）
            key = f"{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            if key in self.edge_attr_mlp:
                # 无论原始维度，都映射到[E, H]
                edge_attr_processed[edge_type] = self.edge_attr_mlp[key](attr)
        ### 

        ### 构造 bipartite edge_index 时融入处理后的边属性 ###
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict, src_offset, dst_offset, 
            edge_attr_dict={**self.p_rel,** edge_attr_processed},  # 合并原有p_rel和新边属性
            num_nodes=k.size(0))
        ###
        
        # edge_index, edge_attr = construct_bipartite_edge_index(
        #     edge_index_dict, src_offset, dst_offset, edge_attr_dict=self.p_rel,
        #     num_nodes=k.size(0))

        out = self.propagate(edge_index, k=k, q=q, v=v, edge_attr=edge_attr)

        # Reconstruct output node embeddings dict:
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings:
        a_dict = self.out_lin({
            k:
            torch.nn.functional.gelu(v) if v is not None else v
            for k, v in out_dict.items()
        })

        # Iterate over node types:
        for node_type, out in out_dict.items():
            out = a_dict[node_type]

            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict
        
    # original message
    # def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
    #             index: Tensor, ptr: Optional[Tensor],
    #             size_i: Optional[int]) -> Tensor:
    #     alpha = (q_i * k_j).sum(dim=-1) * edge_attr
    #     alpha = alpha / math.sqrt(q_i.size(-1))
    #     alpha = softmax(alpha, index, ptr, size_i)
    #     out = v_j * alpha.view(-1, self.heads, 1)
    #     return out.view(-1, self.out_channels)

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        attn_score = (q_i * k_j).sum(dim=-1)  # [E, H]
        
        # 边类型偏置（self.p_rel）和边属性（edge_attr_processed）已合并到edge_attr中
        # 此时edge_attr的维度为[E, H]，直接作为权重
        edge_weight = torch.sigmoid(edge_attr)  # 归一化到0-1
        
        # 融合注意力得分和边权重
        alpha = attn_score * edge_weight
        
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')