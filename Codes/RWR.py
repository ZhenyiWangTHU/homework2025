from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_scatter import scatter_add
from utils import classify_element

class RWRPyG:
    """使用PyTorch Geometric实现的Random Walk with Restart"""
    
    def __init__(self, relationship, feature_matrix, node_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化RWRPyG类
        
        参数:
            relationship: pd dataframe，形状为(n_edges, 3)
            feature_matrix: numpy数组，形状为(n_samples, n_features)
            node_names: 节点名称列表，长度为n_samples
            device: 计算设备，'cpu'或'cuda'
        """
        self.relationship = relationship
        self.feature_matrix = feature_matrix
        self.node_names = node_names  # 新增节点名称列表
        self.name_to_idx = {name: i for i, name in enumerate(node_names)}  # 名称到索引的映射
        self.device = device
        self.n_samples = feature_matrix.shape[0]
        self.edge_index = None
        self.edge_weight = None
        self.rwr_model = None
        
    def build_graph(self, k_neighbors=10, metric='minkowski', p=2, symmetrize=True):
        """
        基于K近邻构建相似性图
        
        参数:
            k_neighbors: K近邻参数
            metric: 距离度量
            p: float, Power parameter for the Minkowski metric
            symmetrize: 是否确保图的对称性
        """
        print(f"构建相似性图 (k={k_neighbors}, metric={metric})...")
        
        # 计算K近邻图
        adj_matrix = kneighbors_graph(
            X=self.feature_matrix, n_neighbors=k_neighbors, mode='distance', 
            metric=metric, p=p, include_self=False
        )
        
        # 获取node1和node2的索引
        node1_indices = [self.name_to_idx.get(name, -1) for name in self.relationship['node1']]
        node2_indices = [self.name_to_idx.get(name, -1) for name in self.relationship['node2']]
        
        # 过滤掉不在node_names中的节点
        valid_pairs = [(i, j) for i, j in zip(node1_indices, node2_indices) if i != -1 and j != -1]
        
        # 创建稀疏矩阵
        rows, cols = zip(*valid_pairs)
        data = np.ones(len(valid_pairs))
        rel_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(self.n_samples, self.n_samples))
        
        # 合并两个矩阵
        adj_matrix = adj_matrix + rel_matrix
        
        # 确保没有重复的边（如果有）
        adj_matrix = adj_matrix.tocsr()
        adj_matrix.sum_duplicates()
        
        # 将所有边权重设置为1
        adj_matrix.data = np.ones_like(adj_matrix.data)
        
        # 确保图的对称性
        if symmetrize:
            adj_matrix = 0.5 * (adj_matrix + adj_matrix.T)
            adj_matrix.data = np.ones_like(adj_matrix.data)
        
        # 转换为PyG格式
        self.edge_index, self.edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(adj_matrix)
        self.edge_index = self.edge_index.to(self.device)
        self.edge_weight = self.edge_weight.to(self.device).float()
        
        print(f"图构建完成: {self.n_samples} 个节点, {self.edge_index.shape[1]} 条边")
        return self
    
    def compute_rwr(self, seeds, alpha=0.85, max_iter=100, tol=1e-6):
        """
        计算RWR得分
        
        参数:
            seeds: 种子节点的索引列表
            alpha: 重启概率
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            numpy数组，每个节点的RWR得分
        """
        if self.edge_index is None:
            raise ValueError("请先调用build_graph方法构建图")
            
        print(f"计算RWR (alpha={alpha}, num of seeds={len(seeds)})...")
        
        # 初始化RWR向量
        r = torch.zeros(self.n_samples, dtype=torch.float, device=self.device)
        r[seeds] = 1.0 / len(seeds)
        r_old = r.clone()
        
        # 构建转移矩阵（归一化邻接矩阵）
        row, col = self.edge_index

        # 按节点度归一化边权重（关键修改）
        # 计算每个节点的度（出边数量）
        deg = scatter_add(self.edge_weight, row, dim=0, dim_size=self.n_samples)
        
        # 避免除零错误（虽然在无向图中每个节点度至少为1）
        deg_safe = torch.clamp(deg, min=1e-10)
        
        # 归一化边权重：每条边的权重 = 1/源节点的度
        edge_weight_norm = self.edge_weight / deg_safe[row]
        
        # 迭代计算RWR
        for i in range(max_iter):
            # 传播步骤
            r_new = scatter_add(
                src=edge_weight_norm * r_old[col],
                index=row, dim=0, dim_size=self.n_samples
            )
            
            # 重启步骤
            r_new = (1 - alpha) * r_new  # 缩放传播结果
            r_new[seeds] += alpha / len(seeds)  
            
            # 检查收敛
            diff = torch.norm(r_new - r_old, p=1)
            if diff < tol:
                print(f"RWR在第 {i+1} 次迭代后收敛 (diff={diff:.6e})")
                break
                
            r_old = r_new.clone()
            
        return r_new.cpu().numpy()

def get_top_nodes(rwr, rwr_scores, select_index, top_k=50, node_type='all', return_stats=False):
    """
    获取Top-K节点，支持按类型筛选，并可返回类型统计信息
    
    参数:
        rwr: RWRPyG实例
        rwr_scores: RWR得分数组
        select_index: 种子节点索引
        top_k: 要返回的节点数量
        node_type: 节点类型，可选值: 'all'、'icd10'、'eid'、'protein'、'metabolite'
        return_stats: 是否返回类型统计信息
        
    返回:
        如果return_stats=False: (top_node_names, top_node_scores)
        如果return_stats=True: (top_node_names, top_node_scores, type_stats)
    """
    # 获取得分大于0的节点索引（排除种子节点自身）
    reachable_indices = np.where((rwr_scores > 0) & (np.arange(len(rwr_scores)) != select_index))[0]
    
    # 按得分排序
    sorted_indices = reachable_indices[np.argsort(rwr_scores[reachable_indices])[::-1]]
    
    # 按类型筛选
    if node_type != 'all':
        filtered_indices = []
        for idx in sorted_indices:
            node_name = rwr.node_names[idx]
            if classify_element(node_name) == node_type:
                filtered_indices.append(idx)
        sorted_indices = np.array(filtered_indices)
    
    # 限制数量
    top_indices = sorted_indices[:top_k]
    
    # 获取对应的节点名称和得分
    top_node_names = [rwr.node_names[idx] for idx in top_indices]
    top_node_scores = rwr_scores[top_indices]
    
    # 统计不同类型的节点数量
    if return_stats:
        type_stats = {
            'icd10': 0,
            'eid': 0,
            'protein': 0,
            'metabolite': 0,
            'total': 0
        }
        
        # 统计所有可达节点（得分>0）
        for idx in reachable_indices:
            node_name = rwr.node_names[idx]
            node_class = classify_element(node_name)
            if node_class in type_stats:
                type_stats[node_class] += 1
            type_stats['total'] += 1
        
        return top_node_names, top_node_scores, type_stats
    else:
        return top_node_names, top_node_scores

## The definition of transition probability did not involve the edges in the original input graph.
# class RWRPyG:
#     """使用PyTorch Geometric实现的Random Walk with Restart"""
    
#     def __init__(self, relationship, feature_matrix, node_names, device='cuda' if torch.cuda.is_available() else 'cpu'):
#         """
#         初始化RWRPyG类
        
#         参数:
#             relationship: 
#             feature_matrix: numpy数组，形状为(n_samples, n_features)
#             node_names: 节点名称列表，长度为n_samples
#             device: 计算设备，'cpu'或'cuda'
#         """
#         self.relationship = relationship
#         self.feature_matrix = feature_matrix
#         self.node_names = node_names  # 新增节点名称列表
#         self.name_to_idx = {name: i for i, name in enumerate(node_names)}  # 名称到索引的映射
#         self.device = device
#         self.n_samples = feature_matrix.shape[0]
#         self.edge_index = None
#         self.edge_weight = None
#         self.rwr_model = None
        
#     def build_graph(self, k_neighbors=10, metric='minkowski', p=2, threshold=None, symmetrize=True):
#         """
#         基于K近邻构建相似性图
        
#         参数:
#             k_neighbors: K近邻参数
#             metric: 距离度量
#             p: float, Power parameter for the Minkowski metric
#             threshold: 相似性阈值，低于此值的边将被忽略
#             symmetrize: 是否确保图的对称性
#         """
#         print(f"构建相似性图 (k={k_neighbors}, metric={metric})...")
        
#         # 计算K近邻图
#         adj_matrix = kneighbors_graph(
#             X=self.feature_matrix, n_neighbors=k_neighbors, mode='distance', 
#             metric=metric, p=p, include_self=False
#         )
        
#         # 转换为相似度

#         # 余弦相似度 = 1 - 余弦距离
#         # adj_matrix.data = 1.0 - adj_matrix.data

#         # 对于欧氏距离等，使用高斯核
#         adj_matrix.data = np.exp(-adj_matrix.data)
        
#         # 应用阈值
#         if threshold is not None:
#             adj_matrix.data[adj_matrix.data < threshold] = 0
#             adj_matrix.eliminate_zeros()
        
#         # 确保图的对称性
#         if symmetrize:
#             adj_matrix = 0.5 * (adj_matrix + adj_matrix.T)
        
#         # 转换为PyG格式
#         self.edge_index, self.edge_weight = torch_geometric.utils.from_scipy_sparse_matrix(adj_matrix)
#         self.edge_index = self.edge_index.to(self.device)
#         self.edge_weight = self.edge_weight.to(self.device).float()
        
#         print(f"图构建完成: {self.n_samples} 个节点, {self.edge_index.shape[1]} 条边")
#         return self
    
#     def compute_rwr(self, seeds, alpha=0.85, max_iter=100, tol=1e-6):
#         """
#         计算RWR得分
        
#         参数:
#             seeds: 种子节点的索引列表
#             alpha: 重启概率
#             max_iter: 最大迭代次数
#             tol: 收敛容差
            
#         返回:
#             numpy数组，每个节点的RWR得分
#         """
#         if self.edge_index is None:
#             raise ValueError("请先调用build_graph方法构建图")
            
#         print(f"计算RWR (alpha={alpha}, seeds={len(seeds)})...")
        
#         # 初始化RWR向量
#         r = torch.zeros(self.n_samples, dtype=torch.float, device=self.device)
#         r[seeds] = 1.0 / len(seeds)
#         r_old = r.clone()
        
#         # 构建转移矩阵（归一化邻接矩阵）
#         row, col = self.edge_index
#         deg = torch_geometric.utils.degree(row, self.n_samples, dtype=self.edge_weight.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         edge_weight_norm = deg_inv_sqrt[row] * self.edge_weight * deg_inv_sqrt[col]
        
#         # 迭代计算RWR
#         for i in range(max_iter):
#             # 传播步骤
#             r_new = scatter_add(
#                 src=edge_weight_norm * r_old[col],
#                 index=row, dim=0, dim_size=self.n_samples
#             )
            
#             # 重启步骤
#             r_new = (1 - alpha) * r_new  # 缩放传播结果
#             r_new[seeds] += alpha / len(seeds)  
            
#             # 检查收敛
#             diff = torch.norm(r_new - r_old, p=1)
#             if diff < tol:
#                 print(f"RWR在第 {i+1} 次迭代后收敛 (diff={diff:.6e})")
#                 break
                
#             r_old = r_new.clone()
            
#         return r_new.cpu().numpy()