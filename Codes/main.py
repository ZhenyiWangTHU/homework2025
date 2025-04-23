from embedding import load_graph_network
from train import fit_transform
import torch
import numpy as np
from scipy.sparse import identity, csr_matrix

if __name__=="__main__":
    torch.cuda.empty_cache()  # 清空缓存
    torch.backends.cuda.max_split_size_mb = 128  # 减少内存碎片
    # relationship_file = './data/relationship_table.txt'
    # node_feature_file = './data/node_feature.npy'
    # embedding_save_file = './results/UKB_feature.npy'
    relationship_file = '../Data/merged_df_long.txt'
    node_feature_file = '../Data/UKB_node_feature.npy'
    embedding_save_file = '../Data/UKB_feature.npy'
    
    features, adj = load_graph_network(relationship_file, node_feature_file)
    
    #adj = adj + np.eye(adj.shape[0])
    # 使用稀疏矩阵的方式添加自环
    adj = adj + identity(adj.shape[0], dtype=adj.dtype, format='csr')
    
    fit_transform(features, adj, embedding_save_file, device='cuda')