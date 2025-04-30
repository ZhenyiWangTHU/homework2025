from embedding import load_graph_network
from train import fit_transform
import torch
import numpy as np
from scipy.sparse import identity, csr_matrix

if __name__=="__main__":
    torch.cuda.empty_cache()  # 清空缓存
    torch.backends.cuda.max_split_size_mb = 128  # 减少内存碎片
    # relationship_file = '/home/llma/wzy/UKB_net/GLIM-main/data/relationship_table_20250427.txt'
    # node_feature_file = '/home/llma/wzy/UKB_net/GLIM-main/data/node_feature_20250427.npy'
    # embedding_save_file = '/home/llma/wzy/UKB_net/GLIM-main/results/hmln_feature_20250427.npy'
    relationship_file = '/home/llma/wzy/comorbidity/Data/merged_df_long.txt'
    node_feature_file = '/home/llma/wzy/comorbidity/Data/UKB_node_feature.npy'
    embedding_save_file = '/home/llma/wzy/comorbidity/Data/UKB_feature.npy'
    
    features, adj = load_graph_network(relationship_file, node_feature_file)
    
    #adj = adj + np.eye(adj.shape[0])
    # 使用稀疏矩阵的方式添加自环
    adj = adj + identity(adj.shape[0], dtype=adj.dtype, format='csr')
    
    fit_transform(features, adj, embedding_save_file, device='cuda', tmp_path='UKB_best_dgi.pkl')