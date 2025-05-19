from embedding import load_graph_network
from train import fit_transform
from embedData import embedData
import torch
import numpy as np
from scipy.sparse import identity, csr_matrix

if __name__=="__main__":
    torch.cuda.empty_cache()  # 清空缓存
    torch.backends.cuda.max_split_size_mb = 128  # 减少内存碎片
    # relationship_file = '/home/llma/wzy/UKB_net/GLIM-main/data/relationship_table_20250427.txt'
    # node_feature_file = '/home/llma/wzy/UKB_net/GLIM-main/data/node_feature_20250427.npy'
    # embedding_save_file = '/home/llma/wzy/UKB_net/GLIM-main/results/hmln_feature_20250427.npy'
    # relationship_file = '/home/llma/wzy/comorbidity/Data/merged_df_long_subset.txt'
    # node_feature_file = '/home/llma/wzy/comorbidity/Data/UKB_node_feature_gpt.npy'
    # embedding_save_file = '/home/llma/wzy/comorbidity/Data/UKB_feature.npy'
    
    # train
    relationship_file_train = '/home/llma/wzy/comorbidity/Data/train_data_splitClusters.txt'
    node_feature_file_train = '/home/llma/wzy/comorbidity/Data/UKB_node_feature_gpt_train_splitClusters.npy'
    features_train, adj_train = load_graph_network(relationship_file_train, node_feature_file_train)
    #adj = adj + np.eye(adj.shape[0])
    # 使用稀疏矩阵的方式添加自环
    adj_train = adj_train + identity(adj_train.shape[0], dtype=adj_train.dtype, format='csr')

    # validate
    relationship_file_val = '/home/llma/wzy/comorbidity/Data/val_data_splitClusters.txt'
    node_feature_file_val = '/home/llma/wzy/comorbidity/Data/UKB_node_feature_gpt_val_splitClusters.npy'
    features_val, adj_val = load_graph_network(relationship_file_val, node_feature_file_val)
    #adj = adj + np.eye(adj.shape[0])
    # 使用稀疏矩阵的方式添加自环
    adj_val = adj_val + identity(adj_val.shape[0], dtype=adj_val.dtype, format='csr')

    fit_transform(features=features_train, 
                  adj=adj_train, 
                  features_validation=features_val,
                  adj_validation=adj_val,
                  patience=100,
                  n_epoch=1000, 
                  n_hid=512,
                  device='cuda', 
                  learning_ratio = 0.00001,
                  tmp_path='UKB_best_dgi.pkl')    

    # embed all data
    relationship_file = '/home/llma/wzy/comorbidity/Data/merged_df_long_subset.txt'
    node_feature_file = '/home/llma/wzy/comorbidity/Data/UKB_node_feature_gpt.npy'
    embedding_save_file = '/home/llma/wzy/comorbidity/Data/UKB_feature.npy'
    features, adj = load_graph_network(relationship_file, node_feature_file)
    # 使用稀疏矩阵的方式添加自环
    adj = adj + identity(adj.shape[0], dtype=adj.dtype, format='csr')

    embedData(modelPath = 'UKB_best_dgi.pkl', 
              features = features, 
              adj = adj, 
              embeddingSavePath = embedding_save_file, 
              n_hid=512,
              device = 'cuda')