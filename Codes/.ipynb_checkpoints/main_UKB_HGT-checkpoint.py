#from embedding import load_graph_network
from train import fit_transform
from embedData import embedData
import torch
import numpy as np
from scipy.sparse import identity, csr_matrix

if __name__=="__main__":
    torch.cuda.empty_cache()  # 清空缓存
    torch.backends.cuda.max_split_size_mb = 128  # 减少内存碎片
    
    # 读取保存的图数据
    # HGData = torch.load('../Data/HGData_Undir_with_self_loops.pt', weights_only=False)
    HGData = torch.load('/home/llma/wzy/comorbidity/Data/HGData_Undir_with_self_loops_add_default_edge_attr.pt', weights_only=False)

    fit_transform(data=HGData,
                  device='cuda', 
                  print_log=True, 
                  patience=30, 
                  n_epoch=300, 
                  learning_ratio = 0.001,
                  tmp_path='UKB_HGT_model.pkl')  

    # embedData(modelPath = './UKB_best_dgi_all_GAT_add_filtered_disGeNet_disease_disease_PPI_proteinTopEid_metaboliteTopEid.pkl', 
    #           features = features, 
    #           adj = adj, 
    #           embeddingSavePath = embedding_save_file, 
    #           n_hid=512,
    #           device = 'cuda')