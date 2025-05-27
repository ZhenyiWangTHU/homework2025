import models
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import issparse
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader, NeighborSampler

# def fit_transform(features, 
#                   adj, 
#                   save_path, 
#                   n_hid=512, 
#                   num_layers=6, 
#                   device='cuda', 
#                   print_log=True, 
#                   patience=50, 
#                   n_epoch=300, 
#                   tmp_path='best_dgi.pkl'):
#     device = torch.device(device)
    
#     # Self loop is required.
#     # 处理稀疏矩阵获取边索引
#     if issparse(adj):
#         edge_index = torch.LongTensor(np.column_stack(adj.nonzero())).T.to(device)
#     else:
#         edge_index = torch.LongTensor(np.where(adj > 0)).to(device)
    
#     # Features_shape = n_sample * n_dim
#     if isinstance(features, np.ndarray):
#         features = torch.FloatTensor(features).to(device)
#     elif issparse(features):
#         features = torch.FloatTensor(features.toarray()).to(device)
    
#     hid_units = n_hid
#     feature_size = features.shape[1]
    
#     summary = models.Summary('max').to(device)
#     encoder = models.Encoder(feature_size, hid_units).to(device)
#     corruption = models.corruption
#     model = models.GraphLocalInfomax(hid_units, encoder, summary, corruption).to(device)
#     #print(model)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
#     def train():
#         model.train()
#         optimizer.zero_grad()
#         pos_z, neg_z, summary = model(features, edge_index)
#         loss = model.loss(pos_z, neg_z, summary)
#         loss.backward()
#         optimizer.step()
#         return loss.item()
    
    
#     best = 1e9
#     for epoch in range(1, n_epoch+1):
#         loss = train()
#         if loss < best:
#             best = loss
#             best_t = epoch
#             cnt_wait = 0
#             torch.save(model.state_dict(), tmp_path)
#         else:
#             cnt_wait += 1

#         if cnt_wait == patience:
#             print('Early stopping!')
#             break

#         if print_log:
#             print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss))
            
#     print('Loading {}th epoch'.format(best_t))
#     model.load_state_dict(torch.load(tmp_path))
    
#     with torch.no_grad():
#         embeds, _, _ = model(features, edge_index)
#         embeds = embeds.detach().cpu().numpy()
#     np.save(save_path, embeds)
    
#     return embeds

## Batch
def fit_transform(features, 
                  adj,
                  features_validation,
                  adj_validation,
                  n_hid=512, 
                  device='cuda', 
                  print_log=True, 
                  patience=100, 
                  n_epoch=300, 
                  learning_ratio = 0.00001,
                  tmp_path='best_dgi.pkl'):
    
    device = torch.device(device)
    
    # Self loop is required.
    # 处理稀疏矩阵获取边索引
    if issparse(adj):
        edge_index = torch.LongTensor(np.column_stack(adj.nonzero())).T
    else:
        edge_index = torch.LongTensor(np.where(adj > 0))
    
    if issparse(adj_validation):
        edge_index_validation = torch.LongTensor(np.column_stack(adj_validation.nonzero())).T
    else:
        edge_index_validation = torch.LongTensor(np.where(adj_validation > 0))
        
    # 确保 edge_index 是连续的
    edge_index = edge_index.contiguous()
    edge_index_validation = edge_index_validation.contiguous()
    
    # Features_shape = n_sample * n_dim
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    elif issparse(features):
        features = torch.FloatTensor(features.toarray())

    if isinstance(features_validation, np.ndarray):
        features_validation = torch.FloatTensor(features_validation)
    elif issparse(features_validation):
        features_validation = torch.FloatTensor(features_validation.toarray())
        
     # 确保 features 是连续的
    features = features.contiguous()
    features_validation = features_validation.contiguous()
    
    hid_units = n_hid
    feature_size = features.shape[1]
    
    summary = models.Summary('max')
    encoder = models.Encoder(feature_size, hid_units, dropout=0.5)
    corruption = models.corruption
    model = models.GraphLocalInfomax(hid_units, encoder, summary, corruption, dropout=0.5)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_ratio)

    # # train for neighborLoader
    # def train():
    #     model.train()
    #     total_loss = 0
    #     for batch in loader:
    #         optimizer.zero_grad()
    #         #print(f"Min node ID: {batch.n_id.min()}, Max node ID: {batch.n_id.max()}, Features shape: {features.shape}")
    #         batch_features = features[batch.n_id].to(device)
    #         # 确保 batch_features 是连续的
    #         batch_features = batch_features.contiguous()
    #         batch_edge_index = batch.edge_index.to(device)
    #         # 确保 batch_edge_index 是连续的
    #         batch_edge_index = batch_edge_index.contiguous()
    #         #print(f"Batch features shape: {batch_features.shape}")
    #         #print(f"batch_edge_index.shape: {batch_edge_index.shape}")
    #         #print(f"Minimum node ID in edge_index: {torch.min(batch_edge_index)}")
    #         #print(f"Maximum node ID in edge_index: {torch.max(batch_edge_index)}")
    #         #print(f"Original node IDs len: {len(batch.n_id)}")
    #         pos_z, neg_z, summary = model(batch_features, batch_edge_index)
    #         loss = model.loss(pos_z, neg_z, summary)
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         del batch_features, batch_edge_index
    #         #print('one batch!')
    #     return total_loss / len(loader)

     # 创建 NeighborLoader
    data = torch_geometric.data.Data(x=features, edge_index=edge_index)
    data.n_id = torch.arange(data.num_nodes)
    data_validation = torch_geometric.data.Data(x=features_validation, edge_index=edge_index_validation)
    data_validation.n_id = torch.arange(data_validation.num_nodes)

    features = features.to(device)
    features = features.contiguous()
    features_validation = features_validation.to(device)
    features_validation = features_validation.contiguous()
    
    # loader = NeighborLoader(
    #     data,
    #     num_neighbors=[2000,2000],  # 每层采样的邻居数量
    #     batch_size=5,  # 批量大小
    #     shuffle=True
    # )

    ## Cluster-GCN
    cluster_data = ClusterData(data, num_parts=2000, recursive=False, save_dir=None)
    train_loader = ClusterLoader(cluster_data, batch_size=200, shuffle=True, num_workers=0)
    #subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024, shuffle=False, num_workers=12)
    
    # train for Cluster-GCN
    def train():
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            batch_features = features[batch.n_id]
            # 确保 batch_features 是连续的
            batch_features = batch_features.contiguous()
            batch_edge_index = batch.edge_index
            # 确保 batch_edge_index 是连续的
            batch_edge_index = batch_edge_index.contiguous()
            pos_z, neg_z, summary = model(batch_features, batch_edge_index)
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            nodes = batch.n_id.size(0)
            total_nodes += nodes
            total_loss += loss.item() * nodes
            del batch_features, batch_edge_index
        print('Total nodes: {:03d}'.format(total_nodes))
        return total_loss / total_nodes

    # loader_for_validation = NeighborLoader(
    #     data_validation,
    #     num_neighbors=[-1],  # all neighbors
    #     batch_size=100,  # 批量大小
    #     shuffle=False
    # )

    # def validation():
    #     model.eval()
    #     with torch.no_grad():
    #         total_loss = 0.0
    #         total_nodes = 0
    #         for batch in loader_for_validation:
    #             batch = batch.to(device)
    #             batch_features = features_validation[batch.n_id]
    #             # 确保 batch_features 是连续的
    #             batch_features = batch_features.contiguous()
    #             batch_edge_index = batch.edge_index
    #             # 确保 batch_edge_index 是连续的
    #             batch_edge_index = batch_edge_index.contiguous()
    #             pos_z, neg_z, summary = model(batch_features, batch_edge_index)
    #             loss = model.loss(pos_z, neg_z, summary)
    #             nodes = batch.n_id.size(0)
    #             total_nodes += nodes
    #             total_loss += loss.item() * nodes
    #             del batch_features, batch_edge_index
    #         return total_loss / total_nodes

    ## Cluster-GCN
    validation_cluster_data = ClusterData(data_validation, num_parts=500, recursive=False, save_dir=None)
    validation_loader = ClusterLoader(validation_cluster_data, batch_size=50, shuffle=False, num_workers=0)
    
    def validation():
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_nodes = 0
            for batch in validation_loader:
                batch = batch.to(device)
                batch_features = features_validation[batch.n_id]
                # 确保 batch_features 是连续的
                batch_features = batch_features.contiguous()
                batch_edge_index = batch.edge_index
                # 确保 batch_edge_index 是连续的
                batch_edge_index = batch_edge_index.contiguous()
                pos_z, neg_z, summary = model(batch_features, batch_edge_index)
                loss = model.loss(pos_z, neg_z, summary)
                nodes = batch.n_id.size(0)
                total_nodes += nodes
                total_loss += loss.item() * nodes
                del batch_features, batch_edge_index
            return total_loss / total_nodes
    
    best_train = 1e9
    best_validation = 1e9
    for epoch in range(1, n_epoch+1):
        loss_train = train()
        loss_validation = validation()
        if loss_train < best_train:
            best_train = loss_train
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), tmp_path)
            if loss_validation < best_validation:
                best_validation = loss_validation
                val_wait = 0
            else:
                val_wait += 1
        else:
            cnt_wait += 1
            
        if val_wait == 10:
            print('Change the learning ratio for validation loss!')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
            
        if cnt_wait == patience:
            print('Early stopping for train loss!')
            break
            
        if print_log:
            print('Epoch: {:03d}, Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, loss_train, loss_validation))
            
    print('The best model: {}th epoch'.format(best_t))