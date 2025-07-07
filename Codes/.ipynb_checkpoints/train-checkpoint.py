import models

import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, HGTLoader

def fit_transform(data,
                  device='cuda', 
                  print_log=True, 
                  patience=30, 
                  n_epoch=300, 
                  learning_ratio = 0.001,
                  tmp_path='best_model.pkl'):
    
    device = torch.device(device)
    data = data.to(device)
    
    train_loader = HGTLoader(data, num_samples={key: [64] * 3 for key in data.node_types}, 
                             batch_size=128, shuffle=True,
                             #input_nodes=('participant', data['participant'].train_mask))
                             input_nodes=('author', data['author'].train_mask))
    val_loader = HGTLoader(data, num_samples={key: [64] * 3 for key in data.node_types}, 
                           batch_size=128,
                           #input_nodes=('participant', data['participant'].val_mask))
                           input_nodes=('author', data['author'].val_mask))
    
    summary = models.Summary(hidden_channels=512, aggr='mean', use_attention=False)
    encoder = models.Encoder(node_types=list(data.node_types),
                             edge_types=list(data.edge_types),
                             #in_channels_dict={'participant':1536, 'disease':1536, 'protein':1536, 'metabolite':1536}, 
                             in_channels_dict={'author':334, 'paper':4231, 'term':50, 'conference':1}, 
                             hidden_channels=512, dropout=0.5, num_layers=2)
    corruption = models.corruption
    model = models.GraphLocalInfomax(hidden_channels=512, encoder=encoder, summary=summary, corruption=corruption, dropout=0.5, loss_type='bce', temperature=1.0)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_ratio)

    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device, 'edge_index')
        model(batch.x_dict, batch.edge_index_dict)

    def train():
        model.train()
        total_loss = 0.0
        total_nodes = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            # # 调试：打印所有节点类型和特征形状
            # print(f"Batch节点类型: {list(batch.x_dict.keys())}")
            # if 'participant' not in batch.x_dict:
            #     print("错误：'participant' 节点特征缺失")
            #     return 0.0  # 临时返回，便于调试
            
            # for node_type, x in batch.x_dict.items():
            #     print(f"{node_type} 特征形状: {x.shape}")

            # # 调试：打印各节点类型的采样数量
            # node_counts = {nt: batch.x_dict[nt].size(0) for nt in batch.x_dict}
            # print(f"批次节点计数: {node_counts}")

            
            pos_z, neg_z, summary = model(batch.x_dict, batch.edge_index_dict)
            
            loss = model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()
            
            # 统计所有类型的节点数
            batch_nodes = 0
            for node_type, x in batch.x_dict.items():
                batch_nodes += x.size(0)  # x.size(0)为该类型节点数
            
            total_nodes += batch_nodes
            total_loss += loss.item() * batch_nodes
            del batch

        print('Total nodes: {:03d}'.format(total_nodes))
        return total_loss / total_nodes

    def validation():
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            total_nodes = 0
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                pos_z, neg_z, summary = model(batch.x_dict, batch.edge_index_dict)
            
                loss = model.loss(pos_z, neg_z, summary)
                
                # 统计所有类型的节点数
                batch_nodes = 0
                for node_type, x in batch.x_dict.items():
                    batch_nodes += x.size(0)  # x.size(0)为该类型节点数
                
                total_nodes += batch_nodes
                total_loss += loss.item() * batch_nodes
                del batch
                
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