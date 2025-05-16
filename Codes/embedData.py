import models
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import issparse
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader, NeighborSampler

def embedData(modelPath, 
              features, 
              adj, 
              embeddingSavePath,
              n_hid=512,
              device = 'cuda'):
    device = torch.device(device)
    
    if issparse(adj):
        edge_index = torch.LongTensor(np.column_stack(adj.nonzero())).T
    else:
        edge_index = torch.LongTensor(np.where(adj > 0))
        
    edge_index = edge_index.contiguous()

    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    elif issparse(features):
        features = torch.FloatTensor(features.toarray())

    features = features.contiguous()

    data = torch_geometric.data.Data(x=features, edge_index=edge_index)
    data.n_id = torch.arange(data.num_nodes)

    features = features.to(device)
    features = features.contiguous()

    hid_units = n_hid
    feature_size = features.shape[1]
    
    summary = models.Summary('max')
    encoder = models.Encoder(feature_size, hid_units)
    corruption = models.corruption
    model = models.GraphLocalInfomax(hid_units, encoder, summary, corruption)
    model.load_state_dict(torch.load(modelPath))
    model = model.to(device)

    loader_for_embeds = NeighborLoader(
        data,
        num_neighbors=[-1],  # all neighbors
        batch_size=100,  # 批量大小
        shuffle=False
    )
    all_embeds = []
    with torch.no_grad():
        model.eval()
        for batch in loader_for_embeds:
            batch_features = features[batch.n_id].to(device)
            #batch_features = batch_features.contiguous()
            batch_edge_index = batch.edge_index.to(device)
            #batch_edge_index = batch_edge_index.contiguous()
            batch_embeds, _, _ = model(batch_features, batch_edge_index)
            # only retain top batch_size embeds
            batch_embeds = batch_embeds[:batch.batch_size].detach().cpu().numpy()
            all_embeds.append(batch_embeds)
            del batch_features, batch_edge_index
    embeds = np.concatenate(all_embeds, axis=0)
    np.save(embeddingSavePath, embeds)