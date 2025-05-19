#import glim
import numpy as np
import os
from scipy.sparse import csr_matrix
from utils import N2V
import pandas as pd
from fastnode2vec import Graph, Node2Vec 
#import glim.utils as utils
#import argparse

# parser = argparse.ArgumentParser(
#   description='GLIM',
#   formatter_class = argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--relationship-file', type=str, default='./data/relationship_table.txt')
# parser.add_argument('--node-feature-file', type=str, default='./data/node_feature.npy')
# parser.add_argument('--embedding-save-file', type=str, default='./results/hmln_feature.npy')

# args = parser.parse_args()

## For dense matrix
# def load_graph_network(adj_path, feature_path):
#     X, A, Y = [], None, []
#     n_node = 0

#     # Acquire Edges
#     edge_list = []
#     node_list = []
#     node_type = {}

#     with open(adj_path, 'rt', encoding='utf-8') as f:
#         next(f)
#         for line in f.readlines():
#             node1, node2, *_ = line.strip().split('\t')
#             edge_list.append((node1, node2))
#             node_list.extend([node1, node2])
                
#     node_map = {item:i for i, item in enumerate(sorted(list(set(node_list))))}
#     n_node = len(node_map)
#     print('Loading {} nodes.'.format(n_node))
#     A = np.zeros((n_node, n_node))
#     for node1, node2 in edge_list:
#         A[node_map[node1], node_map[node2]] = 1
#         A[node_map[node2], node_map[node1]] = 1
#     A = np.float32(A)
    
#     ####################################################
#     #            Acquire Features                      #
#     ####################################################

#     if os.path.exists(feature_path):
#         X = np.load(feature_path)
#     else:
#         X = np.float32(N2V(A, 512, 4, 1))
#         np.save(feature_path, X)
    
#     return X, A

## For sparse matrix
# def load_graph_network(adj_path, feature_path):
#     X, A, Y = [], None, []
#     n_node = 0

#     # Acquire Edges
#     edge_list = []
#     node_list = []
#     node_type = {}
#     relationship_list = []

#     with open(adj_path, 'rt', encoding='utf-8') as f:
#         next(f)
#         for line in f.readlines():
#             node1, node2, relationship, *_ = line.strip().split('\t')
#             edge_list.append((node1, node2))
#             node_list.extend([node1, node2])
#             relationship_list.append(relationship)

#     node_map = {item: i for i, item in enumerate(sorted(list(set(node_list))))}
#     n_node = len(node_map)

#     # 使用稀疏矩阵存储邻接关系
#     row = []
#     col = []
#     data = []
#     for i, (node1, node2) in enumerate(edge_list):
#         row.append(node_map[node1])
#         col.append(node_map[node2])
#         data.append(relationship_list[i])  # 使用实际的关系值
#         row.append(node_map[node2])
#         col.append(node_map[node1])
#         data.append(relationship_list[i])  # 使用实际的关系值
#     A = csr_matrix((data, (row, col)), shape=(n_node, n_node), dtype=np.float32)

#     ####################################################
#     #            Acquire Features                      #
#     ####################################################

#     if os.path.exists(feature_path):
#         X = np.load(feature_path)
#     else:
#         # 这里假设 N2V 函数可以处理稀疏矩阵
#         X = np.float32(N2V(A, 512, 4, 1))
#         np.save(feature_path, X)

#     return X, A

## For fastnode2vec
def load_graph_network(adj_path, feature_path):
    X, A, Y = [], None, []
    n_node = 0

    # Acquire Edges
    edge_list = []
    node_list = []
    node_type = {}
    relationship_list = []

    with open(adj_path, 'rt', encoding='utf-8') as f:
        next(f)
        for line in f.readlines():
            node1, node2, relationship, *_ = line.strip().split('\t')
            edge_list.append((node1, node2))
            node_list.extend([node1, node2])
            relationship_list.append(relationship)

    node_map = {item: i for i, item in enumerate(sorted(list(set(node_list))))}
    n_node = len(node_map)

    # 使用稀疏矩阵存储邻接关系
    row = []
    col = []
    data = []
    for i, (node1, node2) in enumerate(edge_list):
        row.append(node_map[node1])
        col.append(node_map[node2])
        data.append(relationship_list[i])  # 使用实际的关系值
        row.append(node_map[node2])
        col.append(node_map[node1])
        data.append(relationship_list[i])  # 使用实际的关系值
    A = csr_matrix((data, (row, col)), shape=(n_node, n_node), dtype=np.float32)
    
    adj_matrix = pd.read_csv(adj_path, sep='\t')
    # 构建图所需的边列表
    edges = [(row['node1'], row['node2'], row['relationship']) for _, row in adj_matrix.iterrows()]

    # 构建图
    graph = Graph(edges, directed=False, weighted=True)
    
    if os.path.exists(feature_path):
        X = np.load(feature_path)
    else:
        # 这里假设 N2V 函数可以处理稀疏矩阵
        X = np.float32(N2V(graph, 512, 4, 1))
        np.save(feature_path, X)

    return X, A