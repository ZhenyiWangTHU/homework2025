import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
import networkx as nx
#from node2vec import Node2Vec
from fastnode2vec import Graph, Node2Vec 
from scipy.sparse import csr_matrix
import re

def data2tsne(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    tsne = TSNE()
    tsne.fit_transform(embedding)
    return tsne.embedding_

def data2umap(data, n_pca=0):
    if n_pca > 0:
        pca = PCA(n_components=n_pca)
        embedding = pca.fit_transform(data)
    else:
        embedding = data
    embedding_ = umap.UMAP(
        n_neighbors=30,
        min_dist=0.3,
        metric='cosine',
        n_components = 2,
        learning_rate = 1.0,
        spread = 1.0,
        set_op_mix_ratio = 1.0,
        local_connectivity = 1,
        repulsion_strength = 1,
        negative_sample_rate = 5,
        angular_rp_forest = False,
        verbose = False
    ).fit_transform(embedding)
    return embedding_

def umap_plot(data, save_path):
    import seaborn as sns
    plt.figure(figsize=(10,10))
    fig = sns.scatterplot(
        x = 'UMAP_1',
        y = 'UMAP_2',
        data = data,
        hue = 'hue',
        palette="deep"
    )
    fig = plt.gcf()
    fig.savefig(save_path)
    plt.close()
    
def gplot(embedding_, type_info, filename):
    test = pd.DataFrame(embedding_, columns=['UMAP_1', 'UMAP_2'])
    test['hue'] = type_info
    save_path = './pic/'+filename + '.png'
    umap_plot(test, save_path)

def create_plot(features, labels, save_path, style='tsne', n_pca=None):
    if style=='tsne':
        if not n_pca:
            n_pca = 0
        embedding_ = data2tsne(features, n_pca)
    elif style=='umap':
        if not n_pca:
            n_pca = 30
        embedding_ = data2umap(features, n_pca)
    else:
        print(f'No style:{style}!')
        return
    gplot(embedding_, labels, save_path)
    
    
# def N2V(adj, hid_units, p=1, q=1, walk_length=20, num_walks=40):
#     edge_index = np.where(adj>0)
#     edge_index = np.r_[[edge_index[0]], [edge_index[1]]].T
    
#     def create_net(elist):
#         import networkx as nx
#         g = nx.Graph()
#         elist = np.array(elist)
#         g.add_edges_from(elist)
#         for edge in g.edges():
#             g[edge[0]][edge[1]]['weight'] = 1
#         return g
    
#     graph = create_net(edge_index)
#     node2vec = Node2Vec(graph, dimensions=hid_units, walk_length=walk_length, num_walks=num_walks, p=p,q=q)
#     model = node2vec.fit()
#     outputs = np.array([model.wv[str(item)] for item in range(len(adj))])
#     return outputs

# def N2V(adj, hid_units, p=1, q=1, walk_length=20, num_walks=40):
#     import networkx as nx
#     # 处理稀疏矩阵获取边索引
#     if isinstance(adj, csr_matrix):
#         edge_index = np.column_stack((adj.nonzero()[0], adj.nonzero()[1]))
#     else:
#         edge_index = np.where(adj > 0)
#         edge_index = np.r_[[edge_index[0]], [edge_index[1]]].T

#     def create_net(elist):
#         g = nx.Graph()
#         elist = np.array(elist)
#         g.add_edges_from(elist)
#         for edge in g.edges():
#             g[edge[0]][edge[1]]['weight'] = 1
#         return g

#     graph = create_net(edge_index)
#     node2vec = Node2Vec(graph, dimensions=hid_units, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=100)
#     model = node2vec.fit()
#     outputs = np.array([model.wv[str(item)] for item in range(adj.shape[0])])
#     return outputs

def N2V(graph, hid_units, p=1, q=1, walk_length=20, num_walks=40):
    node2vec = Node2Vec(graph, dim=hid_units, walk_length=walk_length, window=10, p=p, q=q, workers=150)
    node2vec.train(epochs=100)
    print("training is completed!")
    if node2vec is None:
        print("model is none!")
    outputs = np.array([node2vec.wv[str(item)] for item in graph.node_names])
    return outputs
    
# if __name__ == '__main__':
#     pass

def classify_element(element):
    # 正则表达式模式
    pattern_id = r'^\d+$'  # 全数字（病人ID）
    pattern_code = r'^[A-Z][A-Z\d]*$'  # 大写字母开头，后跟数字or大写字母(icd10)
    pattern_protein = r'^[a-z\d][a-z\d_]*$'
    
    # 分类判断
    if re.match(pattern_id, element):
        return 'eid'
    elif re.match(pattern_code, element):
        return 'icd10'
    elif re.match(pattern_protein, element):
        return 'protein'
    else:
        return 'metabolite'

def get_edge_types(data):
    """从PyG异构图数据对象中提取所有边类型"""
    edge_types = []
    
    # 遍历数据对象中的所有属性
    for key in data.keys():
        # 检查是否为边索引属性（格式为 (源节点类型, 边关系, 目标节点类型)）
        if isinstance(key, tuple) and len(key) == 3:
            src_type, relation, dst_type = key
            # 检查对应的值是否为边索引张量
            if isinstance(data[key], torch.Tensor) and data[key].dim() == 2:
                edge_types.append((src_type, relation, dst_type))
    
    return edge_types