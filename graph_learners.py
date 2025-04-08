import dgl
import torch
import torch.nn as nn

from layers import Attentive, GCNConv_dense, GCNConv_dgl
from utils import *


# class FGP_learner(nn.Module):
#     def __init__(self, features, k, knn_metric, i, sparse):
#         super(FGP_learner, self).__init__()

#         self.k = k
#         self.knn_metric = knn_metric
#         self.i = i
#         self.sparse = sparse

#         self.Adj = nn.Parameter(
#             torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i)))

#     def forward(self, h):
#         if not self.sparse:
#             Adj = F.elu(self.Adj) + 1
#         else:
#             Adj = self.Adj.coalesce()
#             Adj.values = F.elu(Adj.values()) + 1
#         return Adj

# class FGP_learner(nn.Module):
#     def __init__(self, features, k, knn_metric, i, sparse):
#         super(FGP_learner, self).__init__()

#         self.k = k
#         self.knn_metric = knn_metric
#         self.i = i
#         self.sparse = sparse

#         # Initialize as a regular dense parameter
#         adjacency = torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i))
#         self.Adj = nn.Parameter(adjacency)

#     def forward(self, h):
#         if not self.sparse:
#             # Dense mode
#             Adj = F.elu(self.Adj) + 1
#             return Adj
#         else:
#             # Sparse mode - convert dense to sparse first
#             # Create a sparse tensor from the dense parameter
#             dense_adj = self.Adj
            
#             # Debug: Check the device and shape
#             print("Dense Adj device:", dense_adj.device)
#             print("Dense Adj shape:", dense_adj.shape)
            
#             # Get indices of non-zero elements
#             indices = torch.nonzero(dense_adj, as_tuple=True)
#             values = dense_adj[indices]
            
#             # Debug: Check indices and values 
#             print("Number of non-zero elements:", values.size(0))
            
#             # Apply ELU and add 1
#             values = F.elu(values) + 1
            
#             # Create DGL graph
#             src, dst = indices
#             adj = dgl.graph((src, dst), num_nodes=dense_adj.shape[0], device=dense_adj.device)
#             adj.edata['w'] = values
            
#             return adj

class FGP_learner(nn.Module):
    def __init__(self, features, k, knn_metric, i, sparse):
        super(FGP_learner, self).__init__()

        self.k = k
        self.knn_metric = knn_metric
        self.i = i
        self.sparse = sparse

        
        if self.sparse:
            # Get indices, values and shape directly in sparse format
            indices, values, size = nearest_neighbors_pre_elu_sparse(features, self.k, self.knn_metric, self.i)
            indices = torch.from_numpy(indices)
            values = torch.from_numpy(values).to(torch.float32)
            # Create sparse tensor directly
            sparse_adj = torch.sparse_coo_tensor(indices, values, size).coalesce()
            self.Adj = nn.Parameter(sparse_adj)
        else:
            # Only convert to dense if needed
            adj_dense = torch.from_numpy(nearest_neighbors_pre_elu(features, self.k, self.knn_metric, self.i))
            self.Adj = nn.Parameter(adj_dense)

    def forward(self, h, faiss_index=None):
        if not self.sparse:
            # For dense mode
            Adj = F.elu(self.Adj) + 1
            return Adj
        else:
            # For sparse mode - ensure the tensor is coalesced
            Adj = self.Adj.coalesce()
            
            # Debug info
            print("Is sparse tensor:", Adj.is_sparse)
            print("Is coalesced:", Adj.is_coalesced())
            
            # Get indices and values
            indices = Adj._indices()
            values = Adj._values()
            
            # Apply ELU and add 1
            new_values = F.elu(values) + 1
            
            # Create new sparse tensor with processed values
            processed_adj = torch.sparse_coo_tensor(
                indices, new_values, Adj.size(), device=Adj.device
            )
            
            # For DGL graph creation
            src, dst = indices[0], indices[1]
            graph = dgl.graph((src, dst), num_nodes=Adj.size(0), device=Adj.device)
            graph.edata['w'] = new_values
            
            return graph


class ATT_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act):
        super(ATT_learner, self).__init__()

        self.i = i
        self.layers = nn.ModuleList()
        for _ in range(nlayers):
            self.layers.append(Attentive(isize))
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def forward(self, features, faiss_index=None):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000, faiss_index=faiss_index)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features, faiss_index=None):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000, faiss_index=faiss_index)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, adj):
        super(GNN_learner, self).__init__()

        self.adj = adj
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GCNConv_dgl(isize, isize))
        else:
            self.layers.append(GCNConv_dgl(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(isize, isize))
            self.layers.append(GCNConv_dgl(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h, self.adj)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features, faiss_index=None):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000, faiss_index=faiss_index)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities