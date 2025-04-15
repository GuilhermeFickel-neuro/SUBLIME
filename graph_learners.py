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
    def __init__(self, k, knn_metric, i, sparse, initial_graph_data=None):
        """
        Initializes the FGP Learner.

        Args:
            k (int): Number of neighbors (informational, not used for computation here).
            knn_metric (str): KNN metric used (informational).
            i (int): Parameter for the ELU non-linearity exponent.
            sparse (bool): Whether the graph is sparse.
            initial_graph_data (Tensor or Tuple): The pre-computed graph structure.
                - If sparse: Should be a tuple (indices, values, size).
                - If dense: Should be a dense Tensor.
                If None, an error will be raised as it's now mandatory.
        """
        super(FGP_learner, self).__init__()

        self.k = k # Keep for potential future use or info
        self.knn_metric = knn_metric # Keep for info
        self.i = i
        self.sparse = sparse

        if initial_graph_data is None:
            raise ValueError("FGP_learner now requires pre-computed 'initial_graph_data'.")

        if self.sparse:
            # Expecting a sparse tensor (already created in load_data)
            if not isinstance(initial_graph_data, torch.Tensor) or not initial_graph_data.is_sparse:
                 raise TypeError("Expected a sparse torch Tensor for initial_graph_data in sparse mode.")
            # Initialize Adj directly with the sparse tensor
            # Ensure it's coalesced and requires gradients
            self.Adj = nn.Parameter(initial_graph_data.coalesce().requires_grad_(True))
        else:
            # Expecting a dense tensor
            if not isinstance(initial_graph_data, torch.Tensor) or initial_graph_data.is_sparse:
                 raise TypeError("Expected a dense torch Tensor for initial_graph_data in dense mode.")
            # Initialize Adj directly with the dense tensor
            self.Adj = nn.Parameter(initial_graph_data.requires_grad_(True))

    def forward(self, h=None, faiss_index=None): # h and faiss_index are no longer used
        """Applies ELU activation to the stored graph parameter."""
        if not self.sparse:
            # For dense mode: Apply activation to the dense tensor parameter
            Adj = F.elu(self.Adj, alpha=self.i) + 1 # Use self.i for alpha
            return Adj
        else:
            # For sparse mode: Apply activation to the values of the sparse tensor parameter
            # Clone to avoid in-place modification if Adj is used elsewhere
            Adj_activated = self.Adj.clone()
            new_values = F.elu(Adj_activated.values(), alpha=self.i) + 1 # Use self.i

            # Create a *new* sparse tensor with the activated values
            # This is crucial because modifying Adj_activated.values() in-place might not work correctly
            # or might affect gradient calculation.
            processed_adj_sparse = torch.sparse_coo_tensor(
                 Adj_activated.indices(), new_values, Adj_activated.size(),
                 device=Adj_activated.device, requires_grad=True
            )

            # Convert the processed *sparse tensor* to a DGL graph for compatibility
            # with downstream layers (like GCNConv_dgl)
            src, dst = processed_adj_sparse.indices()
            graph = dgl.graph((src, dst), num_nodes=processed_adj_sparse.size(0), device=processed_adj_sparse.device)
            graph.edata['w'] = processed_adj_sparse.values()

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