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
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, knn_threshold_type='none', knn_std_dev_factor=1.0):
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
        self.knn_threshold_type = knn_threshold_type
        self.knn_std_dev_factor = knn_std_dev_factor

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
        embeddings = self.internal_forward(features)

        if self.sparse:
            # Get neighbor indices without similarity values
            rows, cols = knn_fast(embeddings, self.k, 1000, faiss_index=faiss_index,
                                          knn_threshold_type=self.knn_threshold_type,
                                          knn_std_dev_factor=self.knn_std_dev_factor)

            # Super memory-efficient approach
            device = features.device
            num_nodes = features.shape[0]
            
            # Calculate the total number of edges without storing the full symmetrized tensors
            base_num_edges = rows.size(0)
            
            # Track in-memory edges (each direction separately)
            all_rows = []
            all_cols = []
            all_values = []
            
            # First process the original row, col indices
            # Create much smaller chunks for processing
            base_chunk_size = 100  # Try with very small chunks
            
            for i in range(0, base_num_edges, base_chunk_size):
                # Get indices for this chunk
                end_idx = min(i + base_chunk_size, base_num_edges)
                rows_chunk = rows[i:end_idx].to('cpu')  # Move to CPU temporarily
                cols_chunk = cols[i:end_idx].to('cpu')
                
                # Get only the embeddings we need and normalize only those
                needed_indices = torch.unique(torch.cat([rows_chunk, cols_chunk]))
                needed_indices = needed_indices.to(device)
                
                # Create a mini-embedding matrix for only needed nodes - VECTORIZED VERSION
                mini_embeddings = embeddings[needed_indices]
                mini_embeddings = F.normalize(mini_embeddings, dim=1, p=2)
                
                # Fast vectorized index mapping using scatter (much faster than the loop)
                if offload_to_cpu:
                    # Create a CPU mapping tensor
                    needed_indices_cpu = needed_indices.cpu()
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device='cpu')
                    # Use scatter to build the mapping efficiently
                    position_map.scatter_(0, needed_indices_cpu, torch.arange(len(needed_indices_cpu), device='cpu'))
                    # Map the original row and col indices to positions in mini_embeddings
                    mapped_rows = position_map[rows_chunk]
                    mapped_cols = position_map[cols_chunk]
                else:
                    # Create a GPU mapping tensor (faster on GPU)
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    # Use scatter to build the mapping efficiently
                    position_map.scatter_(0, needed_indices, torch.arange(len(needed_indices), device=device))
                    # Map the original row and col indices to positions in mini_embeddings
                    mapped_rows = position_map[rows_chunk]
                    mapped_cols = position_map[cols_chunk]
                
                # Move back to device for calculation if needed
                if offload_to_cpu:
                    mapped_rows = mapped_rows.to(device)
                    mapped_cols = mapped_cols.to(device)
                    rows_chunk = rows_chunk.to(device)
                    cols_chunk = cols_chunk.to(device)
                
                # Calculate similarity using the mini-embeddings
                emb_rows = mini_embeddings[mapped_rows]
                emb_cols = mini_embeddings[mapped_cols]
                values_chunk = torch.sum(emb_rows * emb_cols, dim=1)
                
                # Store the original rows, cols and values (not the mapped indices)
                all_rows.append(rows_chunk)
                all_cols.append(cols_chunk)
                all_values.append(values_chunk)
                
                # Clear memory
                del mini_embeddings, needed_indices, position_map
                del mapped_rows, mapped_cols, emb_rows, emb_cols
                
                # Periodically run garbage collection and empty cache
                if cleanup_every_n_chunks > 0 and chunk_idx % cleanup_every_n_chunks == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Combine all processed chunks
            rows_sym = torch.cat(all_rows)
            cols_sym = torch.cat(all_cols)
            values_sym = torch.cat(all_values)
            
            # Apply non-linearity to the differentiable similarity values
            values_processed = apply_non_linearity(values_sym, self.non_linearity, self.i)

            # Create DGL graph and assign differentiable edge weights
            adj = dgl.graph((rows_sym, cols_sym), num_nodes=features.shape[0], device=features.device)
            adj.edata['w'] = values_processed
            
            # Clear any remaining tensors
            del all_rows, all_cols, all_values, rows, cols, values_sym
            
            # Final memory cleanup
            if cleanup_every_n_chunks > 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            
            return adj
        else:
            print("Warning: Dense mode in ATT_learner still uses top_k similarity, not knn_fast thresholding.")
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act, knn_threshold_type='none', knn_std_dev_factor=1.0, 
                 chunk_size=100, offload_to_cpu=True, cleanup_every_n_chunks=5):
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
        self.knn_threshold_type = knn_threshold_type
        self.knn_std_dev_factor = knn_std_dev_factor
        
        # Memory optimization parameters
        self.chunk_size = chunk_size
        self.offload_to_cpu = offload_to_cpu
        self.cleanup_every_n_chunks = cleanup_every_n_chunks

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
            layer.weight.data.copy_(torch.eye(self.input_dim))
            if layer.bias is not None:
                 layer.bias.data.fill_(0)

    def forward(self, features, faiss_index=None):
        embeddings = self.internal_forward(features)

        if self.sparse:
            device = features.device
            num_nodes = features.shape[0]

            # --- 1. Get Initial KNN Edges ---
            knn_rows, knn_cols = knn_fast(
                embeddings, self.k,
                faiss_index=faiss_index,
                knn_threshold_type=self.knn_threshold_type,
                knn_std_dev_factor=self.knn_std_dev_factor
            )

            # --- 2. Explicitly Symmetrize Edges ---
            # Combine original and reversed edges for processing
            all_src_nodes = torch.cat([knn_rows, knn_cols])
            all_dst_nodes = torch.cat([knn_cols, knn_rows])
            del knn_rows, knn_cols # Free memory early
            
            num_total_edges = all_src_nodes.size(0)
            if num_total_edges == 0:
                 # Handle case with no edges found
                 print("Warning: No edges found by knn_fast. Returning empty graph.")
                 return dgl.graph(([], []), num_nodes=num_nodes, device=device)


            # --- 3. Process Edges in Chunks for Memory Efficiency ---
            processed_rows = []
            processed_cols = []
            processed_values = []

            # Use memory optimization parameters from instance variables
            chunk_size = self.chunk_size
            offload_to_cpu = self.offload_to_cpu
            cleanup_every_n_chunks = self.cleanup_every_n_chunks

            for chunk_idx, i in enumerate(range(0, num_total_edges, chunk_size)):
                # --- a. Get Chunk ---
                end_idx = min(i + chunk_size, num_total_edges)
                src_chunk = all_src_nodes[i:end_idx]
                dst_chunk = all_dst_nodes[i:end_idx]

                # Optionally offload indices to CPU to save GPU memory during index creation
                src_chunk_mem = src_chunk.to('cpu') if offload_to_cpu else src_chunk
                dst_chunk_mem = dst_chunk.to('cpu') if offload_to_cpu else dst_chunk

                # --- b. Get Embeddings for Nodes in Chunk ---
                # Find unique nodes involved in this chunk
                needed_indices = torch.unique(torch.cat([src_chunk_mem, dst_chunk_mem]))
                # Ensure needed_indices are on the correct device (GPU) for embedding lookup
                needed_indices = needed_indices.to(device)

                # Extract only necessary embeddings and normalize them
                mini_embeddings = embeddings[needed_indices]
                mini_embeddings = F.normalize(mini_embeddings, dim=1, p=2)

                # --- c. Map Chunk Indices to Mini-Embedding Indices ---
                if offload_to_cpu:
                    # Create mapping structures on CPU
                    needed_indices_cpu = needed_indices.cpu()
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device='cpu')
                    position_map.scatter_(0, needed_indices_cpu, torch.arange(len(needed_indices_cpu), device='cpu'))
                    
                    # Perform mapping lookup on CPU using CPU indices
                    mapped_src = position_map[src_chunk_mem] # src_chunk_mem is already on CPU
                    mapped_dst = position_map[dst_chunk_mem] # dst_chunk_mem is already on CPU
                    
                    # Move mapped indices back to GPU for similarity calculation
                    mapped_src = mapped_src.to(device)
                    mapped_dst = mapped_dst.to(device)
                    
                    # Clean up CPU tensors explicitly
                    del needed_indices_cpu, position_map
                else:
                    # Create mapping structures directly on GPU
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    position_map.scatter_(0, needed_indices, torch.arange(len(needed_indices), device=device))
                    
                    # Perform mapping lookup on GPU using GPU indices
                    mapped_src = position_map[src_chunk] # src_chunk is already on GPU
                    mapped_dst = position_map[dst_chunk] # dst_chunk is already on GPU

                    # Clean up GPU tensor explicitly
                    del position_map
                
                # Cleanup mapping-related tensors regardless of path
                del needed_indices 

                # --- d. Calculate Similarities for Chunk ---
                emb_src = mini_embeddings[mapped_src]
                emb_dst = mini_embeddings[mapped_dst]
                # Cosine similarity for normalized embeddings is the dot product
                values_chunk = torch.sum(emb_src * emb_dst, dim=1)

                # --- e. Store Original Indices and Calculated Values ---
                # Store the *original* source/destination indices (on GPU) and computed values
                processed_rows.append(src_chunk)
                processed_cols.append(dst_chunk)
                processed_values.append(values_chunk)

                # --- f. Memory Cleanup for Chunk ---
                del src_chunk, dst_chunk, src_chunk_mem, dst_chunk_mem
                if cleanup_every_n_chunks > 0 and (chunk_idx + 1) % cleanup_every_n_chunks == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()


            # --- 4. Combine Processed Chunks ---
            final_rows = torch.cat(processed_rows)
            final_cols = torch.cat(processed_cols)
            final_values = torch.cat(processed_values)

            del processed_rows, processed_cols, processed_values # Free list memory
            del all_src_nodes, all_dst_nodes # Free combined indices

            # --- 5. Apply Non-linearity ---
            final_values_processed = apply_non_linearity(final_values, self.non_linearity, self.i)

            # --- 6. Create DGL Graph ---
            adj = dgl.graph((final_rows, final_cols), num_nodes=num_nodes, device=device)
            adj.edata['w'] = final_values_processed

            # --- 7. Final Memory Cleanup ---
            del final_rows, final_cols, final_values, final_values_processed, embeddings
            if cleanup_every_n_chunks > 0:
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            return adj
        else:
            # --- Dense Mode ---
            print("Warning: Dense mode in MLP_learner still uses top_k similarity, not knn_fast thresholding.")
            # Normalize embeddings for cosine similarity calculation
            embeddings_norm = F.normalize(embeddings, dim=1, p=2)
            # Calculate dense similarity matrix
            similarities = torch.mm(embeddings_norm, embeddings_norm.t())
            # Keep only top K neighbors for each node
            similarities = top_k(similarities, self.k + 1) # +1 to potentially include self
            # Apply non-linearity
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, adj, knn_threshold_type='none', knn_std_dev_factor=1.0):
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
        self.knn_threshold_type = knn_threshold_type
        self.knn_std_dev_factor = knn_std_dev_factor

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
            layer.weight.data.copy_(torch.eye(self.input_dim))
            if layer.bias is not None:
                 layer.bias.data.fill_(0)

    def forward(self, features, faiss_index=None):
        embeddings = self.internal_forward(features)

        if self.sparse:
            # Get neighbor indices without similarity values
            rows, cols = knn_fast(embeddings, self.k, 1000, faiss_index=faiss_index,
                                          knn_threshold_type=self.knn_threshold_type,
                                          knn_std_dev_factor=self.knn_std_dev_factor)

            # Super memory-efficient approach
            device = features.device
            num_nodes = features.shape[0]
            
            # Calculate the total number of edges without storing the full symmetrized tensors
            base_num_edges = rows.size(0)
            
            # Track in-memory edges (each direction separately)
            all_rows = []
            all_cols = []
            all_values = []
            
            # First process the original row, col indices
            # Create much smaller chunks for processing
            base_chunk_size = 100  # Try with very small chunks
            
            for i in range(0, base_num_edges, base_chunk_size):
                # Get indices for this chunk
                end_idx = min(i + base_chunk_size, base_num_edges)
                rows_chunk = rows[i:end_idx].to('cpu')  # Move to CPU temporarily
                cols_chunk = cols[i:end_idx].to('cpu')
                
                # Get only the embeddings we need and normalize only those
                needed_indices = torch.unique(torch.cat([rows_chunk, cols_chunk]))
                needed_indices = needed_indices.to(device)
                
                # Create a mini-embedding matrix for only needed nodes - VECTORIZED VERSION
                mini_embeddings = embeddings[needed_indices]
                mini_embeddings = F.normalize(mini_embeddings, dim=1, p=2)
                
                # Fast vectorized index mapping using scatter (much faster than the loop)
                if offload_to_cpu:
                    # Create a CPU mapping tensor
                    needed_indices_cpu = needed_indices.cpu()
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device='cpu')
                    # Use scatter to build the mapping efficiently
                    position_map.scatter_(0, needed_indices_cpu, torch.arange(len(needed_indices_cpu), device='cpu'))
                    # Map the original row and col indices to positions in mini_embeddings
                    mapped_rows = position_map[rows_chunk]
                    mapped_cols = position_map[cols_chunk]
                else:
                    # Create a GPU mapping tensor (faster on GPU)
                    position_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
                    # Use scatter to build the mapping efficiently
                    position_map.scatter_(0, needed_indices, torch.arange(len(needed_indices), device=device))
                    # Map the original row and col indices to positions in mini_embeddings
                    mapped_rows = position_map[rows_chunk]
                    mapped_cols = position_map[cols_chunk]
                
                # Move back to device for calculation if needed
                if offload_to_cpu:
                    mapped_rows = mapped_rows.to(device)
                    mapped_cols = mapped_cols.to(device)
                    rows_chunk = rows_chunk.to(device)
                    cols_chunk = cols_chunk.to(device)
                
                # Calculate similarity using the mini-embeddings
                emb_rows = mini_embeddings[mapped_rows]
                emb_cols = mini_embeddings[mapped_cols]
                values_chunk = torch.sum(emb_rows * emb_cols, dim=1)
                
                # Store the original rows, cols and values (not the mapped indices)
                all_rows.append(rows_chunk)
                all_cols.append(cols_chunk)
                all_values.append(values_chunk)
                
                # Clear memory
                del mini_embeddings, needed_indices, position_map
                del mapped_rows, mapped_cols, emb_rows, emb_cols
                
                # Periodically run garbage collection and empty cache
                if cleanup_every_n_chunks > 0 and chunk_idx % cleanup_every_n_chunks == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Combine all processed chunks
            rows_sym = torch.cat(all_rows)
            cols_sym = torch.cat(all_cols)
            values_sym = torch.cat(all_values)
            
            # Apply non-linearity to the differentiable similarity values
            values_processed = apply_non_linearity(values_sym, self.non_linearity, self.i)

            # Create DGL graph and assign differentiable edge weights
            adj = dgl.graph((rows_sym, cols_sym), num_nodes=num_nodes, device=device)
            adj.edata['w'] = values_processed
            
            # Clear any remaining tensors
            del all_rows, all_cols, all_values, rows, cols
            
            return adj
        else:
            print("Warning: Dense mode in GNN_learner still uses top_k similarity, not knn_fast thresholding.")
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities