import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tqdm

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
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act, knn_threshold_type='none', knn_std_dev_factor=1.0, chunk_size=100000):
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
        self.chunk_size = chunk_size # Store chunk size
        


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
        # --- VRAM Measurement Setup ---
        device = features.device
        use_cuda_memory_tracking = torch.cuda.is_available() and device.type == 'cuda'
        initial_allocated_mem = 0
        start_mem_embeddings = 0
        start_mem_block = 0

        if use_cuda_memory_tracking:
            # Record initial memory
            initial_allocated_mem = torch.cuda.memory_allocated(device)
            print(f"[MLP Forward Start] Initial VRAM allocated: {initial_allocated_mem / (1024**2):.2f} MB")
            # Reset peak stats for embedding calculation
            torch.cuda.reset_peak_memory_stats(device)
            start_mem_embeddings = torch.cuda.memory_allocated(device)
        # --- End VRAM Measurement Setup ---

        embeddings = self.internal_forward(features)

        # --- Measure Embeddings VRAM ---
        if use_cuda_memory_tracking:
            peak_mem_embeddings = torch.cuda.max_memory_allocated(device)
            print(f"  [Embeddings Calc] Peak VRAM increase: {(peak_mem_embeddings - start_mem_embeddings) / (1024**2):.2f} MB")
            # Reset peak stats for the main sparse/dense block
            torch.cuda.reset_peak_memory_stats(device)
            start_mem_block = torch.cuda.memory_allocated(device)
        # --- End Embeddings VRAM ---


        if self.sparse:
            # --- Start Sparse Block ---
            device = features.device # Already defined, but good practice
            num_nodes = features.shape[0]
            # self._log_vram("MLP Sparse Start") # REMOVED: Method not in this class

            # --- 1. Get Initial KNN Edges ---
            # self._log_vram("MLP Sparse: Before KNN") # REMOVED
            knn_rows, knn_cols = knn_fast(
                embeddings, self.k,
                faiss_index=faiss_index,
                knn_threshold_type=self.knn_threshold_type,
                knn_std_dev_factor=self.knn_std_dev_factor
            )
            # del knn_rows, knn_cols # Free memory early <-- REMOVE THIS LINE
            # self._log_vram("MLP Sparse: After Symmetrization") # REMOVED
            
            num_total_edges = knn_rows.size(0)
            if num_total_edges == 0:
                 print("Warning: No edges found by knn_fast. Returning empty graph.")
                 return dgl.graph(([], []), num_nodes=num_nodes, device=device)

            # --- 2. Explicitly Symmetrize Edges ---
            all_src_nodes = torch.cat([knn_rows, knn_cols])
            all_dst_nodes = torch.cat([knn_cols, knn_rows])
            del knn_rows, knn_cols # Free memory early <-- ADD THIS LINE HERE
            # self._log_vram("MLP Sparse: After Symmetrization") # REMOVED
            
            # --- 3. Calculate Similarities for ALL Edges ---
            # Find unique nodes involved across all edges
            needed_indices = torch.unique(torch.cat([all_src_nodes, all_dst_nodes]))
            needed_indices = needed_indices.to(device)
            
            # Extract necessary embeddings and normalize (Do this once)
            # Note: If embeddings themselves are huge, this might still be an issue
            all_needed_embeddings = embeddings[needed_indices]
            all_needed_embeddings_norm = F.normalize(all_needed_embeddings, dim=1, p=2)
            # self._log_vram("MLP Sparse: After Mini-Embeddings Extraction") # REMOVED
            
            # Create mapping from global node index to local index within all_needed_embeddings_norm (on GPU)
            position_map = torch.zeros(num_nodes, dtype=torch.long, device=device)
            position_map.scatter_(0, needed_indices, torch.arange(len(needed_indices), device=device))
            # self._log_vram("MLP Sparse: After Position Map Creation") # REMOVED

            # Clear original embeddings if no longer needed elsewhere
            del embeddings, all_needed_embeddings

            # --- 3b. Calculate Similarities in Chunks --- 
            final_src_list = []
            final_dst_list = []
            final_values_list = []

            num_chunks = math.ceil(num_total_edges / self.chunk_size)
            print(f"  [MLP Sparse] Processing {num_total_edges} edges in {num_chunks} chunks of size {self.chunk_size}")

            for i in range(0, num_total_edges, self.chunk_size):
                start_idx = i
                end_idx = min(i + self.chunk_size, num_total_edges)
                chunk_src = all_src_nodes[start_idx:end_idx]
                chunk_dst = all_dst_nodes[start_idx:end_idx]
                # self._log_vram(f"MLP Sparse Chunk {i//self.chunk_size + 1}/{num_chunks}: Start") # REMOVED

                # Map chunk source and destination nodes using the precomputed position_map
                mapped_src_chunk = position_map[chunk_src]
                mapped_dst_chunk = position_map[chunk_dst]
                # self._log_vram(f"MLP Sparse Chunk {i//self.chunk_size + 1}: Mapped Indices") # REMOVED

                # Calculate similarities for the chunk using the normalized mini-embeddings
                emb_src_chunk = all_needed_embeddings_norm[mapped_src_chunk]
                emb_dst_chunk = all_needed_embeddings_norm[mapped_dst_chunk]
                chunk_values = torch.sum(emb_src_chunk * emb_dst_chunk, dim=1)
                # self._log_vram(f"MLP Sparse Chunk {i//self.chunk_size + 1}: Calculated Similarities") # REMOVED

                # Apply Non-linearity to chunk values
                chunk_values_processed = apply_non_linearity(chunk_values, self.non_linearity, self.i)
                # self._log_vram(f"MLP Sparse Chunk {i//self.chunk_size + 1}: Applied Non-linearity") # REMOVED

                # Store the original chunk src/dst and the processed values
                final_src_list.append(chunk_src)
                final_dst_list.append(chunk_dst)
                final_values_list.append(chunk_values_processed)

                # Clean up chunk-specific tensors immediately
                del chunk_src, chunk_dst, mapped_src_chunk, mapped_dst_chunk
                del emb_src_chunk, emb_dst_chunk, chunk_values, chunk_values_processed
                # Optionally clear cache periodically within the loop if memory pressure is still high
                # if (i // self.chunk_size + 1) % 5 == 0: # Example: every 5 chunks
                #     torch.cuda.empty_cache()
                # self._log_vram(f"MLP Sparse Chunk {i//self.chunk_size + 1}: End & Cleanup") # REMOVED

            # --- 4. Combine Chunks and Create DGL Graph --- 
            # self._log_vram("MLP Sparse: Before Combining Chunks") # REMOVED
            final_src_nodes = torch.cat(final_src_list)
            final_dst_nodes = torch.cat(final_dst_list)
            final_values_processed = torch.cat(final_values_list)
            # self._log_vram("MLP Sparse: After Combining Chunks") # REMOVED

            # --- 5. Clean up intermediate tensors --- (Original position map, etc.)
            del all_src_nodes, all_dst_nodes, needed_indices, all_needed_embeddings_norm, position_map
            del final_src_list, final_dst_list, final_values_list # Delete lists
            # self._log_vram("MLP Sparse: After Final Cleanup Pre-Graph") # REMOVED

            # --- 6. Create DGL Graph --- 
            # Use the combined source/destination nodes and processed values
            adj = dgl.graph((final_src_nodes, final_dst_nodes), num_nodes=num_nodes, device=device)
            adj.edata['w'] = final_values_processed
            # self._log_vram("MLP Sparse: After DGL Graph Creation") # REMOVED

            # --- 7. Final Memory Cleanup (Less critical now, but keep) ---
            # del final_src_nodes, final_dst_nodes, final_values_processed # Keep these for the return 'adj'

            # --- Measure Sparse Block VRAM ---
            if use_cuda_memory_tracking:
                peak_mem_block = torch.cuda.max_memory_allocated(device)
                print(f"  [Sparse Block Total] Peak VRAM increase: {(peak_mem_block - start_mem_block) / (1024**2):.2f} MB")
                final_allocated_mem = torch.cuda.memory_allocated(device)
                print(f"[MLP Forward End (Sparse)] Final VRAM allocated: {final_allocated_mem / (1024**2):.2f} MB (Delta: {(final_allocated_mem - initial_allocated_mem) / (1024**2):.2f} MB)")
            # --- End Sparse Block VRAM ---

            return adj
        else:
            # --- Dense Mode (Unchanged) ---
            print("Warning: Dense mode in MLP_learner still uses top_k similarity, not knn_fast thresholding.")
            embeddings_norm = F.normalize(embeddings, dim=1, p=2)
            similarities = torch.mm(embeddings_norm, embeddings_norm.t())
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)

            # --- Measure Dense Block VRAM ---
            if use_cuda_memory_tracking:
                peak_mem_block = torch.cuda.max_memory_allocated(device)
                print(f"  [Dense Block Total] Peak VRAM increase: {(peak_mem_block - start_mem_block) / (1024**2):.2f} MB")
                final_allocated_mem = torch.cuda.memory_allocated(device)
                print(f"[MLP Forward End (Dense)] Final VRAM allocated: {final_allocated_mem / (1024**2):.2f} MB (Delta: {(final_allocated_mem - initial_allocated_mem) / (1024**2):.2f} MB)")
            # --- End Dense Block VRAM ---

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