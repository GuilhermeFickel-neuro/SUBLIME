import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import dgl
from sklearn import metrics
from munkres import Munkres
import faiss
import time
import psutil
import os
import gc

EOS = 1e-10

# Function from test_faiss.py, potentially useful here too
def get_memory_usage():
    """Return the current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB

# New function to build and return a FAISS index
def build_faiss_index(features, k=10, use_gpu=True, nprobe=None):
    """
    Build and return a FAISS index (now using IndexFlatIP) for the given features.
    This can be reused to avoid rebuilding the index on each call.
    
    Args:
        features (torch.Tensor): Input features (num_nodes x num_features).
        k (int): Number of neighbors (no longer used for index tuning with IndexFlat).
        use_gpu (bool): If True, attempt to use GPU for FAISS operations.
        nprobe (int, optional): Ignored for IndexFlat.
                               
    Returns:
        faiss.Index: An FAISS index containing the features (IndexFlatIP or GpuIndexFlatIP).
    """
    print("Building reusable FAISS index (using IndexFlatIP)...")
    t_start = time.time()
    
    # Get device and dimensions
    device = features.device
    n, d = features.shape
    t0 = time.time()
    print(f"  [build_index] Input shape: {n}x{d} on {device}. Time: {t0-t_start:.4f}s")
    
    # Normalize features for cosine similarity (since we use IndexFlatIP)
    X_normalized = F.normalize(features, dim=1, p=2)
    t1 = time.time()
    print(f"  [build_index] Normalization time: {t1-t0:.4f}s")
    
    # Convert to numpy for FAISS
    X_np = X_normalized.cpu().detach().numpy().astype('float32')
    X_np = np.ascontiguousarray(X_np)
    t2 = time.time()
    print(f"  [build_index] Data transfer & conversion time: {t2-t1:.4f}s")
    
    # --- IndexFlat does not require training or complex parameter calculation --- 
    # Removed nlist, M, nbits calculation
    # Removed nprobe setting logic
    
    # Build the index
    actual_use_gpu = use_gpu and faiss.get_num_gpus() > 0
    index_type = "IndexFlatIP"
    faiss_device_info = "CPU"
    index = None # Initialize index

    if actual_use_gpu:
        try:
            faiss_device_info = "GPU"
            print(f"  [build_index] Building FAISS index on GPU with {index_type}")
            t_gpu_start = time.time()
            res = faiss.StandardGpuResources() # Resources needed for index_cpu_to_gpu
            t_res = time.time()
            print(f"    [GPU Build] StandardGpuResources init time: {t_res-t_gpu_start:.4f}s")
            
            # Create CPU index first
            cpu_index = faiss.IndexFlatIP(d) 
            t_cpu_idx = time.time()
            print(f"    [GPU Build] CPU index ({type(cpu_index).__name__}) creation time: {t_cpu_idx-t_res:.4f}s")
            
            # Move index to GPU 
            # Note: GpuIndexFlatIP exists, but index_cpu_to_gpu is often more convenient
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            t_gpu_conv = time.time()
            print(f"    [GPU Build] Index CPU->GPU conversion time: {t_gpu_conv-t_cpu_idx:.4f}s")
            
            # Add data to the index
            print(f"    [GPU Build] Adding data to GPU index...")
            gpu_index.add(X_np)
            t_add = time.time()
            print(f"    [GPU Build] GPU index add time: {t_add-t_gpu_conv:.4f}s")
            
            index = gpu_index
            print(f"  [build_index] Total GPU build time: {t_add - t_gpu_start:.4f}s")
        except Exception as e:
            print(f"FAISS GPU execution failed ({index_type}): {e}. Falling back to CPU.")
            actual_use_gpu = False
            index = None # Reset index if GPU failed

    if not actual_use_gpu:
        faiss_device_info = "CPU"
        print(f"  [build_index] Building FAISS index on CPU with {index_type}")
        t_cpu_build_start = time.time()
        
        # Create IndexFlatIP directly
        index = faiss.IndexFlatIP(d)
        t_idx_create = time.time()
        print(f"    [CPU Build] Index ({type(index).__name__}) creation time: {t_idx_create-t_cpu_build_start:.4f}s")
        
        # Add vectors to the index
        print(f"    [CPU Build] Adding data to CPU index...")
        index.add(X_np)
        t_add = time.time()
        print(f"    [CPU Build] CPU index add time: {t_add-t_idx_create:.4f}s")
        print(f"  [build_index] Total CPU build time: {t_add - t_cpu_build_start:.4f}s")

    if index is None:
         raise RuntimeError("Failed to build index on both CPU and GPU.")

    t_end = time.time()
    print(f"FAISS index built on {faiss_device_info} in {t_end - t_start:.4f} seconds (End-to-End)")
    
    return index

def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def edge_deletion(adj, drop_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    samples = np.random.choice(num_edge, size=int(drop_r * num_edge), replace=False)
    dropped_edge_index = half_edge_index[:, samples].T
    adj[dropped_edge_index[:,0],dropped_edge_index[:,1]] = 0.
    adj[dropped_edge_index[:,1],dropped_edge_index[:,0]] = 0.
    return adj

def edge_addition(adj, add_r):
    edge_index = np.array(np.nonzero(adj))
    half_edge_index = edge_index[:, edge_index[0,:] < edge_index[1,:]]
    num_edge = half_edge_index.shape[1]
    num_node = adj.shape[0]
    added_edge_index_in = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    added_edge_index_out = np.random.choice(num_node, size=int(add_r * num_edge), replace=True)
    adj[added_edge_index_in,added_edge_index_out] = 1.
    adj[added_edge_index_out,added_edge_index_in] = 1.
    return adj


def get_feat_mask(features, mask_rate):
    device = features.device
    feat_node = features.shape[1]
    mask = torch.zeros(features.shape, device=device)
    samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
    mask[:, samples] = 1
    return mask, samples


def accuracy(preds, labels):
    pred_class = torch.max(preds, 1)[1]
    return torch.sum(torch.eq(pred_class, labels)).float() / labels.shape[0]

def faiss_kneighbors_graph(X, k, metric='l2'):
    # Convert to numpy if tensor
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    
    # Ensure correct data type and contiguity
    X = np.ascontiguousarray(X.astype('float32'))
    
    n_samples, n_features = X.shape
    
    # Use a more memory-efficient index for medium-sized datasets
    if metric == 'l2':
        index = faiss.IndexFlatL2(n_features)
    elif metric == 'ip' or metric == 'cosine':
        if metric == 'cosine':
            # Normalize in-place to save memory
            faiss.normalize_L2(X)
        index = faiss.IndexFlatIP(n_features)
    else:
        raise ValueError(f"Metric {metric} not supported")
    
    # Add data to the index
    index.add(X)
    
    # Batch processing to reduce peak memory usage
    batch_size = 10000  # Process in smaller chunks
    rows_list = []
    cols_list = []
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch = X[i:end_idx]
        
        # Search for nearest neighbors for this batch
        distances, indices = index.search(batch, k + 1)
        
        # Filter out any invalid indices (should not happen with proper index)
        valid_mask = indices >= 0
        valid_mask &= indices < n_samples
        
        for j in range(indices.shape[0]):
            row_idx = i + j
            valid_neighbors = indices[j, 1:][valid_mask[j, 1:]]  # Skip self (first column)
            
            if len(valid_neighbors) > 0:
                batch_rows = np.full(len(valid_neighbors), row_idx)
                batch_cols = valid_neighbors
                
                rows_list.append(batch_rows)
                cols_list.append(batch_cols)

        if (i // batch_size) % 5 == 0:
            print(f"Processed {end_idx}/{n_samples} samples. Memory: {get_memory_usage():.2f} GB")
    
    # Combine all batches
    if not rows_list:  # Handle empty case
        return csr_matrix((n_samples, n_samples))
        
    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    data = np.ones_like(cols)
    
    # Create sparse matrix
    from scipy.sparse import csr_matrix
    graph = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    
    return graph

def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = faiss_kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj

def nearest_neighbors_pre_elu_sparse(X, k, metric, i):
    # Get sparse adjacency matrix from FAISS
    adj_sparse = faiss_kneighbors_graph(X, k, metric=metric)
    
    # Add self-loops while keeping it sparse
    n = X.shape[0]
    eye_indices = np.stack([np.arange(n), np.arange(n)])
    eye_values = np.ones(n)
    
    # Get existing indices and values
    adj_coo = adj_sparse.tocoo()
    indices = np.vstack([adj_coo.row, adj_coo.col])
    values = adj_coo.data
    
    # Combine existing edges and self-loops
    indices = np.hstack([indices, eye_indices])
    values = np.hstack([values, eye_values])
    
    # Apply the transformation to the values only
    values = values * i - i
    
    # Return in COO format for easy conversion to PyTorch sparse tensor
    shape = (n, n)
    return indices, values, shape


def normalize(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b=None, use_gpu=True, nprobe=None, faiss_index=None, knn_threshold_type='none', knn_std_dev_factor=1.0):
    """
    Optimized KNN implementation using FAISS.
    Can optionally filter neighbors based on similarity threshold.

    Args:
        X (torch.Tensor): Input features (num_nodes x num_features).
        k (int): Number of neighbors to find for each node.
        b (int): Batch size parameter (kept for backward compatibility, not used).
        use_gpu (bool): If True, attempt to use GPU for FAISS operations when building internally.
        nprobe (int, optional): Ignored for IndexFlatIP.
        faiss_index (faiss.Index, optional): A pre-built FAISS index.
        knn_threshold_type (str): Type of thresholding ('none', 'median_k', 'std_dev_k').
                                  Ensures the closest valid neighbor is kept, others are thresholded based on similarity.
        knn_std_dev_factor (float): Factor for 'std_dev_k' threshold (mean - factor * std_dev of k-th similarities).

    Returns:
        tuple: (rows, cols, values) representing the filtered graph in COO format
               with RAW similarity scores (higher is better).
    """
    t_start_knn = time.time()
    device = X.device
    n, d = X.shape
    # print(f"[knn_fast] Starting KNN search. Threshold type: {knn_threshold_type}, k: {k}, std_factor: {knn_std_dev_factor}") # Less verbose

    # --- Preprocessing, Index Building/Loading, Search ---
    t_preproc_start = time.time()
    X_normalized = F.normalize(X, dim=1, p=2)
    # Ensure data is contiguous AFTER normalization and potential device transfer
    X_np = X_normalized.cpu().detach().numpy().astype('float32')
    X_np = np.ascontiguousarray(X_np) # Ensure contiguous array for FAISS
    # print(f"  [knn_fast] Preprocessing time: {time.time() - t_preproc_start:.4f}s") # Less verbose

    internal_index = None # Keep track if we built one internally
    t_index_start = time.time()
    if faiss_index is None:
        # print("  [knn_fast] Building internal FAISS index (IndexFlatIP)...") # Less verbose
        actual_use_gpu_build = use_gpu and faiss.get_num_gpus() > 0
        faiss_device_info = "CPU"
        if actual_use_gpu_build:
            try:
                # print("    [knn_fast] Attempting GPU index build...") # Less verbose
                faiss_device_info = "GPU"
                res = faiss.StandardGpuResources()
                cpu_index = faiss.IndexFlatIP(d)
                gpu_index_internal = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                gpu_index_internal.add(X_np)
                internal_index = gpu_index_internal
            except Exception as e:
                print(f"    [knn_fast] GPU index build failed: {e}. Falling back to CPU.")
                actual_use_gpu_build = False
                # internal_index remains None, will be built on CPU below
        # Build CPU index if GPU failed or wasn't requested/available
        if not actual_use_gpu_build:
            # print("    [knn_fast] Building CPU index...") # Less verbose
            faiss_device_info = "CPU"
            index_cpu_internal = faiss.IndexFlatIP(d)
            index_cpu_internal.add(X_np)
            internal_index = index_cpu_internal

        if internal_index is None: # Check if index building failed completely
             raise RuntimeError("Failed to build internal FAISS index on both CPU and GPU.")

        faiss_index = internal_index # Use the internally built index
        # print(f"  [knn_fast] Internal index built on {faiss_device_info}.") # Less verbose
    # else: # Using pre-built index
        # Determine device of pre-built index (best effort)
        # try:
        #      gpu_res = faiss.downcast_index(faiss_index).getResources()
        #      faiss_device_info = "GPU (pre-built)"
        # except AttributeError:
        #      faiss_device_info = "CPU or Unknown (pre-built)"
        # print(f"  [knn_fast] Using pre-built FAISS index: {type(faiss_index).__name__} on {faiss_device_info}")

    # print(f"  [knn_fast] Index build/setup time: {time.time() - t_index_start:.4f}s") # Less verbose

    # print(f"  [knn_fast] Searching for k+1={k+1} neighbors...") # Less verbose
    t_search_start = time.time()
    # Perform the search using the determined index (either provided or built)
    vals_np, inds_np = faiss_index.search(X_np, k + 1) # Search for k+1 (includes self)
    # print(f"  [knn_fast] FAISS search time: {time.time() - t_search_start:.4f}s") # Less verbose
    # --- End of setup part ---

    # --- Filtering and Thresholding ---
    # print("  [knn_fast] Filtering neighbors...") # Less verbose
    t_filter_start = time.time()
    filtered_rows_list = []
    filtered_cols_list = []
    filtered_vals_list = []

    similarity_threshold = -np.inf # Default: keep all valid neighbors

    # Calculate threshold if needed (only if type is not 'none' and k > 0)
    if knn_threshold_type != 'none' and k > 0:
        # Get similarities of the k-th neighbor (index k, since index 0 is self)
        # Ensure we don't index out of bounds if k+1 > number of results
        kth_neighbor_col_idx = min(k, vals_np.shape[1] - 1)
        if kth_neighbor_col_idx > 0: # Only calculate if k > 0 and results exist
            kth_similarities = vals_np[:, kth_neighbor_col_idx]
            # Consider only valid neighbors for threshold calculation (index != -1)
            valid_indices_mask = inds_np[:, kth_neighbor_col_idx] != -1
            if np.any(valid_indices_mask):
                valid_kth_similarities = kth_similarities[valid_indices_mask]

                if knn_threshold_type == 'median_k':
                    similarity_threshold = np.median(valid_kth_similarities)
                    print(f"    [knn_fast] Calculated 'median_k' threshold: {similarity_threshold:.4f} (based on {len(valid_kth_similarities)} nodes' k-th neighbor)")

                elif knn_threshold_type == 'std_dev_k':
                    mean_kth_sim = np.mean(valid_kth_similarities)
                    std_kth_sim = np.std(valid_kth_similarities)
                    # Handle std_dev being zero or very small
                    if std_kth_sim < EOS:
                         print(f"    [knn_fast] Warning: Standard deviation of k-th similarities is near zero ({std_kth_sim:.4e}). Using mean ({mean_kth_sim:.4f}) as threshold.")
                         similarity_threshold = mean_kth_sim
                    else:
                         # Higher similarity is better, so threshold is mean - factor * std
                         similarity_threshold = mean_kth_sim - knn_std_dev_factor * std_kth_sim
                         print(f"    [knn_fast] Calculated 'std_dev_k' threshold: {similarity_threshold:.4f} (mean={mean_kth_sim:.4f}, std={std_kth_sim:.4f}, factor={knn_std_dev_factor})")
                else: # Should not happen if choices are enforced by argparse
                     print(f"    [knn_fast] Unknown threshold type '{knn_threshold_type}'. Keeping all neighbors.")
            else: # No valid k-th neighbors found
                 print(f"    [knn_fast] Warning: No valid k-th neighbors found. Cannot compute '{knn_threshold_type}' threshold. Keeping all neighbors.")
        else: # k=0 or no neighbors found beyond self
            print(f"    [knn_fast] Warning: k={k} is too small or not enough neighbors found to compute '{knn_threshold_type}' threshold. Keeping all neighbors.")


    # --- Filtering Logic (Vectorized NumPy) ---
    t_filter_start = time.time()
    # print("  [knn_fast] Filtering neighbors (vectorized)...") # Less verbose

    # Exclude self (column 0)
    neighbor_inds = inds_np[:, 1:] # Shape (n, k)
    neighbor_vals = vals_np[:, 1:] # Shape (n, k)

    # 1. Basic Validity Mask
    valid_mask = (neighbor_inds != -1) & (neighbor_inds < n) # Shape (n, k)

    # 2. Threshold Mask
    # Apply threshold: keep if similarity is >= threshold
    threshold_keep_mask = neighbor_vals >= similarity_threshold # Shape (n, k)

    # 3. Identify neighbors kept based *only* on threshold and validity
    preliminary_keep_mask = valid_mask & threshold_keep_mask # Shape (n, k)

    # 4. Ensure the *first valid* neighbor is always kept (original logic)
    final_keep_mask = preliminary_keep_mask.copy() # Start with thresholded neighbors
    # Find the column index of the first valid neighbor for each row
    # Need np.where to handle rows with no valid neighbors (argmax would return 0)
    rows_with_valid, first_valid_col_indices = np.where(valid_mask)
    # Use unique rows to find the *first* occurrence index per row
    unique_rows, first_occurrence_indices = np.unique(rows_with_valid, return_index=True)
    cols_to_force_keep = first_valid_col_indices[first_occurrence_indices]

    # If there are rows with valid neighbors, force-keep the first valid one
    if len(unique_rows) > 0:
        final_keep_mask[unique_rows, cols_to_force_keep] = True

    # 5. Apply the final mask to get the edges
    # Generate row indices corresponding to the neighbor_inds array
    row_indices_np = np.arange(n).reshape(-1, 1) # Shape (n, 1)
    # Apply the mask
    final_rows_np = np.broadcast_to(row_indices_np, neighbor_inds.shape)[final_keep_mask]
    final_cols_np = neighbor_inds[final_keep_mask]
    final_vals_np = neighbor_vals[final_keep_mask]

    num_kept_edges = final_rows_np.shape[0]
    # --- End Vectorized Filtering ---

    # print(f"  [knn_fast] Filtering done. Kept {num_kept_edges} edges. Time: {time.time() - t_filter_start:.4f}s") # Less verbose

    # Convert lists to tensors
    if num_kept_edges > 0:
        # rows = torch.tensor(filtered_rows_list, dtype=torch.long, device=device)
        # cols = torch.tensor(filtered_cols_list, dtype=torch.long, device=device)
        # values = torch.tensor(filtered_vals_list, dtype=torch.float32, device=device)
        rows = torch.from_numpy(final_rows_np).to(device=device, dtype=torch.long)
        cols = torch.from_numpy(final_cols_np).to(device=device, dtype=torch.long)
        values = torch.from_numpy(final_vals_np).to(device=device, dtype=torch.float32)
    else:
        rows = torch.tensor([], dtype=torch.long, device=device)
        cols = torch.tensor([], dtype=torch.long, device=device)
        values = torch.tensor([], dtype=torch.float32, device=device)

    # print(f"[knn_fast] Finished. Total time: {time.time() - t_start_knn:.4f}s. Returning {rows.shape[0]} edges.") # Less verbose
    return rows, cols, values # Return raw similarities


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    device = values.device  # Get the device from the input tensor
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device=device)
    dgl_graph.edata['w'] = values.detach()  # Remove .cuda() since we'll use the same device as input
    return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_eye(num_nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    indices = torch.arange(num_nodes, device=device).repeat(2, 1)
    values = torch.ones(num_nodes, device=device)
    return torch.sparse.FloatTensor(indices, values)


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluationClusterModelFromLabel(self, print_results=True):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()

        if print_results:
            print('ACC={:.4f}, f1_macro={:.4f}, precision_macro={:.4f}, recall_macro={:.4f}, f1_micro={:.4f}, '
                  .format(acc, f1_macro, precision_macro, recall_macro, f1_micro) +
                  'precision_micro={:.4f}, recall_micro={:.4f}, NMI={:.4f}, ADJ_RAND_SCORE={:.4f}'
                  .format(precision_micro, recall_micro, nmi, adjscore))

        return acc, nmi, f1_macro, adjscore