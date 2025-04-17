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
def build_faiss_index(features, k=10, use_gpu=False, nprobe=None):
    """
    Build and return a FAISS index for the given features.
    This can be reused to avoid rebuilding the index on each call.
    
    Args:
        features (torch.Tensor): Input features (num_nodes x num_features).
        k (int): Number of neighbors to find for each node (used to tune index).
        use_gpu (bool): If True, attempt to use GPU for FAISS operations.
        nprobe (int, optional): Number of clusters to visit (higher improves recall).
                               
    Returns:
        faiss.Index: A trained FAISS index containing the features
    """
    print("Building reusable FAISS index...")
    t_start = time.time()
    
    # Get device and dimensions
    device = features.device
    n, d = features.shape
    t0 = time.time()
    print(f"  [build_index] Input shape: {n}x{d} on {device}. Time: {t0-t_start:.4f}s")
    
    # Normalize features for cosine similarity
    X_normalized = F.normalize(features, dim=1, p=2)
    t1 = time.time()
    print(f"  [build_index] Normalization time: {t1-t0:.4f}s")
    
    # Convert to numpy for FAISS
    X_np = X_normalized.cpu().detach().numpy().astype('float32')
    X_np = np.ascontiguousarray(X_np)
    t2 = time.time()
    print(f"  [build_index] Data transfer & conversion time: {t2-t1:.4f}s")
    
    # Calculate appropriate values for nlist (num of clusters) and M (num of subquantizers)
    def get_valid_pq_m(d, max_m=16):
        divisors = [i for i in range(1, min(max_m+1, d+1)) if d % i == 0]
        return max(divisors) if divisors else 1
    
    if n < 10000:
        nlist = min(int(np.sqrt(n)), n // 20)
        nbits = 6
    else:
        nlist = min(4 * int(np.sqrt(n)), n // 10)
        nbits = 8
    
    M = get_valid_pq_m(d)
    t3 = time.time()
    print(f"  [build_index] Parameter calculation time: {t3-t2:.4f}s")
    print(f"  [build_index] Calculated params: nlist={nlist}, M={M}, nbits={nbits}")
    
    # Set nprobe
    actual_nprobe = nprobe
    if actual_nprobe is None:
        actual_nprobe = min(nlist // 2, 100)
    print(f"  [build_index] Using nprobe={actual_nprobe}")
    
    # Build the index
    actual_use_gpu = use_gpu and faiss.get_num_gpus() > 0
    index_type = "IndexIVFPQ"
    faiss_device_info = "CPU"
    index = None

    if actual_use_gpu:
        try:
            faiss_device_info = "GPU"
            print(f"  [build_index] Building FAISS index on GPU with {index_type}")
            t_gpu_start = time.time()
            res = faiss.StandardGpuResources()
            t_res = time.time()
            print(f"    [GPU Build] StandardGpuResources init time: {t_res-t_gpu_start:.4f}s")
            
            # Create CPU index first
            cpu_quantizer = faiss.IndexFlatIP(d)
            cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
            t_cpu_idx = time.time()
            print(f"    [GPU Build] CPU index ({type(cpu_index).__name__}) creation time: {t_cpu_idx-t_res:.4f}s")
            
            # Train on CPU
            print(f"    [GPU Build] Training CPU index...")
            cpu_index.train(X_np)
            t_train = time.time()
            print(f"    [GPU Build] CPU index training time: {t_train-t_cpu_idx:.4f}s")
            
            # Move index to GPU after training
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            t_gpu_conv = time.time()
            print(f"    [GPU Build] Index CPU->GPU conversion time: {t_gpu_conv-t_train:.4f}s")
            
            # Set search parameters
            gpu_index.nprobe = actual_nprobe
            
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
            index = None

    if not actual_use_gpu:
        faiss_device_info = "CPU"
        print(f"  [build_index] Building FAISS index on CPU with {index_type}")
        t_cpu_build_start = time.time()
        
        # Create quantizer and IVFPQ index
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
        t_idx_create = time.time()
        print(f"    [CPU Build] Index ({type(index).__name__}) creation time: {t_idx_create-t_cpu_build_start:.4f}s")
        
        # Train the index
        print(f"    [CPU Build] Training CPU index...")
        index.train(X_np)
        t_train = time.time()
        print(f"    [CPU Build] CPU index training time: {t_train-t_idx_create:.4f}s")
        
        # Set search parameters
        index.nprobe = actual_nprobe
        
        # Add vectors to the index
        print(f"    [CPU Build] Adding data to CPU index...")
        index.add(X_np)
        t_add = time.time()
        print(f"    [CPU Build] CPU index add time: {t_add-t_train:.4f}s")
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


def knn_fast(X, k, b=None, use_gpu=False, nprobe=None, faiss_index=None):
    """
    Optimized KNN implementation using FAISS with IndexIVFPQ.
    Includes detailed timing prints and option to reuse a pre-built index.

    Args:
        X (torch.Tensor): Input features (num_nodes x num_features).
        k (int): Number of neighbors to find for each node.
        b (int): Batch size parameter (kept for backward compatibility, not used).
        use_gpu (bool): If True, attempt to use GPU for FAISS operations. Defaults to False.
        nprobe (int, optional): Number of clusters to visit (higher improves recall).
                               If None, a suitable value is selected automatically.
        faiss_index (faiss.Index, optional): A pre-built and trained FAISS index.

    Returns:
        tuple: (rows, cols, values) representing the graph in COO format
               with normalized edge weights.
    """
    t_start_knn = time.time()
    device = X.device
    n, d = X.shape
    print(f"\n--- knn_fast Start ---")
    print(f"  Input: {n}x{d} features on {device}, k={k}, use_gpu={use_gpu}, nprobe={nprobe}, index_provided={faiss_index is not None}")

    # --- Preprocessing ---
    t0 = time.time()
    X_normalized = F.normalize(X, dim=1, p=2)
    t1 = time.time()
    print(f"  [knn_fast] Step 1: F.normalize time: {t1-t0:.4f}s")
    X_np = X_normalized.cpu().detach().numpy().astype('float32')
    X_np = np.ascontiguousarray(X_np)
    t2 = time.time()
    print(f"  [knn_fast] Step 2: Torch->CPU->NumPy conversion time: {t2-t1:.4f}s")
    # ---------------------

    index_built_or_trained = False # Flag to track if we built the index here
    index_total_build_time = 0.0 # Track time if index is built here
    internal_index = None # Define here to ensure scope if building

    if faiss_index is None:
        # --- Build and Train Index if not provided ---
        print("  [knn_fast] Step 3a: Building internal FAISS index...")
        index_built_or_trained = True
        t_build_start = time.time()

        def get_valid_pq_m(d, max_m=16): # Inner func ok
            divisors = [i for i in range(1, min(max_m+1, d+1)) if d % i == 0]
            return max(divisors) if divisors else 1

        if n < 10000: nlist = min(int(np.sqrt(n)), n // 20); nbits = 6
        else: nlist = min(4 * int(np.sqrt(n)), n // 10); nbits = 8
        M = get_valid_pq_m(d)
        actual_nprobe = nprobe if nprobe is not None else min(nlist // 2, 100)
        print(f"    Index Params: nlist={nlist}, M={M}, nbits={nbits}, nprobe={actual_nprobe}")

        actual_use_gpu = use_gpu and faiss.get_num_gpus() > 0
        faiss_device_info = "CPU"
        # internal_index already defined

        if actual_use_gpu:
            try:
                faiss_device_info = "GPU"
                t_gpu_build_i_start = time.time()
                res = faiss.StandardGpuResources()
                cpu_quantizer = faiss.IndexFlatIP(d)
                cpu_index = faiss.IndexIVFPQ(cpu_quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT)
                t_gpu_init_done = time.time()
                print(f"      GPU: Resource/Index init time: {t_gpu_init_done - t_gpu_build_i_start:.4f}s")
                cpu_index.train(X_np)
                t_gpu_train_done = time.time()
                print(f"      GPU: CPU training time: {t_gpu_train_done - t_gpu_init_done:.4f}s")
                gpu_index_internal = faiss.index_cpu_to_gpu(res, 0, cpu_index) # Rename var
                t_gpu_conv_done = time.time()
                print(f"      GPU: CPU->GPU conversion time: {t_gpu_conv_done - t_gpu_train_done:.4f}s")
                gpu_index_internal.nprobe = actual_nprobe
                gpu_index_internal.add(X_np)
                t_gpu_add_done = time.time()
                print(f"      GPU: Add data time: {t_gpu_add_done - t_gpu_conv_done:.4f}s")
                internal_index = gpu_index_internal # Assign to local var
                index_total_build_time = t_gpu_add_done - t_gpu_build_i_start
            except Exception as e:
                print(f"    FAISS GPU build failed: {e}. Falling back to CPU.")
                actual_use_gpu = False
                internal_index = None # Reset

        if not actual_use_gpu: # This block will execute if GPU failed or was not requested
            faiss_device_info = "CPU"
            t_cpu_build_i_start = time.time()
            quantizer = faiss.IndexFlatIP(d)
            index_cpu_internal = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits, faiss.METRIC_INNER_PRODUCT) # Rename var
            t_cpu_init_done = time.time()
            print(f"      CPU: Index init time: {t_cpu_init_done - t_cpu_build_i_start:.4f}s")
            index_cpu_internal.train(X_np)
            t_cpu_train_done = time.time()
            print(f"      CPU: Training time: {t_cpu_train_done - t_cpu_init_done:.4f}s")
            index_cpu_internal.nprobe = actual_nprobe
            index_cpu_internal.add(X_np)
            t_cpu_add_done = time.time()
            print(f"      CPU: Add data time: {t_cpu_add_done - t_cpu_train_done:.4f}s")
            internal_index = index_cpu_internal # Assign to local var
            index_total_build_time = t_cpu_add_done - t_cpu_build_i_start

        faiss_index = internal_index # Assign the newly built index to the main variable
        if faiss_index is None: # Check if index build failed completely
             raise RuntimeError("Failed to build internal FAISS index on both CPU and GPU.")
        print(f"  [knn_fast] Step 3a: Index built internally on {faiss_device_info}. Total build/train/add time: {index_total_build_time:.4f}s")
        # ------------------------------------
    else:
        # --- Use Provided Index ---
        print("  [knn_fast] Step 3b: Using pre-built FAISS index.")
        t_prebuilt_start = time.time()
        actual_nprobe = nprobe # Use provided nprobe if any
        if actual_nprobe is None:
             if hasattr(faiss_index, 'nprobe') and faiss_index.nprobe > 0: # Check if nprobe is already set
                 actual_nprobe = faiss_index.nprobe
             else: # Estimate or default if not set
                 if isinstance(faiss_index, faiss.IndexIVFPQ):
                     nlist = faiss_index.nlist
                     # Ensure nlist is positive before division
                     actual_nprobe = min(nlist // 2, 100) if nlist > 0 else 10
                 else: actual_nprobe = 10
        # Ensure nprobe is set on the index for this search
        if hasattr(faiss_index, 'nprobe'):
            # print(f"    Setting nprobe={actual_nprobe} on provided index.")
            faiss_index.nprobe = actual_nprobe
        # else:
            # print(f"    Warning: Cannot set nprobe on provided index type {type(faiss_index).__name__}.")

        # Determine device info for provided index - IMPROVED LOGIC
        faiss_device_info = "Unknown"
        if hasattr(faiss_index, 'getResources'): # Check if it's likely a GpuIndex variant
            try:
                 # Attempt to get device from resources if possible
                 # This is heuristic as structure can vary
                 res = faiss_index.getResources()
                 # Check common attributes or types associated with GPU resources
                 if hasattr(res, 'getGpuDevice'):
                      faiss_device_info = f"GPU (device {res.getGpuDevice()})"
                 elif hasattr(res, 'devices'): # For older/different resource types
                      faiss_device_info = f"GPU (devices {res.devices})"
                 else:
                      faiss_device_info = "GPU (Unknown Device Details)"
            except Exception:
                 faiss_device_info = "GPU (Error getting details)" # Could be GPU but failed inspection
        elif isinstance(faiss_index, faiss.Index) and not hasattr(faiss_index, 'getResources'): # Likely a CPU index
            faiss_device_info = "CPU"
        else:
            faiss_device_info = f"Unknown Type ({type(faiss_index).__name__})"

        print(f"    Index type: {type(faiss_index).__name__} on {faiss_device_info}, using nprobe={actual_nprobe}.")
        t_prebuilt_end = time.time()
        print(f"  [knn_fast] Step 3b: Setup time for pre-built index: {t_prebuilt_end - t_prebuilt_start:.4f}s")
        # --------------------------

    if faiss_index is None:
        raise RuntimeError("FAISS index is None after build/setup phase.")

    # --- Search ---
    t_search_start = time.time()
    # print(f"  [knn_fast] Step 4: Searching neighbors using {faiss_device_info} index...")
    # Search using X_np (features from *this* function call)
    vals_np, inds_np = faiss_index.search(X_np, k + 1) # Search for k+1 neighbors
    t_search_end = time.time()
    search_time = t_search_end - t_search_start
    print(f"  [knn_fast] Step 4: FAISS search ({k+1} neighbors) time: {search_time:.4f}s")
    # --------------

    # Optional: Clean up internally built index immediately after search if desired
    # This might cause issues if the index object is needed elsewhere after knn_fast returns
    # Consider cleanup outside this function if index is built internally but needed later.
    # if index_built_or_trained and internal_index is not None:
    #     del internal_index # Delete the specific local variable used for internal build
    #     # Avoid deleting faiss_index if it was passed externally
    #     gc.collect()
    #     print("  [knn_fast] Cleaned up internally built index.")

    # --- Post-processing ---
    t_post_start = time.time()
    vals = torch.from_numpy(vals_np).to(device)
    inds = torch.from_numpy(inds_np).to(device)
    t_conv_back = time.time()
    print(f"    [PostProc] NumPy->Torch conversion time: {t_conv_back - t_post_start:.4f}s")

    valid_mask_faiss = (inds != -1) # Mask for valid indices
    vals = torch.clamp(vals, min=0.0) # Ensure non-negative similarities
    t_clamp = time.time()
    print(f"    [PostProc] Clamp/Mask creation time: {t_clamp - t_conv_back:.4f}s")

    # Create COO structure from valid entries only
    rows_full = torch.arange(n, device=device).view(-1, 1).repeat(1, k + 1)
    rows = rows_full[valid_mask_faiss]
    cols = inds[valid_mask_faiss]
    values = vals[valid_mask_faiss]
    t_coo = time.time()
    print(f"    [PostProc] COO structure creation time: {t_coo - t_clamp:.4f}s")


    # --- Normalization ---
    # Use original vals/inds for normalization sum, but zero out invalid contributions
    vals_safe = vals.clone()
    vals_safe[~valid_mask_faiss] = 0.0 # Zero out contributions from invalid indices (-1)

    norm_row = torch.sum(vals_safe, dim=1) # Sum across K dim for each node
    t_norm_row = time.time()
    print(f"    [PostProc Norm] Row sum time: {t_norm_row - t_coo:.4f}s")

    # Calculate column sums using index_add_ on the valid edges
    norm_col = torch.zeros(n, device=device)
    # Ensure cols are valid indices before using index_add
    if rows.numel() > 0: # Check if there are any valid edges
         # Ensure cols indices are within the bounds of norm_col
         # This shouldn't be necessary if FAISS inds are correct, but as a safeguard:
         valid_cols_mask = (cols >= 0) & (cols < n)
         if not valid_cols_mask.all():
              print(f"    Warning: {torch.sum(~valid_cols_mask)} invalid column indices found before index_add_!")
              # Filter based on valid cols mask
              rows = rows[valid_cols_mask]
              values = values[valid_cols_mask]
              cols = cols[valid_cols_mask]

         if cols.numel() > 0: # Check again after potential filtering
             norm_col.index_add_(0, cols, values) # Use valid 'cols' and 'values'

    t_norm_col = time.time()
    print(f"    [PostProc Norm] Col sum (index_add) time: {t_norm_col - t_norm_row:.4f}s")

    norm = norm_row + norm_col + EOS # Total degree + EOS

    # Apply symmetric normalization D^(-1/2) * A * D^(-1/2) using valid rows/cols
    # Ensure norm[rows] and norm[cols] don't contain zeros or negatives before pow(-0.5)
    norm_rows_safe = torch.clamp(norm[rows], min=EOS)
    norm_cols_safe = torch.clamp(norm[cols], min=EOS)
    norm_rows_sqrt_inv = torch.pow(norm_rows_safe, -0.5)
    norm_cols_sqrt_inv = torch.pow(norm_cols_safe, -0.5)
    values *= norm_rows_sqrt_inv * norm_cols_sqrt_inv
    t_norm_apply = time.time()
    print(f"    [PostProc Norm] D^-1/2 A D^-1/2 application time: {t_norm_apply - t_norm_col:.4f}s")

    # Ensure indices are long type
    rows = rows.long()
    cols = cols.long()
    # --------------------
    t_post_end = time.time()
    postproc_time = t_post_end - t_post_start
    print(f"  [knn_fast] Step 5: Post-processing & Normalization time: {postproc_time:.4f}s")
    # -----------------------

    t_end_knn = time.time()
    total_knn_time = t_end_knn - t_start_knn
    index_source = "pre-built" if not index_built_or_trained else "built/trained"
    # Calculate overhead only if we have valid timings for build, search, postproc
    if index_total_build_time >= 0 and search_time >= 0 and postproc_time >= 0:
         overhead_time = total_knn_time - index_total_build_time - search_time - postproc_time
    else:
         overhead_time = np.nan # Indicate if timings were invalid
    print(f"--- knn_fast End ({index_source} index, {k} neighbors) ---")
    print(f"  Total Execution Time: {total_knn_time:.4f} seconds")
    print(f"  (Breakdown Est: Build={index_total_build_time:.4f}s, Search={search_time:.4f}s, PostProc={postproc_time:.4f}s, Overhead={overhead_time:.4f}s)")

    # Return only the graph structure
    return rows, cols, values


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