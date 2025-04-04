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

EOS = 1e-10


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

def get_memory_usage():
    import psutil
    import os
    """Return the current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Convert bytes to GB


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


def knn_fast(X, k, b, use_gpu=False):
    """
    Computes the k-nearest neighbors graph using FAISS with an Approximate
    Nearest Neighbor (ANN) index (IndexIVFFlat) for efficiency.
    Can optionally run FAISS operations on GPU if available and requested.

    Args:
        X (torch.Tensor): Input features (num_nodes x num_features).
        k (int): Number of neighbors to find for each node.
        b (int): Batch size parameter (ignored in this FAISS implementation).
        use_gpu (bool): If True, attempt to use GPU for FAISS operations.
                      Defaults to False.

    Returns:
        tuple: (rows, cols, values) representing the graph in COO format
               with normalized edge weights.
    """
    start_time = time.time()
    device = X.device
    n, d = X.shape # number of nodes, dimension of features

    # Normalize features (always on input device, likely GPU)
    X_normalized = F.normalize(X, dim=1, p=2)

    # --- FAISS Setup --- 
    # Heuristic for nlist (number of Voronoi cells/partitions)
    nlist = min(n // 40, max(1, int(4 * np.sqrt(n))))
    nlist = max(1, nlist) # Ensure nlist is at least 1
    # Heuristic for nprobe (number of cells to search)
    nprobe = min(nlist, max(1, nlist // 10))
    nprobe = max(1, nprobe) # Ensure nprobe is at least 1

    actual_use_gpu = use_gpu and faiss.get_num_gpus() > 0
    faiss_device_info = "CPU"
    index_type = "IndexIVFFlat" # Default for CPU

    if actual_use_gpu:
        try:
            # --- GPU FAISS Path (Using GpuIndexFlatIP - No Training Required) ---
            faiss_device_info = "GPU"
            index_type = "GpuIndexFlatIP"
            print(f"Attempting FAISS ({faiss_device_info}, {index_type})")
            res = faiss.StandardGpuResources()
            
            # Create GPU Flat Index directly
            gpu_index = faiss.GpuIndexFlatIP(res, d, faiss.METRIC_INNER_PRODUCT)

            # No training needed for IndexFlatIP

            # Add data to GPU index
            gpu_index.add(X_normalized) # Pass GPU tensor

            # No nprobe needed for IndexFlatIP

            # Search using GPU index and GPU data
            vals, inds = gpu_index.search(X_normalized, k + 1) # Returns tensors on GPU
            # --- End GPU FAISS Path ---

        except Exception as e:
            print(f"FAISS GPU execution failed ({index_type}): {e}. Falling back to CPU.")
            actual_use_gpu = False # Reset flag to trigger CPU path

    # CPU Path (or fallback from GPU error) - Still using IndexIVFFlat
    if not actual_use_gpu:
        # --- CPU FAISS Path (Using IndexIVFFlat) ---
        faiss_device_info = "CPU"
        index_type = "IndexIVFFlat"
        print(f"Using FAISS ({faiss_device_info}, {index_type}) with nlist={nlist}, nprobe={nprobe}")
        # Convert to numpy for CPU FAISS
        X_np = X_normalized.cpu().detach().numpy().astype('float32')
        X_np = np.ascontiguousarray(X_np)

        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

        if not index.is_trained:
            index.train(X_np)
        index.add(X_np)
        index.nprobe = nprobe

        # Search using CPU index
        vals_np, inds_np = index.search(X_np, k + 1)

        # Convert results back to PyTorch tensors on the original device
        vals = torch.from_numpy(vals_np).to(device)
        inds = torch.from_numpy(inds_np).to(device)
        # --- End CPU FAISS Path ---

    # --- Post-processing (Common to both paths) ---

    # Handle potential -1 indices from FAISS search (indices are on 'device' now)
    valid_mask_faiss = (inds != -1).flatten() 
    vals = torch.clamp(vals, min=0.0) # Ensure similarities are non-negative

    # Create COO structure (Initial creation based on expected full shape)
    rows_full = torch.arange(n, device=device).view(-1, 1).repeat(1, k + 1).view(-1)
    cols_full = inds.view(-1)
    values_full = vals.view(-1)

    # Apply the valid mask from FAISS search results
    rows = rows_full[valid_mask_faiss]
    cols = cols_full[valid_mask_faiss]
    values = values_full[valid_mask_faiss]

    # --- Normalization (Common logic, uses tensors on 'device') ---
    vals_safe = vals.clone() # vals has shape (n, k+1) tensor on 'device'
    vals_safe[inds == -1] = 0.0 # inds is (n, k+1) tensor on 'device'

    norm_row = torch.sum(vals_safe, dim=1) # Shape (n,)

    norm_col = torch.zeros(n, device=device)
    # Use the *filtered* cols and values (already on 'device')
    norm_col.index_add_(0, cols, values)

    norm = norm_row + norm_col + EOS # Add EOS for stability

    # Apply symmetric normalization D^(-1/2) * A * D^(-1/2)
    # Use the filtered rows/cols indices to access the correct norm values
    norm_rows_sqrt_inv = torch.pow(norm[rows], -0.5)
    norm_cols_sqrt_inv = torch.pow(norm[cols], -0.5)
    # Apply normalization only to the valid 'values'
    values *= norm_rows_sqrt_inv * norm_cols_sqrt_inv
    # --- End Normalization ---

    # Ensure indices are long type
    rows = rows.long()
    cols = cols.long()

    end_time = time.time()
    print(f"knn_fast ({faiss_device_info}, {index_type}) execution time: {end_time - start_time:.4f} seconds")

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