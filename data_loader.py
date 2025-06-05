import warnings
import pickle as pkl
import sys, os

import scipy.sparse as sp
import networkx as nx
import torch
import numpy as np

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy

from utils import sparse_mx_to_torch_sparse_tensor #, dgl_graph_to_torch_sparse

warnings.simplefilter("ignore")


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_citation_network(dataset_str, sparse=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not sparse:
        adj = np.array(adj.todense(),dtype='float32')
    else:
        adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    features = torch.FloatTensor(features.todense())
    labels = torch.LongTensor(labels)
    train_mask = torch.BoolTensor(train_mask)
    val_mask = torch.BoolTensor(val_mask)
    test_mask = torch.BoolTensor(test_mask)

    nfeats = features.shape[1]
    for i in range(labels.shape[0]):
        sum_ = torch.sum(labels[i])
        if sum_ != 1:
            labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
    labels = (labels == 1).nonzero()[:, 1]
    nclasses = torch.max(labels).item() + 1

    return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj


def load_data(args):
    """
    Load data from a file.
    
    Args:
        args: Arguments containing dataset path and other parameters
        
    Returns:
        Tuple of (features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj)
        For annotated datasets, the binary label is expected to be in the last column of the dataset
    """
    # For standard citation networks
    if args.dataset in ['cora', 'citeseer', 'pubmed']:
        return load_citation_network(args.dataset, args.sparse)
        
    # Check if we're loading a CSV file
    if args.dataset and args.dataset.endswith('.csv'):
        # Load from CSV file
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # Read the CSV file
        try:
            data = pd.read_csv(args.dataset)
        except Exception as e:
            print(f"Error loading {args.dataset}: {str(e)}")
            raise
            
        # Check if this is the annotated dataset (with binary labels)
        if hasattr(args, 'annotated_dataset') and args.dataset == args.annotated_dataset:
            if args.annotation_column in data.columns:
                # Use the specified column as the binary target
                target_column = args.annotation_column
            else:
                # Assume the last column is the binary target
                target_column = data.columns[-1]
                print(f"Warning: annotation_column '{args.annotation_column}' not found, using last column '{target_column}' as binary target")
                
            # Extract features and binary labels
            features = data.drop(columns=[target_column]).values.astype('float32')
            binary_labels = data[target_column].values.astype('int')
            
            # Ensure binary labels are 0 and 1
            unique_labels = np.unique(binary_labels)
            if len(unique_labels) > 2:
                raise ValueError(f"Binary labels should have at most 2 classes, found {len(unique_labels)}: {unique_labels}")
            if not set(unique_labels).issubset({0, 1}):
                print(f"Warning: Converting labels {unique_labels} to 0 and 1")
                # Map the smallest value to 0 and largest to 1
                label_map = {unique_labels[0]: 0, unique_labels[-1]: 1}
                binary_labels = np.array([label_map[lbl] for lbl in binary_labels])
            
            # Append binary labels as the last column of features for downstream processing
            features = np.column_stack((features, binary_labels))
            
            # For annotated data, no need for masks or adjacency matrix
            # We'll return placeholders that will be ignored
            nfeats = features.shape[1] - 1  # Subtract 1 for the appended label
            labels = None
            nclasses = 2  # Binary classification
            train_mask = val_mask = test_mask = None
            adj = None if not args.sparse else torch.sparse.FloatTensor(2, 2)
            
            return torch.FloatTensor(features), nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj
        
        # Regular dataset (not annotated)
        features = data.values.astype('float32')
        nfeats = features.shape[1]
        
        # For self-supervised learning, we don't use explicit labels
        # but can set them to row indices for use with ArcFace if needed
        labels = torch.arange(features.shape[0])
        nclasses = features.shape[0]  # Each node is its own class for contrastive learning
        
        # For regular CSV data, we don't need masks
        train_mask = val_mask = test_mask = None
        
        # For regular CSV data with structure inference, we don't use a predefined adjacency
        adj = None if not args.sparse else torch.sparse.FloatTensor(2, 2)
        
        return torch.FloatTensor(features), nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj
    
    # Fall back to citation network loading if not using CSV
    return load_citation_network(args.dataset, args.sparse)