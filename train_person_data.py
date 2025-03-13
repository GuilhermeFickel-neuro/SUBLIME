import argparse
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from main import Experiment
from utils import sparse_mx_to_torch_sparse_tensor

# Add device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_person_data(args):
    """
    Load person data from CSV and prepare it for SUBLIME self-supervised learning.
    
    Args:
        args: Arguments containing dataset path
    
    Returns:
        Tuple of (features, nfeats, labels, n_clusters, train_mask, val_mask, test_mask, adj)
        matching the structure of load_citation_network
    """
    # Load data
    df = pd.read_csv(args.dataset)
    
    # Get dimensions
    n_samples = df.shape[0]
    
    # Convert features to sparse matrix first for consistency
    features = sp.lil_matrix(df.values, dtype=np.float32)
    
    # Create empty labels (since this is unsupervised)
    labels = None
    
    # Create empty masks (not used in unsupervised setting but needed for consistency)
    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    
    # Handle adjacency matrix
    if args.sparse:
        # Create sparse identity matrix using scipy first
        identity = sp.eye(n_samples, dtype=np.float32)
        # Convert to torch sparse tensor using the same method as in citation network
        adj = sparse_mx_to_torch_sparse_tensor(identity)
    else:
        adj = torch.eye(n_samples).to(device)
    
    # Convert features to dense torch tensor
    features = torch.FloatTensor(features.todense())
    
    # Get feature dimension
    nfeats = features.shape[1]
    
    # Return in the same format as load_citation_network
    return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, adj

def main():
    # Parse arguments (using same defaults as original code)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='person_data.csv')
    parser.add_argument('-ntrials', type=int, default=1)  # Reduced since we don't need multiple trials for self-supervised
    parser.add_argument('-sparse', type=int, default=0)  # Changed default to 0 for CPU compatibility
    parser.add_argument('-gsl_mode', type=str, default="structure_inference")
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='clustering')  # Changed to clustering since we don't have labels
    parser.add_argument('-n_clusters', type=int, default=5, help='Number of clusters for clustering task')
    
    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=10000)
    parser.add_argument('-nlayers', type=int, default=2)
    
    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    
    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp')
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine')
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu')
    
    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)
    
    args = parser.parse_args()
    
    # Train model
    experiment = Experiment(device)
    experiment.train(args, load_data_fn=load_person_data)

if __name__ == "__main__":
    main() 
