import argparse
import torch
import pandas as pd
import numpy as np
from main import Experiment

# Add device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_person_data(args):
    """
    Load person data from CSV and prepare it for SUBLIME self-supervised learning.
    
    Args:
        args: Arguments containing dataset path
    
    Returns:
        Tuple of (features, nfeats, None, n_clusters, None, None, None, adj)
        where None values represent unused label-related data
    """
    # Load data
    df = pd.read_csv(args.dataset)
    
    # Convert features to torch tensor and move to device
    features = torch.FloatTensor(df.values).to(device)
    
    # Get dimensions
    nfeats = features.shape[1]
    n_samples = features.shape[0]
    
    # For initial adjacency, we return identity matrix (no edges)
    # SUBLIME will learn the graph structure
    if args.sparse:
        from utils import torch_sparse_eye
        adj = torch_sparse_eye(n_samples)
    else:
        adj = torch.eye(n_samples).to(device)
    
    # Return None for all label-related data since we're doing self-supervised learning
    # For clustering, we need to specify the number of clusters
    return features, nfeats, None, args.n_clusters, None, None, None, adj

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
    parser.add_argument('-contrast_batch_size', type=int, default=0)
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