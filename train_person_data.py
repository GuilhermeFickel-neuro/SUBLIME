import argparse
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from main import Experiment
from utils import sparse_mx_to_torch_sparse_tensor

# Add device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_mixed_data(df, model_dir='sublime_models', load_transformer=True):
    """
    Preprocess a dataframe with mixed data types (categorical, numerical, missing values)
    
    Args:
        df: Pandas DataFrame with raw data
        model_dir: Directory to save/load transformation models
        load_transformer: Whether to load existing transformer or create a new one
    
    Returns:
        Preprocessed numpy array with all features converted to float and normalized to [-1, 1]
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    transformer_path = os.path.join(model_dir, 'data_transformer.joblib')
    
    # Separate numerical and categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if load_transformer and os.path.exists(transformer_path):
        # Load pre-fitted transformer
        print(f"Loading transformer from {transformer_path}")
        preprocessor = joblib.load(transformer_path)
    else:
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))  # Normalize to [-1, 1]
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode
        ])
        
        # Combine pipelines into a single transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ],
            remainder='drop'  # Drop columns that can't be transformed
        )
        
        # Fit the transformer
        print(f"Fitting new transformer and saving to {transformer_path}")
        preprocessor.fit(df)
        
        # Save the transformer
        joblib.dump(preprocessor, transformer_path)
    
    # Transform the data
    processed_data = preprocessor.transform(df)
    
    # Print data statistics
    print("\nData Transformation Summary:")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"Features retained: {processed_data.shape[1]} out of {df.shape[1]} original columns")
    print(f"Data range: [{processed_data.min():.4f}, {processed_data.max():.4f}]")
    print(f"Mean value: {processed_data.mean():.4f}, Std: {processed_data.std():.4f}")
    
    return processed_data

def load_person_data(args):
    """
    Load person data from CSV and prepare it for SUBLIME self-supervised learning.
    Handles mixed data types, missing values, and normalizes features.
    
    Args:
        args: Arguments containing dataset path
    
    Returns:
        Tuple of (features, nfeats, labels, n_clusters, train_mask, val_mask, test_mask, adj)
        matching the structure of load_citation_network
    """
    # Load data
    print(f"Loading dataset from {args.dataset}")
    df = pd.read_csv(args.dataset, delimiter='\t')
    
    # Get dimensions
    n_samples = df.shape[0]
    
    # Preprocess the data - convert to floats, handle missing values, normalize to [-1, 1]
    processed_data = preprocess_mixed_data(df, model_dir='sublime_models')
    
    # Convert features to sparse matrix for consistency with the original code
    features = sp.lil_matrix(processed_data, dtype=np.float32)
    
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
    
    print(f"Prepared dataset with {n_samples} samples and {nfeats} features")
    
    # Return in the same format as load_citation_network
    return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, adj

def main():
    # Parse arguments (using same defaults as original code)
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='person_data.csv')
    parser.add_argument('-ntrials', type=int, default=1)  # Reduced since we don't need multiple trials for self-supervised
    parser.add_argument('-sparse', type=int, default=0)  # Changed default to 0 for CPU compatibility
    parser.add_argument('-gsl_mode', type=str, default="structure_inference")
    parser.add_argument('-eval_freq', type=int, default=500)
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

    # One Cycle Learning Rate Scheduler parameters
    parser.add_argument('-use_one_cycle', type=int, default=0,
                        help='Whether to use OneCycleLR scheduler (0=disabled, 1=enabled)')
    parser.add_argument('-one_cycle_pct_start', type=float, default=0.3,
                        help='Percentage of cycle spent increasing learning rate (default: 0.3)')
    parser.add_argument('-one_cycle_div_factor', type=float, default=25.0,
                        help='Initial learning rate is max_lr/div_factor (default: 25.0)')
    parser.add_argument('-one_cycle_final_div_factor', type=float, default=10000.0,
                        help='Final learning rate is max_lr/(div_factor*final_div_factor) (default: 10000.0)')
    
    # ArcFace arguments
    parser.add_argument('-use_arcface', type=int, default=0,
                       help='Whether to use ArcFace loss (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_scale', type=float, default=30.0,
                       help='Scale factor for ArcFace (s parameter)')
    parser.add_argument('-arcface_margin', type=float, default=0.5,
                       help='Angular margin for ArcFace (m parameter)')
    parser.add_argument('-arcface_weight', type=float, default=1.0,
                       help='Weight for ArcFace loss when combining with contrastive loss')
    
        # Memory optimization arguments
    parser.add_argument('-use_batched_arcface', type=int, default=0,
                       help='Whether to use memory-efficient batched ArcFace implementation (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_batch_size', type=int, default=1000,
                       help='Number of classes to process in each ArcFace batch (default: 1000)')
    parser.add_argument('-debug_memory', type=int, default=0,
                       help='Run memory debugging before training (0=disabled, 1=enabled)')
    parser.add_argument('-max_debug_samples', type=int, default=10000,
                       help='Maximum samples to use for memory debugging (default: 10000)')
    parser.add_argument('-use_mixed_precision', type=int, default=0,
                       help='Use mixed precision training (FP16) to reduce memory usage (0=disabled, 1=enabled)')
    parser.add_argument('-grad_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before updating weights (1=disabled)')
    parser.add_argument('-memory_efficient_training', type=int, default=0,
                       help='Use memory-efficient training with gradient accumulation and chunked processing (0=disabled, 1=enabled)')

    parser.add_argument('-verbose', type=int, default=1)
    parser.add_argument('-save_model', type=int, default=1)
    parser.add_argument('-output_dir', type=str, default='sublime_models/')
    
    args = parser.parse_args()
    
    # Create experiment and train model
    experiment = Experiment(device)
    experiment.train(args, load_data_fn=load_person_data)

if __name__ == "__main__":
    main() 
