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
from utils import sparse_mx_to_torch_sparse_tensor, faiss_kneighbors_graph
from main import create_parser # Import the shared parser creation function

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
    
    # Handle adjacency matrix: Use k-NN graph instead of identity
    print(f"Constructing k-NN graph with k={args.k} using FAISS (cosine similarity)...")
    # Ensure data is float32 for FAISS
    if processed_data.dtype != np.float32:
        processed_data = processed_data.astype(np.float32)
        
    # Compute k-NN graph using faiss
    knn_graph_sparse = faiss_kneighbors_graph(processed_data, k=args.k, metric='cosine')
    print(f"k-NN graph computed. Shape: {knn_graph_sparse.shape}, Non-zero entries: {knn_graph_sparse.nnz}")

    if args.sparse:
        # Convert scipy sparse matrix to torch sparse tensor
        adj = sparse_mx_to_torch_sparse_tensor(knn_graph_sparse).to(device)
        print("Using sparse adjacency matrix.")
    else:
        # Convert scipy sparse matrix to dense torch tensor
        adj = torch.FloatTensor(knn_graph_sparse.todense()).to(device)
        print("Using dense adjacency matrix.")
    
    # Convert features to dense torch tensor
    features = torch.FloatTensor(features.todense()).to(device) # Move features to device
    
    # Get feature dimension
    nfeats = features.shape[1]
    
    print(f"Prepared dataset with {n_samples} samples and {nfeats} features")
    
    # Return in the same format as load_citation_network
    return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, adj

def main():
    # Get the base parser from main.py
    parent_parser = create_parser()

    # Create a new parser that inherits from the parent, but don't add help to avoid conflicts
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    # Set defaults specific to this script
    parser.set_defaults(
        dataset='person_data.csv',
        ntrials=1,  # Reduced since we don't need multiple trials for self-supervised
        sparse=0,   # Default to dense for broader compatibility initially
        gsl_mode="structure_refinement",
        eval_freq=500,
        downstream_task='clustering',
        epochs=10, # Example: Reduced epochs for person data
        contrast_batch_size=10000, # Example: Specific batch size
        save_model=1,
        output_dir='sublime_models/',
        verbose=1
        # Note: We inherit other defaults like k=30, type_learner='fgp' etc. from main.py
        # unless explicitly overridden here or via command line.
    )

    # Add arguments specific to this script
    parser.add_argument('-n_clusters', type=int, default=5, 
                       help='Number of clusters for clustering task')
    # --- Argument Definitions Removed --- 
    # All the parser.add_argument calls from line 143 to 232 are removed 
    # as they are inherited from the parent_parser.

    args = parser.parse_args()
    
    # Create experiment and train model
    experiment = Experiment(device)
    experiment.train(args, load_data_fn=load_person_data)

if __name__ == "__main__":
    main() 
