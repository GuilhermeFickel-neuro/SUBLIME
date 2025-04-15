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
        Tuple: (processed_data (np.ndarray), preprocessor (ColumnTransformer))
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    transformer_path = os.path.join(model_dir, 'data_transformer.joblib')
    
    # Separate numerical and categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if load_transformer and joblib and os.path.exists(transformer_path):
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
            remainder='passthrough' # Keep other columns if any, though usually we define all
        )
        
        # Fit the transformer
        print(f"Fitting new transformer...")
        preprocessor.fit(df)
        
        # Save the transformer if joblib is available
        if joblib:
            print(f"Saving transformer to {transformer_path}")
            joblib.dump(preprocessor, transformer_path)
        else:
            print("joblib not installed. Cannot save transformer.")
    
    # Transform the data
    processed_data = preprocessor.transform(df)
    
    # Print data statistics
    print("\nData Transformation Summary:")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {processed_data.shape}")
    # Get feature names after transformation to count properly
    try:
        out_features = preprocessor.get_feature_names_out()
        print(f"Features retained/created: {len(out_features)}")
    except Exception: # Handle cases where get_feature_names_out might fail
        print(f"Could not retrieve feature names from transformer.")

    # Check if processed_data is numpy array, if not, convert
    if not isinstance(processed_data, np.ndarray):
        try:
            # Handle sparse matrix output from ColumnTransformer if necessary
            if sp.issparse(processed_data):
                 processed_data = processed_data.toarray()
            else: # Convert potential list of lists or other formats
                 processed_data = np.array(processed_data)
        except Exception as e:
             print(f"Error converting processed data to numpy array: {e}")
             raise TypeError("Processed data could not be converted to a dense numpy array.")

    # Ensure data is float type for calculations
    try:
        processed_data = processed_data.astype(np.float32)
    except ValueError as e:
        print(f"Error converting processed data to float32: {e}. Check data types.")
        # Attempt to identify problematic columns if possible
        for col_idx in range(processed_data.shape[1]):
            try:
                processed_data[:, col_idx].astype(np.float32)
            except ValueError:
                print(f"  Column index {col_idx} likely contains non-numeric data.")
        raise

    print(f"Data range: [{np.min(processed_data):.4f}, {np.max(processed_data):.4f}]")
    print(f"Mean value: {np.mean(processed_data):.4f}, Std: {np.std(processed_data):.4f}")

    # --- Post-processing Checks ---
    print("\n--- Post-processing Checks ---")
    if np.isnan(processed_data).any() or np.isinf(processed_data).any():
        print("Warning: NaN or Infinite values found in processed data!")
        nan_cols = np.where(np.isnan(processed_data).any(axis=0))[0]
        inf_cols = np.where(np.isinf(processed_data).any(axis=0))[0]
        print(f"  Indices of columns with NaN: {nan_cols}")
        print(f"  Indices of columns with Inf: {inf_cols}")
        # Consider adding imputation here if necessary, e.g., replacing NaN/Inf with 0 or mean/median
        # processed_data = np.nan_to_num(processed_data, nan=0.0, posinf=0.0, neginf=0.0)
        # print("  NaN/Inf values replaced with 0.")
    else:
        print("OK: No NaN or Infinite values found.")

    try:
        variances = np.var(processed_data, axis=0)
        low_variance_threshold = 1e-6
        low_variance_indices = np.where(variances < low_variance_threshold)[0]

        if len(low_variance_indices) > 0:
            print(f"Warning: {len(low_variance_indices)} columns in the processed data have variance < {low_variance_threshold}.")
            print(f"  Indices of low-variance columns: {low_variance_indices}")
        else:
            print(f"OK: All processed columns have variance >= {low_variance_threshold}.")
    except Exception as e:
        print(f"Error during variance check: {e}")
    # --- End of Added Post-processing Checks ---

    # Return both processed data and the fitted transformer
    return processed_data, preprocessor

def load_person_data(args):
    """
    Load person data, preprocess, calculate initial KNN graph potentially excluding
    some features, and return data for SUBLIME training.

    Args:
        args: Arguments containing dataset path, k, sparse flag, drop_columns_file, etc.

    Returns:
        Tuple: (
            features: Full preprocessed features (Tensor).
            nfeats: Number of features.
            labels: None (unsupervised).
            n_clusters: Number of clusters target (from args).
            train_mask, val_mask, test_mask: Dummy masks (Tensor).
            initial_graph: The computed initial graph structure (Tensor, dense or sparse).
        )
    """
    # Load data
    print(f"Loading dataset from {args.dataset}")
    try:
        df = pd.read_csv(args.dataset, delimiter='\t')
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset}")
        raise
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    # Preprocess the data - convert to floats, handle missing values, normalize to [-1, 1]
    # Also gets the fitted preprocessor needed for mapping dropped columns
    processed_data, preprocessor = preprocess_mixed_data(df, model_dir='sublime_models')

    # Full features for the GCL model training
    features = torch.FloatTensor(processed_data).to(device)
    nfeats = features.shape[1]
    n_samples = features.shape[0]

    # --- Determine features for initial KNN graph calculation ---
    features_for_knn = processed_data # Default to all features
    if args.drop_columns_file and os.path.exists(args.drop_columns_file):
        print(f"Attempting to exclude columns specified in {args.drop_columns_file} for initial KNN graph construction.")
        try:
            drop_df = pd.read_csv(args.drop_columns_file)
            if 'col' not in drop_df.columns:
                print(f"Error: Column 'col' not found in {args.drop_columns_file}. Using all features for KNN.")
            else:
                cols_to_drop = drop_df['col'].unique().tolist()
                print(f"Columns to exclude from KNN: {cols_to_drop}")

                transformed_feature_names = preprocessor.get_feature_names_out()
                drop_indices = []
                dropped_feature_names = []

                for i, name in enumerate(transformed_feature_names):
                    # Extract base column name (handles 'num__col' and 'cat__col_val')
                    try:
                        # Split by '__', take the second part, then split by '_' and take the first
                        # This aims to get the original column name back
                        # Example: 'num__age' -> 'age', 'cat__country_USA' -> 'country'
                        original_col = name.split('__')[1].split('_')[0]
                    except IndexError:
                        print(f"Warning: Could not parse original column name from transformed feature '{name}'. Skipping.")
                        continue

                    if original_col in cols_to_drop:
                        drop_indices.append(i)
                        dropped_feature_names.append(name)

                if drop_indices:
                    # Sort unique indices to avoid issues with np.delete
                    drop_indices = sorted(list(set(drop_indices)))
                    print(f"Found {len(drop_indices)} transformed features corresponding to columns to drop:")
                    # Limit printing of dropped feature names if too many
                    if len(dropped_feature_names) > 20:
                         print(f"  Features: {dropped_feature_names[:10]}... (Total: {len(dropped_feature_names)})")
                    else:
                         print(f"  Features: {dropped_feature_names}")
                    print(f"  Indices: {drop_indices[:20]}...")
                    features_for_knn = np.delete(processed_data, drop_indices, axis=1)
                    print(f"Shape of features used for KNN calculation: {features_for_knn.shape}")
                    if features_for_knn.shape[1] == 0:
                         print("Error: All features were dropped! Cannot compute KNN graph. Check drop_columns_file.")
                         raise ValueError("All features dropped for KNN graph construction.")
                else:
                    print("Warning: No matching transformed features found for columns specified in drop_columns_file. Using all features for KNN.")

        except FileNotFoundError:
             print(f"Warning: drop_columns_file '{args.drop_columns_file}' not found. Using all features for KNN.")
        except Exception as e:
            print(f"Error processing drop_columns_file '{args.drop_columns_file}': {e}. Using all features for KNN.")

    else:
        print("No drop_columns_file specified or file not found. Using all features for initial KNN graph.")
        print(f"Shape of features used for KNN calculation: {features_for_knn.shape}")


    # Ensure features_for_knn is float32 for FAISS
    if features_for_knn.dtype != np.float32:
        features_for_knn = features_for_knn.astype(np.float32)

    # Check for low variance *only* in the features used for KNN
    try:
        variances_knn = np.var(features_for_knn, axis=0)
        low_variance_threshold = 1e-6
        low_variance_indices_knn = np.where(variances_knn < low_variance_threshold)[0]
        if len(low_variance_indices_knn) > 0:
            print(f"Warning: {len(low_variance_indices_knn)} columns in the data *used for KNN* have variance < {low_variance_threshold}.")
            print(f"  Indices (within KNN features): {low_variance_indices_knn}")
            print("  This might affect KNN graph quality.")
        else:
             print(f"OK: All features used for KNN have variance >= {low_variance_threshold}.")
    except Exception as e:
        print(f"Error during variance check for KNN features: {e}")

    # --- Calculate initial graph structure ---
    print(f"Constructing initial graph structure with k={args.k} using FAISS (cosine similarity) on selected features...")
    initial_graph = None # Initialize to None
    try:
        # Ensure there's at least one feature to compute KNN on
        if features_for_knn.shape[1] == 0:
            raise ValueError("Cannot compute KNN graph with zero features.")

        initial_graph_sparse = faiss_kneighbors_graph(features_for_knn, k=args.k, metric='cosine')
        print(f"Initial graph computed. Shape: {initial_graph_sparse.shape}, Non-zero entries: {initial_graph_sparse.nnz}")

        # Prepare the initial graph data based on sparse flag
        if args.sparse:
            initial_graph = sparse_mx_to_torch_sparse_tensor(initial_graph_sparse).to(device)
            print("Initial graph prepared as sparse torch tensor.")
        else:
            initial_graph = torch.FloatTensor(initial_graph_sparse.todense()).to(device)
            print("Initial graph prepared as dense torch tensor.")

    except Exception as e:
        print(f"Error during FAISS KNN graph construction: {e}")
        print("Falling back to identity matrix for initial graph.")
        if args.sparse:
             initial_graph_sparse_fallback = sp.eye(n_samples, dtype=np.float32).tocoo()
             initial_graph = sparse_mx_to_torch_sparse_tensor(initial_graph_sparse_fallback).to(device)
        else:
             initial_graph = torch.eye(n_samples, dtype=torch.float32).to(device)
        print("Using identity matrix as initial graph.")


    # Create dummy labels and masks (consistent with original structure)
    labels = None # Unsupervised
    train_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)

    print(f"Prepared dataset. Full features shape: {features.shape}. Initial graph type: {'Sparse' if args.sparse else 'Dense'}.")

    # Return full features, metadata, masks, and the computed initial_graph
    # Ensure n_clusters is passed correctly (it's used in clustering evaluation)
    return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, initial_graph

def main():
    # Get the base parser from main.py
    parent_parser = create_parser()

    # Create a new parser that inherits from the parent, but don't add help to avoid conflicts
    # Use argument_default=argparse.SUPPRESS to allow defaults from parent to take precedence
    # unless explicitly overridden here or by command line.
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, argument_default=argparse.SUPPRESS)

    # Set defaults specific to this script *if they need to override the parent*
    parser.set_defaults(
        dataset='person_data.csv', # Keep or change as needed
        ntrials=1,
        sparse=0,
        gsl_mode="structure_inference", # Changed default: inference makes more sense if we always compute KNN
        eval_freq=50,               # Example: adjusted eval freq
        downstream_task='clustering',
        epochs=50,                 # Example: Reduced epochs for person data
        # contrast_batch_size=10000, # Let parent default handle this or set explicitly
        save_model=1,
        output_dir='sublime_models/',
        checkpoint_dir='sublime_checkpoints/', # Give specific checkpoint dir
        checkpoint_freq=25,          # Example: checkpoint freq
        verbose=1,
        k=10,                        # Example: adjusted k
        n_clusters=5                 # Default cluster number
        # type_learner defaults to 'fgp' from parent
        # lr, hidden_dim etc. default from parent
    )

    # Add arguments specific to this script OR override parent defaults forcefully
    # parser.add_argument('-k', type=int, default=10, help='Override k value') # Example override
    parser.add_argument('-n_clusters', type=int, help='Number of clusters for clustering task') # Keep help text


    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Print final config being used
    print("--- Running Training with Configuration ---")
    for arg, value in sorted(vars(args).items()):
         print(f"{arg}: {value}")
    print("-----------------------------------------")


    # Create experiment and train model
    experiment = Experiment(device)
    # Pass the specific load_person_data function
    experiment.train(args, load_data_fn=load_person_data)

if __name__ == "__main__":
    main() 
