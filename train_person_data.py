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
from utils import sparse_mx_to_torch_sparse_tensor, faiss_kneighbors_graph, get_memory_usage
from main import create_parser # Import the shared parser creation function

# Add device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _check_processed_data(data, dataset_name=""):
    """Internal helper to check processed data for issues."""
    print(f"\n--- Post-processing Checks ({dataset_name}) ---")
    if not isinstance(data, np.ndarray):
        print(f"Error ({dataset_name}): Processed data is not a numpy array (type: {type(data)}).")
        return False, data # Return False indicating failure

    try:
        data = data.astype(np.float32) # Ensure float32
    except ValueError as e:
        print(f"Error ({dataset_name}): Could not convert processed data to float32: {e}")
        return False, data # Return False indicating failure

    if np.isnan(data).any() or np.isinf(data).any():
        print(f"Warning ({dataset_name}): NaN or Infinite values found!")
        nan_cols = np.where(np.isnan(data).any(axis=0))[0]
        inf_cols = np.where(np.isinf(data).any(axis=0))[0]
        print(f"  Indices of columns with NaN: {nan_cols}")
        print(f"  Indices of columns with Inf: {inf_cols}")
        # Option to impute: data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # return False, data # Indicate potential issue even if imputed
    else:
        print(f"OK ({dataset_name}): No NaN or Infinite values found.")

    try:
        variances = np.var(data, axis=0)
        low_variance_threshold = 1e-6
        low_variance_indices = np.where(variances < low_variance_threshold)[0]
        if len(low_variance_indices) > 0:
            print(f"Warning ({dataset_name}): {len(low_variance_indices)} columns have variance < {low_variance_threshold}.")
            print(f"  Indices of low-variance columns: {low_variance_indices}")
        else:
            print(f"OK ({dataset_name}): All columns have variance >= {low_variance_threshold}.")
    except Exception as e:
        print(f"Error ({dataset_name}) during variance check: {e}")
        return False, data # Return False indicating failure

    return True, data # Return True indicating success

def preprocess_mixed_data(df_main, df_annotated=None, model_dir='sublime_models', target_column=None):
    """
    Preprocess main and optional annotated dataframes using a single transformer.
    Transformer is fitted ONLY on df_main if not loaded.

    Args:
        df_main: Main Pandas DataFrame.
        df_annotated: Optional annotated Pandas DataFrame.
        model_dir: Directory to save/load transformation models.
        target_column: Name of the target column in df_annotated (if provided).

    Returns:
        Tuple: (
            processed_main (np.ndarray),
            processed_annotated (np.ndarray or None),
            preprocessor (ColumnTransformer)
        )
        Raises ValueError if preprocessing fails critical checks.
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    transformer_path = os.path.join(model_dir, 'data_transformer.joblib')
    preprocessor = None

    # Separate numerical and categorical columns from the *main* dataframe
    # Assume the annotated dataframe has the same relevant columns for transformation
    categorical_cols = df_main.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df_main.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Target column should not be part of numerical/categorical cols used for fitting
    if target_column:
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)

    processed_main = None
    processed_annotated = None

    try:
        # --- Load or Fit Transformer ---
        if joblib and os.path.exists(transformer_path):
            print(f"Loading transformer from {transformer_path}")
            preprocessor = joblib.load(transformer_path)
        else:
            print("Transformer not found. Fitting new transformer ON MAIN DATA...")
            # Create preprocessing pipelines
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', MinMaxScaler(feature_range=(-1, 1)))
            ])
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            # Combine pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_cols),
                    ('cat', categorical_pipeline, categorical_cols)
                ],
                remainder='drop' # Drop columns not specified (like target column if present)
            )
            # Fit ONLY on main data
            preprocessor.fit(df_main)
            print("Transformer fitted.")
            # Save the transformer
            if joblib:
                print(f"Saving transformer to {transformer_path}")
                joblib.dump(preprocessor, transformer_path)
            else:
                print("joblib not installed. Cannot save transformer.")

        # --- Transform Main Data ---
        print("Transforming main data...")
        processed_main = preprocessor.transform(df_main)
        print(f"Original main shape: {df_main.shape}")
        print(f"Processed main shape: {processed_main.shape}")
        ok, processed_main = _check_processed_data(processed_main, "Main Data")
        if not ok:
            raise ValueError("Preprocessing check failed for main data.")

        # --- Transform Annotated Data (if provided) ---
        if df_annotated is not None:
            print("Transforming annotated data...")
            processed_annotated = preprocessor.transform(df_annotated)
            print(f"Original annotated shape: {df_annotated.shape}")
            print(f"Processed annotated shape: {processed_annotated.shape}")
            ok, processed_annotated = _check_processed_data(processed_annotated, "Annotated Data")
            if not ok:
                raise ValueError("Preprocessing check failed for annotated data.")

            # Check if feature dimensions match
            if processed_main.shape[1] != processed_annotated.shape[1]:
                 raise ValueError(f"Feature dimension mismatch after processing: "
                                  f"Main data has {processed_main.shape[1]} features, "
                                  f"Annotated data has {processed_annotated.shape[1]} features. "
                                  "Ensure both datasets have compatible columns and the transformer is applied correctly.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise # Re-raise the exception after logging

    return processed_main, processed_annotated, preprocessor


def load_person_data(args):
    """
    Loads main and annotated person data, preprocesses them using a single
    transformer (fitted on main data if needed), combines them, calculates
    a combined initial KNN graph, and returns data for SUBLIME training.

    Args:
        args: Arguments containing dataset paths, k, sparse flag, etc.

    Returns:
        Tuple: (
            features: Combined preprocessed features (Tensor).
            nfeats: Number of features.
            combined_labels: Combined labels (Tensor, -1 for main, 0/1 for annotated).
            nclasses_or_clusters: Target number for evaluation (based on args.downstream_task).
            train_mask, val_mask, test_mask: Dummy masks (Tensor).
            initial_graph: Combined initial graph structure (Tensor, dense or sparse).
        )
    """
    # === 1. Load Main Data ===
    print(f"Loading main dataset from {args.dataset}")
    try:
        df_main = pd.read_csv(args.dataset, delimiter='\t')
        n_main = len(df_main)
    except FileNotFoundError:
        print(f"Error: Main dataset file not found at {args.dataset}")
        raise
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        raise

    # === 2. Load Annotated Data (if specified) ===
    df_annotated = None
    extracted_binary_labels = None
    n_annotated = 0
    target_column = args.annotation_column if hasattr(args, 'annotation_column') else None

    if args.annotated_dataset:
        print(f"Loading annotated dataset from {args.annotated_dataset}")
        try:
            df_annotated = pd.read_csv(args.annotated_dataset, delimiter='\t')
            n_annotated = len(df_annotated)

            # Check for and extract target column
            if not target_column:
                raise ValueError(f"Annotated dataset '{args.annotated_dataset}' loaded, but 'annotation_column' not specified.")
            if target_column not in df_annotated.columns:
                 raise ValueError(f"Annotation column '{target_column}' not found in annotated dataset '{args.annotated_dataset}'. "
                                  f"Available columns: {df_annotated.columns.tolist()}")

            print(f"Extracting labels from target column '{target_column}'...")
            try:
                # Ensure labels are integer 0 or 1
                label_values = df_annotated[target_column].values
                unique_labels = np.unique(label_values)
                if not np.all(np.isin(unique_labels, [0, 1])):
                     raise ValueError(f"Target column '{target_column}' contains values other than 0 or 1: {unique_labels}")
                extracted_binary_labels = torch.tensor(label_values, dtype=torch.long) # Keep on CPU for now
                print(f"  Found {len(torch.unique(extracted_binary_labels))} unique binary labels.")
                # Drop the target column *before* preprocessing
                df_annotated = df_annotated.drop(columns=[target_column])
            except Exception as e:
                print(f"Error processing target column '{target_column}': {e}")
                raise

        except FileNotFoundError:
            print(f"Error: Annotated dataset file not found at {args.annotated_dataset}")
            raise
        except Exception as e:
            print(f"Error loading or processing annotated dataset: {e}")
            raise
    else:
        print("No annotated dataset specified.")


    # === 3. Preprocess Data ===
    # Pass both dataframes, preprocessor handles fitting/loading and transforming
    processed_main, processed_annotated, preprocessor = preprocess_mixed_data(
        df_main, df_annotated, model_dir='sublime_models', target_column=target_column
    )


    # === 4. Combine Features ===
    if processed_annotated is not None:
        print("Combining processed main and annotated features...")
        combined_features_np = np.vstack((processed_main, processed_annotated))
    else:
        print("Using only main features.")
        combined_features_np = processed_main

    n_total_samples = combined_features_np.shape[0]
    nfeats = combined_features_np.shape[1]
    print(f"Combined features shape: ({n_total_samples}, {nfeats})")
    features = torch.FloatTensor(combined_features_np).to(device)


    # === 5. Combine Labels ===
    if extracted_binary_labels is not None:
        print("Creating combined labels tensor (-1 for main, 0/1 for annotated)...")
        main_placeholders = torch.full((n_main,), -1, dtype=torch.long)
        combined_labels = torch.cat((main_placeholders, extracted_binary_labels)).to(device)
        # Verify shape
        if len(combined_labels) != n_total_samples:
             raise ValueError(f"Combined labels length ({len(combined_labels)}) does not match total samples ({n_total_samples}).")
        print(f"Combined labels shape: {combined_labels.shape}")
    else:
        # If no annotated data, create labels tensor of -1s or handle as needed
        # For consistency, we'll create a tensor of -1s. Loss function needs to handle this.
        print("No annotated data provided, creating labels tensor with -1 placeholders.")
        combined_labels = torch.full((n_total_samples,), -1, dtype=torch.long).to(device)

    # === 6. Determine nclasses/n_clusters for evaluation ===
    if args.downstream_task == 'clustering':
        nclasses_or_clusters = args.n_clusters if hasattr(args, 'n_clusters') else n_total_samples # Default to nodes if not specified
        print(f"Downstream task is clustering. Using n_clusters = {nclasses_or_clusters} for evaluation.")
    elif extracted_binary_labels is not None:
        # If classification task and we have binary labels, nclasses for evaluation is 2
        nclasses_or_clusters = 2
        print(f"Downstream task is classification. Using n_classes = {nclasses_or_clusters} for evaluation.")
    else:
        # Fallback if classification task but no labels somehow (shouldn't happen with checks)
        # Or if another task is added later.
        nclasses_or_clusters = n_total_samples # Default to number of nodes
        print(f"Warning: Downstream task '{args.downstream_task}' but labels are missing or not binary. Using n_nodes = {nclasses_or_clusters} for evaluation.")


    # === 7. Determine Features for Initial KNN Graph ===
    # (Optional: Logic to exclude columns for KNN based on args.drop_columns_file)
    # For simplicity now, using *all* combined features for KNN.
    # If column dropping is needed, apply it to combined_features_np *before* KNN.
    features_for_knn = combined_features_np # Using combined features
    print(f"Using combined features (shape: {features_for_knn.shape}) for initial KNN graph construction.")
    # Add variance check for KNN features
    ok, _ = _check_processed_data(features_for_knn, "KNN Features")
    if not ok:
        raise ValueError("Preprocessing check failed for features used in KNN.")


    # === 8. Calculate Combined Initial Graph Structure ===
    print(f"Constructing initial graph structure with k={args.k} using FAISS (cosine similarity) on COMBINED features ({n_total_samples} nodes)...")
    initial_graph = None
    try:
        if features_for_knn.shape[1] == 0:
            raise ValueError("Cannot compute KNN graph with zero features.")

        initial_graph_sparse = faiss_kneighbors_graph(features_for_knn, k=args.k, metric='cosine')
        print(f"Initial combined graph computed. Shape: {initial_graph_sparse.shape}, Non-zero entries: {initial_graph_sparse.nnz}")

        if args.sparse:
            initial_graph = sparse_mx_to_torch_sparse_tensor(initial_graph_sparse).to(device)
            print("Initial combined graph prepared as sparse torch tensor.")
        else:
            initial_graph = torch.FloatTensor(initial_graph_sparse.todense()).to(device)
            print("Initial combined graph prepared as dense torch tensor.")

    except Exception as e:
        print(f"Error during FAISS KNN graph construction for combined data: {e}")
        print("Falling back to identity matrix for initial combined graph.")
        if args.sparse:
             initial_graph_sparse_fallback = sp.eye(n_total_samples, dtype=np.float32).tocoo()
             initial_graph = sparse_mx_to_torch_sparse_tensor(initial_graph_sparse_fallback).to(device)
        else:
             initial_graph = torch.eye(n_total_samples, dtype=torch.float32).to(device)
        print("Using identity matrix as initial combined graph.")


    # === 9. Create Dummy Masks ===
    train_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device)

    print(f"Prepared combined dataset. Features shape: {features.shape}. Initial graph type: {'Sparse' if args.sparse else 'Dense'}. Labels shape: {combined_labels.shape}.")

    # Return combined data
    return features, nfeats, combined_labels, nclasses_or_clusters, train_mask, val_mask, test_mask, initial_graph

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
        # annotated_dataset=None, # Let parent handle this default or set via command line
        # annotation_column=None, # Let parent handle this default or set via command line
        ntrials=1,
        sparse=0,
        gsl_mode="structure_inference", # Changed default: inference makes more sense if we always compute KNN
        eval_freq=50,               # Example: adjusted eval freq
        downstream_task='clustering', # Default to clustering for this script
        epochs=50,                 # Example: Reduced epochs for person data
        save_model=1,
        output_dir='sublime_models/',
        checkpoint_dir='sublime_checkpoints/', # Give specific checkpoint dir
        checkpoint_freq=25,          # Example: checkpoint freq
        verbose=1,
        k=10,                        # Example: adjusted k
        # n_clusters=5                 # Let parent handle default or set via command line
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

    # --- Configuration Validation ---
    if args.annotated_dataset and not args.annotation_column:
        parser.error("If --annotated_dataset is provided, --annotation_column must also be specified.")
    if not args.annotated_dataset and args.annotation_column:
        print("Warning: --annotation_column specified but --annotated_dataset is missing. The annotation column will be ignored.")
        args.annotation_column = None # Ensure it's None if no annotated dataset
    # --- End Validation ---


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
