import argparse
import torch
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import optuna
import torch.nn.functional as F
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import lightgbm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import faiss
import time
import matplotlib.pyplot as plt
import gc
import sys # Import sys for sys.exit
import scipy.sparse as sp # Added for sparse matrix operations

# Add necessary imports from main and model/graph_learner
from main import Experiment, GCL, FGP_learner, MLP_learner, ATT_learner, GNN_learner, normalize, symmetrize, sparse_mx_to_torch_sparse_tensor, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse
# Attempt to import from utils, assuming it's in the path or same directory structure
from utils import knn_fast # sparse_mx_to_torch_sparse_tensor is already imported from main
import dgl # Make sure dgl is imported
from sklearn.linear_model import LogisticRegression # Add this import

# Define coerce_numeric at the top level
def coerce_numeric(df_):
    """Coerces DataFrame columns to numeric, handling errors."""
    for col in df_.columns:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_[col]):
            continue
        try:
            # Attempt coercion, raising errors for non-numeric
            df_[col] = pd.to_numeric(df_[col], errors='raise')
        except (ValueError, TypeError):
            # If coercion fails, print a message and force to NaN
            print(f"Column {col} could not be coerced to numeric. Forcing errors to NaN.")
            df_[col] = pd.to_numeric(df_[col], errors='coerce')
            # Optional: Add category conversion here if desired, but NaN is needed for imputation
            # if df_[col].nunique() < 50 and len(df_) > 0:
            #     print(f"  Converting column {col} to category.") # This line might not be needed now
            #     df_[col] = df_[col].astype('category')
    return df_

def _find_column_case_insensitive(df_columns, target_name_lower):
    """Finds the actual column name in a list of columns, ignoring case."""
    for col in df_columns:
        if col.lower() == target_name_lower:
            return col
    return None

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- New Anchor Graph Generation Function --- #
def _generate_custom_anchor_graph(
    node_features_tensor: torch.Tensor, 
    cpf_identifiers_df: pd.DataFrame,   
    id_column_name: str,               
    k_knn: int,
    sparse_output: bool,              
    device: torch.device,
    knn_threshold_type: str = 'none',
    knn_std_dev_factor: float = 1.0,
    relationship_csv_path: str = None,
    relationship_cpf1_col: str = 'CPF',
    relationship_cpf2_col: str = 'CPF_VINCULO',
    relationship_weight: float = 1.0
):
    """Generates a new anchor graph based on provided features and relationship data."""
    node_features_tensor = node_features_tensor.to(device)
    n_total_samples = node_features_tensor.shape[0]

    if len(cpf_identifiers_df) != n_total_samples:
        raise ValueError(
            f"Length of cpf_identifiers_df ({len(cpf_identifiers_df)}) "
            f"does not match node_features_tensor rows ({n_total_samples})."
        )

    # 1. Create CPF to Index Mapping
    actual_id_col = _find_column_case_insensitive(cpf_identifiers_df.columns, id_column_name.lower())
    if not actual_id_col:
        raise ValueError(f"ID column '{id_column_name}' not found in cpf_identifiers_df. Available: {cpf_identifiers_df.columns.tolist()}")
    
    cpf_to_index = {cpf: idx for idx, cpf in enumerate(cpf_identifiers_df[actual_id_col])}
    print(f"Created ID-to-index mapping for {len(cpf_to_index)} unique IDs / {n_total_samples} total samples for new anchor graph.")

    # 2. KNN Graph (Feature Similarity)
    print(f"Constructing KNN graph for new anchor (k={k_knn}, sim=cosine, threshold={knn_threshold_type}) ...")
    adj_knn_sparse = None
    try:
        if node_features_tensor.shape[1] == 0:
            raise ValueError("Cannot compute KNN graph with zero features.")
        
        knn_rows, knn_cols, knn_vals = knn_fast(
            node_features_tensor, k=k_knn, use_gpu=(device.type == 'cuda'),
            knn_threshold_type=knn_threshold_type, knn_std_dev_factor=knn_std_dev_factor,
            return_values=True
        )
        adj_knn_sparse = sp.csr_matrix(
            (knn_vals.cpu().numpy(), (knn_rows.cpu().numpy(), knn_cols.cpu().numpy())),
            shape=(n_total_samples, n_total_samples)
        )
        adj_knn_sparse = adj_knn_sparse.maximum(adj_knn_sparse.T) # Symmetrize
        print(f"  KNN graph computed. Shape: {adj_knn_sparse.shape}, Non-zero entries: {adj_knn_sparse.nnz}")
    except Exception as e:
        print(f"Error during KNN graph construction: {e}. Using identity matrix for KNN part.")
        adj_knn_sparse = sp.eye(n_total_samples, dtype=np.float32, format='csr')

    # 3. Relationship Graph
    adj_rel_sparse = sp.csr_matrix((n_total_samples, n_total_samples), dtype=np.float32)
    if relationship_csv_path:
        print(f"Constructing relationship graph from {relationship_csv_path}...")
        try:
            df_relationships = pd.read_csv(relationship_csv_path, sep='\t') 
            
            actual_rel_cpf1_col = _find_column_case_insensitive(df_relationships.columns, relationship_cpf1_col.lower())
            actual_rel_cpf2_col = _find_column_case_insensitive(df_relationships.columns, relationship_cpf2_col.lower())

            if not actual_rel_cpf1_col or not actual_rel_cpf2_col:
                raise ValueError(f"Relationship CSV '{relationship_csv_path}' missing required columns '{relationship_cpf1_col}' or '{relationship_cpf2_col}'. Found: {df_relationships.columns.tolist()}")

            rows, cols, data = [], [], []
            valid_edges = 0
            for _, row_rel in df_relationships.iterrows():
                id1_val = row_rel[actual_rel_cpf1_col]
                id2_val = row_rel[actual_rel_cpf2_col]
                idx1 = cpf_to_index.get(id1_val)
                idx2 = cpf_to_index.get(id2_val)
                if idx1 is not None and idx2 is not None and idx1 != idx2:
                    rows.extend([idx1, idx2])
                    cols.extend([idx2, idx1])
                    data.extend([relationship_weight, relationship_weight])
                    valid_edges +=1
            
            if valid_edges > 0:
                adj_rel_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_total_samples, n_total_samples), dtype=np.float32)
                adj_rel_sparse.sum_duplicates()
                print(f"  Relationship graph computed. Non-zero entries: {adj_rel_sparse.nnz}")
            else:
                print("  No valid relationship edges created.")
        except Exception as e:
            print(f"Error loading or processing relationship graph: {e}. Skipping relationship graph.")
    
    # 4. Combine Graphs
    print("Combining KNN and Relationship graphs for new anchor...")
    combined_graph_sparse = adj_knn_sparse.maximum(adj_rel_sparse)
    # print(f"  Combined graph non-zero entries: {combined_graph_sparse.nnz}") # Optional: for debugging
    
    # Add self-loops with weight 1.0
    combined_graph_sparse = combined_graph_sparse.maximum(sp.eye(n_total_samples, dtype=np.float32, format='csr'))
    print(f"  Final combined graph + self-loops non-zero entries: {combined_graph_sparse.nnz}")

    # 5. Convert to PyTorch Tensor (this is the raw_adj, normalization happens in SublimeHandler)
    if sparse_output:
        final_graph_tensor_raw = sparse_mx_to_torch_sparse_tensor(combined_graph_sparse)
    else:
        final_graph_tensor_raw = torch.FloatTensor(combined_graph_sparse.todense())
    
    return final_graph_tensor_raw.to(device)
# --- End New Anchor Graph Generation Function --- #

# --- Configuration Class ---
class Config:
    """Holds all configuration parameters."""
    def __init__(self, args):
        self.neurolake_csv = args.neurolake_csv
        self.dataset_features_csv = args.dataset_features_csv
        self.model_dir = args.model_dir
        self.target_column = args.target_column
        self.test_csv = args.test_csv
        self.output_dir = args.output_dir
        self.n_trials = args.n_trials
        self.batch_size = args.batch_size
        self.cache_dir = args.cache_dir
        self.embeddings_output = args.embeddings_output
        self.k_neighbors = args.k_neighbors
        self.data_fraction = args.data_fraction
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.using_separate_test = self.test_csv is not None
        self.use_loaded_adj_for_extraction = args.use_loaded_adj_for_extraction
        self.extract_embeddings_only = args.extract_embeddings_only # Added
        self.class_weight_multiplier = args.class_weight_multiplier # Added

        # --- New Anchor Graph Generation Config --- # Updated
        self.generate_new_anchor_adj_for_eval = args.generate_new_anchor_adj_for_eval
        if self.generate_new_anchor_adj_for_eval:
            # self.sublime_model_original_ids_csv = args.sublime_model_original_ids_csv # Removed
            self.anchor_adj_k_knn = args.anchor_adj_k_knn
            self.anchor_adj_use_sparse_format = bool(args.anchor_adj_use_sparse_format)
            self.anchor_adj_id_col_name = args.anchor_adj_id_col_name # This ID col is from the main eval data CSV (neurolake_csv)
            self.anchor_adj_relationship_csv = args.anchor_adj_relationship_csv
            self.anchor_adj_relationship_cpf1_col = args.anchor_adj_relationship_cpf1_col
            self.anchor_adj_relationship_cpf2_col = args.anchor_adj_relationship_cpf2_col
            self.anchor_adj_relationship_weight = args.anchor_adj_relationship_weight
            self.anchor_adj_knn_threshold_type = args.anchor_adj_knn_threshold_type
            self.anchor_adj_knn_std_dev_factor = args.anchor_adj_knn_std_dev_factor
            print("New anchor graph generation for evaluation (using eval dataset features) is ENABLED.")
            if not self.anchor_adj_id_col_name:
                 raise ValueError("anchor_adj_id_col_name must be specified when generate_new_anchor_adj_for_eval is True.")
        # --- End New Anchor Graph Generation Config ---

        self.dataset_name = self._derive_dataset_name()
        self.test_dataset_name = self._derive_test_dataset_name()
        self.active_k_values = sorted([k for k in self.k_neighbors if k > 0])
        self.k_str = '_'.join(map(str, self.active_k_values)) if self.active_k_values else 'none'
        self.model_name_tag = os.path.basename(os.path.normpath(self.model_dir)) # For cache filenames

        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"Using cache directory: {self.cache_dir}")

    def _derive_dataset_name(self):
        if self.dataset_features_csv:
            return os.path.splitext(os.path.basename(self.dataset_features_csv))[0]
        elif self.neurolake_csv:
             return os.path.splitext(os.path.basename(self.neurolake_csv))[0]
        return "unknown_dataset"

    def _derive_test_dataset_name(self):
        if self.using_separate_test:
            return os.path.splitext(os.path.basename(self.test_csv))[0] + "_ext_test"
        return None

# --- Data Management Class ---
class DataManager:
    """Handles loading, preprocessing, splitting, and filtering data."""
    def __init__(self, config):
        self.config = config
        self.neurolake_df = None
        self.dataset_df = None
        self.test_neurolake_df = None
        self.test_dataset_df = None

        self.X_neurolake = None
        self.X_test_neurolake = None
        self.X_dataset = None
        self.X_test_dataset = None
        self.y = None
        self.y_test = None

        self.neurolake_preprocessor = None # Loaded from model_dir
        self.dataset_preprocessor = None # Fitted here

        self.split_data_dict = {} # To store train/val/test splits

    def load_and_sample_data(self):
        """Loads initial dataframes and applies sampling."""
        print(f"Loading neurolake data from {self.config.neurolake_csv}")
        self.neurolake_df = pd.read_csv(self.config.neurolake_csv, delimiter='\t')

        if self.config.dataset_features_csv and self.config.target_column:
            print(f"Loading dataset features from {self.config.dataset_features_csv}")
            self.dataset_df = pd.read_csv(self.config.dataset_features_csv, delimiter='\t')
            if len(self.neurolake_df) != len(self.dataset_df):
                raise ValueError(f"Neurolake ({len(self.neurolake_df)}) and dataset features ({len(self.dataset_df)}) must have the same number of rows!")

        if self.config.using_separate_test:
            print(f"Loading separate test data from {self.config.test_csv}")
            test_df = pd.read_csv(self.config.test_csv, delimiter='\t')
            neurolake_cols = self.neurolake_df.columns
            missing_neurolake = [col for col in neurolake_cols if col not in test_df.columns]
            if missing_neurolake: raise ValueError(f"Test CSV missing neurolake columns: {missing_neurolake}")

            self.test_neurolake_df = test_df[neurolake_cols].copy()

            if self.dataset_df is not None:
                dataset_cols = self.dataset_df.columns
                missing_dataset = [col for col in dataset_cols if col not in test_df.columns]
                if missing_dataset: raise ValueError(f"Test CSV missing dataset columns: {missing_dataset}")
                self.test_dataset_df = test_df[dataset_cols].copy()
            print(f"Test data loaded: {len(test_df)} rows. Neurolake shape: {self.test_neurolake_df.shape}, Dataset shape: {self.test_dataset_df.shape if self.test_dataset_df is not None else 'N/A'}")


        # Apply data sampling (only to training data, not test data)
        if self.config.data_fraction < 1.0:
            if not (0 < self.config.data_fraction <= 1.0):
                 raise ValueError(f"data_fraction must be between 0 and 1, got {self.config.data_fraction}")

            n_samples = int(len(self.neurolake_df) * self.config.data_fraction)
            if n_samples <= 0: raise ValueError("data_fraction resulted in 0 samples.")

            sampled_indices = np.random.RandomState(42).choice(len(self.neurolake_df), size=n_samples, replace=False)
            self.neurolake_df = self.neurolake_df.iloc[sampled_indices].reset_index(drop=True)
            if self.dataset_df is not None:
                self.dataset_df = self.dataset_df.iloc[sampled_indices].reset_index(drop=True)
            print(f"Using {self.config.data_fraction:.1%} of the data: {n_samples} samples for training/validation")

    def filter_target_variable(self):
        """Filters rows where the target variable is not binary (0 or 1)."""
        if self.dataset_df is None or self.config.target_column not in self.dataset_df.columns:
             print("Target column filtering skipped: No dataset features or target column specified.")
             return

        # Filter train/val data
        valid_mask = self.dataset_df[self.config.target_column].isin([0, 1])
        filtered_count = (~valid_mask).sum()
        if filtered_count > 0:
            print(f"Removing {filtered_count} train/val rows where {self.config.target_column} is not 0 or 1")
            self.dataset_df = self.dataset_df[valid_mask].reset_index(drop=True)
            self.neurolake_df = self.neurolake_df[valid_mask].reset_index(drop=True) # Keep aligned
            print(f"After filtering train/val: {len(self.dataset_df)} rows remaining")

        # Filter test data (if applicable)
        if self.test_dataset_df is not None and self.config.target_column in self.test_dataset_df.columns:
            test_valid_mask = self.test_dataset_df[self.config.target_column].isin([0, 1])
            test_filtered_count = (~test_valid_mask).sum()
            if test_filtered_count > 0:
                print(f"Removing {test_filtered_count} test rows where {self.config.target_column} is not 0 or 1")
                self.test_dataset_df = self.test_dataset_df[test_valid_mask].reset_index(drop=True)
                self.test_neurolake_df = self.test_neurolake_df[test_valid_mask].reset_index(drop=True) # Keep aligned
                print(f"After filtering test data: {len(self.test_dataset_df)} rows remaining")

    def _preprocess_mixed_data(self, df, model_dir):
        """Internal helper reused from original script."""
        transformer_path = os.path.join(model_dir, 'data_transformer.joblib')
        if not os.path.exists(transformer_path):
             raise FileNotFoundError(f"Data transformer not found at {transformer_path}")

        print(f"Loading neurolake data transformer from {transformer_path}")
        self.neurolake_preprocessor = joblib.load(transformer_path) # Store loaded preprocessor
        processed_data = self.neurolake_preprocessor.transform(df)
        print(f"Neurolake data shape: {df.shape} -> Processed shape: {processed_data.shape}")
        return processed_data

    def preprocess_neurolake(self):
        """Preprocesses the neurolake data using the loaded transformer."""
        print("Preprocessing neurolake data (for SUBLIME)...")
        self.X_neurolake = self._preprocess_mixed_data(self.neurolake_df, self.config.model_dir)
        if self.test_neurolake_df is not None:
            print("Preprocessing test neurolake data...")
            # Use the same loaded preprocessor
            self.X_test_neurolake = self.neurolake_preprocessor.transform(self.test_neurolake_df)
            print(f"Test neurolake data processed: {self.X_test_neurolake.shape}")

    def _preprocess_dataset_features_internal(self, df, target_column, fit_transform=False):
        """Internal helper reused from original script."""
        df_copy = df.copy()
        y_local = None
        if target_column and target_column in df_copy.columns:
            X = df_copy.drop(columns=[target_column])
            y_local = df_copy[target_column].astype(int) # Ensure target is integer
        else:
            X = df_copy

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # --- Define Transformers with Coercion Inside Pipeline ---
        # Function to coerce to numeric, converting errors to NaN
        # Removed local definition of coerce_numeric here

        # Removed the problematic if-check for self.config.preprocess_mode

        numerical_transformer = Pipeline(steps=[
            ('coerce', FunctionTransformer(coerce_numeric)), # Coerce before imputation - uses top-level function
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        if fit_transform:
             # Define preprocessor *after* defining transformers
             preprocessor = ColumnTransformer(transformers=[
                 ('num', numerical_transformer, numerical_cols),
                 ('cat', categorical_transformer, categorical_cols)
             ])
             print("Fitting dataset feature preprocessor...")
             processed_data = preprocessor.fit_transform(X)
             self.dataset_preprocessor = preprocessor # Store fitted preprocessor
        elif self.dataset_preprocessor:
             print("Transforming dataset features using existing preprocessor...")
             processed_data = self.dataset_preprocessor.transform(X)
        else:
            raise RuntimeError("Preprocessor not fitted. Call with fit_transform=True first.")

        if hasattr(processed_data, 'toarray'): # Handle sparse matrix output
            print("Converting sparse dataset features to dense array.")
            processed_data = processed_data.toarray()

        if len(processed_data.shape) == 1: # Ensure 2D
            processed_data = processed_data.reshape(-1, 1)

        print(f"Dataset features shape: {X.shape} -> Processed shape: {processed_data.shape}")
        return processed_data, y_local

    def preprocess_dataset_features(self):
        """Preprocesses the dataset features, fitting or transforming as needed."""
        if self.dataset_df is None:
             print("Skipping dataset feature preprocessing: No dataset features provided.")
             return

        print("Preprocessing dataset features...")
        self.X_dataset, self.y = self._preprocess_dataset_features_internal(
            self.dataset_df, self.config.target_column, fit_transform=True
        )
        if self.test_dataset_df is not None:
            print("Preprocessing test dataset features...")
            self.X_test_dataset, self.y_test = self._preprocess_dataset_features_internal(
                self.test_dataset_df, self.config.target_column, fit_transform=False # Use fitted preprocessor
            )
            print(f"Test dataset features processed: {self.X_test_dataset.shape}. Test target shape: {self.y_test.shape}")
            
    def get_dataset_feature_names(self):
        """Attempts to get feature names from the fitted dataset preprocessor."""
        if self.dataset_preprocessor and hasattr(self.dataset_preprocessor, 'get_feature_names_out'):
            try:
                return list(self.dataset_preprocessor.get_feature_names_out())
            except Exception as e:
                print(f"Could not get feature names from preprocessor: {e}. Using generic names.")
        
        # Fallback to generic names if preprocessor fails or doesn't exist
        # Try using the split data dictionary which should persist after cleanup
        if self.split_data_dict and 'X_dataset_train' in self.split_data_dict:
            try:
                n_features = self.split_data_dict['X_dataset_train'].shape[1]
                print(f"Generating generic dataset feature names (count: {n_features}).")
                return [f"DatasetFeature_{i}" for i in range(n_features)]
            except Exception as e:
                print(f"Error getting shape from split_data_dict['X_dataset_train']: {e}")
        
        print("Warning: Could not determine dataset feature names.")
        return []

    def perform_train_val_test_split(self, sublime_results, test_sublime_results):
        """Splits the data into train, validation, and test sets."""
        print("Splitting data...")

        sublime_embeddings = sublime_results['embeddings']
        classification_probs = sublime_results.get('classification_probs') # May be None

        if self.config.using_separate_test:
            print("Using separate test set. Splitting remaining data into train/validation.")
            # Test set is already defined: X_test_dataset, y_test, X_test_neurolake
            sublime_test_embeddings = test_sublime_results['embeddings'] if test_sublime_results else None
            test_classification_probs = test_sublime_results.get('classification_probs') if test_sublime_results else None

            # Ensure alignment before split
            if self.X_dataset is None or self.y is None:
                 raise ValueError("Dataset features and target must be processed before splitting.")
            if sublime_embeddings is None or len(sublime_embeddings) != len(self.X_dataset):
                 raise ValueError("Sublime embeddings must be extracted and aligned with dataset features.")

            # Inputs to split for train/val
            split_inputs = [self.X_dataset, sublime_embeddings, self.y]
            outputs_base = ['X_dataset', 'sublime', 'y']
            if classification_probs is not None:
                if len(classification_probs) != len(self.X_dataset):
                    raise ValueError("Classification probs must be aligned with dataset features.")
                split_inputs.insert(2, classification_probs) # Insert before y
                outputs_base.insert(2, 'cls_probs')

            # Split train/val (80/20 split of the non-test data)
            # Use test_size=0.25 because that's 20% of the original 80%
            split_results_train_val = train_test_split(
                *split_inputs, test_size=0.25, random_state=42, stratify=self.y
            )

            # Populate split_data_dict
            for i, name in enumerate(outputs_base):
                self.split_data_dict[f'{name}_train'] = split_results_train_val[i * 2]
                self.split_data_dict[f'{name}_val'] = split_results_train_val[i * 2 + 1]

            # Add the full train+val sets
            self.split_data_dict['X_dataset_train_val'] = self.X_dataset
            self.split_data_dict['sublime_train_val'] = sublime_embeddings
            self.split_data_dict['y_train_val'] = self.y
            if 'cls_probs_train' in self.split_data_dict:
                 self.split_data_dict['cls_probs_train_val'] = classification_probs

            # Add test sets
            self.split_data_dict['X_dataset_test'] = self.X_test_dataset
            self.split_data_dict['sublime_test'] = sublime_test_embeddings
            self.split_data_dict['y_test'] = self.y_test
            if test_classification_probs is not None:
                self.split_data_dict['cls_probs_test'] = test_classification_probs

        else:
            print("No separate test set. Splitting data into train/validation/test (60/20/20).")
            # Ensure alignment
            if self.X_dataset is None or self.y is None:
                 raise ValueError("Dataset features and target must be processed.")
            if sublime_embeddings is None or len(sublime_embeddings) != len(self.X_dataset):
                 raise ValueError("Sublime embeddings must be extracted and aligned.")

            # Inputs for splitting
            split_inputs = [self.X_dataset, sublime_embeddings, self.y]
            outputs_base = ['X_dataset', 'sublime', 'y']
            if classification_probs is not None:
                if len(classification_probs) != len(self.X_dataset):
                    raise ValueError("Classification probs must be aligned.")
                split_inputs.insert(2, classification_probs) # Insert before y
                outputs_base.insert(2, 'cls_probs')


            # First split: 80% train+val, 20% test
            split_results_tv_test = train_test_split(
                *split_inputs, test_size=0.2, random_state=42, stratify=self.y
            )

            # Second split: 75% train, 25% val (of the 80% train+val pool)
            train_val_indices = [i * 2 for i in range(len(outputs_base))]
            split_inputs_train_val = [split_results_tv_test[i] for i in train_val_indices]
            stratify_y_train_val = split_results_tv_test[outputs_base.index('y') * 2]

            split_results_train_val = train_test_split(
                *split_inputs_train_val, test_size=0.25, random_state=42, stratify=stratify_y_train_val
            )

            # Populate split_data_dict
            for i, name in enumerate(outputs_base):
                 self.split_data_dict[f'{name}_train_val'] = split_results_tv_test[i*2]
                 self.split_data_dict[f'{name}_test'] = split_results_tv_test[i*2+1]
                 self.split_data_dict[f'{name}_train'] = split_results_train_val[i*2]
                 self.split_data_dict[f'{name}_val'] = split_results_train_val[i*2+1]

        print(f"Data split complete: Train={len(self.split_data_dict['y_train'])}, Validation={len(self.split_data_dict['y_val'])}, Test={len(self.split_data_dict['y_test'])}")
        # Clean up large intermediate arrays if possible
        del self.X_neurolake, self.X_dataset, self.y, self.neurolake_df, self.dataset_df
        gc.collect()

    def get_split_data(self):
        return self.split_data_dict

    def save_preprocessor(self):
        if self.dataset_preprocessor:
            preprocessor_path = os.path.join(self.config.output_dir, 'dataset_features_preprocessor.joblib')
            joblib.dump(self.dataset_preprocessor, preprocessor_path)
            print(f"Dataset features preprocessor saved to {preprocessor_path}")
        else:
            print("No dataset preprocessor was fitted, skipping save.")

# --- SUBLIME Model Handling Class ---
class SublimeHandler:
    """Handles loading the SUBLIME model and extracting embeddings."""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.model = None
        self.graph_learner = None
        self.original_model_features = None # Features SUBLIME was trained on (from features.pt)
        self.adj = None # Adjacency matrix to be used for extraction
        self.current_extraction_features = None # Features corresponding to self.adj if it's newly generated from eval data
        self.sparse = None # Sparsity of the original SUBLIME model (from its config.txt)
        self.faiss_index = None # For original_model_features if graph_learner is used
        self.has_classification_head = False

    def load_model(self):
        """Loads SUBLIME model components (model, learner, original features). 
        If not generating a new eval graph, loads the original adjacency.
        Does NOT generate the custom eval graph here; that's a separate step."""
        print(f"Loading SUBLIME model from {self.config.model_dir}")
        config_path = os.path.join(self.config.model_dir, 'config.txt')
        model_path = os.path.join(self.config.model_dir, 'model.pt')
        learner_path = os.path.join(self.config.model_dir, 'graph_learner.pt')
        features_path = os.path.join(self.config.model_dir, 'features.pt')
        adj_path = os.path.join(self.config.model_dir, 'adjacency.pt')

        if not all(os.path.exists(p) for p in [config_path, model_path, learner_path, features_path]):
            raise FileNotFoundError(f"One or more essential model files not found in {self.config.model_dir}. Expected config.txt, model.pt, graph_learner.pt, features.pt")

        # 1. Read model's original training config.txt
        model_config = {}
        with open(config_path, 'r') as f:
            for line in f:
                try:
                    key, value = line.strip().split(': ', 1)
                    if value == 'True': model_config[key] = True
                    elif value == 'False': model_config[key] = False
                    elif value == 'None': model_config[key] = None
                    elif '.' in value:
                        try: model_config[key] = float(value)
                        except ValueError: model_config[key] = value
                    else:
                        try: model_config[key] = int(value)
                        except ValueError: model_config[key] = value
                except ValueError:
                    print(f"Warning: Could not parse line in config.txt: {line.strip()}")
                    continue
        self.sparse = model_config.get('sparse', False) # Sparsity of the loaded SUBLIME model

        # 2. Load Original Model Features (features.pt SUBLIME was trained on)
        self.original_model_features = torch.load(features_path, map_location=self.device)
        print(f"Loaded original SUBLIME model features (from features.pt): {self.original_model_features.shape}")

        # 3. Load Original Adjacency (if not generating a new one for eval)
        # self.adj will be set here if using original, or later by generate_and_set_eval_anchor_graph
        if not self.config.generate_new_anchor_adj_for_eval:
            if not os.path.exists(adj_path):
                 raise FileNotFoundError(f"Original adjacency file (adjacency.pt) not found in {self.config.model_dir} and not generating new one for eval.")
            print(f"Loading existing original adjacency matrix from {adj_path} to be used for extraction.")
            adj_data_loaded = torch.load(adj_path, map_location=self.device)
            self._process_and_set_adj(adj_data_loaded, "original SUBLIME model adj", self.sparse)
        else:
            print("Will generate a new anchor graph for evaluation later using current evaluation dataset's features.")
            # self.adj will be set by generate_and_set_eval_anchor_graph

        # 4. Instantiate and Load Graph Learner (operates on original_model_features)
        learner_type = model_config.get('type_learner', 'unknown')
        feature_dim_for_learner = self.original_model_features.shape[1]
        k = model_config.get('k', 30)
        sim_func = model_config.get('sim_function', 'cosine')
        act_learner = model_config.get('activation_learner', 'relu')

        if learner_type == 'fgp':
            fgp_i = model_config.get('fgp_elu_alpha', 6) 
            self.graph_learner = FGP_learner(k=k, knn_metric=sim_func, i=fgp_i, sparse=self.sparse, initial_graph_data=None)
        elif learner_type == 'mlp':
            self.graph_learner = MLP_learner(nlayers=2, isize=feature_dim_for_learner, k=k, knn_metric=sim_func, i=6, sparse=self.sparse, act=act_learner, knn_threshold_type=model_config.get('knn_threshold_type', 'none'), knn_std_dev_factor=model_config.get('knn_std_dev_factor', 1.0), chunk_size=model_config.get('graph_learner_chunk_size', 100))
        elif learner_type == 'att':
             self.graph_learner = ATT_learner(nlayers=2, in_dim=feature_dim_for_learner, k=k, knn_metric=sim_func, i=6, sparse=self.sparse, mlp_act=act_learner)
        elif learner_type == 'gnn':
             self.graph_learner = GNN_learner(nlayers=2, in_dim=feature_dim_for_learner, k=k, knn_metric=sim_func, i=6, sparse=self.sparse, mlp_act=act_learner, anchor_adj=None)
        else:
            raise ValueError(f"Unsupported learner type found in config: {learner_type}")
        self.graph_learner.load_state_dict(torch.load(learner_path, map_location=self.device))
        self.graph_learner = self.graph_learner.to(self.device)
        self.graph_learner.eval()

        # 5. Instantiate and Load GCL Model (dimensions based on original_model_features)
        gcl_params = {
            'nlayers': model_config.get('nlayers', 2),
            'in_dim': self.original_model_features.shape[1], 
            'hidden_dim': model_config['hidden_dim'], 'emb_dim': model_config['emb_dim'],
            'proj_dim': model_config['proj_dim'], 'dropout': model_config.get('dropout', 0.5),
            'dropout_adj': model_config.get('dropout_adj', 0.5),
            'sparse': self.sparse, # GCL model uses sparsity of its original training
            'use_layer_norm': model_config.get('use_layer_norm', False),
            'use_residual': model_config.get('use_residual', False),
            'use_arcface': model_config.get('use_arcface', False),
            'use_classification_head': model_config.get('use_classification_head', False)
        }
        if gcl_params['use_arcface']:
             gcl_params.update({'num_classes': model_config['num_classes'], 'arcface_scale': model_config.get('arcface_scale', 30.0), 'arcface_margin': model_config.get('arcface_margin', 0.5), 'use_sampled_arcface': model_config.get('use_sampled_arcface', False), 'arcface_num_samples': model_config.get('arcface_num_samples', None)})
        if gcl_params['use_classification_head']:
             gcl_params.update({'classification_dropout': model_config.get('classification_dropout', 0.3), 'classification_head_layers': model_config.get('classification_head_layers', 2)})
             self.has_classification_head = True
        
        self.model = GCL(**gcl_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("SUBLIME model (GCL, GraphLearner, original_features) loaded successfully!")
        if self.has_classification_head: print("Model includes a binary classification head.")

    def _process_and_set_adj(self, adj_data_raw, source_description, expected_final_sparse_format):
        """Helper to process raw adjacency (tensor or dict) and set self.adj.
        expected_final_sparse_format: bool, True if the GCL model expects a sparse DGL graph.
        """
        processed_adj = None
        if expected_final_sparse_format: # GCL model expects a sparse DGL graph
            if isinstance(adj_data_raw, dict) and 'edges' in adj_data_raw: # DGL dict from file
                num_nodes = adj_data_raw['num_nodes']
                edges = adj_data_raw['edges']
                weights = adj_data_raw.get('weights')
                processed_adj = dgl.graph(edges, num_nodes=num_nodes).to(self.device)
                if weights is not None: processed_adj.edata['w'] = weights.to(self.device)
            elif isinstance(adj_data_raw, torch.Tensor) and adj_data_raw.is_sparse: # Torch sparse tensor
                temp_adj = normalize(adj_data_raw, 'sym', True) 
                processed_adj = torch_sparse_to_dgl_graph(temp_adj)
            else:
                raise TypeError(f"For sparse GCL model, {source_description} is {type(adj_data_raw)}. Expected DGL dict or sparse torch.Tensor.")
        else: # GCL model expects a dense tensor
            if isinstance(adj_data_raw, torch.Tensor) and not adj_data_raw.is_sparse:
                processed_adj = normalize(adj_data_raw, 'sym', False)
            else:
                raise TypeError(f"For dense GCL model, {source_description} is {type(adj_data_raw)}. Expected dense torch.Tensor.")
        
        if processed_adj is None:
            raise RuntimeError(f"_process_and_set_adj: processed_adj is None before assignment to self.adj. adj_data_raw type: {type(adj_data_raw)}, source: {source_description}, expected_sparse: {expected_final_sparse_format}")

        self.adj = processed_adj.to(self.device)
        print(f"Adjacency matrix ({source_description}) processed and set. Type: {type(self.adj)}, Device: {self.adj.device}")


    def generate_and_set_eval_anchor_graph(self, eval_node_features_np, eval_node_ids_df):
        """Generates a new anchor graph using evaluation data features and sets it for extraction."""
        if not self.config.generate_new_anchor_adj_for_eval:
            print("Skipping generation of new eval anchor graph as flag is False.")
            return

        print("Generating new anchor graph for evaluation using provided evaluation dataset features...")
        eval_node_features_tensor = torch.from_numpy(eval_node_features_np.astype(np.float32)).to(self.device)
        self.current_extraction_features = eval_node_features_tensor # Features for the new graph
        print(f"Set current_extraction_features to eval data features: {self.current_extraction_features.shape}")

        raw_new_adj = _generate_custom_anchor_graph(
            node_features_tensor=self.current_extraction_features, 
            cpf_identifiers_df=eval_node_ids_df,
            id_column_name=self.config.anchor_adj_id_col_name,
            k_knn=self.config.anchor_adj_k_knn,
            sparse_output=self.config.anchor_adj_use_sparse_format, # Format of raw_new_adj
            device=self.device,
            knn_threshold_type=self.config.anchor_adj_knn_threshold_type,
            knn_std_dev_factor=self.config.anchor_adj_knn_std_dev_factor,
            relationship_csv_path=self.config.anchor_adj_relationship_csv,
            relationship_cpf1_col=self.config.anchor_adj_relationship_cpf1_col,
            relationship_cpf2_col=self.config.anchor_adj_relationship_cpf2_col,
            relationship_weight=self.config.anchor_adj_relationship_weight
        )

        if raw_new_adj is None:
            raise RuntimeError("_generate_custom_anchor_graph returned None, cannot proceed to process and set adjacency.")
        print(f"_generate_custom_anchor_graph returned: type={type(raw_new_adj)}, sparse={raw_new_adj.is_sparse if isinstance(raw_new_adj, torch.Tensor) else 'N/A (not a tensor)'}")

        # The newly generated graph's format (sparse/dense as per anchor_adj_use_sparse_format)
        # must be processed into the format expected by the SUBLIME model (self.sparse).
        # Example: if model is sparse (self.sparse=True), but new graph was made dense for some reason
        # (anchor_adj_use_sparse_format=False), _process_and_set_adj needs to handle this.
        # However, _generate_custom_anchor_graph produces raw tensor. _process_and_set_adj handles normalization & DGL conversion.
        # The critical part is that the GCL model's `self.sparse` flag dictates the final format of `self.adj`.
        
        # Check for potential mismatch in expectation vs. generation config
        if self.sparse != self.config.anchor_adj_use_sparse_format:
            print(f"Warning: SUBLIME model's expected graph sparsity (self.sparse={self.sparse}) "
                  f"differs from the new anchor graph's generation config (anchor_adj_use_sparse_format={self.config.anchor_adj_use_sparse_format}). "
                  f"The _process_and_set_adj method will attempt to convert to the model's expected format, but ensure this is intended.")

        self._process_and_set_adj(raw_new_adj, "newly generated eval anchor graph", self.sparse)
        print(f"Successfully generated and set new anchor graph for evaluation. Final self.adj type: {type(self.adj)}")


    def build_faiss_index_if_needed(self):
        """Builds FAISS index from self.original_model_features for graph learner optimization."""
        if self.original_model_features is None: 
             print("Cannot build FAISS index: Original SUBLIME model features not loaded.")
             return
        print("Building FAISS index for SUBLIME's original features (for graph learner if used)...")
        try:
            from utils import build_faiss_index 
            k = getattr(self.graph_learner, 'k', 10) 
            self.faiss_index = build_faiss_index(self.original_model_features, k=k, use_gpu=torch.cuda.is_available())
            print("FAISS index built successfully for original_model_features!")
        except ImportError: # pragma: no cover
            print("FAISS or utils.py not found. Skipping FAISS index optimization.")
            self.faiss_index = None
        except Exception as e: # pragma: no cover
            print(f"Failed to build FAISS index for original_model_features: {str(e)}. Continuing without index optimization.")
            self.faiss_index = None

    def extract_embeddings(self, X_new_points_np, dataset_tag): 
        """Extracts embeddings in batches, handles caching."""
        if self.model is None:
            raise RuntimeError("SUBLIME model not loaded. Call load_model() first.")
        if self.adj is None and self.config.use_loaded_adj_for_extraction:
             raise RuntimeError("Adjacency matrix (self.adj) is not set for anchor mode. Ensure it was loaded or generate_and_set_eval_anchor_graph was called.")

        print(f"Extracting SUBLIME features for dataset tag: {dataset_tag} using features of shape {X_new_points_np.shape}...")

        cache_file_embeddings = None
        cache_file_classifications = None
        can_cache = self.config.cache_dir and dataset_tag
        
        extraction_mode_tag = ""
        if self.config.use_loaded_adj_for_extraction:
            if self.config.generate_new_anchor_adj_for_eval:
                rel_file_hash_part = ""
                if self.config.anchor_adj_relationship_csv and os.path.exists(self.config.anchor_adj_relationship_csv):
                    try:
                        fname = os.path.basename(self.config.anchor_adj_relationship_csv)
                        mtime = str(os.path.getmtime(self.config.anchor_adj_relationship_csv))[:5]
                        sanitized_fname = ''.join(c if c.isalnum() else '-' for c in fname.replace('.csv','').replace('.tsv',''))[:10]
                        rel_file_hash_part = f"_rel_{sanitized_fname}_{mtime}"
                    except Exception: rel_file_hash_part = "_rel_err"
                extraction_mode_tag = f"_anchor_evalset_k{self.config.anchor_adj_k_knn}{rel_file_hash_part}"
            else:
                extraction_mode_tag = "_anchor_orig_loaded" 
        else:
            extraction_mode_tag = "_learner_active"


        if can_cache:
            cache_file_base_embeddings = f"sublime_embeddings_{self.config.model_name_tag}_{dataset_tag}{extraction_mode_tag}.npy"
            cache_file_embeddings = os.path.join(self.config.cache_dir, cache_file_base_embeddings)
            if os.path.exists(cache_file_embeddings):
                print(f"Loading cached embeddings from {cache_file_embeddings}")
                loaded_embeddings = np.load(cache_file_embeddings)
                if len(loaded_embeddings.shape) == 1: loaded_embeddings = loaded_embeddings.reshape(-1, 1)
                loaded_classification_probs = None
                if self.has_classification_head:
                     cache_file_base_classifications = f"sublime_classifications_{self.config.model_name_tag}_{dataset_tag}{extraction_mode_tag}.npy"
                     cache_file_classifications = os.path.join(self.config.cache_dir, cache_file_base_classifications)
                     if os.path.exists(cache_file_classifications):
                          print(f"Loading cached classification results from {cache_file_classifications}")
                          class_results = np.load(cache_file_classifications)
                          if class_results.ndim > 0 and class_results.shape[0] > 0 :
                            if class_results.ndim == 1: loaded_classification_probs = class_results
                            elif class_results.shape[1] >= 1: loaded_classification_probs = class_results[:, 0]
                            else: print("Warning: Classification cache file has unexpected shape (cols).")
                          else: print("Warning: Classification cache file is empty or has unexpected shape (rows/dim).")
                     else: print("Classification cache file not found.")
                print(f"Cached embeddings shape: {loaded_embeddings.shape}")
                return {'embeddings': loaded_embeddings, 'classification_probs': loaded_classification_probs}
            else: print(f"Cache file not found: {cache_file_embeddings}. Extracting embeddings...")
        else: print("Caching disabled or dataset_tag not provided.")
            
        num_batches = (len(X_new_points_np) + self.config.batch_size - 1) // self.config.batch_size
        all_embeddings = []
        all_class_probs = [] if self.has_classification_head else None
        
        self.model.eval()
        if self.graph_learner: self.graph_learner.eval()

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"Extracting {dataset_tag}", unit="batch"):
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, len(X_new_points_np))
                batch_X_new_points_sub_np = X_new_points_np[start_idx:end_idx]
                batch_X_new_points_tensor = torch.FloatTensor(batch_X_new_points_sub_np).to(self.device)

                for j in range(len(batch_X_new_points_tensor)):
                    point_tensor = batch_X_new_points_tensor[j].unsqueeze(0) # This is a preprocessed new person's features

                    try:
                        # Determine which set of features to use for comparison and as input to GCL
                        # And which adjacency to use
                        features_for_gcl_input_and_comparison = None
                        adj_for_gcl_input = None
                        
                        if self.config.use_loaded_adj_for_extraction:
                            if self.config.generate_new_anchor_adj_for_eval and self.current_extraction_features is not None:
                                # Mode 1A: Newly generated anchor graph from EVALUATION data
                                features_for_gcl_input_and_comparison = self.current_extraction_features # Eval data features
                                adj_for_gcl_input = self.adj # Newly generated graph based on eval data features
                            else:
                                # Mode 1B: Original SUBLIME model's loaded anchor graph
                                features_for_gcl_input_and_comparison = self.original_model_features # SUBLIME's original training features
                                adj_for_gcl_input = self.adj # Original loaded graph
                        else:
                            # Mode 2: Graph learner active (operates on original_model_features)
                            features_for_gcl_input_and_comparison = self.original_model_features 
                            # adj_for_gcl_input will be learned by graph_learner dynamically below

                        if features_for_gcl_input_and_comparison is None:
                            raise RuntimeError("features_for_gcl_input_and_comparison is None. Logic error.")
                        if adj_for_gcl_input is None and self.config.use_loaded_adj_for_extraction:
                             raise RuntimeError("adj_for_gcl_input is None in anchor mode. Adj not set.")

                        # 1. Find most similar point in features_for_gcl_input_and_comparison
                        # The point_tensor (new person) is compared against the feature set corresponding to the active graph.
                        normalized_comparison_features = F.normalize(features_for_gcl_input_and_comparison, p=2, dim=1)
                        normalized_point = F.normalize(point_tensor, p=2, dim=1) # point_tensor is from X_new_points_np
                        
                        # Ensure point_tensor and normalized_comparison_features have compatible dimensions for mm
                        if point_tensor.shape[1] != features_for_gcl_input_and_comparison.shape[1]:
                            raise ValueError(f"Dimension mismatch: point_tensor ({point_tensor.shape[1]}) vs features_for_gcl_input_and_comparison ({features_for_gcl_input_and_comparison.shape[1]}). Ensure GCL model's in_dim and input point features align.")

                        similarities = torch.mm(normalized_point, normalized_comparison_features.t())
                        replace_idx = torch.argmax(similarities).item()

                        # Features and Adjacency that actually go into GCL.forward()
                        final_features_for_gcl_model = None
                        final_adj_for_gcl_model = None

                        if self.config.use_loaded_adj_for_extraction:
                            # Anchor graph mode: GCL sees the features corresponding to self.adj
                            final_features_for_gcl_model = features_for_gcl_input_and_comparison 
                            final_adj_for_gcl_model = adj_for_gcl_input
                        else:
                            # Graph learner mode: GCL sees modified original features and newly learned adj
                            # Here, features_for_gcl_input_and_comparison IS self.original_model_features
                            modified_features_for_learner = features_for_gcl_input_and_comparison.clone()
                            modified_features_for_learner[replace_idx] = point_tensor 
                            
                            new_learned_adj = self.graph_learner(modified_features_for_learner, faiss_index=self.faiss_index)
                            if not self.sparse: 
                                if isinstance(new_learned_adj, torch.Tensor):
                                    new_learned_adj = symmetrize(new_learned_adj)
                                    new_learned_adj = normalize(new_learned_adj, 'sym', self.sparse)
                                else: print(f"Warning: Dense graph learner output type {type(new_learned_adj)}.")
                            
                            final_features_for_gcl_model = modified_features_for_learner
                            final_adj_for_gcl_model = new_learned_adj
                        
                        if self.has_classification_head:
                            _, embedding, _, classification_output = self.model(final_features_for_gcl_model, final_adj_for_gcl_model, include_features=True)
                            classification_prob = torch.sigmoid(classification_output[replace_idx]).item() if classification_output is not None else None
                        else:
                            _, embedding = self.model(final_features_for_gcl_model, final_adj_for_gcl_model)
                            classification_prob = None
                        
                        embedding_tensor = embedding[replace_idx].detach()
                        if isinstance(embedding_tensor, torch.Tensor):
                            embedding_vector = embedding_tensor.cpu().numpy().flatten()
                        else: raise TypeError(f"Unexpected type for extracted embedding: {type(embedding_tensor)}")
                        all_embeddings.append(embedding_vector)

                        if self.has_classification_head and classification_prob is not None:
                             if isinstance(classification_prob, (int, float, np.number)):
                                 all_class_probs.append(float(classification_prob))
                             else: raise TypeError(f"Unexpected type for classification_probability: {type(classification_prob)}")
                    except Exception as e:
                        print(f"Error processing point {start_idx + j} (new point features shape: {point_tensor.shape}): {str(e)}")
                        import traceback
                        traceback.print_exc()


        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        classification_probs_array = np.array(all_class_probs, dtype=np.float32) if self.has_classification_head and all_class_probs else None

        if embeddings_array.shape[0] != X_new_points_np.shape[0] and embeddings_array.shape[0] > 0 : 
             print(f"WARNING: Expected {X_new_points_np.shape[0]} embeddings but got {embeddings_array.shape[0]}!")
        elif embeddings_array.shape[0] == 0 and X_new_points_np.shape[0] > 0:
            print(f"CRITICAL WARNING: No embeddings extracted for {X_new_points_np.shape[0]} points. Returning empty. Check errors.")
            return {'embeddings': np.array([], dtype=np.float32).reshape(0,0), 'classification_probs': None}

        print("L2 Normalizing SUBLIME embeddings...")
        if embeddings_array.shape[0] > 0: 
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10 
            embeddings_array = embeddings_array / norms
            print(f"Normalization complete. Final embeddings shape: {embeddings_array.shape}")
        else:
            print("Skipping normalization as no embeddings were extracted.")

        if can_cache:
            if cache_file_embeddings: 
                print(f"Saving embeddings to cache: {cache_file_embeddings}")
                np.save(cache_file_embeddings, embeddings_array)
            if self.has_classification_head and classification_probs_array is not None and cache_file_classifications:
                np.save(cache_file_classifications, classification_probs_array.reshape(-1, 1) if classification_probs_array.ndim == 1 else classification_probs_array)
                print(f"Saving classification probabilities to cache: {cache_file_classifications}")

        return {'embeddings': embeddings_array, 'classification_probs': classification_probs_array}

    def evaluate_builtin_classifier(self, sublime_results, y_true, dataset_tag):
        """Evaluates the model's built-in classifier if present."""
        if not self.has_classification_head or 'classification_probs' not in sublime_results or sublime_results['classification_probs'] is None:
            print("No built-in classifier results to evaluate.")
            return

        print(f"{'='*80}\nBUILT-IN BINARY CLASSIFIER PERFORMANCE ({dataset_tag})\n{'='*80}")
        probabilities = sublime_results['classification_probs']
        # Assuming binary classification, threshold at 0.5 for predictions
        predictions = (probabilities >= 0.5).astype(int)

        if len(probabilities) != len(y_true):
             print(f"Warning: Mismatch between probabilities ({len(probabilities)}) and true labels ({len(y_true)}). Skipping built-in evaluation.")
             return

        try:
            accuracy = accuracy_score(y_true, predictions)
            auc = roc_auc_score(y_true, probabilities)
            report = classification_report(y_true, predictions)

            print(f"Accuracy: {accuracy:.4f}")
            print(f"AUC-ROC: {auc:.4f}")
            print("Classification Report:")
            print(report)

            # Save results
            results_df = pd.DataFrame({'target': y_true, 'model_prediction': predictions, 'model_probability': probabilities})
            results_path = os.path.join(self.config.output_dir, f"{self.config.dataset_name}_{dataset_tag}_model_classification_results.csv")
            results_df.to_csv(results_path, index=False)
            print(f"Built-in classifier results saved to {results_path}")

        except Exception as e:
            print(f"Error during built-in classifier evaluation: {e}")
        print("="*80)

# --- Feature Engineering Class ---
class FeatureEngineer:
    """Handles creation of KNN features and combined feature sets."""
    def __init__(self, config):
        self.config = config

    def _calculate_knn_features_single(self, query_embeddings, index_embeddings, index_labels, k, query_is_index=False):
        """Calculates KNN features for one set (e.g., train vs train, val vs train)."""
        # This function contains the core logic from the original calculate_knn_features
        t_start = time.time()
        n_query, n_index = query_embeddings.shape[0], index_embeddings.shape[0]
        print(f"Calculating KNN features for {n_query} query points using {n_index} index points (k={k}, query_is_index={query_is_index}).")

        if n_query == 0 or n_index == 0:
            print("Warning: Empty query or index embeddings provided to calculate_knn_features.")
            return np.zeros((n_query, 8)) # Return dummy features (8 features)

        # Embeddings should already be L2 normalized numpy arrays on CPU
        query_np = query_embeddings.astype('float32')
        index_np = index_embeddings.astype('float32')
        index_labels_np = np.array(index_labels).astype(float) # Ensure float for mean

        # Adjust k if necessary
        original_k = k
        if query_is_index and k >= n_index:
            k = max(1, n_index - 1) if n_index > 1 else 0
            if k != original_k: print(f"Warning: Adjusted k from {original_k} to {k} for self-query.")
        elif not query_is_index and k > n_index:
            k = n_index
            if k != original_k: print(f"Warning: Adjusted k from {original_k} to {k} due to index size.")

        if k <= 0:
            print("Warning: k <= 0 after adjustment. Returning zeros.")
            return np.zeros((n_query, 8))

        # Build FAISS index (IP for Cosine Similarity on normalized vectors)
        d = index_np.shape[1]
        index = faiss.IndexFlatIP(d)
        
        # Use GPU if available
        gpu_res = None
        if 'cuda' in str(self.config.device) and hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
                # print("  Using GPU for FAISS index.") # Less verbose
            except Exception as e:
                print(f"  GPU FAISS failed: {e}. Using CPU.")
                index = faiss.IndexFlatIP(d) # Fallback to CPU index

        index.add(index_np)

        # Search
        search_k = k + 1 if query_is_index else k
        similarities_np, indices_np = index.search(query_np, search_k)

        # Handle self-query results
        if query_is_index:
            if indices_np.shape[1] > 1:
                 similarities_np = similarities_np[:, 1:]
                 indices_np = indices_np[:, 1:]
            elif indices_np.shape[1] == 1 and n_query > 0: # Only found self
                 print("  Warning: Only self found during self-query search. KNN features based on 0 neighbors.")
                 similarities_np = np.zeros((n_query, 0))
                 indices_np = np.zeros((n_query, 0), dtype=int)
            else: # Found 0 neighbors (unlikely for IndexFlatIP if index not empty)
                 print("  Warning: Found 0 neighbors during self-query search.")
                 similarities_np = np.zeros((n_query, 0))
                 indices_np = np.zeros((n_query, 0), dtype=int)

        actual_k_found = similarities_np.shape[1]
        if actual_k_found < k and actual_k_found > 0: print(f"  Warning: Found only {actual_k_found} valid neighbors (k={k} requested).")
        elif actual_k_found == 0 and k > 0: print(f"  Warning: Found 0 valid neighbors.")

        # Calculate distances (d = sqrt(2 - 2 * cos_sim))
        clipped_similarities = np.clip(similarities_np, -1.0, 1.0)
        distances_np = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * clipped_similarities))

        # Pre-fetch labels efficiently
        valid_indices_flat = indices_np.ravel()
        valid_indices_mask = (valid_indices_flat >= 0) & (valid_indices_flat < len(index_labels_np))
        unique_valid_indices = np.unique(valid_indices_flat[valid_indices_mask])
        labels_map = {idx: index_labels_np[idx] for idx in unique_valid_indices} if len(unique_valid_indices) > 0 else {}

        # Initialize feature arrays
        nan_fill = np.full(n_query, np.nan)
        features = {
            'mean_dist': nan_fill.copy(), 'std_dist': nan_fill.copy(),
            'min_dist': nan_fill.copy(), 'max_dist': nan_fill.copy(),
            'mean_label': nan_fill.copy(), 'weighted_mean_label': nan_fill.copy(),
            'label_var': nan_fill.copy(), 'class_margin': nan_fill.copy()
        }

        # Calculate features per query point
        for i in range(n_query):
            if actual_k_found == 0: continue
            row_indices = indices_np[i]
            row_distances = distances_np[i]
            
            # Get valid neighbors for this row
            row_valid_mask = (row_indices >= 0) & (row_indices < len(index_labels_np))
            if not np.any(row_valid_mask): continue
            
            valid_row_indices = row_indices[row_valid_mask]
            valid_row_distances = row_distances[row_valid_mask]
            
            # Lookup labels, handle potential misses (though shouldn't happen with map)
            valid_row_labels = np.array([labels_map.get(idx, np.nan) for idx in valid_row_indices])
            valid_label_mask = ~np.isnan(valid_row_labels)

            # Calculate distance stats (always possible if neighbors exist)
            if len(valid_row_distances) > 0:
                features['mean_dist'][i] = np.mean(valid_row_distances)
                features['std_dist'][i] = np.std(valid_row_distances) if len(valid_row_distances) > 1 else 0
                features['min_dist'][i] = np.min(valid_row_distances)
                features['max_dist'][i] = np.max(valid_row_distances)

            # Calculate label stats (only if valid labels exist)
            if np.any(valid_label_mask):
                filtered_labels = valid_row_labels[valid_label_mask]
                filtered_distances = valid_row_distances[valid_label_mask] # Use corresponding distances

                if len(filtered_labels) > 0:
                    mean_lbl = np.mean(filtered_labels)
                    features['mean_label'][i] = mean_lbl
                    features['label_var'][i] = np.var(filtered_labels)
                    features['class_margin'][i] = 2.0 * mean_lbl - 1.0 # Assumes binary 0/1

                    # Weighted mean
                    if len(filtered_distances) > 0:
                        weights = 1.0 / (filtered_distances + 1e-10) # Avoid division by zero
                        weights_sum = np.sum(weights)
                        if weights_sum > 1e-10: # Avoid division by zero if all distances were huge
                            features['weighted_mean_label'][i] = np.sum(filtered_labels * weights) / weights_sum
                        else: 
                             features['weighted_mean_label'][i] = mean_lbl # Fallback to unweighted mean

        # Impute NaNs using the median of calculated values (more robust)
        imputation_values = {}
        for key, arr in features.items():
            if np.all(np.isnan(arr)): # Handle case where no features could be calculated
                 median_val = 0.5 if 'label' in key else 0 # Default imputation
            else:
                 median_val = np.nanmedian(arr)
            imputation_values[key] = median_val
            features[key] = np.nan_to_num(arr, nan=median_val)

        # Combine features into the final array
        knn_features_array = np.column_stack(list(features.values()))

        # Cleanup FAISS index (especially GPU memory)
        del index
        if gpu_res is not None:
            del gpu_res # Free GPU resources
        gc.collect()

        t_end = time.time()
        print(f"KNN features calculated. Shape: {knn_features_array.shape}. Time: {t_end - t_start:.4f}s")
        return knn_features_array

    def create_all_feature_sets(self, split_data):
        """Creates 'dataset', 'sublime', 'concat', and 'knn_kX' feature sets for train/val/test."""
        feature_sets = {}
        y_train = split_data['y_train']
        y_train_val = split_data['y_train_val']

        # --- Base: Dataset Features ---
        if 'X_dataset_train' in split_data:
            feature_sets['dataset'] = {
                'train': split_data['X_dataset_train'],
                'val': split_data['X_dataset_val'],
                'test': split_data['X_dataset_test'],
                'train_val': split_data['X_dataset_train_val']
            }
            print(f"Dataset features shapes: Train={feature_sets['dataset']['train'].shape}, Val={feature_sets['dataset']['val'].shape}, Test={feature_sets['dataset']['test'].shape}")
        else:
             print("Warning: Dataset features not found in split_data. Skipping 'dataset' feature set.")

        # --- Base: SUBLIME Embeddings Only ---
        if 'sublime_train' in split_data:
            feature_sets['sublime'] = {
                'train': split_data['sublime_train'],
                'val': split_data['sublime_val'],
                'test': split_data['sublime_test'],
                'train_val': split_data['sublime_train_val']
            }
            print(f"SUBLIME features shapes: Train={feature_sets['sublime']['train'].shape}, Val={feature_sets['sublime']['val'].shape}, Test={feature_sets['sublime']['test'].shape}")
        else:
            print("Warning: Sublime embeddings not found in split_data. Skipping 'sublime' feature set.")


        # --- Concatenated Features ---
        # Ensure 'dataset' and 'sublime' sets exist before creating 'concat'
        if 'dataset' in feature_sets and 'sublime' in feature_sets:
            try:
                # Base concatenation using the newly created 'dataset' and 'sublime' sets
                concat_train = np.hstack((feature_sets['dataset']['train'], feature_sets['sublime']['train']))
                concat_val = np.hstack((feature_sets['dataset']['val'], feature_sets['sublime']['val']))
                concat_test = np.hstack((feature_sets['dataset']['test'], feature_sets['sublime']['test']))
                concat_train_val = np.hstack((feature_sets['dataset']['train_val'], feature_sets['sublime']['train_val']))

                # Add classification probabilities if available
                if 'cls_probs_train' in split_data:
                     print("Adding SUBLIME classification probability feature.")
                     # Ensure shapes match before hstack
                     cls_prob_train = split_data['cls_probs_train'].reshape(-1, 1)
                     cls_prob_val = split_data['cls_probs_val'].reshape(-1, 1)
                     cls_prob_train_val = split_data['cls_probs_train_val'].reshape(-1, 1)

                     if cls_prob_train.shape[0] == concat_train.shape[0]:
                         concat_train = np.hstack((concat_train, cls_prob_train))
                     if cls_prob_val.shape[0] == concat_val.shape[0]:
                         concat_val = np.hstack((concat_val, cls_prob_val))
                     if cls_prob_train_val.shape[0] == concat_train_val.shape[0]:
                         concat_train_val = np.hstack((concat_train_val, cls_prob_train_val))

                     # Add for test if available
                     if 'cls_probs_test' in split_data:
                          cls_prob_test = split_data['cls_probs_test'].reshape(-1, 1)
                          if cls_prob_test.shape[0] == concat_test.shape[0]:
                              concat_test = np.hstack((concat_test, cls_prob_test))
                          else: # Impute test if shapes mismatch or missing
                               print("Warning: Test classification probabilities shape mismatch or missing for concatenation. Filling with 0.5.")
                               prob_col = np.full((concat_test.shape[0], 1), 0.5)
                               concat_test = np.hstack((concat_test, prob_col))
                     else:
                          # If test probs missing entirely
                          print("Warning: Test classification probabilities missing for concatenation. Filling with 0.5.")
                          prob_col = np.full((concat_test.shape[0], 1), 0.5)
                          concat_test = np.hstack((concat_test, prob_col))

                feature_sets['concat'] = {
                     'train': concat_train, 'val': concat_val, 'test': concat_test, 'train_val': concat_train_val
                }
                print(f"Concatenated features shapes: Train={concat_train.shape}, Val={concat_val.shape}, Test={concat_test.shape}")

            except ValueError as e:
                 print(f"Error during feature concatenation: {e}. Check shapes:")
                 print(f"  Dataset train: {feature_sets.get('dataset', {}).get('train', np.array([])).shape}")
                 print(f"  Sublime train: {feature_sets.get('sublime', {}).get('train', np.array([])).shape}")
                 if 'cls_probs_train' in split_data: print(f"  ClsProb train: {split_data['cls_probs_train'].shape}")
                 # Skip concat if failed
                 if 'concat' in feature_sets: del feature_sets['concat']
            except Exception as e:
                 print(f"Unexpected error during concatenation: {e}")
                 if 'concat' in feature_sets: del feature_sets['concat']
        elif 'sublime' not in feature_sets:
             print("Warning: Sublime embeddings missing. Skipping 'concat' and 'knn' feature sets.")
        elif 'dataset' not in feature_sets:
             print("Warning: Base 'dataset' features missing. Skipping 'concat' and 'knn' feature sets.")


        # --- KNN Features ---
        # Ensure 'concat' exists before creating KNN features
        if 'concat' in feature_sets and self.config.active_k_values:
            # Use the 'sublime' features from split_data directly for KNN calculation
            sublime_train = split_data['sublime_train']
            sublime_val = split_data['sublime_val']
            sublime_test = split_data['sublime_test']
            sublime_train_val = split_data['sublime_train_val']

            # Embeddings should already be normalized by SublimeHandler

            for k in self.config.active_k_values:
                print(f"Calculating KNN features for k={k}...")
                knn_key = f'knn_k{k}'
                try:
                    # Calculate KNN features for each split
                    knn_feat_train = self._calculate_knn_features_single(sublime_train, sublime_train, y_train, k, query_is_index=True)
                    knn_feat_val = self._calculate_knn_features_single(sublime_val, sublime_train, y_train, k, query_is_index=False)
                    # Use train+val as the index for test set KNN features
                    knn_feat_test = self._calculate_knn_features_single(sublime_test, sublime_train_val, y_train_val, k, query_is_index=False)

                    # Combine train and val KNN features for the final training step
                    knn_feat_train_val = np.vstack((knn_feat_train, knn_feat_val))

                    # Concatenate KNN features with the 'concat' set
                    feature_sets[knn_key] = {
                        'train': np.hstack((feature_sets['concat']['train'], knn_feat_train)),
                        'val': np.hstack((feature_sets['concat']['val'], knn_feat_val)),
                        'test': np.hstack((feature_sets['concat']['test'], knn_feat_test)),
                        'train_val': np.hstack((feature_sets['concat']['train_val'], knn_feat_train_val))
                    }
                    print(f"KNN (k={k}) enhanced shapes: Train={feature_sets[knn_key]['train'].shape}, Val={feature_sets[knn_key]['val'].shape}, Test={feature_sets[knn_key]['test'].shape}")
                except Exception as e:
                    print(f"Error calculating or combining KNN features for k={k}: {e}")
                    # Ensure partial results for this k are removed if error occurs
                    if knn_key in feature_sets: del feature_sets[knn_key]

        elif not self.config.active_k_values:
             print("No valid k values provided. Skipping KNN feature generation.")
        elif 'concat' not in feature_sets:
             print("Warning: 'concat' feature set missing. Skipping KNN feature generation.")


        return feature_sets

# --- Evaluation Class ---
class Evaluator:
    """Runs Optuna optimization, trains final models, evaluates performance, and runs stacking."""
    def __init__(self, config):
        self.config = config
        self.class_weight_multiplier = config.class_weight_multiplier # Store the multiplier
        self.results = {} # {model_name: {feature_set: {metric: val, ...}}}
        self.best_params = {} # {model_name: {feature_set: {param: val, ...}}}
        self.stacking_results = {} # Store stacking results separately
        self.model_configs = {
            'xgboost': {
                'class': XGBClassifier,
                'base_params': {'random_state': 42, 'use_label_encoder': False, 'eval_metric': 'auc'},
                'trial_params': {'n_estimators': ['int', 50, 500], 'max_depth': ['int', 3, 10], 'learning_rate': ['float', 0.01, 0.3], 'subsample': ['float', 0.6, 1.0], 'colsample_bytree': ['float', 0.3, 1.0], 'min_child_weight': ['int', 1, 10], 'gamma': ['float', 0, 5], 'reg_alpha': ['float', 0, 10], 'reg_lambda': ['float', 0, 15]}
            },
            'catboost': {
                'class': CatBoostClassifier, 'base_params': {'random_seed': 42, 'verbose': False, 'eval_metric': 'AUC'},
                'trial_params': {'iterations': ['int', 50, 500], 'depth': ['int', 3, 10], 'learning_rate': ['float', 0.01, 0.3], 'l2_leaf_reg': ['float', 1, 10], 'random_strength': ['float', 0, 10], 'bagging_temperature': ['float', 0, 10]}
            },
            'lightgbm': {
                'class': LGBMClassifier, 'base_params': {'random_state': 42, 'verbosity': -1}, # Added verbosity=-1
                'trial_params': {'n_estimators': ['int', 50, 500], 'max_depth': ['int', 3, 10], 'learning_rate': ['float', 0.01, 0.3], 'num_leaves': ['int', 20, 100], 'subsample': ['float', 0.6, 1.0], 'colsample_bytree': ['float', 0.3, 1.0], 'reg_alpha': ['float', 0, 10], 'reg_lambda': ['float', 0, 15]}
            }
        }

    def _create_objective(self, model_class, train_features, train_labels, val_features, val_labels, base_params, trial_params):
        """Creates the Optuna objective function for a given model setup."""

        # --- Calculate Class Weights --- # Added
        calculated_weight = None
        if self.class_weight_multiplier > 1.0:
            neg_count = np.sum(train_labels == 0)
            pos_count = np.sum(train_labels == 1)
            if pos_count > 0:
                base_weight = neg_count / pos_count
                calculated_weight = base_weight * self.class_weight_multiplier
                print(f"  Applying class weight multiplier: {self.class_weight_multiplier:.2f}. Base weight: {base_weight:.2f}, Final weight for class 1: {calculated_weight:.2f}")
            else:
                print("  Warning: No positive samples in training labels for objective. Cannot calculate class weight.")
        # --- End Class Weight Calculation --- #

        def objective(trial):
            params = base_params.copy()
            for name, suggester_args in trial_params.items():
                 suggester_func = getattr(trial, f"suggest_{suggester_args[0]}")
                 params[name] = suggester_func(name, *suggester_args[1:])

            # Add class weights to initialization params if applicable # MODIFIED
            init_params = params.copy()
            if calculated_weight is not None:
                 if model_class == LGBMClassifier:
                      init_params['class_weight'] = {0: 1, 1: calculated_weight}
                 elif model_class == CatBoostClassifier:
                      init_params['class_weights'] = [1, calculated_weight]
                 elif model_class == XGBClassifier:
                      init_params['scale_pos_weight'] = calculated_weight

            model = model_class(**init_params) # Use modified init_params
            try:
                if isinstance(model, LGBMClassifier):
                    eval_metric = 'auc'
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, eval_metric)
                    early_stopping_callback = lightgbm.early_stopping(stopping_rounds=10, verbose=False) # Use verbose=False

                    # Apply class weight if calculated # Added
                    fit_params = {
                        'eval_set': [(val_features, val_labels)],
                        'eval_metric': eval_metric,
                        'callbacks': [pruning_callback, early_stopping_callback]
                    }
                    # REMOVED class weight from fit_params for LGBM

                    model.fit(train_features, train_labels, **fit_params)
                          # Removed direct early_stopping_rounds=10 argument
                elif isinstance(model, CatBoostClassifier):
                    # Apply class weight if calculated # Added
                    fit_params = {
                        'eval_set': (val_features, val_labels),
                        'verbose': False,
                        'early_stopping_rounds': 10
                    }
                    # REMOVED class weight from fit_params for CatBoost

                    model.fit(train_features, train_labels, **fit_params)
                elif isinstance(model, XGBClassifier):
                    # Manual Pruning Implementation for XGBoost >= 1.6
                    # XGBoost weight already handled during init_params creation above

                    # Removed application here

                    # eval_set calculation remains the same
                    eval_set = [(val_features, val_labels)]
                    eval_metric_name = params.get('eval_metric', 'auc') # Ensure AUC is used

                    # Add early stopping rounds for manual pruning logic
                    early_stopping_rounds = 20 # Set desired rounds

                    # Instantiate model with params AND early stopping
                    model = XGBClassifier(**params,
                                          early_stopping_rounds=early_stopping_rounds)
                                          # No Optuna callback needed

                    # Fit with eval_set. Early stopping happens internally.
                    model.fit(train_features, train_labels,
                              eval_set=eval_set,
                              verbose=False)

                    # --- Manual Pruning Logic ---
                    # Access evaluation results
                    results = model.evals_result()
                    # Check if results are available before accessing - Use 'validation_0'
                    if 'validation_0' in results and eval_metric_name in results['validation_0']:
                        validation_scores = results['validation_0'][eval_metric_name]

                        # Report intermediate scores to Optuna for pruning
                        for step, score in enumerate(validation_scores):
                            trial.report(score, step)
                            # Check if pruning is suggested
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                    else:
                        # Handle case where results are missing (e.g., finished in 1 iteration)
                        print(f"Warning: Trial {trial.number} - No validation scores found in evals_result for pruning. Results: {results}")
                        # Optionally, report a single score if available or just skip reporting
                        # Here, we'll proceed to calculate final AUC without intermediate reporting/pruning
                        pass
                    # --- End Manual Pruning Logic ---

                    # If loop completes without pruning, calculate final AUC for objective
                    preds_proba = model.predict_proba(val_features)[:, 1]
                    final_auc = roc_auc_score(val_labels, preds_proba)
                    return final_auc

                else:
                    model.fit(train_features, train_labels) # Generic fit

                # Check for pruning (for models using Optuna callbacks)
                # Note: XGBoost pruning is handled manually above
                if not isinstance(model, XGBClassifier) and trial.should_prune():
                     raise optuna.TrialPruned()

                # Calculate AUC for non-XGBoost models or if XGBoost wasn't pruned
                if not isinstance(model, XGBClassifier):
                     preds_proba = model.predict_proba(val_features)[:, 1]
                     return roc_auc_score(val_labels, preds_proba)
                # If XGBoost survived pruning, its final_auc is returned above

            except optuna.TrialPruned:
                 raise # Re-raise prune exceptions
            except KeyError as e:
                 # Handle potential issues accessing evals_result for XGBoost
                 if isinstance(model, XGBClassifier):
                     # Corrected print statement: remove trailing ()
                     print(f"KeyError accessing XGBoost evals_result: {e}. Results dict: {getattr(model, 'evals_result_', 'N/A')}")
                     return 0.0 # Return low score
                 else:
                     # Reraise if not XGBoost related
                     raise
            except Exception as e:
                 print(f"Warning: Trial failed with error: {e}")

        return objective

    def run_evaluation(self, feature_sets, split_data):
        """Runs the full evaluation loop: Optuna, train final, test."""
        print("Running model evaluation loop...")
        labels = {split: split_data[f'y_{split}'] for split in ['train', 'val', 'test', 'train_val']}

        # Make sure 'sublime' feature set exists if embeddings are available
        if 'sublime' not in feature_sets and 'sublime_train' in split_data:
            print("Warning: 'sublime' feature set missing from FeatureEngineer output but data exists.")
            # Attempt to add it manually, though it should be there
            feature_sets['sublime'] = {
                'train': split_data['sublime_train'],
                'val': split_data['sublime_val'],
                'test': split_data['sublime_test'],
                'train_val': split_data['sublime_train_val']
            }

        # Ensure feature_sets isn't empty before proceeding
        if not feature_sets:
            print("Error: No feature sets available to evaluate. Aborting.")
            return # Exit evaluation if no features

        for model_name, config in self.model_configs.items():
            print(f"{'='*50}\n{model_name.upper()} Models\n{'='*50}")
            self.results[model_name] = {}
            self.best_params[model_name] = {}
            model_class = config['class']
            base_params = config['base_params']
            trial_params = config['trial_params']

            # Now includes 'sublime' if available
            for feature_set_name, features in feature_sets.items():
                print(f"--- Evaluating Feature Set: {feature_set_name} ---")

                # Handle potential missing splits within a feature set gracefully
                required_splits = ['train', 'val', 'test', 'train_val']
                if not all(s in features for s in required_splits):
                    print(f"Warning: Feature set '{feature_set_name}' is missing required data splits. Skipping evaluation for this set.")
                    continue
                if not all(l in labels for l in required_splits):
                     print(f"Warning: Labels are missing required data splits. Skipping evaluation for this set.")
                     continue


                X_train, X_val, X_test, X_train_val = features['train'], features['val'], features['test'], features['train_val']
                y_train, y_val, y_test, y_train_val = labels['train'], labels['val'], labels['test'], labels['train_val']

                # Check for empty arrays before proceeding
                if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                    print(f"Warning: Empty train ({X_train.shape}) or validation ({X_val.shape}) features for {model_name} - {feature_set_name}. Skipping Optuna.")
                    self.results[model_name][feature_set_name] = {'error': 'Empty train/val features'}
                    continue


                # --- Optuna Optimization ---
                print(f"Optimizing {model_name.upper()} for {feature_set_name} features...")
                try:
                    objective_func = self._create_objective(model_class, X_train, y_train, X_val, y_val, base_params, trial_params)
                    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                    study.optimize(objective_func, n_trials=self.config.n_trials, n_jobs=1, show_progress_bar=True)
                    best_trial_params = study.best_params
                    self.best_params[model_name][feature_set_name] = best_trial_params
                    best_val_auc = study.best_value
                    print(f"Optimization complete. Best Validation AUC: {best_val_auc:.4f}")

                except Exception as e:
                    print(f"ERROR during Optuna optimization for {model_name} - {feature_set_name}: {e}")
                    print("Skipping evaluation for this combination.")
                    self.results[model_name][feature_set_name] = {'error': str(e)}
                    continue # Skip to next feature set

                # --- Final Model Training & Testing ---
                print("Training final model on train+validation data...")
                final_params = {**base_params, **best_trial_params}
                # final_model = model_class(**final_params)

                # --- Calculate Class Weights for Final Training --- # Added
                calculated_weight_final = None
                if self.class_weight_multiplier > 1.0:
                     neg_count_final = np.sum(y_train_val == 0)
                     pos_count_final = np.sum(y_train_val == 1)
                     if pos_count_final > 0:
                          base_weight_final = neg_count_final / pos_count_final
                          calculated_weight_final = base_weight_final * self.class_weight_multiplier
                          print(f"  Applying final class weight multiplier: {self.class_weight_multiplier:.2f}. Base weight: {base_weight_final:.2f}, Final weight for class 1: {calculated_weight_final:.2f}")
                     else:
                          print("  Warning: No positive samples in train+val labels. Cannot calculate final class weight.")
                 # --- End Final Class Weight Calculation --- #

                # Add weights to init params for final model # MODIFIED
                final_init_params = final_params.copy()
                if calculated_weight_final is not None:
                     if model_class == LGBMClassifier:
                          final_init_params['class_weight'] = {0: 1, 1: calculated_weight_final}
                     elif model_class == CatBoostClassifier:
                          final_init_params['class_weights'] = [1, calculated_weight_final]
                     elif model_class == XGBClassifier:
                          final_init_params['scale_pos_weight'] = calculated_weight_final

                final_model = model_class(**final_init_params) # Instantiate with final weights

                try:
                    # Check for empty train_val data
                    if X_train_val.shape[0] == 0 or X_test.shape[0] == 0:
                         print(f"Warning: Empty train_val ({X_train_val.shape}) or test ({X_test.shape}) features for {model_name} - {feature_set_name}. Skipping final training/testing.")
                         self.results[model_name][feature_set_name] = {'error': 'Empty train_val/test features'}
                         continue

                    # Apply weights during final fit # Added
                    # Fit without extra params, weights are in the model object now
                    if isinstance(final_model, LGBMClassifier):
                         # if calculated_weight_final is not None:
                         #      fit_params_final['class_weight'] = {0: 1, 1: calculated_weight_final}
                         # final_model.fit(X_train_val, y_train_val, **fit_params_final)
                         final_model.fit(X_train_val, y_train_val)
                    elif isinstance(final_model, CatBoostClassifier):
                         # if calculated_weight_final is not None:
                         #      fit_params_final['class_weights'] = [1, calculated_weight_final]
                         # final_model.fit(X_train_val, y_train_val, **fit_params_final)
                         final_model.fit(X_train_val, y_train_val)
                    elif isinstance(final_model, XGBClassifier):
                         # Weight already set during init, no need for set_params
                         # if calculated_weight_final is not None:
                         #      final_model.set_params(scale_pos_weight=calculated_weight_final)
                         final_model.fit(X_train_val, y_train_val) # Fit after setting param
                    else: # Others without specific weight params
                         final_model.fit(X_train_val, y_train_val)

                    print("Evaluating final model on test data...")
                    preds_test = final_model.predict(X_test)
                    preds_proba_test = final_model.predict_proba(X_test)[:, 1]

                    acc_test = accuracy_score(y_test, preds_test)
                    auc_test = roc_auc_score(y_test, preds_proba_test)
                    fpr_test, tpr_test, _ = roc_curve(y_test, preds_proba_test)
                    ks_test = np.max(tpr_test - fpr_test) if len(tpr_test) > 0 and len(fpr_test) > 0 else 0.0

                    print(f"{model_name.upper()} - {feature_set_name} - Test AUC: {auc_test:.4f}, KS: {ks_test:.4f}")
                    print("Test Set Classification Report:")
                    print(classification_report(y_test, preds_test))

                    self.results[model_name][feature_set_name] = {
                        'acc': acc_test, 'auc': auc_test, 'ks': ks_test,
                        'fpr': fpr_test, 'tpr': tpr_test,
                        'final_model': final_model # Store the trained model object
                    }
                except Exception as e:
                    print(f"ERROR during final model training/testing for {model_name} - {feature_set_name}: {e}")
                    self.results[model_name][feature_set_name] = {'error': str(e)}

    def run_stacking_evaluation(self, feature_sets, split_data):
        """Trains base models and a meta-model for stacking."""
        print(f"\n{'='*50}\nSTACKING EVALUATION\n{'='*50}")

        # --- Configuration ---
        # Define which models and feature sets to use as base learners
        # Example: Use best XGBoost on 'dataset' and best LightGBM on 'sublime'
        base_model_configs = [
            {'model_name': 'xgboost', 'feature_set': 'dataset'},
            {'model_name': 'lightgbm', 'feature_set': 'sublime'} # Use 'sublime' set now
        ]
        meta_model_class = LogisticRegression
        meta_model_params = {'random_state': 42, 'solver': 'liblinear'} # Simple params

        # --- Check Availability ---
        oof_preds_list = []
        test_preds_list = []
        labels = {split: split_data[f'y_{split}'] for split in ['train', 'val', 'test', 'train_val']}
        y_val = labels['val']
        y_test = labels['test']

        # --- Calculate Meta-Model Class Weights --- # Added
        meta_class_weight_dict = None
        if self.class_weight_multiplier > 1.0:
             neg_count_meta = np.sum(y_val == 0) # Meta model trained on val predictions/labels
             pos_count_meta = np.sum(y_val == 1)
             if pos_count_meta > 0:
                 base_weight_meta = neg_count_meta / pos_count_meta
                 calculated_weight_meta = base_weight_meta * self.class_weight_multiplier
                 meta_class_weight_dict = {0: 1, 1: calculated_weight_meta}
                 print(f"Applying meta-model class weight multiplier: {self.class_weight_multiplier:.2f}. Final weight for class 1: {calculated_weight_meta:.2f}")
             else:
                 print("Warning: No positive samples in validation labels for meta-model. Cannot calculate meta class weight.")

        # Define meta model params, including class weight
        meta_model_params = {
             'random_state': 42,
             'solver': 'liblinear',
             'class_weight': meta_class_weight_dict # Will be None if multiplier is 1 or pos_count is 0
        }
        # --- End Meta-Model Weight Calculation --- #

        print("Preparing base models for stacking...")
        for config in base_model_configs:
            model_name = config['model_name']
            feature_set = config['feature_set']
            print(f"  Base Model: {model_name.upper()} on Feature Set: {feature_set}")

            # Check if features and results exist
            if feature_set not in feature_sets:
                print(f"    ERROR: Feature set '{feature_set}' not found. Skipping this base model.")
                continue
            if model_name not in self.best_params or feature_set not in self.best_params[model_name]:
                print(f"    ERROR: Best parameters for {model_name}-{feature_set} not found (likely failed during optimization). Skipping this base model.")
                continue
            if 'train' not in feature_sets[feature_set] or 'val' not in feature_sets[feature_set] or 'test' not in feature_sets[feature_set] or 'train_val' not in feature_sets[feature_set]:
                 print(f"    ERROR: Missing data splits for feature set '{feature_set}'. Skipping this base model.")
                 continue


            # Get features
            X_train = feature_sets[feature_set]['train']
            X_val = feature_sets[feature_set]['val']
            X_test = feature_sets[feature_set]['test']
            X_train_val = feature_sets[feature_set]['train_val'] # For final prediction on test
            y_train = labels['train']
            y_train_val = labels['train_val']

            # --- Calculate Base Model Weights --- # Added
            calculated_weight_oof = None # For OOF prediction model (trained on train)
            calculated_weight_test = None # For test prediction model (trained on train+val)
            if self.class_weight_multiplier > 1.0:
                # Weight for OOF model (using y_train)
                neg_count_oof = np.sum(y_train == 0)
                pos_count_oof = np.sum(y_train == 1)
                if pos_count_oof > 0:
                    base_weight_oof = neg_count_oof / pos_count_oof
                    calculated_weight_oof = base_weight_oof * self.class_weight_multiplier
                # Weight for test model (using y_train_val)
                neg_count_test = np.sum(y_train_val == 0)
                pos_count_test = np.sum(y_train_val == 1)
                if pos_count_test > 0:
                    base_weight_test = neg_count_test / pos_count_test
                    calculated_weight_test = base_weight_test * self.class_weight_multiplier
            # --- End Base Model Weight Calculation --- #

            # Get model class and best params
            model_class = self.model_configs[model_name]['class']
            base_params = self.model_configs[model_name]['base_params']
            best_trial_params = self.best_params[model_name][feature_set]
            final_params = {**base_params, **best_trial_params}

            try:
                # --- Generate Validation (OOF) Predictions --- # Modified fit calls
                print(f"    Training {model_name}-{feature_set} on train split for validation predictions...")
                # model_for_oof = model_class(**final_params)
                # Add weights to init params for OOF model # MODIFIED
                init_params_oof = final_params.copy()
                if calculated_weight_oof is not None:
                    if model_class == LGBMClassifier:
                         init_params_oof['class_weight'] = {0: 1, 1: calculated_weight_oof}
                    elif model_class == CatBoostClassifier:
                         init_params_oof['class_weights'] = [1, calculated_weight_oof]
                    elif model_class == XGBClassifier:
                         init_params_oof['scale_pos_weight'] = calculated_weight_oof

                model_for_oof = model_class(**init_params_oof)

                # Fit without extra params
                if isinstance(model_for_oof, LGBMClassifier):
                    model_for_oof.fit(X_train, y_train)
                elif isinstance(model_for_oof, CatBoostClassifier):
                    model_for_oof.fit(X_train, y_train)
                elif isinstance(model_for_oof, XGBClassifier):
                    model_for_oof.fit(X_train, y_train)
                else: # Others without specific weight params
                    model_for_oof.fit(X_train, y_train)

                oof_preds = model_for_oof.predict_proba(X_val)[:, 1]
                oof_preds_list.append(oof_preds)
                print(f"    Generated validation predictions. Shape: {oof_preds.shape}")

                # --- Generate Test Predictions --- # Modified fit calls
                print(f"    Training {model_name}-{feature_set} on train+val split for test predictions...")
                # model_for_test = model_class(**final_params)
                # Add weights to init params for test model # MODIFIED
                init_params_test = final_params.copy()
                if calculated_weight_test is not None:
                     if model_class == LGBMClassifier:
                          init_params_test['class_weight'] = {0: 1, 1: calculated_weight_test}
                     elif model_class == CatBoostClassifier:
                          init_params_test['class_weights'] = [1, calculated_weight_test]
                     elif model_class == XGBClassifier:
                          init_params_test['scale_pos_weight'] = calculated_weight_test

                model_for_test = model_class(**init_params_test)

                # Fit without extra params
                if isinstance(model_for_test, LGBMClassifier):
                    model_for_test.fit(X_train_val, y_train_val)
                elif isinstance(model_for_test, CatBoostClassifier):
                    model_for_test.fit(X_train_val, y_train_val)
                elif isinstance(model_for_test, XGBClassifier):
                    model_for_test.fit(X_train_val, y_train_val)
                else: # Others without specific weight params
                    model_for_test.fit(X_train_val, y_train_val)

                test_preds = model_for_test.predict_proba(X_test)[:, 1]
                test_preds_list.append(test_preds)
                print(f"    Generated test predictions. Shape: {test_preds.shape}")

            except Exception as e:
                print(f"    ERROR training or predicting base model {model_name}-{feature_set}: {e}")
                # Need to handle this - maybe remove the partial predictions?
                # For simplicity, let's just skip this base model if it fails.
                # Ensure lists are consistent if a model fails mid-way
                if len(oof_preds_list) > len(test_preds_list): oof_preds_list.pop()
                if len(test_preds_list) > len(oof_preds_list): test_preds_list.pop()
                continue

        # --- Train Meta-Model ---
        if len(oof_preds_list) < 2: # Need at least two base models for stacking
            print("\nERROR: Not enough successful base models (< 2) to perform stacking. Aborting stacking evaluation.")
            self.stacking_results = {'error': 'Insufficient base models'}
            return

        print("\nTraining meta-model...")
        try:
            meta_X_train = np.column_stack(oof_preds_list)
            meta_X_test = np.column_stack(test_preds_list)

            print(f"  Meta-model input shape (validation set): {meta_X_train.shape}")
            print(f"  Meta-model input shape (test set): {meta_X_test.shape}")

            meta_model = meta_model_class(**meta_model_params)
            meta_model.fit(meta_X_train, y_val)
            print("  Meta-model trained.")

            # --- Evaluate Meta-Model ---
            print("Evaluating stacked model on test data...")
            final_preds_proba = meta_model.predict_proba(meta_X_test)[:, 1]
            final_preds = (final_preds_proba >= 0.5).astype(int) # Standard threshold

            acc_stack = accuracy_score(y_test, final_preds)
            auc_stack = roc_auc_score(y_test, final_preds_proba)
            fpr_stack, tpr_stack, _ = roc_curve(y_test, final_preds_proba)
            ks_stack = np.max(tpr_stack - fpr_stack) if len(tpr_stack) > 0 and len(fpr_stack) > 0 else 0.0

            print(f"STACKING MODEL - Test AUC: {auc_stack:.4f}, KS: {ks_stack:.4f}")
            print("Test Set Classification Report (Stacked Model):")
            print(classification_report(y_test, final_preds))

            self.stacking_results = {
                'acc': acc_stack, 'auc': auc_stack, 'ks': ks_stack,
                'fpr': fpr_stack, 'tpr': tpr_stack,
                'base_model_configs': base_model_configs # Store config for reference
            }

        except Exception as e:
            print(f"ERROR during meta-model training or evaluation: {e}")
            self.stacking_results = {'error': str(e)}


    def get_results(self):
        return self.results

    def get_best_params(self):
        return self.best_params

    def get_stacking_results(self):
        return self.stacking_results

# --- Reporting Class ---
class Reporter:
    """Handles generation of plots, summary CSVs, and console output."""
    def __init__(self, config):
        self.config = config

    def _get_feature_names(self, feature_set_name, data_manager, sublime_embeddings_dim, has_cls_prob, k=None):
        """Constructs feature names for plotting."""
        base_dataset_names = data_manager.get_dataset_feature_names()
        sublime_names = [f"SUBLIME_{i}" for i in range(sublime_embeddings_dim)]
        cls_prob_name = ["Model_Cls_Prob"] if has_cls_prob else []
        knn_base_names = ['knn_mean_dist', 'knn_std_dist', 'knn_min_dist', 'knn_max_dist',
                          'knn_mean_label', 'knn_weighted_mean_label', 'knn_label_var', 'knn_class_margin']

        if feature_set_name == 'dataset':
            return base_dataset_names
        elif feature_set_name == 'concat':
            return base_dataset_names + sublime_names + cls_prob_name
        elif feature_set_name.startswith('knn_k'):
            k_val = k or feature_set_name.split('knn_k')[1] # Get k if not passed
            specific_knn_names = [f'{name}_k{k_val}' for name in knn_base_names]
            return base_dataset_names + sublime_names + cls_prob_name + specific_knn_names
        else:
            return [] # Unknown feature set

    def generate_roc_plots(self, eval_results, stacking_results=None): # Add stacking_results parameter
        """Generates and saves ROC curve plots for each model, including stacking."""
        print("Generating ROC plots...")
        for model_name, model_type_results in eval_results.items():
            plt.figure(figsize=(10, 8))
            has_plot_data = False
            # Sort keys for consistent plot legend order
            # Ensure 'sublime' is also included in sorting logic if present
            sorted_feature_keys = sorted(model_type_results.keys(), key=lambda k: (k.startswith('knn'), k == 'concat', k == 'sublime', k))

            for feature_key in sorted_feature_keys:
                 data = model_type_results[feature_key]
                 if 'fpr' in data and 'tpr' in data and 'auc' in data and 'ks' in data:
                     label = feature_key.replace('_', ' ').title() # Nicer labels
                     if feature_key.startswith('knn_k'): label = f"KNN (k={feature_key.split('knn_k')[1]})"
                     elif feature_key == 'concat': label = "Dataset + SUBLIME" + (" + ClsProb" if 'cls_probs_train' in self.config.__dict__ else "") # Rough check
                     elif feature_key == 'dataset': label = "Dataset Features"
                     elif feature_key == 'sublime': label = "SUBLIME Features" # Label for sublime only

                     plt.plot(data['fpr'], data['tpr'], lw=2,
                              label=f'{label} (AUC = {data["auc"]:.4f}, KS = {data["ks"]:.4f})')
                     has_plot_data = True
                 else:
                     # print(f"Skipping ROC plot for {model_name} - {feature_key} due to missing data.") # Less verbose
                     pass

            # Add Stacking results to the plot of the *first* base model (e.g., xgboost) for simplicity
            # Or create a separate stacking plot? Let's add to first plot.
            if model_name == list(eval_results.keys())[0] and stacking_results and 'fpr' in stacking_results:
                 plt.plot(stacking_results['fpr'], stacking_results['tpr'], lw=2.5, linestyle='--', color='black',
                          label=f'Stacking Model (AUC = {stacking_results["auc"]:.4f}, KS = {stacking_results["ks"]:.4f})')
                 has_plot_data = True


            if has_plot_data:
                 plt.plot([0, 1], [0, 1], 'k--', lw=1)
                 plt.xlabel('False Positive Rate')
                 plt.ylabel('True Positive Rate')
                 # Adjust title if stacking included
                 plot_title = f'ROC Curves on Test Set - {self.config.dataset_name} - {model_name.upper()}'
                 if model_name == list(eval_results.keys())[0] and stacking_results and 'fpr' in stacking_results:
                     plot_title += " (+ Stacking)"

                 plt.title(plot_title)
                 plt.legend(loc='lower right')
                 plt.grid(alpha=0.4)
                 plot_filename = f"{self.config.dataset_name}_{model_name}_test_roc_curves_k_{self.config.k_str}.png"
                 # Modify filename if stacking is included on this plot
                 if model_name == list(eval_results.keys())[0] and stacking_results and 'fpr' in stacking_results:
                      plot_filename = plot_filename.replace('.png', '_with_stacking.png')

                 plot_path = os.path.join(self.config.plots_dir, plot_filename)
                 plt.savefig(plot_path)
                 print(f"Saved ROC plot: {plot_path}")
            else:
                 print(f"No valid data to plot ROC curve for {model_name}.")
            plt.close()

    def generate_feature_importance_plots(self, eval_results, data_manager, split_data):
         """Generates and saves feature importance plots."""
         print("Generating feature importance plots...")
         # Need dimensions and whether class prob was used
         sublime_dim = split_data.get('sublime_train', np.array([])).shape[1]
         has_cls_prob = 'cls_probs_train' in split_data

         for model_name, model_type_results in eval_results.items():
             # Sort keys for consistent plot order
             sorted_feature_keys = sorted(model_type_results.keys(), key=lambda k: (k.startswith('knn'), k))
             for feature_key in sorted_feature_keys:
                 data = model_type_results[feature_key]
                 model = data.get('final_model')

                 if not model or not hasattr(model, 'feature_importances_'):
                     # print(f"Skipping importance plot for {model_name} - {feature_key}: Model or importances missing.")
                     continue # Silently skip if no model/importance

                 importances = model.feature_importances_
                 k_val = int(feature_key.split('knn_k')[1]) if feature_key.startswith('knn_k') else None
                 current_feature_names = self._get_feature_names(feature_key, data_manager, sublime_dim, has_cls_prob, k=k_val)

                 if len(importances) != len(current_feature_names):
                      print(f"Warning: Mismatch between importance count ({len(importances)}) and feature names ({len(current_feature_names)}) for {model_name} - {feature_key}. Skipping plot.")
                      continue

                 plt.figure(figsize=(12, max(8, len(importances) * 0.3))) # Adjust height slightly
                 indices = np.argsort(importances)[::-1]
                 n_to_plot = min(30, len(importances)) # Plot top 30

                 plot_labels = [current_feature_names[i] for i in indices[:n_to_plot]]

                 plt.barh(range(n_to_plot), importances[indices[:n_to_plot]][::-1], align='center') # Horizontal bar plot
                 plt.yticks(range(n_to_plot), plot_labels[::-1]) # Labels on y-axis
                 plt.xlabel('Feature Importance')

                 title_suffix = feature_key.replace('_', ' ').title()
                 if feature_key.startswith('knn_k'): title_suffix = f"KNN Enhanced (k={k_val})"
                 elif feature_key == 'concat': title_suffix = "Dataset + SUBLIME" + (" + ClsProb" if has_cls_prob else "")
                 elif feature_key == 'dataset': title_suffix = "Dataset Features"

                 plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - {title_suffix}) - {self.config.dataset_name}")
                 plt.tight_layout()
                 plot_filename = f"{self.config.dataset_name}_{model_name}_{feature_key}_feature_importance_k_{self.config.k_str}.png"
                 plot_path = os.path.join(self.config.plots_dir, plot_filename)
                 plt.savefig(plot_path)
                 # print(f"Saved importance plot: {plot_path}") # Can be verbose
                 plt.close()

    def save_results_summary(self, eval_results, stacking_results=None): # Add stacking_results
        """Compiles and saves the summary metrics to a CSV file, including stacking."""
        print("Saving results summary...")
        summary_list = []
        for model_name, model_type_results in eval_results.items():
            # Check required base metrics first
            base_metrics = model_type_results.get('dataset', {})
            sublime_metrics = model_type_results.get('sublime', {}) # Get sublime-only results
            concat_metrics = model_type_results.get('concat', {})

            result_row = {
                'dataset': self.config.dataset_name,
                'model': model_name,
                'k_values_used': '_'.join(map(str, self.config.active_k_values)) if self.config.active_k_values else 'none',
            }

            # Add Dataset metrics
            if 'error' in base_metrics: base_metrics = {}
            result_row.update({
                'dataset_acc': base_metrics.get('acc'),
                'dataset_auc': base_metrics.get('auc'),
                'dataset_ks': base_metrics.get('ks'),
            })

            # Add SUBLIME-only metrics
            if 'error' in sublime_metrics: sublime_metrics = {}
            result_row.update({
                'sublime_acc': sublime_metrics.get('acc'),
                'sublime_auc': sublime_metrics.get('auc'),
                'sublime_ks': sublime_metrics.get('ks'),
            })

            # Add Concat metrics
            if 'error' in concat_metrics: concat_metrics = {}
            result_row.update({
                'concat_acc': concat_metrics.get('acc'),
                'concat_auc': concat_metrics.get('auc'),
                'concat_ks': concat_metrics.get('ks'),
            })
            # Removed concat_vs_dataset_improvement_auc and similar lines as they can be calculated later

            # Add KNN metrics
            for k in self.config.active_k_values:
                knn_key = f'knn_k{k}'
                metric_prefix = f'knn_k{k}'
                knn_metrics = model_type_results.get(knn_key, {})
                if 'error' in knn_metrics: knn_metrics = {} # Treat error as missing

                result_row.update({
                    f'{metric_prefix}_acc': knn_metrics.get('acc'),
                    f'{metric_prefix}_auc': knn_metrics.get('auc'),
                    f'{metric_prefix}_ks': knn_metrics.get('ks'),
                })
                # Removed knn_vs_dataset/concat improvement calculations

            summary_list.append(result_row)

        # Add Stacking results as a separate row (or column?) Let's do a row.
        if stacking_results and 'error' not in stacking_results:
             stacking_row = {
                 'dataset': self.config.dataset_name,
                 'model': 'Stacking', # Special model name
                 'k_values_used': '_'.join(map(str, self.config.active_k_values)) if self.config.active_k_values else 'none',
                 # Add metrics, leaving others blank
                 'stacking_acc': stacking_results.get('acc'),
                 'stacking_auc': stacking_results.get('auc'),
                 'stacking_ks': stacking_results.get('ks'),
             }
             # Optionally detail which base models were used
             base_info = " | ".join([f"{c['model_name']}-{c['feature_set']}" for c in stacking_results.get('base_model_configs', [])])
             stacking_row['stacking_base_models'] = base_info
             summary_list.append(stacking_row)
        elif stacking_results and 'error' in stacking_results:
             summary_list.append({
                 'dataset': self.config.dataset_name, 'model': 'Stacking', 'error': stacking_results['error']
             })


        if not summary_list:
            print("No valid results to save in summary.")
            return

        results_df = pd.DataFrame(summary_list)
        # Define columns explicitly for consistent order, adding new ones
        base_cols = ['dataset', 'model', 'k_values_used']
        dataset_cols = ['dataset_acc', 'dataset_auc', 'dataset_ks']
        sublime_cols = ['sublime_acc', 'sublime_auc', 'sublime_ks']
        concat_cols = ['concat_acc', 'concat_auc', 'concat_ks']
        knn_cols = []
        for k in self.config.active_k_values:
            knn_cols.extend([f'knn_k{k}_acc', f'knn_k{k}_auc', f'knn_k{k}_ks'])
        stacking_cols = ['stacking_acc', 'stacking_auc', 'stacking_ks', 'stacking_base_models', 'error']

        all_cols_ordered = base_cols + dataset_cols + sublime_cols + concat_cols + knn_cols + stacking_cols
        # Filter cols to only those present in the dataframe
        final_cols = [col for col in all_cols_ordered if col in results_df.columns]


        results_filename = f"{self.config.dataset_name}_all_models_test_results_k_{self.config.k_str}.csv"
        results_path = os.path.join(self.config.output_dir, results_filename)
        results_df[final_cols].to_csv(results_path, index=False, float_format='%.4f') # Use float format
        print(f"Test results summary saved to {results_path}")

    def save_best_params(self, best_params):
        """Saves the best hyperparameters found by Optuna to CSV files."""
        print("Saving best hyperparameters...")
        for model_name, model_params in best_params.items():
            params_dict = {}
            # Ensure consistent column order
            all_feature_keys = sorted(model_params.keys(), key=lambda k: (k.startswith('knn'), k))
            for feature_key in all_feature_keys:
                params_dict[f'{feature_key}_params'] = [str(model_params[feature_key])]

            if not params_dict: continue # Skip if no params for this model

            params_df = pd.DataFrame(params_dict)
            params_filename = f"{self.config.dataset_name}_{model_name}_best_params_k_{self.config.k_str}.csv"
            params_path = os.path.join(self.config.output_dir, params_filename)
            params_df.to_csv(params_path, index=False)
        print(f"Best parameters saved to {self.config.output_dir}")

    def print_final_summary(self, eval_results, stacking_results=None): # Add stacking_results
        """Prints a summary of the best performing models to the console, including stacking."""
        print("" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Dataset: {self.config.dataset_name}")

        # --- Detailed Per-Setup Summary ---
        print("Detailed Results per Setup:")
        all_results_for_comparison = [] # Collect tuples (model, feature_set, auc, ks)
        for model_name, model_type_results in eval_results.items():
            print(f"--- {model_name.upper()} ---")
            # Add 'sublime' to sorting logic
            sorted_keys = sorted(model_type_results.keys(), key=lambda k: (k.startswith('knn'), k == 'concat', k == 'sublime', k))
            for feature_key in sorted_keys:
                data = model_type_results[feature_key]
                label = feature_key.replace('_', ' ').title()
                if feature_key.startswith('knn_k'): label = f"KNN (k={feature_key.split('knn_k')[1]})"
                elif feature_key == 'concat': label = "Dataset + SUBLIME" # Simplified label
                elif feature_key == 'dataset': label = "Dataset Features"
                elif feature_key == 'sublime': label = "SUBLIME Features"

                print(f"  {label:<30}: ", end="")
                if 'error' in data:
                    print(f"ERROR ({data['error'][:50]}...)")
                elif 'auc' in data:
                    auc_val = data.get('auc', float('nan'))
                    ks_val = data.get('ks', float('nan'))
                    acc_val = data.get('acc', float('nan'))
                    print(f"Acc={acc_val:.4f}, AUC={auc_val:.4f}, KS={ks_val:.4f}")
                    if not np.isnan(auc_val) and not np.isnan(ks_val):
                        all_results_for_comparison.append((model_name, feature_key, auc_val, ks_val))
                else:
                    print("Metrics N/A")

        # Add Stacking results to detailed summary
        if stacking_results:
             print(f"--- Stacking ---")
             print(f"  {'Stacked Model':<30}: ", end="")
             if 'error' in stacking_results:
                 print(f"ERROR ({stacking_results['error'][:50]}...)")
             elif 'auc' in stacking_results:
                 auc_val = stacking_results.get('auc', float('nan'))
                 ks_val = stacking_results.get('ks', float('nan'))
                 acc_val = stacking_results.get('acc', float('nan'))
                 print(f"Acc={acc_val:.4f}, AUC={auc_val:.4f}, KS={ks_val:.4f}")
                 if not np.isnan(auc_val) and not np.isnan(ks_val):
                     all_results_for_comparison.append(('Stacking', 'Stacking', auc_val, ks_val))
                 base_info = " | ".join([f"{c['model_name']}-{c['feature_set']}" for c in stacking_results.get('base_model_configs', [])])
                 print(f"  {'Base Models Used':<30}: {base_info}")
             else:
                 print("Metrics N/A")

        print("="*80)

        # --- Best Performance Summary (based on collected results) ---
        if not all_results_for_comparison:
            print("No valid results found for performance comparison.")
            print("="*80)
            return

        # Find best overall AUC
        best_auc_res = max(all_results_for_comparison, key=lambda item: item[2])
        # Find best overall KS
        best_ks_res = max(all_results_for_comparison, key=lambda item: item[3])
        # Find best baseline (dataset only) AUC and KS across models
        baseline_results = [res for res in all_results_for_comparison if res[1] == 'dataset']
        best_baseline_auc_res = max(baseline_results, key=lambda item: item[2]) if baseline_results else None
        best_baseline_ks_res = max(baseline_results, key=lambda item: item[3]) if baseline_results else None


        print("Best Overall Performance (Test Set):")
        print(f"  Highest AUC: {best_auc_res[2]:.4f} (Model: {best_auc_res[0].upper()}, Features: {best_auc_res[1]})")
        if best_baseline_auc_res:
             auc_improvement = best_auc_res[2] - best_baseline_auc_res[2]
             print(f"    vs. Best Dataset AUC ({best_baseline_auc_res[2]:.4f}, Model: {best_baseline_auc_res[0].upper()}): {auc_improvement:+.4f}")
        else:
             print("    vs. Best Dataset AUC: N/A (No baseline results)")

        print(f"  Highest KS:  {best_ks_res[3]:.4f} (Model: {best_ks_res[0].upper()}, Features: {best_ks_res[1]})")
        if best_baseline_ks_res:
            ks_improvement = best_ks_res[3] - best_baseline_ks_res[3]
            print(f"    vs. Best Dataset KS ({best_baseline_ks_res[3]:.4f}, Model: {best_baseline_ks_res[0].upper()}):  {ks_improvement:+.4f}")
        else:
            print("    vs. Best Dataset KS: N/A (No baseline results)")

        print("="*80)

# --- Main Execution Logic ---
def main(args):
    """Main function to orchestrate the evaluation pipeline."""
    config = Config(args)

    # 1. Data Management
    data_manager = DataManager(config)
    # ... (loading, filtering, preprocessing) ...
    try:
        data_manager.load_and_sample_data()
        data_manager.filter_target_variable() # Apply target filtering early

        if data_manager.neurolake_df is None or data_manager.neurolake_df.empty:
            print(f"WARNING: Neurolake DataFrame for dataset {config.dataset_name} is empty after loading/filtering. Skipping evaluation.")
            # Exit cleanly for this dataset iteration. The calling shell script will continue.
            sys.exit(0)
        if config.dataset_features_csv and (data_manager.dataset_df is None or data_manager.dataset_df.empty):
            print(f"WARNING: Dataset Features DataFrame for dataset {config.dataset_name} is empty after loading/filtering. Skipping evaluation.")
            sys.exit(0)

        data_manager.preprocess_neurolake()
        # Only preprocess dataset features if needed for evaluation
        if config.dataset_features_csv and config.target_column:
            data_manager.preprocess_dataset_features()
        else:
            print("Skipping dataset feature preprocessing as CSV or target column not provided.")

    except Exception as e:
        print(f"ERROR during Data Management phase: {e}")
        sys.exit(1) # Exit if data handling fails

    # 2. SUBLIME Handling
    sublime_handler = SublimeHandler(config)
    # ... (load model, extract embeddings, save if requested) ...
    try:
        sublime_handler.load_model()
        sublime_handler.build_faiss_index_if_needed()

        if config.generate_new_anchor_adj_for_eval:
            # Ensure data_manager has the necessary attributes populated
            if hasattr(data_manager, 'X_neurolake') and data_manager.X_neurolake is not None and \
               hasattr(data_manager, 'neurolake_df') and data_manager.neurolake_df is not None:
                print("Calling generate_and_set_eval_anchor_graph for the main train/val dataset...")
                sublime_handler.generate_and_set_eval_anchor_graph(
                    data_manager.X_neurolake, 
                    data_manager.neurolake_df   
                )
            else:
                # This case should ideally not be reached if preprocessing was successful and generate_new_anchor_adj_for_eval is true.
                # The script might fail later if adj is still None and use_loaded_adj_for_extraction is true.
                print("WARNING: Skipping generate_and_set_eval_anchor_graph due to missing X_neurolake or neurolake_df from DataManager.")


        train_val_sublime_results = sublime_handler.extract_embeddings(
            data_manager.X_neurolake, dataset_tag=config.dataset_name
        )
        test_sublime_results = None
        if config.using_separate_test and data_manager.X_test_neurolake is not None:
             test_sublime_results = sublime_handler.extract_embeddings(
                 data_manager.X_test_neurolake, dataset_tag=config.test_dataset_name
             )
        # ... (optional embeddings save) ...
        if config.embeddings_output:
             print(f"Saving SUBLIME embeddings...")
             # Train/Val embeddings
             embeddings_df = pd.DataFrame(
                 train_val_sublime_results['embeddings'],
                 columns=[f"sublime_{i}" for i in range(train_val_sublime_results['embeddings'].shape[1])]
             )
             if 'id' in data_manager.neurolake_df.columns: # Use original unsampled/unfiltered df for IDs? Or filtered? Use filtered.
                  # Need to re-attach IDs carefully if filtering/sampling happened
                  # This part might need adjustment depending on exact ID requirement
                  if len(embeddings_df) == len(data_manager.neurolake_df):
                       embeddings_df['id'] = data_manager.neurolake_df['id'].values
                  else:
                       print("Warning: Length mismatch between embeddings and current neurolake dataframe state "
                             f"({len(embeddings_df)} vs {len(data_manager.neurolake_df)}). "
                             "This can happen due to sampling or filtering. Cannot reliably attach original IDs.")

             if sublime_handler.has_classification_head and train_val_sublime_results.get('classification_probs') is not None:
                 embeddings_df['classification_probability'] = train_val_sublime_results['classification_probs']
                 embeddings_df['classification_prediction'] = (train_val_sublime_results['classification_probs'] >= 0.5).astype(int)

             embeddings_df.to_csv(config.embeddings_output, index=False)
             print(f"Train/Val embeddings saved to {config.embeddings_output}")

             # Test embeddings (if applicable)
             if test_sublime_results:
                 test_embeddings_df = pd.DataFrame(
                     test_sublime_results['embeddings'],
                     columns=[f"sublime_{i}" for i in range(test_sublime_results['embeddings'].shape[1])]
                 )
                 # Add ID and classification if available for test set
                 if data_manager.test_neurolake_df is not None and 'id' in data_manager.test_neurolake_df.columns:
                      if len(test_embeddings_df) == len(data_manager.test_neurolake_df):
                          test_embeddings_df['id'] = data_manager.test_neurolake_df['id'].values
                      else:
                          print("Warning: Length mismatch between test embeddings and test neurolake dataframe state "
                                f"({len(test_embeddings_df)} vs {len(data_manager.test_neurolake_df)}). "
                                "Cannot reliably attach original IDs.")

                 if sublime_handler.has_classification_head and test_sublime_results.get('classification_probs') is not None:
                     test_embeddings_df['classification_probability'] = test_sublime_results['classification_probs']
                     test_embeddings_df['classification_prediction'] = (test_sublime_results['classification_probs'] >= 0.5).astype(int)

                 test_output_path = config.embeddings_output.replace('.csv', f'_test_{config.k_str}.csv')
                 test_embeddings_df.to_csv(test_output_path, index=False)
                 print(f"Test embeddings saved to {test_output_path}")
             # Exit if only saving embeddings
             if not (config.dataset_features_csv and config.target_column):
                 print("Embeddings saved. Evaluation skipped as dataset features/target not provided.")
                 return # Exit after saving embeddings

        # --- Added: Exit if only extracting embeddings ---
        if config.extract_embeddings_only:
            print("Embeddings extracted and potentially saved/cached. Skipping model training as requested (--extract-embeddings-only).")
            return

    except Exception as e:
        import logging
        logging.error(f"ERROR during SUBLIME Handling phase: {e}", exc_info=True)
        sys.exit(1)


    # --- Evaluation Setup ---
    if not (config.dataset_features_csv and config.target_column):
         print("Cannot proceed with evaluation: Dataset features CSV or target column missing.")
         return
    if data_manager.X_dataset is None or data_manager.y is None:
         print("Cannot proceed with evaluation: Dataset features or target variable not processed correctly.")
         return

    # --- Evaluate Built-in Classifier ---
    try:
        sublime_handler.evaluate_builtin_classifier(train_val_sublime_results, data_manager.y, "train_val_data")
        if config.using_separate_test and test_sublime_results and data_manager.y_test is not None:
             sublime_handler.evaluate_builtin_classifier(test_sublime_results, data_manager.y_test, "external_test_data")
    except Exception as e:
         print(f"Warning: Error during built-in classifier evaluation: {e}")


    # 3. Split Data for Evaluation Models
    try:
        data_manager.perform_train_val_test_split(train_val_sublime_results, test_sublime_results)
        split_data = data_manager.get_split_data()
    except Exception as e:
        print(f"ERROR during Data Splitting phase: {e}")
        sys.exit(1)
    # Evaluate built-in on the actual test split if *not* using external test
    if not config.using_separate_test:
         # Need to reconstruct test sublime results from the split
         # test_indices = data_manager.neurolake_df.index # Get indices corresponding to the split test set - REMOVED as neurolake_df is deleted after split
         # This requires more complex index tracking during split, let's skip for now
         # or re-extract for the test split? Re-extracting is inefficient.
         print("Skipping built-in classifier evaluation on internal test split (requires more complex index tracking).")


    # 4. Feature Engineering
    try:
        feature_engineer = FeatureEngineer(config)
        feature_sets = feature_engineer.create_all_feature_sets(split_data)
        if not feature_sets:
            print("No feature sets were generated. Aborting evaluation.")
            return
    except Exception as e:
        print(f"ERROR during Feature Engineering phase: {e}")
        sys.exit(1)


    # 5. Run Evaluation (Base Models)
    evaluator = Evaluator(config)
    # Ensure feature sets were created successfully
    if not feature_sets:
         print("No feature sets were generated. Aborting evaluation.")
         return
    try:
        evaluator.run_evaluation(feature_sets, split_data)
    except Exception as e:
         print(f"ERROR during Base Model Evaluation phase: {e}")
         # Continue to stacking if possible, but log the error


    # --- Run Stacking Evaluation ---
    # This runs after the main loop, using the results/params found
    try:
        evaluator.run_stacking_evaluation(feature_sets, split_data)
    except Exception as e:
        print(f"ERROR during Stacking Evaluation phase: {e}")
        # Stacking failed, but might still have base results


    # 6. Reporting
    reporter = Reporter(config)
    eval_results = evaluator.get_results()
    best_params = evaluator.get_best_params()
    stacking_results = evaluator.get_stacking_results() # Get stacking results

    try:
        reporter.generate_roc_plots(eval_results, stacking_results) # Pass stacking results
        # Feature importance plots only make sense for base models, not the meta-model directly
        reporter.generate_feature_importance_plots(eval_results, data_manager, split_data) # Pass data_manager for names
        reporter.save_results_summary(eval_results, stacking_results) # Pass stacking results
        reporter.save_best_params(best_params)
        reporter.print_final_summary(eval_results, stacking_results) # Pass stacking results
    except Exception as e:
        print(f"ERROR during Reporting phase: {e}")


    # 7. Save Dataset Preprocessor
    try:
        data_manager.save_preprocessor()
    except Exception as e:
        print(f"Warning: Failed to save dataset preprocessor: {e}")

    print("Evaluation script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SUBLIME features with dataset features using a structured class-based approach.")
    parser.add_argument('--neurolake-csv', type=str, required=True, help='Path to the neurolake CSV file for SUBLIME')
    parser.add_argument('--dataset-features-csv', type=str, help='Path to the dataset features CSV file (required for evaluation)')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory where the SUBLIME model and neurolake transformer are saved')
    parser.add_argument('--target-column', type=str, help='Name of the target column in the dataset features CSV (required for evaluation)')
    parser.add_argument('--test-csv', type=str, help='Path to a pre-combined CSV file to use as a separate test set.')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials for hyperparameter optimization')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for SUBLIME feature extraction')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory to cache SUBLIME embeddings and classifications')
    parser.add_argument('--embeddings-output', type=str, help='Path to save the extracted SUBLIME embeddings CSV. If specified without evaluation args, only embeddings are saved.')
    parser.add_argument('--k-neighbors', type=int, nargs='+', default=[], help='List of neighbor counts (k) for KNN features. Values <= 0 are ignored.')
    parser.add_argument('--data-fraction', type=float, default=1.0, help='Fraction of the input training/validation datasets to use (0 to 1).')
    parser.add_argument('--use-loaded-adj-for-extraction', action='store_true', help='Use the loaded adjacency matrix for embedding extraction')
    parser.add_argument('--extract-embeddings-only', action='store_true', help='Calculate and cache embeddings, then exit without training downstream models.')
    parser.add_argument('--class-weight-multiplier', type=float, default=1.0, help='Multiplier for the weight of the positive class (class 1) during training. Default is 1 (no extra weight).')

    anchor_group = parser.add_argument_group('New Anchor Graph Generation (for evaluation)')
    anchor_group.add_argument('--generate-new-anchor-adj-for-eval', action='store_true',
                               help='Generate a new anchor adjacency matrix for evaluation (using current eval dataset features) instead of using the one from model_dir.')
    anchor_group.add_argument('--anchor-adj-k-knn', type=int, default=10,
                               help='K for KNN when generating new anchor graph from evaluation dataset features.')
    anchor_group.add_argument('--anchor-adj-use-sparse-format', type=int, default=1,
                               help='Use sparse format for the new anchor graph (1 for sparse, 0 for dense).')
    anchor_group.add_argument('--anchor-adj-id-col-name', type=str, default='id',
                               help="Name of the ID column in neurolake_csv for mapping relationships, e.g., 'id' or 'CPF'.")
    anchor_group.add_argument('--anchor-adj-relationship-csv', type=str, default=None,
                               help='Path to CSV file for relationship data for the new anchor graph.')
    anchor_group.add_argument('--anchor-adj-relationship-cpf1-col', type=str, default='CPF',
                               help='Name of the first ID column in the relationship CSV.')
    anchor_group.add_argument('--anchor-adj-relationship-cpf2-col', type=str, default='CPF_VINCULO',
                               help='Name of the second ID column in the relationship CSV.')
    anchor_group.add_argument('--anchor-adj-relationship-weight', type=float, default=1.0,
                               help='Weight for edges from the relationship data.')
    anchor_group.add_argument('--anchor-adj-knn-threshold-type', type=str, default='none', choices=['none', 'median_k', 'std_dev_k'],
                               help='Type of thresholding for KNN graph generation (none, median_k, std_dev_k).')
    anchor_group.add_argument('--anchor-adj-knn-std-dev-factor', type=float, default=1.0,
                               help='Factor (alpha) for std_dev_k threshold (mean - alpha*std_dev).')

    args = parser.parse_args()

    # --- Argument Validation ---
    is_evaluation_run = args.dataset_features_csv and args.target_column
    is_embedding_run = args.embeddings_output

    if not is_evaluation_run and not is_embedding_run:
        parser.error("Must provide either (--dataset-features-csv AND --target-column) for evaluation OR --embeddings-output for embedding extraction.")

    if is_evaluation_run and not os.path.exists(args.dataset_features_csv):
         parser.error(f"Dataset features file not found: {args.dataset_features_csv}")
    if not os.path.exists(args.neurolake_csv):
         parser.error(f"Neurolake file not found: {args.neurolake_csv}")
    if not os.path.isdir(args.model_dir):
         parser.error(f"Model directory not found: {args.model_dir}")
    if args.test_csv and not os.path.exists(args.test_csv):
         parser.error(f"Separate test file not found: {args.test_csv}")
    main(args) 