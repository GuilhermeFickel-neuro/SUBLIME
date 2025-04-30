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

# Add necessary imports from main and model/graph_learner
from main import Experiment, GCL, FGP_learner, MLP_learner, ATT_learner, GNN_learner, normalize, symmetrize, sparse_mx_to_torch_sparse_tensor, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse
import dgl # Make sure dgl is imported

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

from main import Experiment  # Assuming main.py and Experiment class exist

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
        self.features = None
        self.adj = None
        self.sparse = None
        # self.experiment = None # Remove experiment instance variable, not needed for loading here
        self.faiss_index = None
        self.has_classification_head = False

    def load_model(self):
        """Loads SUBLIME model components by reading config and files."""
        print(f"Loading SUBLIME model from {self.config.model_dir}")
        config_path = os.path.join(self.config.model_dir, 'config.txt')
        model_path = os.path.join(self.config.model_dir, 'model.pt')
        learner_path = os.path.join(self.config.model_dir, 'graph_learner.pt')
        features_path = os.path.join(self.config.model_dir, 'features.pt')
        adj_path = os.path.join(self.config.model_dir, 'adjacency.pt')

        if not all(os.path.exists(p) for p in [config_path, model_path, learner_path, features_path, adj_path]):
            raise FileNotFoundError(f"One or more model files not found in {self.config.model_dir}. Expected config.txt, model.pt, graph_learner.pt, features.pt, adjacency.pt")

        # 1. Read config.txt
        model_config = {}
        with open(config_path, 'r') as f:
            for line in f:
                try:
                    key, value = line.strip().split(': ', 1)
                    # Attempt to infer type (basic types only)
                    if value == 'True': model_config[key] = True
                    elif value == 'False': model_config[key] = False
                    elif value == 'None': model_config[key] = None
                    elif '.' in value:
                        try: model_config[key] = float(value)
                        except ValueError: model_config[key] = value # Keep as string if float fails
                    else:
                        try: model_config[key] = int(value)
                        except ValueError: model_config[key] = value # Keep as string if int fails
                except ValueError:
                    print(f"Warning: Could not parse line in config.txt: {line.strip()}")
                    continue # Skip malformed lines

        # Store essential config items
        self.sparse = model_config.get('sparse', False) # Default to False if missing

        # 2. Load Features and Adjacency Matrix first (needed for some learners)
        self.features = torch.load(features_path, map_location=self.device)

        adj_data = torch.load(adj_path, map_location=self.device)
        if self.sparse:
            if isinstance(adj_data, dict) and 'edges' in adj_data: # Saved DGL graph dictionary
                 num_nodes = adj_data['num_nodes']
                 edges = adj_data['edges']
                 weights = adj_data.get('weights') # May not have weights
                 self.adj = dgl.graph(edges, num_nodes=num_nodes).to(self.device)
                 if weights is not None:
                      self.adj.edata['w'] = weights.to(self.device)
                 # Normalize the loaded DGL graph (as done in original training)
                 # self.adj = normalize(self.adj, 'sym', self.sparse) - REMOVED, assumed already normalized
            elif isinstance(adj_data, torch.Tensor) and adj_data.is_sparse: # Saved torch sparse tensor
                 # This was the format before DGL saving was explicit
                 # Normalize directly
                 self.adj = normalize(adj_data, 'sym', self.sparse)
                 # Convert to DGL graph for model forward pass
                 self.adj = torch_sparse_to_dgl_graph(self.adj)
            else:
                 raise TypeError(f"Unexpected sparse adjacency format loaded from {adj_path}: {type(adj_data)}")
        else: # Dense
            if isinstance(adj_data, torch.Tensor) and not adj_data.is_sparse:
                 self.adj = normalize(adj_data, 'sym', self.sparse) # Normalize dense tensor
            else:
                 raise TypeError(f"Expected dense tensor for adjacency, got {type(adj_data)} from {adj_path}")
        self.adj = self.adj.to(self.device) # Ensure final adj is on device

        # 3. Instantiate Graph Learner
        learner_type = model_config.get('type_learner', 'unknown')
        k = model_config.get('k', 30)
        sim_func = model_config.get('sim_function', 'cosine')
        act_learner = model_config.get('activation_learner', 'relu')

        if learner_type == 'fgp':
            # FGP needs the *initial* graph data, which isn't saved separately by default.
            # Using the loaded 'adj' (which is the final bootstrapped one) might be incorrect.
            # Let's assume for evaluation, we use the final saved 'adj' as the input if needed.
            # Or better: FGP doesn't need initial_graph_data *after* being trained.
            # It learns weights based on the structure it started with.
            # For evaluation/inference, it just needs the feature dimension.
            fgp_i = model_config.get('fgp_elu_alpha', 6) # Get 'i' param if saved
            self.graph_learner = FGP_learner(
                k=k, knn_metric=sim_func, i=fgp_i, sparse=self.sparse,
                initial_graph_data=None # Don't pass initial graph for loading trained state
            )
        elif learner_type == 'mlp':
            self.graph_learner = MLP_learner(
                nlayers=2, # Assume 2 layers? Config doesn't save this explicitly
                isize=model_config['feature_dim'], # Changed 'in_dim' to 'isize'
                k=k, knn_metric=sim_func, i=6, # Use default 'i'?
                sparse=self.sparse, act=act_learner, # Changed 'mlp_act' to 'act'
                knn_threshold_type=model_config.get('knn_threshold_type', 'none'),
                knn_std_dev_factor=model_config.get('knn_std_dev_factor', 1.0),
                chunk_size=model_config.get('graph_learner_chunk_size', 100) # Use saved or default chunk size
                # Removed offload_to_cpu and cleanup_every_n_chunks as they are not in the MLP_learner constructor
            )
        elif learner_type == 'att':
             self.graph_learner = ATT_learner(
                 nlayers=2, # Assume 2?
                 in_dim=model_config['feature_dim'], k=k, knn_metric=sim_func, i=6, # Default 'i'?
                 sparse=self.sparse, mlp_act=act_learner
             )
        elif learner_type == 'gnn':
             # GNN Learner needs the anchor_adj during *training*.
             # For loading, it just needs to be instantiated correctly.
             self.graph_learner = GNN_learner(
                 nlayers=2, # Assume 2?
                 in_dim=model_config['feature_dim'], k=k, knn_metric=sim_func, i=6, # Default 'i'?
                 sparse=self.sparse, mlp_act=act_learner,
                 anchor_adj=None # Don't pass anchor adj for loading trained state
             )
        else:
            raise ValueError(f"Unsupported learner type found in config: {learner_type}")

        self.graph_learner.load_state_dict(torch.load(learner_path, map_location=self.device))
        self.graph_learner = self.graph_learner.to(self.device)
        self.graph_learner.eval() # Set to eval mode

        # 4. Instantiate GCL Model
        gcl_params = {
            'nlayers': model_config.get('nlayers', 2),
            'in_dim': model_config['feature_dim'],
            'hidden_dim': model_config['hidden_dim'],
            'emb_dim': model_config['emb_dim'],
            'proj_dim': model_config['proj_dim'],
            'dropout': model_config.get('dropout', 0.5),
            'dropout_adj': model_config.get('dropout_adj', 0.5), # Note: config saves dropout_adj, GCL takes dropout_adj
            'sparse': self.sparse,
            'use_layer_norm': model_config.get('use_layer_norm', False),
            'use_residual': model_config.get('use_residual', False),
            'use_arcface': model_config.get('use_arcface', False),
            'use_classification_head': model_config.get('use_classification_head', False)
        }
        # Add ArcFace specific params if used
        if gcl_params['use_arcface']:
             gcl_params.update({
                 'num_classes': model_config['num_classes'],
                 'arcface_scale': model_config.get('arcface_scale', 30.0),
                 'arcface_margin': model_config.get('arcface_margin', 0.5),
                 'use_sampled_arcface': model_config.get('use_sampled_arcface', False),
                 'arcface_num_samples': model_config.get('arcface_num_samples', None)
             })
        # Add Classification head specific params if used
        if gcl_params['use_classification_head']:
             gcl_params.update({
                 'classification_dropout': model_config.get('classification_dropout', 0.3),
                 'classification_head_layers': model_config.get('classification_head_layers', 2)
             })
             self.has_classification_head = True # Set flag

        self.model = GCL(**gcl_params)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval() # Set to eval mode

        print("SUBLIME model loaded successfully!")
        if self.has_classification_head: print("Model includes a binary classification head.")

        # self.model, self.graph_learner, self.features, self.adj, self.sparse = self.experiment.load_model(input_dir=self.config.model_dir) # REMOVE THIS LINE
        # print("SUBLIME model loaded successfully!")
        # self.has_classification_head = hasattr(self.model, 'use_classification_head') and self.model.use_classification_head
        # if self.has_classification_head: print("Model includes a binary classification head.")
        # except FileNotFoundError as e:
             # print(f"Error loading model: {e}. Ensure model files exist in {self.config.model_dir}")
             # raise
        # except Exception as e:
             # print(f"An unexpected error occurred loading the model: {e}")
             # raise # Keep raising other unexpected errors

    def build_faiss_index_if_needed(self):
        """Builds FAISS index from loaded features for optimization."""
        if self.features is None:
             print("Cannot build FAISS index: Features not loaded.")
             return

        print("Building FAISS index for faster embedding extraction...")
        try:
            from utils import build_faiss_index # Assuming utils.py exists
            k = getattr(self.graph_learner, 'k', 10) # Get k from graph learner if possible
            self.faiss_index = build_faiss_index(self.features, k=k, use_gpu=torch.cuda.is_available())
            print("FAISS index built successfully!")
        except ImportError:
            print("FAISS or utils.py not found. Skipping FAISS index optimization.")
            self.faiss_index = None
        except Exception as e:
            print(f"Failed to build FAISS index: {str(e)}. Continuing without index optimization.")
            self.faiss_index = None

    def extract_embeddings(self, X, dataset_tag):
        """Extracts embeddings in batches, handles caching."""
        if self.model is None:
            raise RuntimeError("SUBLIME model not loaded. Call load_model() first.")

        print(f"Extracting SUBLIME features for dataset tag: {dataset_tag}...")

        # --- Caching Logic ---
        cache_file_embeddings = None
        cache_file_classifications = None
        can_cache = self.config.cache_dir and dataset_tag
        extraction_mode_tag = "_anchor" if self.config.use_loaded_adj_for_extraction else ""

        if can_cache:
            cache_file_base_embeddings = f"sublime_embeddings_{self.config.model_name_tag}_{dataset_tag}{extraction_mode_tag}.npy"
            cache_file_embeddings = os.path.join(self.config.cache_dir, cache_file_base_embeddings)

            if os.path.exists(cache_file_embeddings):
                print(f"Loading cached embeddings from {cache_file_embeddings}")
                loaded_embeddings = np.load(cache_file_embeddings)
                if len(loaded_embeddings.shape) == 1: # Ensure 2D
                    loaded_embeddings = loaded_embeddings.reshape(-1, 1)

                # Check for cached classification results
                loaded_classification_probs = None
                if self.has_classification_head:
                     cache_file_base_classifications = f"sublime_classifications_{self.config.model_name_tag}_{dataset_tag}{extraction_mode_tag}.npy"
                     cache_file_classifications = os.path.join(self.config.cache_dir, cache_file_base_classifications)
                     if os.path.exists(cache_file_classifications):
                          print(f"Loading cached classification results from {cache_file_classifications}")
                          class_results = np.load(cache_file_classifications)
                          if class_results.shape[1] >= 1:
                               loaded_classification_probs = class_results[:, 0]
                          else:
                               print("Warning: Classification cache file has unexpected shape.")
                     else:
                          print("Classification cache file not found.")

                print(f"Cached embeddings shape: {loaded_embeddings.shape}")
                return {'embeddings': loaded_embeddings, 'classification_probs': loaded_classification_probs}
            else:
                print(f"Cache file not found at {cache_file_embeddings}. Extracting embeddings...")
        else:
            print("Caching disabled or dataset_tag not provided.")
            
        # --- Extraction Logic ---
        num_batches = (len(X) + self.config.batch_size - 1) // self.config.batch_size
        all_embeddings = []
        all_class_probs = [] if self.has_classification_head else None
        
        self.model.eval()
        if self.graph_learner: self.graph_learner.eval()

        with torch.no_grad(): # Disable gradient calculation
            for i in tqdm(range(num_batches), desc=f"Extracting {dataset_tag}", unit="batch"):
                start_idx = i * self.config.batch_size
                end_idx = min((i + 1) * self.config.batch_size, len(X))
                batch_X = X[start_idx:end_idx] # These are the preprocessed neurolake features for the current dataset

                # Convert batch_X to tensor on the correct device once per batch
                batch_X_tensor = torch.FloatTensor(batch_X).to(self.device)

                for j in range(len(batch_X_tensor)):
                    point_tensor = batch_X_tensor[j].unsqueeze(0) # Process one point at a time

                    try:
                        # --- Replicate Experiment.process_new_point logic ---
                        # 1. Find most similar point in loaded features (self.features)
                        # This is needed regardless of the adj used, to know *which* embedding to extract.
                        normalized_features = F.normalize(self.features, p=2, dim=1)
                        normalized_point = F.normalize(point_tensor, p=2, dim=1)
                        similarities = torch.mm(normalized_point, normalized_features.t())
                        replace_idx = torch.argmax(similarities).item()

                        # --- Decide which features and adjacency to use ---
                        if self.config.use_loaded_adj_for_extraction:
                            # Mode 1: Use original features and loaded adjacency
                            features_to_use = self.features
                            adj_to_use = self.adj
                            
                        else:
                            # Mode 2 (Original): Use modified features and learned adjacency
                            # 2. Create modified features (temporarily replace the point)
                            modified_features = self.features.clone()
                            modified_features[replace_idx] = point_tensor
                            
                            # 3. Generate new adjacency using the graph learner
                            new_adj = self.graph_learner(modified_features, faiss_index=self.faiss_index)
                            if not self.sparse:
                                if not isinstance(new_adj, torch.Tensor):
                                    print(f"Warning: Dense graph learner output type is {type(new_adj)}, expected Tensor. Normalization might fail.")
                                else:
                                    new_adj = symmetrize(new_adj)
                                    new_adj = normalize(new_adj, 'sym', self.sparse)
                            # else: Sparse DGL graph handled by GCL forward

                            features_to_use = modified_features
                            adj_to_use = new_adj
                            view_type = 'learner' # Use 'learner' view

                        # 4. Run the model forward pass with the selected features and adjacency
                        if self.has_classification_head:
                            _, embedding, _, classification_output = self.model(
                                features_to_use, adj_to_use, include_features=True
                            )
                            if classification_output is not None:
                                # Classification output corresponds to nodes in features_to_use.
                                # We extract the one corresponding to the most similar original node.
                                classification_prob_tensor = torch.sigmoid(classification_output[replace_idx])
                                classification_prob = classification_prob_tensor.item()
                            else:
                                classification_prob = None
                        else:
                            _, embedding = self.model(features_to_use, adj_to_use)
                            classification_prob = None
                        # --- End Modified Logic ---

                        # Extract the embedding for the *replaced* index
                        # Embedding tensor corresponds to nodes in features_to_use.
                        # We extract the one corresponding to the most similar original node.
                        embedding_tensor = embedding[replace_idx].detach()

                        # Check type before calling .cpu() or .numpy()
                        # embedding_raw = result_dict['embedding_vector'] # Old way
                        # Use the extracted embedding_tensor
                        if isinstance(embedding_tensor, torch.Tensor):
                            embedding_vector = embedding_tensor.cpu().numpy().flatten()
                        else:
                            # This case should ideally not happen if model returns tensor
                            raise TypeError(f"Unexpected type for extracted embedding: {type(embedding_tensor)}")
                        all_embeddings.append(embedding_vector)

                        # Check type for classification probability as well
                        # if self.has_classification_head and 'classification_probability' in result_dict: # Old way
                        if self.has_classification_head and classification_prob is not None:
                             # prob_raw = result_dict['classification_probability'] # Old way
                             # classification_prob is already a float scalar from .item() above
                             if isinstance(classification_prob, (int, float, np.number)):
                                 all_class_probs.append(float(classification_prob))
                             else:
                                 # Should not happen if .item() was used correctly
                                 raise TypeError(f"Unexpected type for classification_probability: {type(classification_prob)}")

                    except Exception as e:
                        print(f"Error processing point {start_idx + j}: {str(e)}")

        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        classification_probs_array = np.array(all_class_probs, dtype=np.float32) if self.has_classification_head else None

        if embeddings_array.shape[0] != X.shape[0]:
             print(f"WARNING: Expected {X.shape[0]} embeddings but got {embeddings_array.shape[0]}!")
        
        # --- Normalization ---
        print("L2 Normalizing SUBLIME embeddings...")
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms[norms == 0] = 1e-10 
        embeddings_array = embeddings_array / norms
        print(f"Normalization complete. Final embeddings shape: {embeddings_array.shape}")

        # --- Save to Cache ---
        if can_cache:
            # Use the correctly tagged filenames defined earlier
            if cache_file_embeddings:
                print(f"Saving embeddings to cache: {cache_file_embeddings}")
                np.save(cache_file_embeddings, embeddings_array)
            if self.has_classification_head and classification_probs_array is not None:
                # Use the correctly tagged classification filename defined earlier
                if cache_file_classifications:
                    # Save probabilities and maybe predictions if needed (currently just probs)
                    # For simplicity, just saving probabilities for now
                    np.save(cache_file_classifications, classification_probs_array.reshape(-1, 1)) # Save as Nx1 array
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
        """Creates 'dataset', 'concat', and 'knn_kX' feature sets for train/val/test."""
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
             # Create empty placeholders if needed downstream? Or raise error? For now, just skip.

        # --- Concatenated Features ---
        if 'sublime_train' in split_data and 'dataset' in feature_sets:
            try:
                # Base concatenation
                concat_train = np.hstack((feature_sets['dataset']['train'], split_data['sublime_train']))
                concat_val = np.hstack((feature_sets['dataset']['val'], split_data['sublime_val']))
                concat_test = np.hstack((feature_sets['dataset']['test'], split_data['sublime_test']))
                concat_train_val = np.hstack((feature_sets['dataset']['train_val'], split_data['sublime_train_val']))

                # Add classification probabilities if available
                if 'cls_probs_train' in split_data:
                     print("Adding SUBLIME classification probability feature.")
                     concat_train = np.hstack((concat_train, split_data['cls_probs_train'].reshape(-1, 1)))
                     concat_val = np.hstack((concat_val, split_data['cls_probs_val'].reshape(-1, 1)))
                     concat_train_val = np.hstack((concat_train_val, split_data['cls_probs_train_val'].reshape(-1, 1)))
                     # Add for test if available
                     if 'cls_probs_test' in split_data:
                          concat_test = np.hstack((concat_test, split_data['cls_probs_test'].reshape(-1, 1)))
                     else:
                          # If test probs missing, impute with 0.5
                          print("Warning: Test classification probabilities missing for concatenation. Filling with 0.5.")
                          # nan_col = np.full((concat_test.shape[0], 1), np.nan) # Old line
                          prob_col = np.full((concat_test.shape[0], 1), 0.5) # New line
                          concat_test = np.hstack((concat_test, prob_col)) # Use prob_col

                feature_sets['concat'] = {
                     'train': concat_train, 'val': concat_val, 'test': concat_test, 'train_val': concat_train_val
                }
                print(f"Concatenated features shapes: Train={concat_train.shape}, Val={concat_val.shape}, Test={concat_test.shape}")

            except ValueError as e:
                 print(f"Error during feature concatenation: {e}. Check shapes:")
                 print(f"  Dataset train: {feature_sets['dataset']['train'].shape}")
                 print(f"  Sublime train: {split_data['sublime_train'].shape}")
                 if 'cls_probs_train' in split_data: print(f"  ClsProb train: {split_data['cls_probs_train'].shape}")
                 # Skip concat if failed
                 if 'concat' in feature_sets: del feature_sets['concat']
            except Exception as e:
                 print(f"Unexpected error during concatenation: {e}")
                 if 'concat' in feature_sets: del feature_sets['concat']
        elif 'sublime_train' not in split_data:
             print("Warning: Sublime embeddings not found in split_data. Skipping 'concat' and 'knn' feature sets.")
        elif 'dataset' not in feature_sets:
             print("Warning: Base 'dataset' features missing. Skipping 'concat' and 'knn' feature sets.")


        # --- KNN Features ---
        if 'concat' in feature_sets and self.config.active_k_values:
            sublime_train = split_data['sublime_train']
            sublime_val = split_data['sublime_val']
            sublime_test = split_data['sublime_test']
            sublime_train_val = split_data['sublime_train_val']
            
            # Embeddings are already normalized by SublimeHandler. No need to re-normalize here.
            # sublime_train = sublime_train / np.linalg.norm(sublime_train, axis=1, keepdims=True)
            # sublime_val = sublime_val / np.linalg.norm(sublime_val, axis=1, keepdims=True)
            # sublime_test = sublime_test / np.linalg.norm(sublime_test, axis=1, keepdims=True)
            # sublime_train_val = sublime_train_val / np.linalg.norm(sublime_train_val, axis=1, keepdims=True)
            
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

        return feature_sets

# --- Evaluation Class ---
class Evaluator:
    """Runs Optuna optimization, trains final models, and evaluates performance."""
    def __init__(self, config):
        self.config = config
        self.results = {} # {model_name: {feature_set: {metric: val, ...}}}
        self.best_params = {} # {model_name: {feature_set: {param: val, ...}}}
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
                'class': LGBMClassifier, 'base_params': {'random_state': 42},
                'trial_params': {'n_estimators': ['int', 50, 500], 'max_depth': ['int', 3, 10], 'learning_rate': ['float', 0.01, 0.3], 'num_leaves': ['int', 20, 100], 'subsample': ['float', 0.6, 1.0], 'colsample_bytree': ['float', 0.3, 1.0], 'reg_alpha': ['float', 0, 10], 'reg_lambda': ['float', 0, 15]}
            }
        }

    def _create_objective(self, model_class, train_features, train_labels, val_features, val_labels, base_params, trial_params):
        """Creates the Optuna objective function for a given model setup."""
        def objective(trial):
            params = base_params.copy()
            for name, suggester_args in trial_params.items():
                 suggester_func = getattr(trial, f"suggest_{suggester_args[0]}")
                 params[name] = suggester_func(name, *suggester_args[1:])

            model = model_class(**params)
            try:
                if isinstance(model, LGBMClassifier):
                    eval_metric = 'auc'
                    # Use optuna callback for pruning
                    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, eval_metric)
                    model.fit(train_features, train_labels, eval_set=[(val_features, val_labels)],
                              eval_metric=eval_metric, callbacks=[pruning_callback], verbose=-1, # Suppress verbose
                              early_stopping_rounds=10) # Added early stopping
                elif isinstance(model, CatBoostClassifier):
                    # Removed early_stopping_rounds=10 as per user request -> Re-adding it
                    model.fit(train_features, train_labels, eval_set=(val_features, val_labels), verbose=False,
                              early_stopping_rounds=10) # Added early stopping
                elif isinstance(model, XGBClassifier):
                    # Manual Pruning Implementation for XGBoost >= 1.6
                    # Combine base and trial parameters
                    params.update(base_params) # Use base_params as default

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
                    # Check if results are available before accessing
                    if 'eval_0' in results and eval_metric_name in results['eval_0']:
                        validation_scores = results['eval_0'][eval_metric_name]

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

        for model_name, config in self.model_configs.items():
            print(f"{'='*50}\n{model_name.upper()} Models\n{'='*50}")
            self.results[model_name] = {}
            self.best_params[model_name] = {}
            model_class = config['class']
            base_params = config['base_params']
            trial_params = config['trial_params']

            for feature_set_name, features in feature_sets.items():
                print(f"--- Evaluating Feature Set: {feature_set_name} ---")
                X_train, X_val, X_test, X_train_val = features['train'], features['val'], features['test'], features['train_val']
                y_train, y_val, y_test, y_train_val = labels['train'], labels['val'], labels['test'], labels['train_val']

                # --- Optuna Optimization ---
                print(f"Optimizing {model_name.upper()} for {feature_set_name} features...")
                try:
                    objective_func = self._create_objective(model_class, X_train, y_train, X_val, y_val, base_params, trial_params)
                    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
                    study.optimize(objective_func, n_trials=self.config.n_trials, n_jobs=1, show_progress_bar=True) # Use n_jobs=1 for safer debugging/logging
                    best_trial_params = study.best_params
                    self.best_params[model_name][feature_set_name] = best_trial_params
                    best_val_auc = study.best_value
                    print(f"Optimization complete. Best Validation AUC: {best_val_auc:.4f}")

                except Exception as e:
                    print(f"ERROR during Optuna optimization for {model_name} - {feature_set_name}: {e}")
                    print("Skipping evaluation for this combination.")
                    self.results[model_name][feature_set_name] = {'error': str(e)} # Store error marker
                    continue # Skip to next feature set

                # --- Final Model Training & Testing ---
                print("Training final model on train+validation data...")
                final_params = {**base_params, **best_trial_params}
                final_model = model_class(**final_params)
                try:
                    # Some models might benefit from fitting on train+val with early stopping against test?
                    # Standard practice is train on train+val without early stopping using best params.
                    if isinstance(final_model, (CatBoostClassifier, LGBMClassifier)):
                         # No eval set needed here, just fit on combined data
                         final_model.fit(X_train_val, y_train_val)
                    else: # XGBoost, others
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
                    self.results[model_name][feature_set_name] = {'error': str(e)} # Store error marker
                    # Don't store best_params if final training failed? Or keep them? Keep for now.

    def get_results(self):
        return self.results

    def get_best_params(self):
        return self.best_params

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

    def generate_roc_plots(self, eval_results):
        """Generates and saves ROC curve plots for each model."""
        print("Generating ROC plots...")
        for model_name, model_type_results in eval_results.items():
            plt.figure(figsize=(10, 8))
            has_plot_data = False
            # Sort keys for consistent plot legend order
            sorted_feature_keys = sorted(model_type_results.keys(), key=lambda k: (k.startswith('knn'), k))

            for feature_key in sorted_feature_keys:
                 data = model_type_results[feature_key]
                 if 'fpr' in data and 'tpr' in data and 'auc' in data and 'ks' in data:
                     label = feature_key.replace('_', ' ').title() # Nicer labels
                     if feature_key.startswith('knn_k'): label = f"KNN (k={feature_key.split('knn_k')[1]})"
                     elif feature_key == 'concat': label = "Dataset + SUBLIME" + (" + ClsProb" if 'cls_probs_train' in self.config.__dict__ else "") # Rough check
                     elif feature_key == 'dataset': label = "Dataset Features"

                     plt.plot(data['fpr'], data['tpr'], lw=2,
                              label=f'{label} (AUC = {data["auc"]:.4f}, KS = {data["ks"]:.4f})')
                     has_plot_data = True
                 else:
                     print(f"Skipping ROC plot for {model_name} - {feature_key} due to missing data.")

            if has_plot_data:
                 plt.plot([0, 1], [0, 1], 'k--', lw=1)
                 plt.xlabel('False Positive Rate')
                 plt.ylabel('True Positive Rate')
                 plt.title(f'ROC Curves on Test Set - {self.config.dataset_name} - {model_name.upper()}')
                 plt.legend(loc='lower right')
                 plt.grid(alpha=0.4)
                 plot_path = os.path.join(self.config.plots_dir, f"{self.config.dataset_name}_{model_name}_test_roc_curves_k_{self.config.k_str}.png")
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

    def save_results_summary(self, eval_results):
        """Compiles and saves the summary metrics to a CSV file."""
        print("Saving results summary...")
        summary_list = []
        for model_name, model_type_results in eval_results.items():
            # Ensure base 'dataset' results exist before processing this model
            base_metrics = model_type_results.get('dataset', {})
            if not base_metrics or 'error' in base_metrics:
                print(f"Warning: Base 'dataset' results missing or invalid for {model_name}. Skipping.")
                continue

            result_row = {
                'dataset': self.config.dataset_name,
                'model': model_name,
                'k_values_used': '_'.join(map(str, self.config.active_k_values)),
                'dataset_acc': base_metrics.get('acc'),
                'dataset_auc': base_metrics.get('auc'),
                'dataset_ks': base_metrics.get('ks'),
            }

            # Add concat metrics
            concat_metrics = model_type_results.get('concat', {})
            if 'error' in concat_metrics: concat_metrics = {} # Treat error as missing
            result_row.update({
                'concat_acc': concat_metrics.get('acc'),
                'concat_auc': concat_metrics.get('auc'),
                'concat_ks': concat_metrics.get('ks'),
                'concat_vs_dataset_improvement_auc': (concat_metrics.get('auc') - base_metrics.get('auc')) if concat_metrics.get('auc') is not None and base_metrics.get('auc') is not None else np.nan,
                # Add other improvement metrics similarly if needed (acc, ks)
            })

            # Add KNN metrics
            for k in self.config.active_k_values:
                knn_key = f'knn_k{k}'
                metric_prefix = f'knn_k{k}'
                knn_metrics = model_type_results.get(knn_key, {})
                if 'error' in knn_metrics: knn_metrics = {} # Treat error as missing

                base_auc = base_metrics.get('auc')
                concat_auc = concat_metrics.get('auc')
                knn_auc = knn_metrics.get('auc')

                result_row.update({
                    f'{metric_prefix}_acc': knn_metrics.get('acc'),
                    f'{metric_prefix}_auc': knn_auc,
                    f'{metric_prefix}_ks': knn_metrics.get('ks'),
                    f'{metric_prefix}_vs_dataset_improvement_auc': (knn_auc - base_auc) if knn_auc is not None and base_auc is not None else np.nan,
                    f'{metric_prefix}_vs_concat_improvement_auc': (knn_auc - concat_auc) if knn_auc is not None and concat_auc is not None else np.nan,
                     # Add other improvement metrics similarly if needed (acc, ks)
                })

            summary_list.append(result_row)

        if not summary_list:
            print("No valid results to save in summary.")
            return

        results_df = pd.DataFrame(summary_list)
        results_filename = f"{self.config.dataset_name}_all_models_test_results_k_{self.config.k_str}.csv"
        results_path = os.path.join(self.config.output_dir, results_filename)
        results_df.to_csv(results_path, index=False)
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

    def print_final_summary(self, eval_results):
        """Prints a summary of the best performing models to the console."""
        print("" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Dataset: {self.config.dataset_name}")

        best_overall_auc_improvement = -float('inf')
        best_model_name_auc = 'N/A'
        best_feature_key_auc = 'N/A'
        best_model_auc = 0
        best_base_auc = 0

        best_overall_ks = -1.0
        best_model_name_ks = 'N/A'
        best_feature_key_ks = 'N/A'
        best_base_ks = -1.0
        best_base_ks_model = 'N/A'

        for model_name, model_type_results in eval_results.items():
            base_metrics = model_type_results.get('dataset', {})
            current_base_auc = base_metrics.get('auc')
            current_base_ks = base_metrics.get('ks')

            if current_base_ks is not None and current_base_ks > best_base_ks:
                 best_base_ks = current_base_ks
                 best_base_ks_model = model_name

            if current_base_auc is None: continue # Cannot compare if base failed

            for feature_key, data in model_type_results.items():
                 current_auc = data.get('auc')
                 current_ks = data.get('ks')

                 # Check AUC improvement
                 if current_auc is not None:
                      improvement = current_auc - current_base_auc
                      if improvement > best_overall_auc_improvement:
                          best_overall_auc_improvement = improvement
                          best_model_name_auc = model_name
                          best_feature_key_auc = feature_key
                          best_model_auc = current_auc
                          best_base_auc = current_base_auc # Store base AUC of the best improved model

                 # Check overall best KS
                 if feature_key != 'dataset' and current_ks is not None:
                      if current_ks > best_overall_ks:
                           best_overall_ks = current_ks
                           best_model_name_ks = model_name
                           best_feature_key_ks = feature_key

        # --- AUC Summary ---
        print("Best Performing Model Setup (Based on AUC Improvement vs. Dataset Features):")
        if best_model_name_auc != 'N/A':
            print(f"  Best Model Type: {best_model_name_auc.upper()}")
            print(f"  Best Feature Set: {best_feature_key_auc}")
            print(f"  Test AUC: {best_model_auc:.4f}")
            print(f"  Improvement vs Dataset AUC ({best_base_auc:.4f}): {best_overall_auc_improvement:.4f}")
            # Add comparison vs concat if KNN was best
            if best_feature_key_auc.startswith('knn_k'):
                 concat_auc = eval_results.get(best_model_name_auc, {}).get('concat', {}).get('auc')
                 if concat_auc is not None:
                     knn_vs_concat_improvement = best_model_auc - concat_auc
                     print(f"  Improvement vs Concat AUC ({concat_auc:.4f}): {knn_vs_concat_improvement:.4f}")
                 else:
                     print("  Improvement vs Concat AUC: N/A (concat results missing)")
        else:
            print("  No improvement found over dataset features for any setup.")

        # --- KS Summary ---
        print("Overall KS Performance Comparison (Across All Models):")
        if best_base_ks >= 0:
            print(f"Best KS using Dataset Features Only:      {best_base_ks:.4f} (Model: {best_base_ks_model.upper()})")
        else:
            print("Best KS using Dataset Features Only:      N/A")

        if best_overall_ks >= 0:
            print(f"Best KS using Enhanced Features:          {best_overall_ks:.4f} (Model: {best_model_name_ks.upper()}, Features: {best_feature_key_ks})")
            if best_base_ks >= 0:
                ks_improvement = best_overall_ks - best_base_ks
                print(f"Improvement in KS vs. Best Dataset Only:  {ks_improvement:.4f}")
            else:
                print("Improvement in KS vs. Best Dataset Only:  N/A")
        else:
            print("Best KS using Enhanced Features:          N/A")
        print("="*80)

        # --- Detailed Per-Setup Summary ---
        print("Detailed Results per Setup:")
        for model_name, model_type_results in eval_results.items():
            print(f"--- {model_name.upper()} ---")
            sorted_keys = sorted(model_type_results.keys(), key=lambda k: (k.startswith('knn'), k))
            for feature_key in sorted_keys:
                data = model_type_results[feature_key]
                label = feature_key.replace('_', ' ').title()
                if feature_key.startswith('knn_k'): label = f"KNN (k={feature_key.split('knn_k')[1]})"
                elif feature_key == 'concat': label = "Dataset + SUBLIME" # Simplified label
                elif feature_key == 'dataset': label = "Dataset Features"
                
                print(f"  {label:<30}: ", end="")
                if 'error' in data:
                    print(f"ERROR ({data['error'][:50]}...)") # Show truncated error
                elif 'auc' in data:
                    print(f"Acc={data.get('acc', float('nan')):.4f}, AUC={data.get('auc', float('nan')):.4f}, KS={data.get('ks', float('nan')):.4f}")
                else:
                    print("Metrics N/A")
        print("="*80)


# --- Main Execution Logic ---
def main(args):
    """Main function to orchestrate the evaluation pipeline."""
    config = Config(args)

    # 1. Data Management
    data_manager = DataManager(config)
    data_manager.load_and_sample_data()
    data_manager.filter_target_variable() # Apply target filtering early

    # Check if dataframes are empty after loading/filtering
    if data_manager.neurolake_df is None or data_manager.neurolake_df.empty:
        print(f"WARNING: Neurolake DataFrame for dataset {config.dataset_name} is empty after loading/filtering. Skipping evaluation for this dataset.")
        # Exit cleanly for this dataset iteration. The calling shell script will continue.
        sys.exit(0)
    # Also check the primary dataset df if it exists (it should mirror neurolake_df's row count after filtering)
    if config.dataset_features_csv and (data_manager.dataset_df is None or data_manager.dataset_df.empty):
        print(f"WARNING: Dataset Features DataFrame for dataset {config.dataset_name} is empty after loading/filtering. Skipping evaluation for this dataset.")
        sys.exit(0)

    data_manager.preprocess_neurolake()
    # Only preprocess dataset features if needed for evaluation
    if config.dataset_features_csv and config.target_column:
        data_manager.preprocess_dataset_features()
    else:
        print("Skipping dataset feature preprocessing as CSV or target column not provided.")


    # 2. SUBLIME Handling
    sublime_handler = SublimeHandler(config)
    sublime_handler.load_model()
    sublime_handler.build_faiss_index_if_needed() # Build index after loading features

    # Extract embeddings for train/val data
    train_val_sublime_results = sublime_handler.extract_embeddings(
        data_manager.X_neurolake, dataset_tag=config.dataset_name
    )
    # Extract for test data if separate test set exists
    test_sublime_results = None
    if config.using_separate_test and data_manager.X_test_neurolake is not None:
         test_sublime_results = sublime_handler.extract_embeddings(
             data_manager.X_test_neurolake, dataset_tag=config.test_dataset_name
         )
    elif not config.using_separate_test and data_manager.X_neurolake is not None:
        # If not using separate test, test embeddings will be handled during split
        pass


    # --- Optional: Save Embeddings ---
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

    # --- Evaluation Setup ---
    if not (config.dataset_features_csv and config.target_column):
         print("Cannot proceed with evaluation: Dataset features CSV or target column missing.")
         return
    if data_manager.X_dataset is None or data_manager.y is None:
         print("Cannot proceed with evaluation: Dataset features or target variable not processed correctly.")
         return

    # --- Evaluate Built-in Classifier (if exists) ---
    # Evaluate on train/val data first
    sublime_handler.evaluate_builtin_classifier(train_val_sublime_results, data_manager.y, "train_val_data")
    # Evaluate on test data if separate test set used
    if config.using_separate_test and test_sublime_results and data_manager.y_test is not None:
         sublime_handler.evaluate_builtin_classifier(test_sublime_results, data_manager.y_test, "external_test_data")
    # Note: If not using external test, built-in classifier on the test split will be evaluated later if needed


    # 3. Split Data for Evaluation Models
    data_manager.perform_train_val_test_split(train_val_sublime_results, test_sublime_results)
    split_data = data_manager.get_split_data()

    # Evaluate built-in on the actual test split if *not* using external test
    if not config.using_separate_test:
         # Need to reconstruct test sublime results from the split
         # test_indices = data_manager.neurolake_df.index # Get indices corresponding to the split test set - REMOVED as neurolake_df is deleted after split
         # This requires more complex index tracking during split, let's skip for now
         # or re-extract for the test split? Re-extracting is inefficient.
         print("Skipping built-in classifier evaluation on internal test split (requires more complex index tracking).")


    # 4. Feature Engineering
    feature_engineer = FeatureEngineer(config)
    feature_sets = feature_engineer.create_all_feature_sets(split_data)

    # 5. Run Evaluation
    evaluator = Evaluator(config)
    # Ensure feature sets were created successfully
    if not feature_sets:
         print("No feature sets were generated. Aborting evaluation.")
         return
    evaluator.run_evaluation(feature_sets, split_data)

    # 6. Reporting
    reporter = Reporter(config)
    eval_results = evaluator.get_results()
    best_params = evaluator.get_best_params()

    reporter.generate_roc_plots(eval_results)
    reporter.generate_feature_importance_plots(eval_results, data_manager, split_data) # Pass data_manager for names
    reporter.save_results_summary(eval_results)
    reporter.save_best_params(best_params)
    reporter.print_final_summary(eval_results)

    # 7. Save Dataset Preprocessor
    data_manager.save_preprocessor()

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