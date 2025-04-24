import argparse
import torch
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import faiss
import time

import matplotlib.pyplot as plt

from main import Experiment

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_mixed_data(df, model_dir):
    """
    Preprocess a dataframe with mixed data types (categorical, numerical, missing values)
    Reused from train_person_data.py
    
    Args:
        df: Pandas DataFrame with raw data
        model_dir: Directory where transformation models are saved
    
    Returns:
        Preprocessed numpy array with all features converted to float and normalized
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    transformer_path = os.path.join(model_dir, 'data_transformer.joblib')
    
    # Separate numerical and categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Load pre-fitted transformer
    print(f"Loading transformer from {transformer_path}")
    preprocessor = joblib.load(transformer_path)
    
    # Transform the data
    processed_data = preprocessor.transform(df)
    
    # Print data statistics
    print(f"Data shape: {df.shape} -> Processed shape: {processed_data.shape}")
    
    return processed_data

def preprocess_dataset_features(df, target_column=None, fit_transform=False):
    """
    Preprocess dataset_features to make them better suited for XGBoost
    
    Args:
        df: Pandas DataFrame with dataset features
        target_column: Name of the target column to exclude from preprocessing
        fit_transform: Whether to fit a new transformer (True) or just transform (False)
        
    Returns:
        tuple: (preprocessed_data, preprocessor)
    """
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Drop target column if provided
    if target_column and target_column in df_copy.columns:
        X = df_copy.drop(columns=[target_column])
        y = df_copy[target_column] if target_column else None
    else:
        X = df_copy
        y = None
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create transformers for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor with ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    if fit_transform:
        # Fit and transform the data
        preprocessed_data = preprocessor.fit_transform(X)
        print(f"Dataset features shape: {X.shape} -> Processed shape: {preprocessed_data.shape}")
        return preprocessed_data, preprocessor, y
    else:
        # Only transform the data
        preprocessed_data = preprocessor.transform(X)
        print(f"Dataset features shape: {X.shape} -> Processed shape: {preprocessed_data.shape}")
        return preprocessed_data, y

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=16, cache_dir=None, model_dir=None, dataset_name=None, faiss_index=None):
    """
    Extract features in batches to avoid memory issues (from test_sublime_features.py)
    
    Args:
        X: Features to extract embeddings for
        model: Trained model
        graph_learner: Trained graph learner
        features: Original features used for training
        adj: Adjacency matrix
        sparse: Whether adjacency is sparse
        experiment: Experiment instance
        batch_size: Batch size for processing
        cache_dir: Directory to cache results. If None, no caching is performed.
        model_dir: Directory where the model is stored, used for cache file naming
        dataset_name: Name of the dataset, used for cache file naming. If None, no caching is performed.
        faiss_index: Optional pre-built FAISS index. If None, will build one from the features.
        
    Returns:
        numpy.ndarray: Extracted features
    """
    # Try to load from cache if cache_dir, model_dir AND dataset_name are provided
    cache_file = None
    if cache_dir is not None and model_dir is not None and dataset_name is not None:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use model_dir as part of the cache filename
        # Extract the base model directory name without the full path
        model_name = os.path.basename(os.path.normpath(model_dir))
        
        # Always include dataset_name in the cache filename
        cache_file = os.path.join(cache_dir, f"sublime_embeddings_{model_name}_{dataset_name}.npy")
        
        # If cache file exists, load and return it
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)
        else:
            print(f"Cache file not found. Extracting embeddings and saving to {cache_file}")
    elif cache_dir is not None and (model_dir is None or dataset_name is None):
        print("Caching disabled: Both model_dir and dataset_name must be provided to enable caching.")
    
    # Build FAISS index from features if not provided (performance optimization)
    if faiss_index is None and hasattr(graph_learner, 'k'):
        try:
            from utils import build_faiss_index
            print("Building FAISS index for graph generation...")
            # Extract parameters from graph_learner if available
            k = getattr(graph_learner, 'k', 10)
            use_gpu = torch.cuda.is_available()
            faiss_index = build_faiss_index(features, k=k, use_gpu=use_gpu)
            print("FAISS index built successfully.")
        except Exception as e:
            print(f"Failed to build FAISS index: {str(e)}. Continuing without index optimization.")
            faiss_index = None
    
    num_batches = (len(X) + batch_size - 1) // batch_size
    all_embeddings = []
    # For storing classification results if available
    all_classifications = []
    has_classification = False
    
    # Set models to evaluation mode
    model.eval()
    if graph_learner is not None:
        graph_learner.eval()
    
    # Process each batch using tqdm for progress tracking
    for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        
        # Process each point individually for better error handling
        batch_embeddings = []
        batch_classifications = []
        
        for j in range(len(batch_X)):
            try:
                point_tensor = torch.FloatTensor(batch_X[j]).to(device)
                
                # Extract embedding using process_new_point method, passing our pre-built FAISS index
                # The method now returns a dictionary with embeddings and possibly classification info
                result_dict = experiment.process_new_point(
                    point_tensor, model, graph_learner, features, adj, sparse, faiss_index=faiss_index
                )
                
                embedding_vector = result_dict['embedding_vector']
                
                # Append the embedding
                batch_embeddings.append(embedding_vector)
                
                # Check if classification data is available (only needed on first point)
                if j == 0 and i == 0 and 'classification_probability' in result_dict:
                    has_classification = True
                
                # Store classification info if available
                if has_classification and 'classification_probability' in result_dict:
                    batch_classifications.append({
                        'probability': result_dict['classification_probability'],
                        'prediction': result_dict['classification_prediction']
                    })
                
            except Exception as e:
                print(f"Error processing point {start_idx + j}: {str(e)}")
                raise e
                
        # Add the batch embeddings to the overall results
        all_embeddings.extend(batch_embeddings)
        if has_classification:
            all_classifications.extend(batch_classifications)
    
    if len(all_embeddings) != len(X):
        print(f"WARNING: Expected {len(X)} embeddings but only got {len(all_embeddings)}!")
    
    embeddings_array = np.array(all_embeddings)
    
    # Save to cache only if cache_file is defined (requires cache_dir, model_dir, and dataset_name)
    if cache_file is not None:
        print(f"Saving embeddings to cache: {cache_file}")
        np.save(cache_file, embeddings_array)
        
        # If classification results are available, save them too
        if has_classification:
            classification_cache_file = os.path.join(cache_dir, f"sublime_classifications_{model_name}_{dataset_name}.npy")
            classification_array = np.array([
                [item['probability'], item['prediction']] 
                for item in all_classifications
            ])
            np.save(classification_cache_file, classification_array)
            print(f"Saving classification results to cache: {classification_cache_file}")
    
    # Optionally, we could return classification info too, but for now just return embeddings
    # to maintain backward compatibility with the rest of the script
    return embeddings_array

# Helper function to calculate KNN features using FAISS
def calculate_knn_features(query_embeddings, index_embeddings, index_labels, k, device, query_is_index=False):
    """
    Calculates KNN features (mean distance, mean label) using FAISS.

    Args:
        query_embeddings (np.ndarray or torch.Tensor): Embeddings to find neighbors for.
        index_embeddings (np.ndarray or torch.Tensor): Embeddings to search within (the index).
        index_labels (np.ndarray or torch.Tensor): Target labels corresponding to index_embeddings.
        k (int): Number of neighbors.
        device: PyTorch device ('cuda' or 'cpu').
        query_is_index (bool): True if query_embeddings are the same as index_embeddings.

    Returns:
        np.ndarray: Array of shape (num_query_points, 2) with [mean_distance, mean_label].
    """
    t_start = time.time()
    print(f"Calculating KNN features for {query_embeddings.shape[0]} query points using {index_embeddings.shape[0]} index points (k={k}, query_is_index={query_is_index}).")

    if query_embeddings.shape[0] == 0 or index_embeddings.shape[0] == 0:
        print("Warning: Empty query or index embeddings provided to calculate_knn_features.")
        return np.zeros((query_embeddings.shape[0], 2)) # Return dummy features

    # Ensure embeddings are numpy arrays on CPU for FAISS
    query_np = query_embeddings.cpu().detach().numpy().astype('float32') if torch.is_tensor(query_embeddings) else np.array(query_embeddings, dtype='float32')
    index_np = index_embeddings.cpu().detach().numpy().astype('float32') if torch.is_tensor(index_embeddings) else np.array(index_embeddings, dtype='float32')

    # Ensure labels are numpy array
    index_labels_np = index_labels.cpu().detach().numpy() if torch.is_tensor(index_labels) else np.array(index_labels)
    index_labels_np = index_labels_np.astype(float) # Ensure labels are float for mean calculation

    if k <= 0:
        print("Warning: k <= 0 requested for KNN features. Returning zeros.")
        return np.zeros((query_np.shape[0], 2))

    if query_is_index and k >= index_np.shape[0]:
         print(f"Warning: k={k} is >= number of index points ({index_np.shape[0]}) for self-query. Setting k to {index_np.shape[0] - 1}.")
         k = max(1, index_np.shape[0] - 1) # Ensure k is at least 1 if index has points
    elif not query_is_index and k > index_np.shape[0]:
         print(f"Warning: k={k} is > number of index points ({index_np.shape[0]}). Setting k to {index_np.shape[0]}.")
         k = index_np.shape[0] # Max k is the number of points in the index

    if k <= 0: # Re-check k after potential adjustment
         print("Warning: k became <= 0 after adjustment. Returning zeros.")
         return np.zeros((query_np.shape[0], 2))

    # Normalize for IP -> Cosine Sim (FAISS uses IP, normalization makes it cosine)
    # It's crucial that the input embeddings (sublime_embeddings) are already normalized before calling this function.
    # Re-normalizing here just in case, but ideally done beforehand.
    faiss.normalize_L2(query_np)
    faiss.normalize_L2(index_np)

    # Build FAISS index (use GPU if available)
    d = index_np.shape[1]
    index = None
    gpu_available = 'cuda' in str(device) and hasattr(faiss, 'StandardGpuResources')
    if gpu_available:
        try:
            res = faiss.StandardGpuResources()
            cpu_index = faiss.IndexFlatIP(d)
            index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            # print("  Using GPU for FAISS index in KNN features.") # Less verbose
        except Exception as e:
            print(f"  GPU FAISS failed in KNN features: {e}. Using CPU.")
            index = faiss.IndexFlatIP(d)
            gpu_available = False # Mark GPU as unavailable
    else:
        # print("  Using CPU for FAISS index in KNN features.") # Less verbose
        index = faiss.IndexFlatIP(d)

    index.add(index_np)

    # Search for neighbors
    search_k = k + 1 if query_is_index else k
    # print(f"  Searching for {search_k} neighbors (query_is_index={query_is_index}).") # Less verbose

    similarities_np, indices_np = index.search(query_np, search_k)

    if query_is_index:
        # Exclude the first neighbor (expected to be self)
        if indices_np.shape[1] > 1:
            # Check if the first result is indeed self (or very close) - optional sanity check
            # self_indices = np.arange(query_np.shape[0])
            # is_self = indices_np[:, 0] == self_indices
            # if not np.all(is_self):
            #     print("  Warning: First neighbor in self-query wasn't always self!")
            similarities_np = similarities_np[:, 1:]
            indices_np = indices_np[:, 1:]
        elif indices_np.shape[1] == 1: # Only found self?
            print("  Warning: Only self found during query_is_index search. KNN features will be based on 0 neighbors.")
            similarities_np = np.zeros((query_np.shape[0], 0))
            indices_np = np.zeros((query_np.shape[0], 0), dtype=int)
        else: # Found 0 neighbors (shouldn't happen with IndexFlat if index has points)
             print("  Warning: Found 0 neighbors during query_is_index search.")
             # Ensure arrays are empty
             similarities_np = np.zeros((query_np.shape[0], 0))
             indices_np = np.zeros((query_np.shape[0], 0), dtype=int)


    # Handle cases where search returns fewer than k neighbors
    actual_k_found = similarities_np.shape[1]
    if actual_k_found < k and actual_k_found > 0:
        print(f"  Warning: Found only {actual_k_found} valid neighbors (k={k} requested).")
    elif actual_k_found == 0:
        print(f"  Warning: Found 0 valid neighbors.")

    # Calculate distances: d = sqrt(2 - 2 * similarity)
    # Clip similarity to avoid numerical issues with sqrt (e.g., similarity slightly > 1 due to precision)
    clipped_similarities = np.clip(similarities_np, -1.0, 1.0)
    distances_np = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * clipped_similarities)) # Ensure non-negative argument for sqrt

    # Aggregate features (handle invalid indices potentially returned by FAISS)
    mean_distances = np.full(query_np.shape[0], np.nan)
    mean_labels = np.full(query_np.shape[0], np.nan)

    # Pre-fetch all required labels to potentially speed up the loop
    valid_indices_flat = indices_np.ravel()
    # Create a mask for valid indices before fetching labels
    valid_indices_mask = valid_indices_flat != -1
    unique_valid_indices = np.unique(valid_indices_flat[valid_indices_mask])
    # Fetch labels only for the unique valid indices found across all queries
    if len(unique_valid_indices) > 0:
        labels_map = {idx: index_labels_np[idx] for idx in unique_valid_indices}
    else:
        labels_map = {}


    for i in range(query_np.shape[0]):
        if actual_k_found == 0: continue # Skip if no neighbors were found for any query point initially

        row_indices = indices_np[i]
        row_distances = distances_np[i]
        row_valid_mask = (row_indices != -1) & (row_indices < len(index_labels_np)) # Ensure index is valid

        if np.any(row_valid_mask):
            valid_row_indices = row_indices[row_valid_mask]
            valid_row_distances = row_distances[row_valid_mask]
            # Lookup labels from the pre-fetched map
            valid_row_labels = np.array([labels_map.get(idx, np.nan) for idx in valid_row_indices])

            # Ensure we only use non-NaN labels for mean calculation (in case map lookup failed somehow)
            valid_label_mask = ~np.isnan(valid_row_labels)
            if np.any(valid_label_mask):
                mean_distances[i] = np.mean(valid_row_distances) # Use distances even if label lookup failed
                mean_labels[i] = np.mean(valid_row_labels[valid_label_mask])
            elif len(valid_row_distances) > 0: # If only distances are valid
                 mean_distances[i] = np.mean(valid_row_distances)
                 # mean_labels[i] remains NaN

        # else: features remain NaN if no valid neighbors found

    # Impute NaNs (e.g., if no valid neighbors found for a point)
    # Use median of calculated means for imputation.
    median_dist = np.nanmedian(mean_distances) if not np.all(np.isnan(mean_distances)) else 0
    median_label = np.nanmedian(mean_labels) if not np.all(np.isnan(mean_labels)) else 0.5 # If all fail, assume neutral label prob

    mean_distances = np.nan_to_num(mean_distances, nan=median_dist)
    mean_labels = np.nan_to_num(mean_labels, nan=median_label)

    # Combine features
    knn_features = np.vstack([mean_distances, mean_labels]).T # Shape: (num_query, 2)

    # Cleanup GPU memory if index was created internally on GPU
    if gpu_available and hasattr(index, 'free'):
         try:
             index.free() # Free GPU resources
         except Exception as e:
             print(f"  Error freeing GPU resources: {e}")
    del index # Ensure index object is released
    import gc
    gc.collect() # Suggest garbage collection

    t_end = time.time()
    print(f"KNN features calculated. Shape: {knn_features.shape}. Time: {t_end - t_start:.4f}s")
    return knn_features

def evaluate_features(dataset_features, sublime_embeddings, y, dataset_name, preprocessor=None, n_trials=50, classification_probs=None, k_neighbors_list=[5], device=device):
    """
    Train XGBoost, CatBoost and LightGBM classifiers on different feature sets and compare performance
    using a train/validation/test split.
    Feature Sets:
    1. Dataset features only
    2. Dataset features + SUBLIME embeddings (+ optional classification probability)
    3. For each k in k_neighbors_list: Dataset features + SUBLIME embeddings + KNN features(k) (+ optional classification probability)

    Args:
        dataset_features: Preprocessed dataset features
        sublime_embeddings: SUBLIME extracted embeddings (MUST be L2-normalized before this function)
        y: Target labels
        dataset_name: Name of the dataset
        preprocessor: The column transformer used to preprocess dataset features (optional)
        n_trials: Number of optimization trials for Optuna
        classification_probs: Model's binary classification probabilities (optional)
        k_neighbors_list (list[int]): List of neighbor counts (k) for KNN feature calculation.
                                     Values <= 0 will be skipped.
        device: PyTorch device ('cuda' or 'cpu').

    Returns:
        dict: Nested results dictionary. Outer keys are model types ('xgboost', 'catboost', 'lightgbm').
              Inner keys are feature set types ('dataset', 'concat', 'knn_k{k}').
              Values contain test metrics, params, models, etc.
    """
    # --- Feature Set Preparation ---
    feature_sets = {'dataset': dataset_features} # Start with base features

    base_concat_features = np.hstack((dataset_features, sublime_embeddings))
    if classification_probs is not None:
        classification_probs_reshaped = classification_probs.reshape(-1, 1)
        concat_features = np.hstack((base_concat_features, classification_probs_reshaped))
        print(f"Including model's classification probability as an additional feature for concat")
    else:
        concat_features = base_concat_features
        classification_probs_reshaped = None

    feature_sets['concat'] = concat_features

    print(f"Dataset features shape: {dataset_features.shape}")
    print(f"SUBLIME features shape: {sublime_embeddings.shape}")
    if classification_probs is not None:
        print(f"Classification probability shape: {classification_probs_reshaped.shape}")
    print(f"Concatenated features shape: {concat_features.shape}")

    # --- Data Splitting ---
    from sklearn.model_selection import train_test_split
    split_inputs = [dataset_features, sublime_embeddings, concat_features, y]
    split_outputs = ['X', 'sublime', 'concat', 'y']
    if classification_probs_reshaped is not None:
        split_inputs.insert(2, classification_probs_reshaped)
        split_outputs.insert(2, 'cls_probs')

    split_results_tv_test = train_test_split(*split_inputs, test_size=0.2, random_state=42, stratify=y)
    train_val_indices = [i * 2 for i in range(len(split_outputs))]
    split_inputs_train_val = [split_results_tv_test[i] for i in train_val_indices]
    stratify_y_train_val = split_results_tv_test[split_outputs.index('y') * 2]
    split_results_train_val = train_test_split(*split_inputs_train_val, test_size=0.25, random_state=42, stratify=stratify_y_train_val)

    split_data = {}
    for i, name in enumerate(split_outputs):
         split_data[f'{name}_train_val'] = split_results_tv_test[i*2]
         split_data[f'{name}_test'] = split_results_tv_test[i*2+1]
         split_data[f'{name}_train'] = split_results_train_val[i*2]
         split_data[f'{name}_val'] = split_results_train_val[i*2+1]

    y_train, y_val, y_test, y_train_val = split_data['y_train'], split_data['y_val'], split_data['y_test'], split_data['y_train_val']
    print(f"Data split: Train ({len(y_train)}), Validation ({len(y_val)}), Test ({len(y_test)})")

    # --- Calculate KNN Features for each k --- 
    knn_features_store = {}
    active_k_values = []
    for k in k_neighbors_list:
        if k <= 0:
             print(f"Skipping k={k} as it is <= 0.")
             continue
        active_k_values.append(k)
        print(f"\nCalculating KNN features for k={k}...")
        knn_key = f'knn_k{k}'
        
        knn_features_train = calculate_knn_features(
            split_data['sublime_train'], split_data['sublime_train'], y_train, k, device, query_is_index=True
        )
        knn_features_val = calculate_knn_features(
            split_data['sublime_val'], split_data['sublime_train'], y_train, k, device, query_is_index=False
        )
        knn_features_test = calculate_knn_features(
            split_data['sublime_test'], split_data['sublime_train_val'], y_train_val, k, device, query_is_index=False
        )

        knn_enhanced_train = np.hstack((split_data['concat_train'], knn_features_train))
        knn_enhanced_val = np.hstack((split_data['concat_val'], knn_features_val))
        knn_enhanced_test = np.hstack((split_data['concat_test'], knn_features_test))
        knn_enhanced_train_val = np.hstack((split_data['concat_train_val'], np.vstack((knn_features_train, knn_features_val))))
        
        knn_features_store[k] = {
            'train': knn_enhanced_train,
            'val': knn_enhanced_val,
            'test': knn_enhanced_test,
            'train_val': knn_enhanced_train_val
        }
        feature_sets[knn_key] = knn_enhanced_train_val
        print(f"KNN (k={k}) features shapes: Train={knn_enhanced_train.shape}, Val={knn_enhanced_val.shape}, Test={knn_enhanced_test.shape}")

    # --- Optuna Objectives Factory --- 
    def create_objective(model_class, train_features, train_labels, val_features, val_labels, base_params={}, trial_params={}):
        def objective(trial):
            params = base_params.copy()
            for name, suggester_args in trial_params.items():
                 suggester_func = getattr(trial, f"suggest_{suggester_args[0]}")
                 params[name] = suggester_func(name, *suggester_args[1:])
            
            model = model_class(**params)
            
            if isinstance(model, LGBMClassifier):
                 eval_metric = 'auc'
                 model.fit(train_features, train_labels, eval_set=[(val_features, val_labels)], 
                           eval_metric=eval_metric, callbacks=[optuna.integration.lightgbm.LightGBMPruningCallback(trial, eval_metric)])
            elif isinstance(model, CatBoostClassifier):
                 model.fit(train_features, train_labels, eval_set=(val_features, val_labels), early_stopping_rounds=10, verbose=False)
            else: # XGBoost or others
                 model.fit(train_features, train_labels)
            
            preds_proba = model.predict_proba(val_features)[:, 1]
            return roc_auc_score(val_labels, preds_proba)
        return objective

    # --- Model Training & Evaluation Loop --- 
    model_configs = {
        'xgboost': {
            'class': XGBClassifier,
            'base_params': {'random_state': 42},
            'trial_params': {
                'n_estimators': ['int', 50, 500],
                'max_depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.6, 1.0],
                'min_child_weight': ['int', 1, 10],
                'gamma': ['float', 0, 5],
                'reg_alpha': ['float', 0, 10],
                'reg_lambda': ['float', 1, 10],
            }
        },
        'catboost': {
            'class': CatBoostClassifier,
            'base_params': {'random_seed': 42, 'verbose': False},
            'trial_params': {
                'iterations': ['int', 50, 500],
                'depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'l2_leaf_reg': ['float', 1, 10],
                'random_strength': ['float', 0, 10],
                'bagging_temperature': ['float', 0, 10],
            }
        },
        'lightgbm': {
            'class': LGBMClassifier,
            'base_params': {'random_state': 42},
            'trial_params': {
                'n_estimators': ['int', 50, 500],
                'max_depth': ['int', 3, 10],
                'learning_rate': ['float', 0.01, 0.3],
                'num_leaves': ['int', 20, 100],
                'subsample': ['float', 0.6, 1.0],
                'colsample_bytree': ['float', 0.6, 1.0],
                'reg_alpha': ['float', 0, 10],
                'reg_lambda': ['float', 0, 10],
            }
        }
    }

    all_results = {}

    for model_name, config in model_configs.items():
        print(f"\n" + "="*50)
        print(f"{model_name.upper()} Models")
        print("="*50)
        model_class = config['class']
        base_params = config['base_params']
        trial_params = config['trial_params']
        model_type_results = {}

        # --- Dataset Features --- 
        print(f"\nOptimizing {model_name.upper()} for dataset features...")
        objective_dataset = create_objective(model_class, split_data['X_train'], y_train, split_data['X_val'], y_val, base_params, trial_params)
        study_dataset = optuna.create_study(direction='maximize')
        study_dataset.optimize(objective_dataset, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params_dataset = study_dataset.best_params
        final_clf_dataset = model_class(**{**base_params, **best_params_dataset})
        final_clf_dataset.fit(split_data['X_train_val'], y_train_val)
        preds_test = final_clf_dataset.predict(split_data['X_test'])
        preds_proba_test = final_clf_dataset.predict_proba(split_data['X_test'])[:, 1]
        acc_test = accuracy_score(y_test, preds_test)
        auc_test = roc_auc_score(y_test, preds_proba_test)
        fpr_test, tpr_test, _ = roc_curve(y_test, preds_proba_test)
        ks_test = max(tpr_test - fpr_test)
        print(f"{model_name.upper()} - Dataset features Test AUC: {auc_test:.4f}, KS: {ks_test:.4f}")
        print(classification_report(y_test, preds_test))
        model_type_results['dataset'] = {
            'acc': acc_test, 'auc': auc_test, 'ks': ks_test,
            'best_params': best_params_dataset,
            'fpr': fpr_test, 'tpr': tpr_test,
            'final_model': final_clf_dataset
        }

        # --- Concatenated Features --- 
        print(f"\nOptimizing {model_name.upper()} for concatenated features...")
        objective_concat = create_objective(model_class, split_data['concat_train'], y_train, split_data['concat_val'], y_val, base_params, trial_params)
        study_concat = optuna.create_study(direction='maximize')
        study_concat.optimize(objective_concat, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params_concat = study_concat.best_params
        final_clf_concat = model_class(**{**base_params, **best_params_concat})
        final_clf_concat.fit(split_data['concat_train_val'], y_train_val)
        preds_test = final_clf_concat.predict(split_data['concat_test'])
        preds_proba_test = final_clf_concat.predict_proba(split_data['concat_test'])[:, 1]
        acc_test = accuracy_score(y_test, preds_test)
        auc_test = roc_auc_score(y_test, preds_proba_test)
        fpr_test, tpr_test, _ = roc_curve(y_test, preds_proba_test)
        ks_test = max(tpr_test - fpr_test)
        print(f"{model_name.upper()} - Concatenated features Test AUC: {auc_test:.4f}, KS: {ks_test:.4f}")
        print(classification_report(y_test, preds_test))
        model_type_results['concat'] = {
            'acc': acc_test, 'auc': auc_test, 'ks': ks_test,
            'best_params': best_params_concat,
            'fpr': fpr_test, 'tpr': tpr_test,
            'final_model': final_clf_concat
        }

        # --- KNN-Enhanced Features (Loop through k) --- 
        for k in active_k_values:
            knn_key = f'knn_k{k}'
            print(f"\nOptimizing {model_name.upper()} for KNN-enhanced features (k={k})...")
            knn_data = knn_features_store[k]
            objective_knn = create_objective(model_class, knn_data['train'], y_train, knn_data['val'], y_val, base_params, trial_params)
            study_knn = optuna.create_study(direction='maximize')
            study_knn.optimize(objective_knn, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
            best_params_knn = study_knn.best_params
            final_clf_knn = model_class(**{**base_params, **best_params_knn})
            final_clf_knn.fit(knn_data['train_val'], y_train_val)
            preds_test = final_clf_knn.predict(knn_data['test'])
            preds_proba_test = final_clf_knn.predict_proba(knn_data['test'])[:, 1]
            acc_test = accuracy_score(y_test, preds_test)
            auc_test = roc_auc_score(y_test, preds_proba_test)
            fpr_test, tpr_test, _ = roc_curve(y_test, preds_proba_test)
            ks_test = max(tpr_test - fpr_test)
            print(f"{model_name.upper()} - KNN-enhanced (k={k}) features Test AUC: {auc_test:.4f}, KS: {ks_test:.4f}")
            print(classification_report(y_test, preds_test))
            model_type_results[knn_key] = {
                'acc': acc_test, 'auc': auc_test, 'ks': ks_test,
                'best_params': best_params_knn,
                'fpr': fpr_test, 'tpr': tpr_test,
                'final_model': final_clf_knn
            }
            
        all_results[model_name] = model_type_results

    # --- Plotting --- 
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ROC Curves
    for model_name, model_type_results in all_results.items():
        plt.figure(figsize=(10, 8))
        for feature_key, data in model_type_results.items():
            label = feature_key
            if feature_key.startswith('knn_k'):
                 k_val = feature_key.split('knn_k')[1]
                 label = f"KNN (k={k_val})"
            elif feature_key == 'dataset':
                 label = "Dataset"
            elif feature_key == 'concat':
                 label = "Concat"
            
            plt.plot(data['fpr'], data['tpr'], 
                     label=f'{label} (AUC = {data["auc"]:.4f}, KS = {data["ks"]:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves on Test Set - {dataset_name} - {model_name.upper()}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_test_roc_curves.png"))
        plt.close()

    # Feature Importance Names Setup
    dataset_feature_names = []
    if preprocessor is not None:
        try:
            dataset_feature_names = list(preprocessor.get_feature_names_out())
        except Exception as e:
            print(f"Couldn't get feature names from preprocessor automatically: {e}")
            dataset_feature_names = [f"DatasetFeature_{i}" for i in range(split_data['X_train'].shape[1])]
    else:
        dataset_feature_names = [f"DatasetFeature_{i}" for i in range(split_data['X_train'].shape[1])]

    sublime_feature_names = [f"SUBLIME_{i}" for i in range(split_data['sublime_train'].shape[1])]
    knn_base_feature_names = ['knn_mean_distance', 'knn_mean_label']

    feature_names_map = {'dataset': list(dataset_feature_names)}
    feature_names_map['concat'] = list(dataset_feature_names) + list(sublime_feature_names)
    if classification_probs is not None:
        feature_names_map['concat'] += ["Model_Classification_Probability"]

    for k in active_k_values:
        knn_key = f'knn_k{k}'
        # Append k to KNN feature names to distinguish them
        specific_knn_names = [f'{name}_k{k}' for name in knn_base_feature_names]
        feature_names_map[knn_key] = feature_names_map['concat'] + specific_knn_names

    # Feature Importance Plots
    for model_name, model_type_results in all_results.items():
        for feature_key, data in model_type_results.items():
            model = data['final_model']
            current_feature_names = feature_names_map.get(feature_key, [])
            
            if model and hasattr(model, 'feature_importances_') and len(current_feature_names) > 0:
                # Ensure feature importances and names align if model provides them differently
                if hasattr(model, 'feature_names_in_') and len(model.feature_names_in_) == len(model.feature_importances_):
                    # Use model's internal names if available and matching
                    importances_dict = dict(zip(model.feature_names_in_, model.feature_importances_))
                    importances = np.array([importances_dict.get(name, 0) for name in current_feature_names])
                else:
                    # Assume order matches if names aren't available or don't align
                    if len(model.feature_importances_) != len(current_feature_names):
                         print(f"Warning: Mismatch between feature importance count ({len(model.feature_importances_)}) and expected feature names ({len(current_feature_names)}) for {model_name} - {feature_key}. Skipping importance plot.")
                         continue
                    importances = model.feature_importances_

                plt.figure(figsize=(12, 8))
                indices = np.argsort(importances)[::-1]
                n_to_plot = min(20, len(importances))
                
                plot_labels = [current_feature_names[i] if i < len(current_feature_names) else f"Unknown_{i}" 
                               for i in indices[:n_to_plot]]
                               
                plt.bar(range(n_to_plot), importances[indices[:n_to_plot]])
                plt.xticks(range(n_to_plot), plot_labels, rotation=45, ha='right')
                
                title_suffix = feature_key
                if feature_key.startswith('knn_k'):
                     k_val = feature_key.split('knn_k')[1]
                     title_suffix = f"KNN-Enhanced (k={k_val})"
                elif feature_key == 'dataset':
                     title_suffix = "Dataset Features"
                elif feature_key == 'concat':
                     title_suffix = "Concatenated Features"
                
                plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - {title_suffix}) - {dataset_name}")
                plt.tight_layout()
                plot_filename = f"{dataset_name}_{model_name}_{feature_key}_feature_importance.png"
                plt.savefig(os.path.join(plots_dir, plot_filename))
                plt.close()
            elif model and hasattr(model, 'feature_importances_') and len(current_feature_names) == 0:
                 print(f"Warning: Could not get feature names for {model_name} - {feature_key}. Skipping importance plot.")

    # --- Save Combined Results --- 
    combined_results_list = []
    for model_name, model_type_results in all_results.items():
        # Ensure base 'dataset' results exist before proceeding
        if 'dataset' not in model_type_results:
            print(f"Warning: Base 'dataset' results missing for model {model_name}. Skipping this model in combined results.")
            continue
        base_metrics = model_type_results['dataset']
        
        # Initialize result row with base metrics
        result_row = {
            'dataset': dataset_name,
            'model': model_name,
            'dataset_acc': base_metrics['acc'],
            'dataset_auc': base_metrics['auc'],
            'dataset_ks': base_metrics['ks'],
        }

        # Add concat metrics if available
        if 'concat' in model_type_results:
             concat_metrics = model_type_results['concat']
             result_row.update({
                 'concat_acc': concat_metrics['acc'],
                 'concat_auc': concat_metrics['auc'],
                 'concat_ks': concat_metrics['ks'],
                 'concat_vs_dataset_improvement_acc': concat_metrics['acc'] - base_metrics['acc'],
                 'concat_vs_dataset_improvement_auc': concat_metrics['auc'] - base_metrics['auc'],
                 'concat_vs_dataset_improvement_ks': concat_metrics['ks'] - base_metrics['ks'],
             })
        else:
             # Fill with NaN or 0 if concat results are missing
             result_row.update({
                 'concat_acc': np.nan, 'concat_auc': np.nan, 'concat_ks': np.nan,
                 'concat_vs_dataset_improvement_acc': np.nan, 
                 'concat_vs_dataset_improvement_auc': np.nan,
                 'concat_vs_dataset_improvement_ks': np.nan,
             })
             concat_metrics = {'acc': np.nan, 'auc': np.nan, 'ks': np.nan} # Placeholder for KNN comparison

        # Add KNN metrics for each k that was active
        for k in active_k_values:
             knn_key = f'knn_k{k}'
             metric_prefix = f'knn_k{k}'
             if knn_key in model_type_results:
                 knn_metrics = model_type_results[knn_key]
                 result_row.update({
                     f'{metric_prefix}_acc': knn_metrics['acc'],
                     f'{metric_prefix}_auc': knn_metrics['auc'],
                     f'{metric_prefix}_ks': knn_metrics['ks'],
                     f'{metric_prefix}_vs_dataset_improvement_acc': knn_metrics['acc'] - base_metrics['acc'],
                     f'{metric_prefix}_vs_dataset_improvement_auc': knn_metrics['auc'] - base_metrics['auc'],
                     f'{metric_prefix}_vs_dataset_improvement_ks': knn_metrics['ks'] - base_metrics['ks'],
                     f'{metric_prefix}_vs_concat_improvement_acc': knn_metrics['acc'] - concat_metrics['acc'],
                     f'{metric_prefix}_vs_concat_improvement_auc': knn_metrics['auc'] - concat_metrics['auc'],
                     f'{metric_prefix}_vs_concat_improvement_ks': knn_metrics['ks'] - concat_metrics['ks'],
                 })
             else:
                 # Fill with NaN if k results are missing
                 result_row.update({
                     f'{metric_prefix}_acc': np.nan, f'{metric_prefix}_auc': np.nan, f'{metric_prefix}_ks': np.nan,
                     f'{metric_prefix}_vs_dataset_improvement_acc': np.nan,
                     f'{metric_prefix}_vs_dataset_improvement_auc': np.nan,
                     f'{metric_prefix}_vs_dataset_improvement_ks': np.nan,
                     f'{metric_prefix}_vs_concat_improvement_acc': np.nan,
                     f'{metric_prefix}_vs_concat_improvement_auc': np.nan,
                     f'{metric_prefix}_vs_concat_improvement_ks': np.nan,
                 })

        combined_results_list.append(result_row)

    results_df = pd.DataFrame(combined_results_list)
    k_str = '_'.join(map(str, sorted(active_k_values))) if active_k_values else 'none'
    results_filename = f"{dataset_name}_all_models_test_results_k_{k_str}.csv"
    results_path = os.path.join(args.output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\nTest results saved to {results_path}")

    # Save best parameters
    for model_name, model_type_results in all_results.items():
        params_dict = {}
        for feature_key, data in model_type_results.items():
             # Ensure 'best_params' exists before trying to access it
             if 'best_params' in data:
                 params_dict[f'{feature_key}_params'] = [str(data['best_params'])]
             else:
                 params_dict[f'{feature_key}_params'] = [None] # Or indicate missing params
        
        params_df = pd.DataFrame(params_dict)
        params_filename = f"{dataset_name}_{model_name}_best_params_k_{k_str}.csv"
        params_path = os.path.join(args.output_dir, params_filename)
        params_df.to_csv(params_path, index=False)
    print(f"Best parameters saved to {os.path.dirname(params_path)}")

    # --- Find Best Performing Setup --- 
    best_overall_auc_improvement = -float('inf')
    best_model_name = 'N/A'
    best_feature_key = 'N/A'
    best_model_data = None
    best_base_auc = 0 

    for model_name, model_type_results in all_results.items():
        if 'dataset' not in model_type_results: continue 
        base_auc = model_type_results['dataset']['auc']
        
        if 'concat' in model_type_results:
             concat_improvement = model_type_results['concat']['auc'] - base_auc
             if concat_improvement > best_overall_auc_improvement:
                 best_overall_auc_improvement = concat_improvement
                 best_model_name = model_name
                 best_feature_key = 'concat'
                 best_model_data = model_type_results['concat']
                 best_base_auc = base_auc
            
        for k in active_k_values:
            knn_key = f'knn_k{k}'
            if knn_key in model_type_results:
                knn_improvement = model_type_results[knn_key]['auc'] - base_auc
                if knn_improvement > best_overall_auc_improvement:
                    best_overall_auc_improvement = knn_improvement
                    best_model_name = model_name
                    best_feature_key = knn_key
                    best_model_data = model_type_results[knn_key]
                    best_base_auc = base_auc

    print("\nBest performing model setup based on TEST SET AUC-ROC improvement vs. Dataset Features:")
    if best_model_name != 'N/A' and best_model_data is not None:
        print(f"  Best Model Type: {best_model_name.upper()}")
        print(f"  Best Feature Set: {best_feature_key}")
        print(f"  Test AUC: {best_model_data['auc']:.4f}")
        print(f"  Improvement vs Dataset AUC ({best_base_auc:.4f}): {best_overall_auc_improvement:.4f}")
        if best_feature_key.startswith('knn_k'):
             # Ensure concat results exist for the best model before comparing
             if 'concat' in all_results.get(best_model_name, {}):
                  concat_auc = all_results[best_model_name]['concat']['auc']
                  # Check if best_model_data['auc'] is valid before subtraction
                  if best_model_data.get('auc') is not None and not np.isnan(best_model_data['auc']) and not np.isnan(concat_auc):
                       knn_vs_concat_improvement = best_model_data['auc'] - concat_auc
                       print(f"  Improvement vs Concat AUC ({concat_auc:.4f}): {knn_vs_concat_improvement:.4f}")
                  else:
                       print("  Improvement vs Concat AUC: N/A (missing data)")
             else:
                  print("  Improvement vs Concat AUC: N/A (concat results missing)")
    else:
        print("  No improvement found over dataset features for any setup.")

    # Return the nested results dictionary
    return all_results

def main(args):
    """Main function to run the evaluation pipeline"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cache directory if caching is enabled
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        print(f"Using cache directory: {args.cache_dir}")
    
    # 1. Load and preprocess neurolake data (for SUBLIME)
    print(f"\nLoading neurolake data from {args.neurolake_csv}")
    neurolake_df = pd.read_csv(args.neurolake_csv, delimiter='\t')
    
    # 2. Process neurolake data with preprocess_mixed_data for SUBLIME
    X_neurolake = preprocess_mixed_data(neurolake_df, model_dir=args.model_dir)
    print(f"Neurolake data processed: {X_neurolake.shape}")
    
    # 3. Load the SUBLIME model
    print(f"\nLoading SUBLIME model from {args.model_dir}")
    experiment = Experiment(device)
    model, graph_learner, features, adj, sparse = experiment.load_model(input_dir=args.model_dir)
    print("Model loaded successfully!")
    
    # Check if the model has a classification head
    has_classification_head = hasattr(model, 'use_classification_head') and model.use_classification_head
    if has_classification_head:
        print("Model includes a binary classification head")
    
    # 4. Build FAISS index once for the entire dataset (optimization)
    print("\nBuilding FAISS index for faster embedding extraction...")
    try:
        from utils import build_faiss_index
        # Extract parameters from graph_learner if available
        k = getattr(graph_learner, 'k', 10)
        faiss_index = build_faiss_index(features, k=k, use_gpu=torch.cuda.is_available())
        print("FAISS index built successfully!")
    except Exception as e:
        print(f"Failed to build FAISS index: {str(e)}. Continuing without index optimization.")
        faiss_index = None
    
    # Dataset name for cache files
    dataset_name = os.path.basename(args.neurolake_csv).split('.')[0]
    
    # 5. Extract SUBLIME features from neurolake data
    print("\nExtracting SUBLIME features...")
    sublime_embeddings = extract_in_batches(
        X_neurolake, model, graph_learner, features, adj, sparse, experiment, 
        batch_size=args.batch_size, cache_dir=args.cache_dir, model_dir=args.model_dir, 
        dataset_name=dataset_name,
        faiss_index=faiss_index
    )
    # Normalize SUBLIME embeddings
    sublime_embeddings = sublime_embeddings / np.linalg.norm(sublime_embeddings, axis=1, keepdims=True)
    
    print(f"Feature extraction complete. Extracted shape: {sublime_embeddings.shape}")
    
    # Check if classification results are available in cache
    classification_results = None
    classification_probs = None
    if has_classification_head and args.cache_dir:
        model_name = os.path.basename(os.path.normpath(args.model_dir))
        classification_cache_file = os.path.join(args.cache_dir, f"sublime_classifications_{model_name}_{dataset_name}.npy")
        if os.path.exists(classification_cache_file):
            classification_results = np.load(classification_cache_file)
            print(f"Loaded classification results from cache: {classification_cache_file}")
            print(f"Classification results shape: {classification_results.shape}")
            # Extract probabilities to use as features
            classification_probs = classification_results[:, 0]
            print(f"Extracted classification probabilities for use as features")
    
    # 5. If embeddings_output is provided, save embeddings to CSV
    if args.embeddings_output:
        # Create embeddings DataFrame with column names
        embeddings_df = pd.DataFrame(
            sublime_embeddings, 
            columns=[f"sublime_embedding_{i}" for i in range(sublime_embeddings.shape[1])]
        )
        
        # Add index from original neurolake data if it exists
        if 'id' in neurolake_df.columns:
            embeddings_df['id'] = neurolake_df['id'].values
        
        # Add classification results if available
        if classification_results is not None:
            embeddings_df['classification_probability'] = classification_results[:, 0]
            embeddings_df['classification_prediction'] = classification_results[:, 1]
            print("Added classification results to embeddings output")
            
        # Save to CSV
        embeddings_df.to_csv(args.embeddings_output, index=False)
        print(f"\nSUBLIME embeddings saved to {args.embeddings_output}")
        
        # If no evaluation parameters provided, exit now
        if not (args.dataset_features_csv and args.target_column):
            return
    
    # 6. Load and process dataset features (only needed for evaluation)
    print(f"\nLoading dataset features from {args.dataset_features_csv}")
    dataset_df = pd.read_csv(args.dataset_features_csv, delimiter='\t')
    
    # Verify that both datasets have the same number of rows (must be aligned)
    if len(neurolake_df) != len(dataset_df):
        raise ValueError(f"Neurolake data ({len(neurolake_df)} rows) and dataset features ({len(dataset_df)} rows) must have the same number of rows!")
    
    # Filter out rows where target values are not 0 or 1
    if args.target_column in dataset_df.columns:
        # Create a mask for valid target values (0 or 1)
        valid_mask = dataset_df[args.target_column].isin([0, 1])
        
        # Count filtered rows
        filtered_count = (~valid_mask).sum()
        if filtered_count > 0:
            print(f"\nRemoving {filtered_count} rows where {args.target_column} is not 0 or 1")
            
            # Apply the same filtering to both dataframes to keep them aligned
            dataset_df = dataset_df[valid_mask].reset_index(drop=True)
            neurolake_df = neurolake_df[valid_mask].reset_index(drop=True)
            
            # Also filter the embeddings and classification results
            sublime_embeddings = sublime_embeddings[valid_mask]
            if classification_results is not None:
                classification_results = classification_results[valid_mask]
                classification_probs = classification_results[:, 0] # Re-assign classification_probs after filtering

            print(f"After filtering: {len(dataset_df)} rows remaining")
    else:
        raise ValueError(f"Target column '{args.target_column}' not found in the dataset features")
    
    # Process dataset features with a new preprocessing pipeline
    X_dataset, preprocessor, y = preprocess_dataset_features(dataset_df, args.target_column, fit_transform=True)
    
    # Compare model's built-in classification with target values if available
    if classification_results is not None:
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        model_predictions = classification_results[:, 1].astype(int)
        model_probabilities = classification_results[:, 0]
        
        # Calculate accuracy and AUC for the model's built-in classifier
        model_accuracy = accuracy_score(y, model_predictions)
        model_auc = roc_auc_score(y, model_probabilities)
        
        print("\n" + "="*80)
        print("BUILT-IN BINARY CLASSIFIER PERFORMANCE")
        print("="*80)
        print(f"Accuracy: {model_accuracy:.4f}")
        print(f"AUC-ROC: {model_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y, model_predictions))
        print("="*80)
        
        # Save results to CSV
        model_results_df = pd.DataFrame({
            'target': y,
            'model_prediction': model_predictions,
            'model_probability': model_probabilities
        })
        model_results_path = os.path.join(args.output_dir, f"{dataset_name}_model_classification_results.csv")
        model_results_df.to_csv(model_results_path, index=False)
        print(f"Model classification results saved to {model_results_path}")
    
    # 7. Evaluate using dataset features and SUBLIME embeddings (and classification probabilities if available)
    print("\nEvaluating features with XGBoost, CatBoost, and LightGBM...")
    dataset_name = os.path.basename(args.dataset_features_csv).split('.')[-1]
    # Pass the list of k values from args
    all_results = evaluate_features(X_dataset, sublime_embeddings, y, dataset_name,
                               preprocessor=preprocessor, n_trials=args.n_trials,
                               classification_probs=classification_probs,
                               k_neighbors_list=args.k_neighbors, # Pass the list using the correct keyword
                               device=device)
    
    # 8. Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")

    # Determine the best performing model setup based on AUC improvement vs. Dataset Features
    best_overall_auc_improvement = -float('inf')
    best_model_name = 'N/A'
    best_feature_key = 'N/A'
    best_model_data = None
    best_base_auc = 0 # Store base AUC of best model for comparison

    for model_name, model_type_results in all_results.items():
        if 'dataset' not in model_type_results: continue # Skip if base model failed
        base_auc = model_type_results['dataset']['auc']
        
        # Check concat improvement
        if 'concat' in model_type_results:
             concat_improvement = model_type_results['concat']['auc'] - base_auc
             if concat_improvement > best_overall_auc_improvement:
                 best_overall_auc_improvement = concat_improvement
                 best_model_name = model_name
                 best_feature_key = 'concat'
                 best_model_data = model_type_results['concat']
                 best_base_auc = base_auc
            
        # Check KNN improvements for each k
        # Ensure active_k_values exists or derive it from keys if needed (safer)
        current_active_k = [int(k.split('knn_k')[1]) for k in model_type_results if k.startswith('knn')]
        for k in current_active_k: # Use k values found in results for this model
            knn_key = f'knn_k{k}'
            if knn_key in model_type_results:
                knn_improvement = model_type_results[knn_key]['auc'] - base_auc
                if knn_improvement > best_overall_auc_improvement:
                    best_overall_auc_improvement = knn_improvement
                    best_model_name = model_name
                    best_feature_key = knn_key
                    best_model_data = model_type_results[knn_key]
                    best_base_auc = base_auc

    print("\nBest Performing Model Setup (vs. Dataset Features):")
    if best_model_name != 'N/A' and best_model_data is not None:
        print(f"  Best Model Type: {best_model_name.upper()}")
        print(f"  Best Feature Set: {best_feature_key}")
        print(f"  Test AUC: {best_model_data['auc']:.4f}")
        print(f"  Improvement vs Dataset AUC ({best_base_auc:.4f}): {best_overall_auc_improvement:.4f}")
        if best_feature_key.startswith('knn_k'):
             # Ensure concat results exist for the best model before comparing
             if 'concat' in all_results.get(best_model_name, {}):
                  concat_auc = all_results[best_model_name]['concat']['auc']
                  # Check if best_model_data['auc'] is valid before subtraction
                  if best_model_data.get('auc') is not None and not np.isnan(best_model_data['auc']) and not np.isnan(concat_auc):
                       knn_vs_concat_improvement = best_model_data['auc'] - concat_auc
                       print(f"  Improvement vs Concat AUC ({concat_auc:.4f}): {knn_vs_concat_improvement:.4f}")
                  else:
                       print("  Improvement vs Concat AUC: N/A (missing data)")
             else:
                  print("  Improvement vs Concat AUC: N/A (concat results missing)")
    else:
        print("  No improvement found over dataset features for any setup.")

    # Print detailed summary for all evaluated setups
    print("\nDetailed Results per Setup:")
    for model_name, model_type_results in all_results.items():
        print("\n" + "-"*60)
        print(f"{model_name.upper()} MODEL")
        print("-"*60)
        # Sort keys: dataset, concat, knn_k5, knn_k10, ...
        for feature_key, data in sorted(model_type_results.items(), key=lambda item: (item[0].startswith('knn'), item[0])): 
             label = feature_key
             if feature_key.startswith('knn_k'):
                  k_val = feature_key.split('knn_k')[1]
                  label = f"KNN-Enhanced (k={k_val})"
             elif feature_key == 'dataset':
                  label = "Dataset Features Only"
             elif feature_key == 'concat':
                  label = "Concatenated Features"
             print(f"  Feature Set: {label}")
             # Check if data is valid before printing metrics
             if data and data.get('auc') is not None:
                 print(f"    Test Accuracy: {data.get('acc', float('nan')):.4f}")
                 print(f"    Test AUC-ROC:  {data.get('auc', float('nan')):.4f}")
                 print(f"    Test KS:       {data.get('ks', float('nan')):.4f}")
             else:
                 print("    Metrics: N/A (Training/Evaluation might have failed)")
             # Optionally print improvement metrics here if desired
    
    print("\n" + "="*80)

    # 9. Save the preprocessor for future use
    preprocessor_path = os.path.join(args.output_dir, 'dataset_features_preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Dataset features preprocessor saved to {preprocessor_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SUBLIME features with dataset features")
    parser.add_argument('--neurolake-csv', type=str, required=True, help='Path to the neurolake CSV file for SUBLIME')
    parser.add_argument('--dataset-features-csv', type=str, required=False, help='Path to the dataset features CSV file')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory where the SUBLIME model is saved')
    parser.add_argument('--target-column', type=str, required=False, help='Name of the target column in the dataset features CSV')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for feature extraction')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory to cache SUBLIME embeddings for faster subsequent runs')
    parser.add_argument('--embeddings-output', type=str, help='Path to save the extracted embeddings CSV. If specified without evaluation parameters, only embeddings will be extracted.')
    parser.add_argument('--k-neighbors', type=int, nargs='+', default=[5, 10, 20], 
                        help='List of neighbor counts (k) for KNN features. Set to 0 in the list to disable KNN for that value, though typically just omit 0.')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.embeddings_output and (not args.dataset_features_csv or not args.target_column):
        parser.error("Either provide --embeddings-output to extract embeddings, or provide --dataset-features-csv and --target-column for evaluation")
    
    main(args) 