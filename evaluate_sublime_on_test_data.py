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
    results = {}

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
    train_val_indices = [i for i, name in enumerate(split_outputs) for _ in range(2)][::2]
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
    # Store KNN features per k to avoid recalculation
    knn_features_store = {}
    active_k_values = []
    for k in k_neighbors_list:
        if k <= 0:
             print(f"Skipping k={k} as it is <= 0.")
             continue
        active_k_values.append(k)
        print(f"\nCalculating KNN features for k={k}...")
        knn_key = f'knn_k{k}'

        # Calculate features for train/val/test splits
        knn_features_train = calculate_knn_features(
            split_data['sublime_train'], split_data['sublime_train'], y_train, k, device, query_is_index=True
        )
        knn_features_val = calculate_knn_features(
            split_data['sublime_val'], split_data['sublime_train'], y_train, k, device, query_is_index=False
        )
        knn_features_test = calculate_knn_features(
            split_data['sublime_test'], split_data['sublime_train_val'], y_train_val, k, device, query_is_index=False
        )

        # Create combined KNN-enhanced feature sets
        knn_enhanced_train = np.hstack((split_data['concat_train'], knn_features_train))
        knn_enhanced_val = np.hstack((split_data['concat_val'], knn_features_val))
        knn_enhanced_test = np.hstack((split_data['concat_test'], knn_features_test))
        knn_enhanced_train_val = np.hstack((split_data['concat_train_val'], np.vstack((knn_features_train, knn_features_val))))

        # Store these features
        knn_features_store[k] = {
            'train': knn_enhanced_train,
            'val': knn_enhanced_val,
            'test': knn_enhanced_test,
            'train_val': knn_enhanced_train_val
        }
        feature_sets[knn_key] = knn_enhanced_train_val # Reference for feature importance names
        print(f"KNN (k={k}) features shapes: Train={knn_enhanced_train.shape}, Val={knn_enhanced_val.shape}, Test={knn_enhanced_test.shape}")

    # --- Optuna Objectives ---
    # (These are generic, they take feature data as input)
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
    results = {}

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
    train_val_indices = [i for i, name in enumerate(split_outputs) for _ in range(2)][::2]
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
    # Store KNN features per k to avoid recalculation
    knn_features_store = {}
    active_k_values = []
    for k in k_neighbors_list:
        if k <= 0:
             print(f"Skipping k={k} as it is <= 0.")
             continue
        active_k_values.append(k)
        print(f"\nCalculating KNN features for k={k}...")
        knn_key = f'knn_k{k}'
        
        # Calculate features for train/val/test splits
        knn_features_train = calculate_knn_features(
            split_data['sublime_train'], split_data['sublime_train'], y_train, k, device, query_is_index=True
        )
        knn_features_val = calculate_knn_features(
            split_data['sublime_val'], split_data['sublime_train'], y_train, k, device, query_is_index=False
        )
        knn_features_test = calculate_knn_features(
            split_data['sublime_test'], split_data['sublime_train_val'], y_train_val, k, device, query_is_index=False
        )

        # Create combined KNN-enhanced feature sets
        knn_enhanced_train = np.hstack((split_data['concat_train'], knn_features_train))
        knn_enhanced_val = np.hstack((split_data['concat_val'], knn_features_val))
        knn_enhanced_test = np.hstack((split_data['concat_test'], knn_features_test))
        knn_enhanced_train_val = np.hstack((split_data['concat_train_val'], np.vstack((knn_features_train, knn_features_val))))
        
        # Store these features
        knn_features_store[k] = {
            'train': knn_enhanced_train,
            'val': knn_enhanced_val,
            'test': knn_enhanced_test,
            'train_val': knn_enhanced_train_val
        }
        feature_sets[knn_key] = knn_enhanced_train_val # Reference for feature importance names
        print(f"KNN (k={k}) features shapes: Train={knn_enhanced_train.shape}, Val={knn_enhanced_val.shape}, Test={knn_enhanced_test.shape}")

    # --- Optuna Objectives --- 
    # (These are generic, they take feature data as input)
    def create_objective(model_class, train_features, train_labels, val_features, val_labels, base_params={}, trial_params={}):
        def objective(trial):
            params = base_params.copy()
            for name, suggester_args in trial_params.items():
                 suggester_func = getattr(trial, f"suggest_{suggester_args[0]}")
                 params[name] = suggester_func(name, *suggester_args[1:])
            
            model = model_class(**params)
            
            if isinstance(model, LGBMClassifier):
                 # Specific handling for LightGBM Pruning Callback
                 eval_metric = 'auc'
                 model.fit(train_features, train_labels, eval_set=[(val_features, val_labels)], 
                           eval_metric=eval_metric, callbacks=[optuna.integration.lightgbm.LightGBMPruningCallback(trial, eval_metric)])
            elif isinstance(model, CatBoostClassifier):
                 # Specific handling for CatBoost Early Stopping
                 model.fit(train_features, train_labels, eval_set=(val_features, val_labels), early_stopping_rounds=10, verbose=False)
            else: # XGBoost or others
                 model.fit(train_features, train_labels)
            
            preds_proba = model.predict_proba(val_features)[:, 1]
            return roc_auc_score(val_labels, preds_proba)
        return objective

    # --- Model Training Loop --- 
    # Iterate through model types (XGB, CatBoost, LGBM)
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
        # Create KNN-enhanced feature sets by appending KNN features to the 'concat' features
        split_data['knn_enhanced_train'] = np.hstack((split_data['concat_train'], knn_features_train))
        split_data['knn_enhanced_val'] = np.hstack((split_data['concat_val'], knn_features_val))
        split_data['knn_enhanced_test'] = np.hstack((split_data['concat_test'], knn_features_test))
        split_data['knn_enhanced_train_val'] = np.hstack((split_data['concat_train_val'], np.vstack((knn_features_train, knn_features_val)))) # Combine train/val KNN features

        feature_sets['knn_enhanced'] = split_data['knn_enhanced_train_val'] # Add full knn set reference
        print(f"KNN-enhanced features shapes: Train={split_data['knn_enhanced_train'].shape}, Val={split_data['knn_enhanced_val'].shape}, Test={split_data['knn_enhanced_test'].shape}")

    else:
         print("\nSkipping KNN feature calculation as k_neighbors <= 0.")
         # Set KNN-related splits to None or handle appropriately later
         split_data['knn_enhanced_train'] = None
         split_data['knn_enhanced_val'] = None
         split_data['knn_enhanced_test'] = None
         split_data['knn_enhanced_train_val'] = None


    # --- Optuna Objectives (Train on Train set, Validate on Validation set) ---

    # Define Optuna objective for dataset features - XGBoost
    def xgb_dataset_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(split_data['X_train'], y_train) # Use split data
        preds_proba = model.predict_proba(split_data['X_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for concatenated features - XGBoost
    def xgb_concat_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(split_data['concat_train'], y_train) # Use split data
        preds_proba = model.predict_proba(split_data['concat_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for KNN-enhanced features - XGBoost
    def xgb_knn_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(split_data['knn_enhanced_train'], y_train) # Use KNN features
        preds_proba = model.predict_proba(split_data['knn_enhanced_val'])[:, 1] # Use KNN features
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for dataset features - CatBoost
    def cat_dataset_objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_seed': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**param)
        model.fit(split_data['X_train'], y_train, eval_set=(split_data['X_val'], y_val), early_stopping_rounds=10, verbose=False) # Use split data
        preds_proba = model.predict_proba(split_data['X_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for concatenated features - CatBoost
    def cat_concat_objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_seed': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**param)
        model.fit(split_data['concat_train'], y_train, eval_set=(split_data['concat_val'], y_val), early_stopping_rounds=10, verbose=False) # Use split data
        preds_proba = model.predict_proba(split_data['concat_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for KNN-enhanced features - CatBoost
    def cat_knn_objective(trial):
        param = {
            'iterations': trial.suggest_int('iterations', 50, 500),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_seed': 42,
            'verbose': False
        }
        model = CatBoostClassifier(**param)
        model.fit(split_data['knn_enhanced_train'], y_train, eval_set=(split_data['knn_enhanced_val'], y_val), early_stopping_rounds=10, verbose=False) # Use KNN features
        preds_proba = model.predict_proba(split_data['knn_enhanced_val'])[:, 1] # Use KNN features
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for dataset features - LightGBM
    def lgbm_dataset_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        model = LGBMClassifier(**param)
        model.fit(split_data['X_train'], y_train, eval_set=[(split_data['X_val'], y_val)], eval_metric='auc', callbacks=[optuna.integration.lightgbm.LightGBMPruningCallback(trial, 'auc')]) # Use split data
        preds_proba = model.predict_proba(split_data['X_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for concatenated features - LightGBM
    def lgbm_concat_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        model = LGBMClassifier(**param)
        model.fit(split_data['concat_train'], y_train, eval_set=[(split_data['concat_val'], y_val)], eval_metric='auc', callbacks=[optuna.integration.lightgbm.LightGBMPruningCallback(trial, 'auc')]) # Use split data
        preds_proba = model.predict_proba(split_data['concat_val'])[:, 1] # Use split data
        return roc_auc_score(y_val, preds_proba)

    # Define Optuna objective for KNN-enhanced features - LightGBM
    def lgbm_knn_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': 42
        }
        model = LGBMClassifier(**param)
        model.fit(split_data['knn_enhanced_train'], y_train, eval_set=[(split_data['knn_enhanced_val'], y_val)], eval_metric='auc', callbacks=[optuna.integration.lightgbm.LightGBMPruningCallback(trial, 'auc')]) # Use KNN features
        preds_proba = model.predict_proba(split_data['knn_enhanced_val'])[:, 1] # Use KNN features
        return roc_auc_score(y_val, preds_proba)


    # Dictionary to store results for all models
    model_results = {}

    # --- XGBoost ---
    print("\n" + "="*50)
    print("XGBoost Models")
    print("="*50)
    
    # Tune hyperparameters for dataset features - XGBoost (using Train/Val sets)
    print("\nOptimizing XGBoost for dataset features (using AUC-ROC on validation set)...")
    study_xgb_dataset = optuna.create_study(direction='maximize')
    # Use clone to avoid modifying original objective function for LightGBM callback compatibility if needed later
    study_xgb_dataset.optimize(xgb_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_xgb_dataset = study_xgb_dataset.best_params

    # Train final XGBoost model on dataset features using Train+Validation set and best params
    print("\nTraining final XGBoost on dataset features (Train+Val set) with best hyperparameters...")
    final_xgb_dataset_clf = XGBClassifier(**best_params_xgb_dataset)
    # Train on the combined training and validation set
    final_xgb_dataset_clf.fit(split_data['X_train_val'], y_train_val) 
    
    # Evaluate final XGBoost model on Test set
    print("\nEvaluating final XGBoost on dataset features using Test set...")
    xgb_dataset_preds_test = final_xgb_dataset_clf.predict(split_data['X_test'])
    xgb_dataset_preds_proba_test = final_xgb_dataset_clf.predict_proba(split_data['X_test'])[:, 1]
    xgb_dataset_acc_test = accuracy_score(y_test, xgb_dataset_preds_test)
    xgb_dataset_auc_test = roc_auc_score(y_test, xgb_dataset_preds_proba_test)
    
    # Calculate KS statistic for XGBoost dataset features model on Test set
    fpr_xgb_dataset_test, tpr_xgb_dataset_test, _ = roc_curve(y_test, xgb_dataset_preds_proba_test)
    ks_xgb_dataset_test = max(tpr_xgb_dataset_test - fpr_xgb_dataset_test)
    
    print(f"XGBoost - Dataset features Test accuracy: {xgb_dataset_acc_test:.4f}")
    print(f"XGBoost - Dataset features Test AUC-ROC: {xgb_dataset_auc_test:.4f}")
    print(f"XGBoost - Dataset features Test KS statistic: {ks_xgb_dataset_test:.4f}")
    print(classification_report(y_test, xgb_dataset_preds_test))

    # Tune hyperparameters for concatenated features - XGBoost (using Train/Val sets)
    print("\nOptimizing XGBoost for concatenated features (using AUC-ROC on validation set)...")
    study_xgb_concat = optuna.create_study(direction='maximize')
    study_xgb_concat.optimize(xgb_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_xgb_concat = study_xgb_concat.best_params

    # Train final XGBoost model on concatenated features using Train+Validation set and best params
    print("\nTraining final XGBoost on concatenated features (Train+Val set) with best hyperparameters...")
    final_xgb_concat_clf = XGBClassifier(**best_params_xgb_concat)
    # Train on the combined training and validation set
    final_xgb_concat_clf.fit(split_data['concat_train_val'], y_train_val)

    # Evaluate final XGBoost model on concatenated features using Test set
    print("\nEvaluating final XGBoost on concatenated features using Test set...")
    xgb_concat_preds_test = final_xgb_concat_clf.predict(split_data['concat_test'])
    xgb_concat_preds_proba_test = final_xgb_concat_clf.predict_proba(split_data['concat_test'])[:, 1]
    xgb_concat_acc_test = accuracy_score(y_test, xgb_concat_preds_test)
    xgb_concat_auc_test = roc_auc_score(y_test, xgb_concat_preds_proba_test)

    # Calculate KS statistic for XGBoost concatenated features model on Test set
    fpr_xgb_concat_test, tpr_xgb_concat_test, _ = roc_curve(y_test, xgb_concat_preds_proba_test)
    ks_xgb_concat_test = max(tpr_xgb_concat_test - fpr_xgb_concat_test)

    print(f"XGBoost - Concatenated features Test accuracy: {xgb_concat_acc_test:.4f}")
    print(f"XGBoost - Concatenated features Test AUC-ROC: {xgb_concat_auc_test:.4f}")
    print(f"XGBoost - Concatenated features Test KS statistic: {ks_xgb_concat_test:.4f}")
    print(classification_report(y_test, xgb_concat_preds_test))

    # --- KNN-Enhanced Features ---
    best_params_xgb_knn = {}
    final_xgb_knn_clf = None
    xgb_knn_acc_test, xgb_knn_auc_test, ks_xgb_knn_test = 0, 0, 0
    fpr_xgb_knn_test, tpr_xgb_knn_test = np.array([0, 1]), np.array([0, 1]) # Defaults for plotting
    if k_neighbors > 0 and split_data['knn_enhanced_train'] is not None:
        print("\nOptimizing XGBoost for KNN-enhanced features...")
        study_xgb_knn = optuna.create_study(direction='maximize')
        study_xgb_knn.optimize(xgb_knn_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params_xgb_knn = study_xgb_knn.best_params
        final_xgb_knn_clf = XGBClassifier(**best_params_xgb_knn)
        final_xgb_knn_clf.fit(split_data['knn_enhanced_train_val'], y_train_val) # Train on train+val
        xgb_knn_preds_test = final_xgb_knn_clf.predict(split_data['knn_enhanced_test'])
        xgb_knn_preds_proba_test = final_xgb_knn_clf.predict_proba(split_data['knn_enhanced_test'])[:, 1]
        xgb_knn_acc_test = accuracy_score(y_test, xgb_knn_preds_test)
        xgb_knn_auc_test = roc_auc_score(y_test, xgb_knn_preds_proba_test)
        fpr_xgb_knn_test, tpr_xgb_knn_test, _ = roc_curve(y_test, xgb_knn_preds_proba_test)
        ks_xgb_knn_test = max(tpr_xgb_knn_test - fpr_xgb_knn_test)
        print(f"XGBoost - KNN-enhanced features Test AUC: {xgb_knn_auc_test:.4f}, KS: {ks_xgb_knn_test:.4f}")
        print(classification_report(y_test, xgb_knn_preds_test))

    # Store XGBoost results
    model_results['xgboost'] = {
        'dataset_acc': xgb_dataset_acc_test, 'dataset_auc': xgb_dataset_auc_test, 'dataset_ks': ks_xgb_dataset_test,
        'concat_acc': xgb_concat_acc_test, 'concat_auc': xgb_concat_auc_test, 'concat_ks': ks_xgb_concat_test,
        'knn_acc': xgb_knn_acc_test, 'knn_auc': xgb_knn_auc_test, 'knn_ks': ks_xgb_knn_test,
        'improvement_concat_vs_dataset_acc': xgb_concat_acc_test - xgb_dataset_acc_test,
        'improvement_concat_vs_dataset_auc': xgb_concat_auc_test - xgb_dataset_auc_test,
        'improvement_concat_vs_dataset_ks': ks_xgb_concat_test - ks_xgb_dataset_test,
        'best_params_dataset': best_params_xgb_dataset,
        'best_params_concat': best_params_xgb_concat,
        'best_params_knn': best_params_xgb_knn,
        'fpr_dataset': fpr_xgb_dataset_test, 'tpr_dataset': tpr_xgb_dataset_test,
        'fpr_concat': fpr_xgb_concat_test, 'tpr_concat': tpr_xgb_concat_test,
        'fpr_knn': fpr_xgb_knn_test, 'tpr_knn': tpr_xgb_knn_test,
        'final_model_dataset': final_xgb_dataset_clf,
        'final_model_concat': final_xgb_concat_clf,
        'final_model_knn': final_xgb_knn_clf
    }

    # --- CatBoost ---
    print("\n" + "="*50)
    print("CatBoost Models")
    print("="*50)
    
    # Tune hyperparameters for dataset features - CatBoost (using Train/Val sets)
    print("\nOptimizing CatBoost for dataset features (using AUC-ROC on validation set)...")
    study_cat_dataset = optuna.create_study(direction='maximize')
    study_cat_dataset.optimize(cat_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_cat_dataset = study_cat_dataset.best_params

    # Train final CatBoost model on dataset features using Train+Validation set and best params
    print("\nTraining final CatBoost on dataset features (Train+Val set) with best hyperparameters...")
    final_cat_dataset_clf = CatBoostClassifier(**best_params_cat_dataset, verbose=False)
    # Train on the combined training and validation set
    final_cat_dataset_clf.fit(split_data['X_train_val'], y_train_val)

    # Evaluate final CatBoost model on Test set
    print("\nEvaluating final CatBoost on dataset features using Test set...")
    cat_dataset_preds_test = final_cat_dataset_clf.predict(split_data['X_test'])
    cat_dataset_preds_proba_test = final_cat_dataset_clf.predict_proba(split_data['X_test'])[:, 1]
    cat_dataset_acc_test = accuracy_score(y_test, cat_dataset_preds_test)
    cat_dataset_auc_test = roc_auc_score(y_test, cat_dataset_preds_proba_test)
    
    # Calculate KS statistic for CatBoost dataset features model on Test set
    fpr_cat_dataset_test, tpr_cat_dataset_test, _ = roc_curve(y_test, cat_dataset_preds_proba_test)
    ks_cat_dataset_test = max(tpr_cat_dataset_test - fpr_cat_dataset_test)

    print(f"CatBoost - Dataset features Test accuracy: {cat_dataset_acc_test:.4f}")
    print(f"CatBoost - Dataset features Test AUC-ROC: {cat_dataset_auc_test:.4f}")
    print(f"CatBoost - Dataset features Test KS statistic: {ks_cat_dataset_test:.4f}")
    print(classification_report(y_test, cat_dataset_preds_test))

    # Tune hyperparameters for concatenated features - CatBoost (using Train/Val sets)
    print("\nOptimizing CatBoost for concatenated features (using AUC-ROC on validation set)...")
    study_cat_concat = optuna.create_study(direction='maximize')
    study_cat_concat.optimize(cat_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_cat_concat = study_cat_concat.best_params

    # Train final CatBoost model on concatenated features using Train+Validation set and best params
    print("\nTraining final CatBoost on concatenated features (Train+Val set) with best hyperparameters...")
    final_cat_concat_clf = CatBoostClassifier(**best_params_cat_concat, verbose=False)
    # Train on the combined training and validation set
    final_cat_concat_clf.fit(split_data['concat_train_val'], y_train_val)

    # Evaluate final CatBoost model on concatenated features using Test set
    print("\nEvaluating final CatBoost on concatenated features using Test set...")
    cat_concat_preds_test = final_cat_concat_clf.predict(split_data['concat_test'])
    cat_concat_preds_proba_test = final_cat_concat_clf.predict_proba(split_data['concat_test'])[:, 1]
    cat_concat_acc_test = accuracy_score(y_test, cat_concat_preds_test)
    cat_concat_auc_test = roc_auc_score(y_test, cat_concat_preds_proba_test)

    # Calculate KS statistic for CatBoost concatenated features model on Test set
    fpr_cat_concat_test, tpr_cat_concat_test, _ = roc_curve(y_test, cat_concat_preds_proba_test)
    ks_cat_concat_test = max(tpr_cat_concat_test - fpr_cat_concat_test)

    print(f"CatBoost - Concatenated features Test accuracy: {cat_concat_acc_test:.4f}")
    print(f"CatBoost - Concatenated features Test AUC-ROC: {cat_concat_auc_test:.4f}")
    print(f"CatBoost - Concatenated features Test KS statistic: {ks_cat_concat_test:.4f}")
    print(classification_report(y_test, cat_concat_preds_test))

    # --- KNN-Enhanced Features ---
    best_params_cat_knn = {}
    final_cat_knn_clf = None
    cat_knn_acc_test, cat_knn_auc_test, ks_cat_knn_test = 0, 0, 0
    fpr_cat_knn_test, tpr_cat_knn_test = np.array([0, 1]), np.array([0, 1]) # Defaults for plotting
    if k_neighbors > 0 and split_data['knn_enhanced_train'] is not None:
        print("\nOptimizing CatBoost for KNN-enhanced features...")
        study_cat_knn = optuna.create_study(direction='maximize')
        study_cat_knn.optimize(cat_knn_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params_cat_knn = study_cat_knn.best_params
        final_cat_knn_clf = CatBoostClassifier(**best_params_cat_knn, verbose=False)
        final_cat_knn_clf.fit(split_data['knn_enhanced_train_val'], y_train_val) # Train on train+val
        cat_knn_preds_test = final_cat_knn_clf.predict(split_data['knn_enhanced_test'])
        cat_knn_preds_proba_test = final_cat_knn_clf.predict_proba(split_data['knn_enhanced_test'])[:, 1]
        cat_knn_acc_test = accuracy_score(y_test, cat_knn_preds_test)
        cat_knn_auc_test = roc_auc_score(y_test, cat_knn_preds_proba_test)
        fpr_cat_knn_test, tpr_cat_knn_test, _ = roc_curve(y_test, cat_knn_preds_proba_test)
        ks_cat_knn_test = max(tpr_cat_knn_test - fpr_cat_knn_test)
        print(f"CatBoost - KNN-enhanced features Test AUC: {cat_knn_auc_test:.4f}, KS: {ks_cat_knn_test:.4f}")
        print(classification_report(y_test, cat_knn_preds_test))

    # Store CatBoost results
    model_results['catboost'] = {
        'dataset_acc': cat_dataset_acc_test, 'dataset_auc': cat_dataset_auc_test, 'dataset_ks': ks_cat_dataset_test,
        'concat_acc': cat_concat_acc_test, 'concat_auc': cat_concat_auc_test, 'concat_ks': ks_cat_concat_test,
        'knn_acc': cat_knn_acc_test, 'knn_auc': cat_knn_auc_test, 'knn_ks': ks_cat_knn_test,
        'improvement_concat_vs_dataset_acc': cat_concat_acc_test - cat_dataset_acc_test,
        'improvement_concat_vs_dataset_auc': cat_concat_auc_test - cat_dataset_auc_test,
        'improvement_concat_vs_dataset_ks': ks_cat_concat_test - ks_cat_dataset_test,
        'improvement_knn_vs_dataset_acc': cat_knn_acc_test - cat_dataset_acc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_dataset_auc': cat_knn_auc_test - cat_dataset_auc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_dataset_ks': ks_cat_knn_test - ks_cat_dataset_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_acc': cat_knn_acc_test - cat_concat_acc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_auc': cat_knn_auc_test - cat_concat_auc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_ks': ks_cat_knn_test - ks_cat_concat_test if k_neighbors > 0 else 0,
        'best_params_dataset': best_params_cat_dataset,
        'best_params_concat': best_params_cat_concat,
        'best_params_knn': best_params_cat_knn,
        'fpr_dataset': fpr_cat_dataset_test, 'tpr_dataset': tpr_cat_dataset_test,
        'fpr_concat': fpr_cat_concat_test, 'tpr_concat': tpr_cat_concat_test,
        'fpr_knn': fpr_cat_knn_test, 'tpr_knn': tpr_cat_knn_test,
        'final_model_dataset': final_cat_dataset_clf,
        'final_model_concat': final_cat_concat_clf,
        'final_model_knn': final_cat_knn_clf
    }


    # --- LightGBM ---
    print("\n" + "="*50)
    print("LightGBM Models")
    print("="*50)

    # Tune hyperparameters for dataset features - LightGBM (using Train/Val sets)
    print("\nOptimizing LightGBM for dataset features (using AUC-ROC on validation set)...")
    study_lgbm_dataset = optuna.create_study(direction='maximize')
    study_lgbm_dataset.optimize(lgbm_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_lgbm_dataset = study_lgbm_dataset.best_params

    # Train final LightGBM model on dataset features using Train+Validation set and best params
    print("\nTraining final LightGBM on dataset features (Train+Val set) with best hyperparameters...")
    final_lgbm_dataset_clf = LGBMClassifier(**best_params_lgbm_dataset)
    # Train on the combined training and validation set
    final_lgbm_dataset_clf.fit(split_data['X_train_val'], y_train_val)

    # Evaluate final LightGBM model on Test set
    print("\nEvaluating final LightGBM on dataset features using Test set...")
    lgbm_dataset_preds_test = final_lgbm_dataset_clf.predict(split_data['X_test'])
    lgbm_dataset_preds_proba_test = final_lgbm_dataset_clf.predict_proba(split_data['X_test'])[:, 1]
    lgbm_dataset_acc_test = accuracy_score(y_test, lgbm_dataset_preds_test)
    lgbm_dataset_auc_test = roc_auc_score(y_test, lgbm_dataset_preds_proba_test)
    
    # Calculate KS statistic for LightGBM dataset features model on Test set
    fpr_lgbm_dataset_test, tpr_lgbm_dataset_test, _ = roc_curve(y_test, lgbm_dataset_preds_proba_test)
    ks_lgbm_dataset_test = max(tpr_lgbm_dataset_test - fpr_lgbm_dataset_test)

    print(f"LightGBM - Dataset features Test accuracy: {lgbm_dataset_acc_test:.4f}")
    print(f"LightGBM - Dataset features Test AUC-ROC: {lgbm_dataset_auc_test:.4f}")
    print(f"LightGBM - Dataset features Test KS statistic: {ks_lgbm_dataset_test:.4f}")
    print(classification_report(y_test, lgbm_dataset_preds_test))

    # Tune hyperparameters for concatenated features - LightGBM (using Train/Val sets)
    print("\nOptimizing LightGBM for concatenated features (using AUC-ROC on validation set)...")
    study_lgbm_concat = optuna.create_study(direction='maximize')
    study_lgbm_concat.optimize(lgbm_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    best_params_lgbm_concat = study_lgbm_concat.best_params

    # Train final LightGBM model on concatenated features using Train+Validation set and best params
    print("\nTraining final LightGBM on concatenated features (Train+Val set) with best hyperparameters...")
    final_lgbm_concat_clf = LGBMClassifier(**best_params_lgbm_concat)
    # Train on the combined training and validation set
    final_lgbm_concat_clf.fit(split_data['concat_train_val'], y_train_val)

    # Evaluate final LightGBM model on concatenated features using Test set
    print("\nEvaluating final LightGBM on concatenated features using Test set...")
    lgbm_concat_preds_test = final_lgbm_concat_clf.predict(split_data['concat_test'])
    lgbm_concat_preds_proba_test = final_lgbm_concat_clf.predict_proba(split_data['concat_test'])[:, 1]
    lgbm_concat_acc_test = accuracy_score(y_test, lgbm_concat_preds_test)
    lgbm_concat_auc_test = roc_auc_score(y_test, lgbm_concat_preds_proba_test)

    # Calculate KS statistic for LightGBM concatenated features model on Test set
    fpr_lgbm_concat_test, tpr_lgbm_concat_test, _ = roc_curve(y_test, lgbm_concat_preds_proba_test)
    ks_lgbm_concat_test = max(tpr_lgbm_concat_test - fpr_lgbm_concat_test)

    print(f"LightGBM - Concatenated features Test accuracy: {lgbm_concat_acc_test:.4f}")
    print(f"LightGBM - Concatenated features Test AUC-ROC: {lgbm_concat_auc_test:.4f}")
    print(f"LightGBM - Concatenated features Test KS statistic: {ks_lgbm_concat_test:.4f}")
    print(classification_report(y_test, lgbm_concat_preds_test))

    # --- KNN-Enhanced Features ---
    best_params_lgbm_knn = {}
    final_lgbm_knn_clf = None
    lgbm_knn_acc_test, lgbm_knn_auc_test, ks_lgbm_knn_test = 0, 0, 0
    fpr_lgbm_knn_test, tpr_lgbm_knn_test = np.array([0, 1]), np.array([0, 1]) # Defaults for plotting
    if k_neighbors > 0 and split_data['knn_enhanced_train'] is not None:
        print("\nOptimizing LightGBM for KNN-enhanced features...")
        study_lgbm_knn = optuna.create_study(direction='maximize')
        study_lgbm_knn.optimize(lgbm_knn_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
        best_params_lgbm_knn = study_lgbm_knn.best_params
        final_lgbm_knn_clf = LGBMClassifier(**best_params_lgbm_knn)
        final_lgbm_knn_clf.fit(split_data['knn_enhanced_train_val'], y_train_val) # Train on train+val
        lgbm_knn_preds_test = final_lgbm_knn_clf.predict(split_data['knn_enhanced_test'])
        lgbm_knn_preds_proba_test = final_lgbm_knn_clf.predict_proba(split_data['knn_enhanced_test'])[:, 1]
        lgbm_knn_acc_test = accuracy_score(y_test, lgbm_knn_preds_test)
        lgbm_knn_auc_test = roc_auc_score(y_test, lgbm_knn_preds_proba_test)
        fpr_lgbm_knn_test, tpr_lgbm_knn_test, _ = roc_curve(y_test, lgbm_knn_preds_proba_test)
        ks_lgbm_knn_test = max(tpr_lgbm_knn_test - fpr_lgbm_knn_test)
        print(f"LightGBM - KNN-enhanced features Test AUC: {lgbm_knn_auc_test:.4f}, KS: {ks_lgbm_knn_test:.4f}")
        print(classification_report(y_test, lgbm_knn_preds_test))

    # Store LightGBM results
    model_results['lightgbm'] = {
        'dataset_acc': lgbm_dataset_acc_test, 'dataset_auc': lgbm_dataset_auc_test, 'dataset_ks': ks_lgbm_dataset_test,
        'concat_acc': lgbm_concat_acc_test, 'concat_auc': lgbm_concat_auc_test, 'concat_ks': ks_lgbm_concat_test,
        'knn_acc': lgbm_knn_acc_test, 'knn_auc': lgbm_knn_auc_test, 'knn_ks': ks_lgbm_knn_test,
        'improvement_concat_vs_dataset_acc': lgbm_concat_acc_test - lgbm_dataset_acc_test,
        'improvement_concat_vs_dataset_auc': lgbm_concat_auc_test - lgbm_dataset_auc_test,
        'improvement_concat_vs_dataset_ks': ks_lgbm_concat_test - ks_lgbm_dataset_test,
        'improvement_knn_vs_dataset_acc': lgbm_knn_acc_test - lgbm_dataset_acc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_dataset_auc': lgbm_knn_auc_test - lgbm_dataset_auc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_dataset_ks': ks_lgbm_knn_test - ks_lgbm_dataset_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_acc': lgbm_knn_acc_test - lgbm_concat_acc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_auc': lgbm_knn_auc_test - lgbm_concat_auc_test if k_neighbors > 0 else 0,
        'improvement_knn_vs_concat_ks': ks_lgbm_knn_test - ks_lgbm_concat_test if k_neighbors > 0 else 0,
        'best_params_dataset': best_params_lgbm_dataset,
        'best_params_concat': best_params_lgbm_concat,
        'best_params_knn': best_params_lgbm_knn,
        'fpr_dataset': fpr_lgbm_dataset_test, 'tpr_dataset': tpr_lgbm_dataset_test,
        'fpr_concat': fpr_lgbm_concat_test, 'tpr_concat': tpr_lgbm_concat_test,
        'fpr_knn': fpr_lgbm_knn_test, 'tpr_knn': tpr_lgbm_knn_test,
        'final_model_dataset': final_lgbm_dataset_clf,
        'final_model_concat': final_lgbm_concat_clf,
        'final_model_knn': final_lgbm_knn_clf
    }


    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot ROC curves for each model using Test set results
    for model_name, model_data in model_results.items():
        plt.figure(figsize=(10, 8))
        # Plot dataset features
        plt.plot(model_data['fpr_dataset'], model_data['tpr_dataset'],
                 label=f'Dataset (Test AUC = {model_data["dataset_auc"]:.4f}, Test KS = {model_data["dataset_ks"]:.4f})')
        # Plot concatenated features
        plt.plot(model_data['fpr_concat'], model_data['tpr_concat'],
                 label=f'Concat (Test AUC = {model_data["concat_auc"]:.4f}, Test KS = {model_data["concat_ks"]:.4f})')
        # Plot KNN-enhanced features if available
        if k_neighbors > 0 and 'fpr_knn' in model_data:
            plt.plot(model_data['fpr_knn'], model_data['tpr_knn'],
                     label=f'KNN-Enhanced (k={k_neighbors}) (Test AUC = {model_data["knn_auc"]:.4f}, Test KS = {model_data["knn_ks"]:.4f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves on Test Set - {dataset_name} - {model_name.upper()}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_test_roc_curves.png"))
        plt.close()

    # Get feature names (simpler approach)
    feature_names = []
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception as e:
            print(f"Couldn't get feature names from preprocessor automatically: {e}")
            feature_names = [f"DatasetFeature_{i}" for i in range(split_data['X_train'].shape[1])] # Use train shape
    else:
        feature_names = [f"DatasetFeature_{i}" for i in range(split_data['X_train'].shape[1])] # Use train shape

    sublime_feature_names = [f"SUBLIME_{i}" for i in range(split_data['sublime_train'].shape[1])] # Use train shape

    # Combined feature names for concatenated model
    concat_feature_names = list(feature_names) + list(sublime_feature_names)
    if classification_probs is not None:
        concat_feature_names += ["Model_Classification_Probability"]

    # Combined feature names for KNN-enhanced model
    knn_enhanced_feature_names = []
    if k_neighbors > 0:
        knn_feature_names = ['knn_mean_distance', 'knn_mean_label']
        knn_enhanced_feature_names = list(concat_feature_names) + list(knn_feature_names)


    # Plot feature importance for each model type using the FINAL models trained on Train+Val data
    for model_name, model_data in model_results.items():

        # --- Dataset Features Importance ---
        if model_data['final_model_dataset'] and hasattr(model_data['final_model_dataset'], 'feature_importances_'):
             plt.figure(figsize=(12, 8))
             importances = model_data['final_model_dataset'].feature_importances_
             indices = np.argsort(importances)[::-1]
             n_to_plot = min(20, len(importances))
             plt.bar(range(n_to_plot), importances[indices[:n_to_plot]])
             plt.xticks(range(n_to_plot), [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                                         for i in indices[:n_to_plot]], rotation=45, ha='right')
             plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - Dataset Features) - {dataset_name}")
             plt.tight_layout()
             plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_dataset_feature_importance.png"))
             plt.close()

        # --- Concatenated Features Importance ---
        if model_data['final_model_concat'] and hasattr(model_data['final_model_concat'], 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            concat_importances = model_data['final_model_concat'].feature_importances_
            concat_indices = np.argsort(concat_importances)[::-1]
            n_to_plot = min(20, len(concat_importances))
            plt.bar(range(n_to_plot), concat_importances[concat_indices[:n_to_plot]])
            plt.xticks(range(n_to_plot), [concat_feature_names[i] if i < len(concat_feature_names) else f"Feature_{i}"
                                        for i in concat_indices[:n_to_plot]], rotation=45, ha='right')
            plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - Concatenated Features) - {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_concat_feature_importance.png"))
            plt.close()

        # --- KNN-Enhanced Features Importance ---
        if k_neighbors > 0 and model_data['final_model_knn'] and hasattr(model_data['final_model_knn'], 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            knn_importances = model_data['final_model_knn'].feature_importances_
            knn_indices = np.argsort(knn_importances)[::-1]
            n_to_plot = min(20, len(knn_importances))
            plt.bar(range(n_to_plot), knn_importances[knn_indices[:n_to_plot]])
            # Use the combined knn_enhanced_feature_names list
            plt.xticks(range(n_to_plot), [knn_enhanced_feature_names[i] if i < len(knn_enhanced_feature_names) else f"Feature_{i}"
                                        for i in knn_indices[:n_to_plot]], rotation=45, ha='right')
            plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - KNN-Enhanced Features, k={k_neighbors}) - {dataset_name}")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_knn_enhanced_feature_importance.png"))
            plt.close()


    # Create combined results for all models based on TEST SET performance
    combined_results = []
    for model_name, model_data in model_results.items():
        result_row = {
            'dataset': dataset_name,
            'model': model_name,
            'k_neighbors': k_neighbors, # Add k used
            # Dataset metrics
            'dataset_features_test_accuracy': model_data['dataset_acc'],
            'dataset_features_test_auc': model_data['dataset_auc'],
            'dataset_features_test_ks': model_data['dataset_ks'],
            # Concat metrics
            'concat_test_accuracy': model_data['concat_acc'],
            'concat_test_auc': model_data['concat_auc'],
            'concat_test_ks': model_data['concat_ks'],
            'concat_vs_dataset_improvement_test_acc': model_data['improvement_concat_vs_dataset_acc'],
            'concat_vs_dataset_improvement_test_auc': model_data['improvement_concat_vs_dataset_auc'],
            'concat_vs_dataset_improvement_test_ks': model_data['improvement_concat_vs_dataset_ks'],
            # KNN metrics (check if k > 0)
            'knn_enhanced_test_accuracy': model_data.get('knn_acc', 0) if k_neighbors > 0 else 0,
            'knn_enhanced_test_auc': model_data.get('knn_auc', 0) if k_neighbors > 0 else 0,
            'knn_enhanced_test_ks': model_data.get('knn_ks', 0) if k_neighbors > 0 else 0,
            'knn_vs_dataset_improvement_test_acc': model_data.get('improvement_knn_vs_dataset_acc', 0) if k_neighbors > 0 else 0,
            'knn_vs_dataset_improvement_test_auc': model_data.get('improvement_knn_vs_dataset_auc', 0) if k_neighbors > 0 else 0,
            'knn_vs_dataset_improvement_test_ks': model_data.get('improvement_knn_vs_dataset_ks', 0) if k_neighbors > 0 else 0,
            'knn_vs_concat_improvement_test_acc': model_data.get('improvement_knn_vs_concat_acc', 0) if k_neighbors > 0 else 0,
            'knn_vs_concat_improvement_test_auc': model_data.get('improvement_knn_vs_concat_auc', 0) if k_neighbors > 0 else 0,
            'knn_vs_concat_improvement_test_ks': model_data.get('improvement_knn_vs_concat_ks', 0) if k_neighbors > 0 else 0,
        }
        combined_results.append(result_row)

    # Save the detailed results (now based on test set) to a CSV file
    results_df = pd.DataFrame(combined_results)
    results_filename = f"{dataset_name}_all_models_test_results"
    if k_neighbors > 0:
        results_filename += f"_k{k_neighbors}"
    results_filename += ".csv"
    results_path = os.path.join(args.output_dir, results_filename)
    results_df.to_csv(results_path, index=False)
    print(f"\nTest results saved to {results_path}")

    # Also save the best parameters for each model and feature set
    for model_name, model_data in model_results.items():
        params_dict = {
            'dataset_params': [str(model_data['best_params_dataset'])],
            'concat_params': [str(model_data['best_params_concat'])]
        }
        if k_neighbors > 0 and 'best_params_knn' in model_data:
             params_dict['knn_params'] = [str(model_data['best_params_knn'])]

        params_df = pd.DataFrame(params_dict)
        params_filename = f"{dataset_name}_{model_name}_best_params"
        if k_neighbors > 0:
            params_filename += f"_k{k_neighbors}"
        params_filename += ".csv"
        params_path = os.path.join(args.output_dir, params_filename)
        params_df.to_csv(params_path, index=False)
    print(f"Best parameters saved to {os.path.dirname(params_path)}")

    # Determine the best model based on AUC improvement over dataset features
    best_auc_improvement_source = 'concat' # Default
    best_auc_improvement = max(model_results.items(), key=lambda x: x[1]['improvement_concat_vs_dataset_auc'])

    if k_neighbors > 0:
        best_knn_improvement = max(model_results.items(), key=lambda x: x[1].get('improvement_knn_vs_dataset_auc', -float('inf')))
        if best_knn_improvement[1].get('improvement_knn_vs_dataset_auc', -float('inf')) > best_auc_improvement[1]['improvement_concat_vs_dataset_auc']:
            best_auc_improvement = best_knn_improvement
            best_auc_improvement_source = 'knn_enhanced'

    best_model_name = best_auc_improvement[0]
    best_model_data = best_auc_improvement[1]

    print("\nBest performing model setup based on TEST SET AUC-ROC improvement vs. Dataset Features:")
    print(f"Best Model Type: {best_model_name.upper()}")
    print(f"Best Feature Set: {best_auc_improvement_source}")
    print(f"  Dataset features Test AUC-ROC: {best_model_data['dataset_auc']:.4f}")
    if best_auc_improvement_source == 'concat':
        print(f"  Concatenated features Test AUC-ROC: {best_model_data['concat_auc']:.4f}")
        print(f"  Improvement vs Dataset: {best_model_data['improvement_concat_vs_dataset_auc']:.4f}")
    elif best_auc_improvement_source == 'knn_enhanced':
        print(f"  KNN-Enhanced features Test AUC-ROC: {best_model_data['knn_auc']:.4f}")
        print(f"  Improvement vs Dataset: {best_model_data['improvement_knn_vs_dataset_auc']:.4f}")
        print(f"  Improvement vs Concat: {best_model_data['improvement_knn_vs_concat_auc']:.4f}")

    # Return the results dictionary containing test metrics for all setups
    return model_results

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
    dataset_name = os.path.basename(args.dataset_features_csv).split('.')[0]
    results = evaluate_features(X_dataset, sublime_embeddings, y, dataset_name,
                               preprocessor=preprocessor, n_trials=args.n_trials,
                               classification_probs=classification_probs,
                               k_neighbors=args.k_neighbors,
                               device=device)
    
    # 8. Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    
    # Find the best performing model based on improvement in AUC-ROC
    best_model = max(results.items(), key=lambda x: x[1]['improvement_auc'])
    best_model_name = best_model[0]
    
    # Print results for each model
    for model_name, model_data in results.items():
        is_best = model_name == best_model_name
        best_indicator = "★ BEST ★" if is_best else ""
        
        print("\n" + "-"*60)
        print(f"{model_name.upper()} MODEL {best_indicator}")
        print("-"*60)
        print(f"Dataset features Test accuracy: {model_data['dataset_acc']:.4f}")
        print(f"Dataset features Test AUC-ROC: {model_data['dataset_auc']:.4f}")
        print(f"Dataset features Test KS statistic: {model_data['dataset_ks']:.4f}")
        print(f"Improvement in accuracy: {model_data['improvement_acc']:.4f} (Test Set)")
        print(f"Improvement in AUC-ROC: {model_data['improvement_auc']:.4f} (Test Set)")
    
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