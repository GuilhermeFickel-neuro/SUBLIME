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

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=16, cache_dir=None, model_dir=None, dataset_name=None):
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
    
    num_batches = (len(X) + batch_size - 1) // batch_size
    all_embeddings = []
    
    # Set models to evaluation mode
    model.eval()
    if graph_learner is not None:
        graph_learner.eval()
    
    # Variable to store and reuse the KNN graph between points
    knn_graph = None
    
    # Process each batch using tqdm for progress tracking
    for i in tqdm(range(num_batches), desc="Processing batches", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        
        # Process each point individually for better error handling
        batch_embeddings = []
        
        for j in range(len(batch_X)):
            try:
                point_tensor = torch.FloatTensor(batch_X[j]).to(device)
                
                # Extract embedding using process_new_point method
                embedding = experiment.process_new_point(
                    point_tensor, model, graph_learner, features, adj, sparse
                )
                
                # Append the embedding
                batch_embeddings.append(embedding.cpu().detach().numpy())
                
            except Exception as e:
                print(f"Error processing point {start_idx + j}: {str(e)}")
                # Reset KNN graph if we encounter an error
                knn_graph = None
                raise e
                
        # Add the batch embeddings to the overall results
        all_embeddings.extend(batch_embeddings)
    
    if len(all_embeddings) != len(X):
        print(f"WARNING: Expected {len(X)} embeddings but only got {len(all_embeddings)}!")
    
    embeddings_array = np.array(all_embeddings)
    
    # Save to cache only if cache_file is defined (requires cache_dir, model_dir, and dataset_name)
    if cache_file is not None:
        print(f"Saving embeddings to cache: {cache_file}")
        np.save(cache_file, embeddings_array)
    
    return embeddings_array

def evaluate_features(dataset_features, sublime_embeddings, y, dataset_name, preprocessor=None, n_trials=50):
    """
    Train XGBoost, CatBoost and LightGBM classifiers on two different feature sets and compare performance
    
    Args:
        dataset_features: Preprocessed dataset features
        sublime_embeddings: SUBLIME extracted embeddings
        y: Target labels
        dataset_name: Name of the dataset
        preprocessor: The column transformer used to preprocess dataset features (optional)
        n_trials: Number of optimization trials for Optuna
        
    Returns:
        dict: Results dictionary with accuracies and hyperparameters
    """
    results = {}
    
    # Create concatenated features (dataset features + SUBLIME)
    concat_features = np.hstack((dataset_features, sublime_embeddings))
    
    print(f"Dataset features shape: {dataset_features.shape}")
    print(f"SUBLIME features shape: {sublime_embeddings.shape}")
    print(f"Concatenated features shape: {concat_features.shape}")
    
    # Split the data into training and validation sets (70/30 split)
    from sklearn.model_selection import train_test_split
    X_train, X_val, sublime_train, sublime_val, concat_train, concat_val, y_train, y_val = train_test_split(
        dataset_features, sublime_embeddings, concat_features, y, 
        test_size=0.3, random_state=42, stratify=y
    )
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
        model.fit(X_train, y_train)
        # Use predict_proba to get probability scores for AUC calculation
        preds_proba = model.predict_proba(X_val)[:, 1]
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
        model.fit(concat_train, y_train)
        # Use predict_proba to get probability scores for AUC calculation
        preds_proba = model.predict_proba(concat_val)[:, 1]
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
        model.fit(X_train, y_train, verbose=False)
        preds_proba = model.predict_proba(X_val)[:, 1]
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
        model.fit(concat_train, y_train, verbose=False)
        preds_proba = model.predict_proba(concat_val)[:, 1]
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
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_val)[:, 1]
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
        model.fit(concat_train, y_train)
        preds_proba = model.predict_proba(concat_val)[:, 1]
        return roc_auc_score(y_val, preds_proba)
        return roc_auc_score(y_val, preds_proba)
    
    # Dictionary to store results for all models
    model_results = {}
    
    # Train and evaluate XGBoost models
    print("\n" + "="*50)
    print("XGBoost Models")
    print("="*50)
    
    # Tune hyperparameters for dataset features - XGBoost
    print("\nOptimizing XGBoost for dataset features (using AUC-ROC)...")
    study_xgb_dataset = optuna.create_study(direction='maximize')
    study_xgb_dataset.optimize(xgb_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_xgb_dataset = study_xgb_dataset.best_params
    
    # Train XGBoost on dataset features with tuned hyperparameters
    print("\nTraining XGBoost on dataset features with best hyperparameters...")
    xgb_dataset_clf = XGBClassifier(**best_params_xgb_dataset)
    xgb_dataset_clf.fit(X_train, y_train)
    
    # Evaluate XGBoost on dataset features
    xgb_dataset_preds = xgb_dataset_clf.predict(X_val)
    xgb_dataset_preds_proba = xgb_dataset_clf.predict_proba(X_val)[:, 1]
    xgb_dataset_acc = accuracy_score(y_val, xgb_dataset_preds)
    xgb_dataset_auc = roc_auc_score(y_val, xgb_dataset_preds_proba)
    
    # Calculate KS statistic for XGBoost dataset features model
    fpr_xgb_dataset, tpr_xgb_dataset, _ = roc_curve(y_val, xgb_dataset_preds_proba)
    ks_xgb_dataset = max(tpr_xgb_dataset - fpr_xgb_dataset)
    
    print(f"XGBoost - Dataset features accuracy: {xgb_dataset_acc:.4f}")
    print(f"XGBoost - Dataset features AUC-ROC: {xgb_dataset_auc:.4f}")
    print(f"XGBoost - Dataset features KS statistic: {ks_xgb_dataset:.4f}")
    print(classification_report(y_val, xgb_dataset_preds))
    
    # Tune hyperparameters for concatenated features - XGBoost
    print("\nOptimizing XGBoost for concatenated features (using AUC-ROC)...")
    study_xgb_concat = optuna.create_study(direction='maximize')
    study_xgb_concat.optimize(xgb_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_xgb_concat = study_xgb_concat.best_params
    
    # Train XGBoost on concatenated features with tuned hyperparameters
    print("\nTraining XGBoost on concatenated features with best hyperparameters...")
    xgb_concat_clf = XGBClassifier(**best_params_xgb_concat)
    xgb_concat_clf.fit(concat_train, y_train)
    
    # Evaluate XGBoost on concatenated features
    xgb_concat_preds = xgb_concat_clf.predict(concat_val)
    xgb_concat_preds_proba = xgb_concat_clf.predict_proba(concat_val)[:, 1]
    xgb_concat_acc = accuracy_score(y_val, xgb_concat_preds)
    xgb_concat_auc = roc_auc_score(y_val, xgb_concat_preds_proba)
    
    # Calculate KS statistic for XGBoost concatenated features model
    fpr_xgb_concat, tpr_xgb_concat, _ = roc_curve(y_val, xgb_concat_preds_proba)
    ks_xgb_concat = max(tpr_xgb_concat - fpr_xgb_concat)
    
    print(f"XGBoost - Concatenated features accuracy: {xgb_concat_acc:.4f}")
    print(f"XGBoost - Concatenated features AUC-ROC: {xgb_concat_auc:.4f}")
    print(f"XGBoost - Concatenated features KS statistic: {ks_xgb_concat:.4f}")
    print(classification_report(y_val, xgb_concat_preds))
    
    # Store XGBoost results
    model_results['xgboost'] = {
        'dataset_acc': xgb_dataset_acc,
        'dataset_auc': xgb_dataset_auc,
        'dataset_ks': ks_xgb_dataset,
        'concat_acc': xgb_concat_acc,
        'concat_auc': xgb_concat_auc,
        'concat_ks': ks_xgb_concat,
        'improvement_acc': xgb_concat_acc - xgb_dataset_acc,
        'improvement_auc': xgb_concat_auc - xgb_dataset_auc,
        'improvement_ks': ks_xgb_concat - ks_xgb_dataset,
        'best_params_dataset': best_params_xgb_dataset,
        'best_params_concat': best_params_xgb_concat,
        'fpr_dataset': fpr_xgb_dataset,
        'tpr_dataset': tpr_xgb_dataset,
        'fpr_concat': fpr_xgb_concat,
        'tpr_concat': tpr_xgb_concat
    }
    
    # Train and evaluate CatBoost models
    print("\n" + "="*50)
    print("CatBoost Models")
    print("="*50)
    
    # Tune hyperparameters for dataset features - CatBoost
    print("\nOptimizing CatBoost for dataset features (using AUC-ROC)...")
    study_cat_dataset = optuna.create_study(direction='maximize')
    study_cat_dataset.optimize(cat_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_cat_dataset = study_cat_dataset.best_params
    
    # Train CatBoost on dataset features with tuned hyperparameters
    print("\nTraining CatBoost on dataset features with best hyperparameters...")
    cat_dataset_clf = CatBoostClassifier(**best_params_cat_dataset)
    cat_dataset_clf.fit(X_train, y_train, verbose=False)
    
    # Evaluate CatBoost on dataset features
    cat_dataset_preds = cat_dataset_clf.predict(X_val)
    cat_dataset_preds_proba = cat_dataset_clf.predict_proba(X_val)[:, 1]
    cat_dataset_acc = accuracy_score(y_val, cat_dataset_preds)
    cat_dataset_auc = roc_auc_score(y_val, cat_dataset_preds_proba)
    
    # Calculate KS statistic for CatBoost dataset features model
    fpr_cat_dataset, tpr_cat_dataset, _ = roc_curve(y_val, cat_dataset_preds_proba)
    ks_cat_dataset = max(tpr_cat_dataset - fpr_cat_dataset)
    
    print(f"CatBoost - Dataset features accuracy: {cat_dataset_acc:.4f}")
    print(f"CatBoost - Dataset features AUC-ROC: {cat_dataset_auc:.4f}")
    print(f"CatBoost - Dataset features KS statistic: {ks_cat_dataset:.4f}")
    print(classification_report(y_val, cat_dataset_preds))
    
    # Tune hyperparameters for concatenated features - CatBoost
    print("\nOptimizing CatBoost for concatenated features (using AUC-ROC)...")
    study_cat_concat = optuna.create_study(direction='maximize')
    study_cat_concat.optimize(cat_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_cat_concat = study_cat_concat.best_params
    
    # Train CatBoost on concatenated features with tuned hyperparameters
    print("\nTraining CatBoost on concatenated features with best hyperparameters...")
    cat_concat_clf = CatBoostClassifier(**best_params_cat_concat)
    cat_concat_clf.fit(concat_train, y_train, verbose=False)
    
    # Evaluate CatBoost on concatenated features
    cat_concat_preds = cat_concat_clf.predict(concat_val)
    cat_concat_preds_proba = cat_concat_clf.predict_proba(concat_val)[:, 1]
    cat_concat_acc = accuracy_score(y_val, cat_concat_preds)
    cat_concat_auc = roc_auc_score(y_val, cat_concat_preds_proba)
    
    # Calculate KS statistic for CatBoost concatenated features model
    fpr_cat_concat, tpr_cat_concat, _ = roc_curve(y_val, cat_concat_preds_proba)
    ks_cat_concat = max(tpr_cat_concat - fpr_cat_concat)
    
    print(f"CatBoost - Concatenated features accuracy: {cat_concat_acc:.4f}")
    print(f"CatBoost - Concatenated features AUC-ROC: {cat_concat_auc:.4f}")
    print(f"CatBoost - Concatenated features KS statistic: {ks_cat_concat:.4f}")
    print(classification_report(y_val, cat_concat_preds))
    
    # Store CatBoost results
    model_results['catboost'] = {
        'dataset_acc': cat_dataset_acc,
        'dataset_auc': cat_dataset_auc,
        'dataset_ks': ks_cat_dataset,
        'concat_acc': cat_concat_acc,
        'concat_auc': cat_concat_auc,
        'concat_ks': ks_cat_concat,
        'improvement_acc': cat_concat_acc - cat_dataset_acc,
        'improvement_auc': cat_concat_auc - cat_dataset_auc,
        'improvement_ks': ks_cat_concat - ks_cat_dataset,
        'best_params_dataset': best_params_cat_dataset,
        'best_params_concat': best_params_cat_concat,
        'fpr_dataset': fpr_cat_dataset,
        'tpr_dataset': tpr_cat_dataset,
        'fpr_concat': fpr_cat_concat,
        'tpr_concat': tpr_cat_concat
    }
    
    # Train and evaluate LightGBM models
    print("\n" + "="*50)
    print("LightGBM Models")
    print("="*50)
    
    # Tune hyperparameters for dataset features - LightGBM
    print("\nOptimizing LightGBM for dataset features (using AUC-ROC)...")
    study_lgbm_dataset = optuna.create_study(direction='maximize')
    study_lgbm_dataset.optimize(lgbm_dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_lgbm_dataset = study_lgbm_dataset.best_params
    
    # Train LightGBM on dataset features with tuned hyperparameters
    print("\nTraining LightGBM on dataset features with best hyperparameters...")
    lgbm_dataset_clf = LGBMClassifier(**best_params_lgbm_dataset)
    lgbm_dataset_clf.fit(X_train, y_train)
    
    # Evaluate LightGBM on dataset features
    lgbm_dataset_preds = lgbm_dataset_clf.predict(X_val)
    lgbm_dataset_preds_proba = lgbm_dataset_clf.predict_proba(X_val)[:, 1]
    lgbm_dataset_acc = accuracy_score(y_val, lgbm_dataset_preds)
    lgbm_dataset_auc = roc_auc_score(y_val, lgbm_dataset_preds_proba)
    
    # Calculate KS statistic for LightGBM dataset features model
    fpr_lgbm_dataset, tpr_lgbm_dataset, _ = roc_curve(y_val, lgbm_dataset_preds_proba)
    ks_lgbm_dataset = max(tpr_lgbm_dataset - fpr_lgbm_dataset)
    
    print(f"LightGBM - Dataset features accuracy: {lgbm_dataset_acc:.4f}")
    print(f"LightGBM - Dataset features AUC-ROC: {lgbm_dataset_auc:.4f}")
    print(f"LightGBM - Dataset features KS statistic: {ks_lgbm_dataset:.4f}")
    print(classification_report(y_val, lgbm_dataset_preds))
    
    # Tune hyperparameters for concatenated features - LightGBM
    print("\nOptimizing LightGBM for concatenated features (using AUC-ROC)...")
    study_lgbm_concat = optuna.create_study(direction='maximize')
    study_lgbm_concat.optimize(lgbm_concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_lgbm_concat = study_lgbm_concat.best_params
    
    # Train LightGBM on concatenated features with tuned hyperparameters
    print("\nTraining LightGBM on concatenated features with best hyperparameters...")
    lgbm_concat_clf = LGBMClassifier(**best_params_lgbm_concat)
    lgbm_concat_clf.fit(concat_train, y_train)
    
    # Evaluate LightGBM on concatenated features
    lgbm_concat_preds = lgbm_concat_clf.predict(concat_val)
    lgbm_concat_preds_proba = lgbm_concat_clf.predict_proba(concat_val)[:, 1]
    lgbm_concat_acc = accuracy_score(y_val, lgbm_concat_preds)
    lgbm_concat_auc = roc_auc_score(y_val, lgbm_concat_preds_proba)
    
    # Calculate KS statistic for LightGBM concatenated features model
    fpr_lgbm_concat, tpr_lgbm_concat, _ = roc_curve(y_val, lgbm_concat_preds_proba)
    ks_lgbm_concat = max(tpr_lgbm_concat - fpr_lgbm_concat)
    
    print(f"LightGBM - Concatenated features accuracy: {lgbm_concat_acc:.4f}")
    print(f"LightGBM - Concatenated features AUC-ROC: {lgbm_concat_auc:.4f}")
    print(f"LightGBM - Concatenated features KS statistic: {ks_lgbm_concat:.4f}")
    print(classification_report(y_val, lgbm_concat_preds))
    
    # Store LightGBM results
    model_results['lightgbm'] = {
        'dataset_acc': lgbm_dataset_acc,
        'dataset_auc': lgbm_dataset_auc,
        'dataset_ks': ks_lgbm_dataset,
        'concat_acc': lgbm_concat_acc,
        'concat_auc': lgbm_concat_auc,
        'concat_ks': ks_lgbm_concat,
        'improvement_acc': lgbm_concat_acc - lgbm_dataset_acc,
        'improvement_auc': lgbm_concat_auc - lgbm_dataset_auc,
        'improvement_ks': ks_lgbm_concat - ks_lgbm_dataset,
        'best_params_dataset': best_params_lgbm_dataset,
        'best_params_concat': best_params_lgbm_concat,
        'fpr_dataset': fpr_lgbm_dataset,
        'tpr_dataset': tpr_lgbm_dataset,
        'fpr_concat': fpr_lgbm_concat,
        'tpr_concat': tpr_lgbm_concat
    }
    
    # Use the best model's results for the plot (XGBoost for backward compatibility)
    dataset_acc = xgb_dataset_acc
    dataset_auc = xgb_dataset_auc
    ks_dataset = ks_xgb_dataset
    concat_acc = xgb_concat_acc
    concat_auc = xgb_concat_auc
    ks_concat = ks_xgb_concat
    fpr_dataset = fpr_xgb_dataset
    tpr_dataset = tpr_xgb_dataset
    fpr_concat = fpr_xgb_concat
    tpr_concat = tpr_xgb_concat
    
    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot ROC curves for each model
    for model_name, model_data in model_results.items():
        plt.figure(figsize=(10, 8))
        plt.plot(model_data['fpr_dataset'], model_data['tpr_dataset'],
                label=f'Dataset features (AUC = {model_data["dataset_auc"]:.4f}, KS = {model_data["dataset_ks"]:.4f})')
        plt.plot(model_data['fpr_concat'], model_data['tpr_concat'],
                label=f'Concatenated features (AUC = {model_data["concat_auc"]:.4f}, KS = {model_data["concat_ks"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {dataset_name} - {model_name.upper()}')
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_roc_curves.png"))
    
    # Get feature names (simpler approach)
    feature_names = []
    if preprocessor is not None:
        # Try to get feature names from preprocessor if available
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            # If get_feature_names_out() not available (older sklearn)
            print("Couldn't get feature names from preprocessor automatically")
            # Use generic feature names
            feature_names = [f"Feature_{i}" for i in range(dataset_features.shape[1])]
    else:
        # Use generic feature names
        feature_names = [f"Feature_{i}" for i in range(dataset_features.shape[1])]
    
    # SUBLIME feature names
    sublime_feature_names = [f"SUBLIME_{i}" for i in range(sublime_embeddings.shape[1])]
    
    # Combined feature names for concatenated model
    concat_feature_names = list(feature_names) + list(sublime_feature_names)
    
    # Define model classifiers for feature importance plots
    model_classifiers = {
        'xgboost': {
            'dataset': xgb_dataset_clf,
            'concat': xgb_concat_clf
        },
        'catboost': {
            'dataset': cat_dataset_clf,
            'concat': cat_concat_clf
        },
        'lightgbm': {
            'dataset': lgbm_dataset_clf,
            'concat': lgbm_concat_clf
        }
    }
    
    # Plot feature importance for each model type
    for model_name, classifiers in model_classifiers.items():
        # Feature importance for dataset features
        plt.figure(figsize=(12, 8))
        importances = classifiers['dataset'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top 20 features or all if less than 20
        n_to_plot = min(20, len(importances))
        plt.bar(range(n_to_plot), importances[indices[:n_to_plot]])
        plt.xticks(range(n_to_plot), [feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                                    for i in indices[:n_to_plot]], rotation=45, ha='right')
        plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - Dataset Features) - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_dataset_feature_importance.png"))
        
        # Feature importance for concatenated features
        plt.figure(figsize=(12, 8))
        concat_importances = classifiers['concat'].feature_importances_
        concat_indices = np.argsort(concat_importances)[::-1]
        
        # Plot top 20 features or all if less than 20
        n_to_plot = min(20, len(concat_importances))
        plt.bar(range(n_to_plot), concat_importances[concat_indices[:n_to_plot]])
        plt.xticks(range(n_to_plot), [concat_feature_names[i] if i < len(concat_feature_names) else f"Feature_{i}"
                                    for i in concat_indices[:n_to_plot]], rotation=45, ha='right')
        plt.title(f"Top {n_to_plot} Feature Importance ({model_name.upper()} - Concatenated Features) - {dataset_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{dataset_name}_{model_name}_concat_feature_importance.png"))
    
    # Create combined results for all models
    combined_results = []
    
    for model_name, model_data in model_results.items():
        result_row = {
            'dataset': dataset_name,
            'model': model_name,
            'dataset_features_accuracy': model_data['dataset_acc'],
            'dataset_features_auc': model_data['dataset_auc'],
            'dataset_features_ks': model_data['dataset_ks'],
            'concat_accuracy': model_data['concat_acc'],
            'concat_auc': model_data['concat_auc'],
            'concat_ks': model_data['concat_ks'],
            'concat_vs_dataset_improvement_acc': model_data['improvement_acc'],
            'concat_vs_dataset_improvement_auc': model_data['improvement_auc'],
            'concat_vs_dataset_improvement_ks': model_data['improvement_ks']
        }
        combined_results.append(result_row)
    
    # Save the detailed results to a CSV file
    results_df = pd.DataFrame(combined_results)
    results_path = os.path.join(args.output_dir, f"{dataset_name}_all_models_results.csv")
    results_df.to_csv(results_path, index=False)
    
    # Also save the best parameters for each model
    for model_name, model_data in model_results.items():
        params_df = pd.DataFrame({
            'dataset_params': [str(model_data['best_params_dataset'])],
            'concat_params': [str(model_data['best_params_concat'])]
        })
        params_path = os.path.join(args.output_dir, f"{dataset_name}_{model_name}_best_params.csv")
        params_df.to_csv(params_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Find the best performing model based on improvement in AUC-ROC
    best_model = max(model_results.items(), key=lambda x: x[1]['improvement_auc'])
    best_model_name = best_model[0]
    best_model_data = best_model[1]
    
    print("\nBest performing model based on AUC-ROC improvement:")
    print(f"Model: {best_model_name.upper()}")
    print(f"Dataset features AUC-ROC: {best_model_data['dataset_auc']:.4f}")
    print(f"Concatenated features AUC-ROC: {best_model_data['concat_auc']:.4f}")
    print(f"Improvement: {best_model_data['improvement_auc']:.4f}")
    
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
    
    # 4. Extract SUBLIME features from neurolake data
    print("\nExtracting SUBLIME features...")
    sublime_embeddings = extract_in_batches(
        X_neurolake, model, graph_learner, features, adj, sparse, experiment, 
        batch_size=args.batch_size, cache_dir=args.cache_dir, model_dir=args.model_dir, dataset_name=os.path.basename(args.neurolake_csv).split('.')[0]
    )
    # Normalize the SUBLIME embeddings
    sublime_embeddings = sublime_embeddings / np.maximum(np.linalg.norm(sublime_embeddings, axis=1, keepdims=True), 1e-10)
    print(f"Feature extraction complete. Extracted shape: {sublime_embeddings.shape}")
    
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
            
            print(f"After filtering: {len(dataset_df)} rows remaining")
    else:
        raise ValueError(f"Target column '{args.target_column}' not found in the dataset features")
    
    # Process dataset features with a new preprocessing pipeline
    X_dataset, preprocessor, y = preprocess_dataset_features(dataset_df, args.target_column, fit_transform=True)
    
    # 7. Evaluate using dataset features and SUBLIME embeddings
    print("\nEvaluating features with XGBoost, CatBoost, and LightGBM...")
    dataset_name = os.path.basename(args.dataset_features_csv).split('.')[0]
    results = evaluate_features(X_dataset, sublime_embeddings, y, dataset_name, preprocessor=preprocessor, n_trials=args.n_trials)
    
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
        print(f"Dataset features accuracy: {model_data['dataset_acc']:.4f}")
        print(f"Dataset features AUC-ROC: {model_data['dataset_auc']:.4f}")
        print(f"Dataset features KS: {model_data['dataset_ks']:.4f}")
        print(f"Concatenated features accuracy: {model_data['concat_acc']:.4f}")
        print(f"Concatenated features AUC-ROC: {model_data['concat_auc']:.4f}")
        print(f"Concatenated features KS: {model_data['concat_ks']:.4f}")
        print(f"Improvement in accuracy: {model_data['improvement_acc']:.4f}")
        print(f"Improvement in AUC-ROC: {model_data['improvement_auc']:.4f}")
        print(f"Improvement in KS: {model_data['improvement_ks']:.4f}")
    
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.embeddings_output and (not args.dataset_features_csv or not args.target_column):
        parser.error("Either provide --embeddings-output to extract embeddings, or provide --dataset-features-csv and --target-column for evaluation")
    
    main(args) 