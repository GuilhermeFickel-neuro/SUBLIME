import argparse
import torch
import pandas as pd
import numpy as np
import os
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
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

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=16, cache_dir=None, model_dir=None):
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
        
    Returns:
        numpy.ndarray: Extracted features
    """
    # Try to load from cache if cache_dir is provided
    if cache_dir is not None and model_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use model_dir as part of the cache filename
        # Extract the base model directory name without the full path
        model_name = os.path.basename(os.path.normpath(model_dir))
        cache_file = os.path.join(cache_dir, f"sublime_embeddings_{model_name}.npy")
        
        # If cache file exists, load and return it
        if os.path.exists(cache_file):
            print(f"Loading cached embeddings from {cache_file}")
            return np.load(cache_file)
        else:
            print(f"Cache file not found. Extracting embeddings and saving to {cache_file}")
    
    num_batches = (len(X) + batch_size - 1) // batch_size
    all_embeddings = []
    
    # Set models to evaluation mode
    model.eval()
    if graph_learner is not None:
        graph_learner.eval()
    
    # Process each batch
    for i in range(num_batches):
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
                raise e
                
        # Add the batch embeddings to the overall results
        all_embeddings.extend(batch_embeddings)
        
        # Print progress every 10 batches
        if i % 10 == 0:
            print(f"Processed {end_idx}/{len(X)} samples")
    
    if len(all_embeddings) != len(X):
        print(f"WARNING: Expected {len(X)} embeddings but only got {len(all_embeddings)}!")
    
    embeddings_array = np.array(all_embeddings)
    
    # Save to cache if cache_dir is provided
    if cache_dir is not None and model_dir is not None:
        print(f"Saving embeddings to cache: {cache_file}")
        np.save(cache_file, embeddings_array)
    
    return embeddings_array

def evaluate_features(dataset_features, sublime_embeddings, y, dataset_name, n_trials=50):
    """
    Train XGBoost classifiers on two different feature sets and compare performance
    
    Args:
        dataset_features: Preprocessed dataset features
        sublime_embeddings: SUBLIME extracted embeddings
        y: Target labels
        dataset_name: Name of the dataset
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
    
    # Define Optuna objective for dataset features
    def dataset_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
    # Define Optuna objective for concatenated features
    def concat_objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(concat_train, y_train)
        preds = model.predict(concat_val)
        return accuracy_score(y_val, preds)
    
    # Tune hyperparameters for dataset features
    print("\nOptimizing model for dataset features...")
    study_dataset = optuna.create_study(direction='maximize')
    study_dataset.optimize(dataset_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_dataset = study_dataset.best_params
    
    # Train XGBoost on dataset features with tuned hyperparameters
    print("\nTraining model on dataset features with best hyperparameters...")
    dataset_clf = XGBClassifier(**best_params_dataset)
    dataset_clf.fit(X_train, y_train)
    
    # Evaluate on dataset features
    dataset_preds = dataset_clf.predict(X_val)
    dataset_acc = accuracy_score(y_val, dataset_preds)
    print(f"Dataset features accuracy: {dataset_acc:.4f}")
    print(classification_report(y_val, dataset_preds))
    
    # Tune hyperparameters for concatenated features
    print("\nOptimizing model for concatenated features...")
    study_concat = optuna.create_study(direction='maximize')
    study_concat.optimize(concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_concat = study_concat.best_params
    
    # Train XGBoost on concatenated features with tuned hyperparameters
    print("\nTraining model on concatenated features with best hyperparameters...")
    concat_clf = XGBClassifier(**best_params_concat)
    concat_clf.fit(concat_train, y_train)
    
    # Evaluate on concatenated features
    concat_preds = concat_clf.predict(concat_val)
    concat_acc = accuracy_score(y_val, concat_preds)
    print(f"Concatenated features accuracy: {concat_acc:.4f}")
    print(classification_report(y_val, concat_preds))
    
    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Feature importance for dataset features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(dataset_clf.feature_importances_)), dataset_clf.feature_importances_)
    plt.title(f"Feature Importance (Dataset Features) - {dataset_name}")
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_dataset_feature_importance.png"))
    
    # Feature importance for concatenated features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(concat_clf.feature_importances_)), concat_clf.feature_importances_)
    plt.title(f"Feature Importance (Concatenated Features) - {dataset_name}")
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_concat_feature_importance.png"))
    
    # Save results
    results = {
        'dataset': dataset_name,
        'dataset_features_accuracy': dataset_acc,
        'concat_accuracy': concat_acc,
        'concat_vs_dataset_improvement': concat_acc - dataset_acc,
        'best_params_dataset': best_params_dataset,
        'best_params_concat': best_params_concat
    }
    
    # Save the detailed results to a CSV file
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f"{dataset_name}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    return results

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
    
    # 3. Load dataset features
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
    
    # 2. Process neurolake data with preprocess_mixed_data for SUBLIME
    X_neurolake = preprocess_mixed_data(neurolake_df, model_dir=args.model_dir)
    print(f"Neurolake data processed: {X_neurolake.shape}")
    
    # 4. Process dataset features with a new preprocessing pipeline
    X_dataset, preprocessor, y = preprocess_dataset_features(dataset_df, args.target_column, fit_transform=True)
    
    # 5. Load the SUBLIME model
    print(f"\nLoading SUBLIME model from {args.model_dir}")
    experiment = Experiment(device)
    model, graph_learner, features, adj, sparse = experiment.load_model(input_dir=args.model_dir)
    print("Model loaded successfully!")
    
    # 6. Extract SUBLIME features from neurolake data
    print("\nExtracting SUBLIME features...")
    sublime_embeddings = extract_in_batches(
        X_neurolake, model, graph_learner, features, adj, sparse, experiment, 
        batch_size=args.batch_size, cache_dir=args.cache_dir, model_dir=args.model_dir
    )
    print(f"Feature extraction complete. Extracted shape: {sublime_embeddings.shape}")
    
    # 7. Evaluate using dataset features and SUBLIME embeddings
    print("\nEvaluating features with XGBoost...")
    dataset_name = os.path.basename(args.dataset_features_csv).split('.')[0]
    results = evaluate_features(X_dataset, sublime_embeddings, y, dataset_name, n_trials=args.n_trials)
    
    # 8. Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Dataset features accuracy: {results['dataset_features_accuracy']:.4f}")
    print(f"Concatenated features accuracy: {results['concat_accuracy']:.4f}")
    print(f"Concatenated vs Dataset features improvement: {results['concat_vs_dataset_improvement']:.4f}")
    print("="*80)
    
    # 9. Save the preprocessor for future use
    preprocessor_path = os.path.join(args.output_dir, 'dataset_features_preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Dataset features preprocessor saved to {preprocessor_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SUBLIME features with dataset features")
    parser.add_argument('--neurolake-csv', type=str, required=True, help='Path to the neurolake CSV file for SUBLIME')
    parser.add_argument('--dataset-features-csv', type=str, required=True, help='Path to the dataset features CSV file')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory where the SUBLIME model is saved')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column in the dataset features CSV')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for feature extraction')
    parser.add_argument('--cache-dir', type=str, default='cache', help='Directory to cache SUBLIME embeddings for faster subsequent runs')
    
    args = parser.parse_args()
    main(args) 