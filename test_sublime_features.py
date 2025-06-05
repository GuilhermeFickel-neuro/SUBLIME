import argparse
import torch
import pandas as pd
import numpy as np
import time
import logging
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from main import Experiment
import os
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def get_available_datasets():
    """
    Returns a dictionary of available datasets for testing.
    
    Returns:
        dict: Dictionary with dataset names as keys and dataset loading functions as values
    """
    datasets = {
        # Original smaller datasets
        'breast_cancer': load_breast_cancer,
        'iris': load_iris,
        'wine': load_wine,
        'digits': load_digits,
        
        # Larger datasets from OpenML
        'mnist': lambda: fetch_openml('mnist_784', version=1, as_frame=False),
        'fashion_mnist': lambda: fetch_openml('Fashion-MNIST', version=1, as_frame=False),
        'adult': lambda: fetch_openml('adult', version=2, as_frame=False),
        'covertype': lambda: fetch_openml('covertype', version=1, as_frame=False, parser='auto'),
        'credit_g': lambda: fetch_openml('credit-g', version=1, as_frame=False),
        'vehicle': lambda: fetch_openml('vehicle', version=1, as_frame=False),
        'satimage': lambda: fetch_openml('satimage', version=1, as_frame=False),
        'cifar_10': lambda: fetch_openml('CIFAR_10', version=1, as_frame=False),
    }
    return datasets

def prepare_data(dataset_name, dataset_loader):
    """
    Load the specified dataset and prepare it for training.
    
    Args:
        dataset_name: Name of the dataset
        dataset_loader: Function to load the dataset
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            feature_names: Names of the features
    """
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    data = dataset_loader()
    
    X, y = data.data, data.target
    
    # Handle feature names (some datasets might not have them)
    if hasattr(data, 'feature_names'):
        feature_names = data.feature_names
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Convert target to numeric if it's not already
    if y.dtype == 'O':  # Object dtype, likely strings
        print("Converting string labels to numeric...")
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Data shape: {X.shape}")
    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    # Safely print class distribution
    try:
        print(f"Class distribution in training: {np.bincount(y_train)}")
        print(f"Class distribution in testing: {np.bincount(y_test)}")
    except TypeError:
        # If bincount still fails, just show the unique values counts
        print(f"Class distribution in training: {pd.Series(y_train).value_counts().sort_index()}")
        print(f"Class distribution in testing: {pd.Series(y_test).value_counts().sort_index()}")
    
    return X_train, X_test, y_train, y_test, feature_names

def save_data_as_csv(X_train, y_train, feature_names, dataset_name, output_dir='data'):
    """
    Save the training data to a CSV file for SUBLIME training.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: Names of the features
        dataset_name: Name of the dataset
        output_dir: Directory to save the CSV file
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{dataset_name}_train.csv")
    
    # Create DataFrame with features
    df = pd.DataFrame(X_train, columns=feature_names)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Training data saved to {output_file}")
    
    return output_file

def train_sublime(data_file, n_clusters, output_dir, hyperparams=None):
    """
    Train SUBLIME model on the given data in self-supervised mode.
    
    Args:
        data_file: Path to the CSV file containing the data
        n_clusters: Number of clusters for SUBLIME clustering
        output_dir: Directory to save the trained model
        hyperparams: Dictionary of hyperparameters to override defaults
    """
    # Configure SUBLIME arguments
    args = argparse.Namespace(
        dataset=data_file,
        ntrials=1,
        sparse=0,
        gsl_mode="structure_inference",
        eval_freq=500,
        downstream_task="clustering",
        n_clusters=n_clusters,
        
        # GCL Module parameters
        epochs=1500,  # Reduced for testing, increase for better results
        lr=0.001244,
        w_decay=0.0099,
        hidden_dim=256,  # Reduced for smaller dataset
        rep_dim=32,      # Reduced for smaller dataset
        proj_dim=128,     # Reduced for smaller dataset
        dropout=0.41,
        contrast_batch_size=0,
        nlayers=2,
        
        # Augmentation parameters
        maskfeat_rate_learner=0.3077,
        maskfeat_rate_anchor=0.2617,
        dropedge_rate=0.1,
        
        # GSL Module parameters
        type_learner="fgp",
        k=10,  # Reduced for smaller dataset
        sim_function="cosine",
        gamma=0.9,
        activation_learner="relu",
        
        # Structure Bootstrapping
        tau=0.9,
        c=10,
        
        # Save model
        save_model=1,
        output_dir=output_dir,
        verbose=0
    )
    
    # Override with provided hyperparameters if any
    if hyperparams:
        for key, value in hyperparams.items():
            setattr(args, key, value)
    
    # Create experiment and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(device)
    
    # Print the hyperparameters being used
    print(f"Training SUBLIME with hyperparameters: epochs={args.epochs}, lr={args.lr}, w_decay={args.w_decay}, "
          f"hidden_dim={args.hidden_dim}, rep_dim={args.rep_dim}, proj_dim={args.proj_dim}, "
          f"dropout={args.dropout}, maskfeat_rate_learner={args.maskfeat_rate_learner}, "
          f"maskfeat_rate_anchor={args.maskfeat_rate_anchor}, dropedge_rate={args.dropedge_rate}")
    
    # Define a custom data loading function for our CSV
    def load_csv_data(args):
        # Load data
        df = pd.read_csv(args.dataset)
        
        # Get dimensions
        n_samples = df.shape[0]
        
        # Convert features to tensor
        features = torch.FloatTensor(df.values)
        
        # Create empty labels (since this is unsupervised)
        labels = None
        
        # Create empty masks (not used in unsupervised setting but needed for consistency)
        train_mask = torch.zeros(n_samples, dtype=torch.bool)
        val_mask = torch.zeros(n_samples, dtype=torch.bool)
        test_mask = torch.zeros(n_samples, dtype=torch.bool)
        
        # Create identity adjacency matrix
        adj = torch.eye(n_samples).to(device)
        
        # Get feature dimension
        nfeats = features.shape[1]
        
        return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, adj
    
    # Train the model
    print(f"Training SUBLIME on {data_file}...")
    experiment.train(args, load_data_fn=load_csv_data)
    print(f"SUBLIME model trained and saved to {output_dir}")
    
    return output_dir

def extract_features(model_dir, X_train, X_test, dataset_name=None, cache_dir=None):
    """
    Extract features from both training and testing data using the trained SUBLIME model.
    
    Args:
        model_dir: Directory where the SUBLIME model is saved
        X_train: Training features
        X_test: Testing features
        dataset_name: Name of the dataset. If None, no caching is performed.
        cache_dir: Directory to cache results. If None, no caching is performed.
        
    Returns:
        tuple: (train_embeddings, test_embeddings)
            train_embeddings: Extracted features for training data
            test_embeddings: Extracted features for testing data
    """
    # Load the saved model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(device)
    
    # Load the model using parameters from config file
    print(f"Loading model from {model_dir}...")
    model, graph_learner, features, adj, sparse = experiment.load_model(input_dir=model_dir)
    
    print("Model loaded successfully!")
    
    # Process batches to avoid memory issues with large datasets
    batch_size = 32  # Smaller batch size
    
    # Extract features for training data
    print(f"Extracting features for {len(X_train)} training points...")
    train_embeddings = extract_in_batches(
        X_train, model, graph_learner, features, adj, sparse, experiment, 
        batch_size, dataset_name, cache_dir, model_dir
    )
    
    # Extract features for testing data 
    print(f"Extracting features for {len(X_test)} testing points...")
    test_embeddings = extract_in_batches(
        X_test, model, graph_learner, features, adj, sparse, experiment, 
        batch_size, dataset_name, cache_dir, model_dir
    )
    
    print(f"Feature extraction complete. Shapes: train={train_embeddings.shape}, test={test_embeddings.shape}")
    
    return train_embeddings, test_embeddings

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=16, dataset_name=None, cache_dir=None, model_dir=None):
    """
    Helper function to extract features in batches to avoid memory issues.
    
    Args:
        X: Features to extract embeddings for
        model: Trained model
        graph_learner: Trained graph learner
        features: Original features used for training
        adj: Adjacency matrix
        sparse: Whether adjacency is sparse
        experiment: Experiment instance
        batch_size: Batch size for processing
        dataset_name: Name of the dataset. If None, no caching is performed.
        cache_dir: Directory to cache results. If None, no caching is performed.
        model_dir: Directory where the model is stored, used for cache file naming
        
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = (len(X) + batch_size - 1) // batch_size
    all_embeddings = []
    
    # Set models to evaluation mode
    model.eval()
    if graph_learner is not None:
        graph_learner.eval()
    
    success_count = 0
    error_count = 0
    
    
    # Track which batch is being processed
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        
        # Process each point individually for better error handling
        batch_embeddings = []
        
        for j in range(len(batch_X)):
            point_idx = start_idx + j
            
            # Convert to tensor and ensure correct shape
            try:
                point_tensor = torch.FloatTensor(batch_X[j]).to(device)
                
                # Extract embedding using our improved process_new_point method
                embedding, _ = experiment.process_new_point(
                    point_tensor, model, graph_learner, features, adj, sparse
                )
                
                # Append the embedding
                batch_embeddings.append(embedding.cpu().detach().numpy())
                success_count += 1

                
            except Exception as e:
                error_count += 1
                print(f"Error processing point {point_idx}: {str(e)}")
                
                raise e
        # Add the batch embeddings to the overall results
        all_embeddings.extend(batch_embeddings)
        
        # Print progress
        # CRITICAL DEBUG: Check if loop should continue but will stop early
    if len(all_embeddings) != len(X):
        print(f"DEBUG: ERROR! Expected {len(X)} embeddings but only got {len(all_embeddings)}!")
        print(f"DEBUG: This explains the dimension mismatch error in your XGBoost training.")
    
    embeddings_array = np.array(all_embeddings)
    
    # Save to cache only if cache_file is defined (requires cache_dir, model_dir, and dataset_name)
    if cache_file is not None:
        print(f"Saving embeddings to cache: {cache_file}")
        np.save(cache_file, embeddings_array)
    
    return embeddings_array

def evaluate_features(X_train, X_test, train_embeddings, test_embeddings, y_train, y_test, dataset_name, n_trials=50):
    """
    Train XGBoost classifiers on both original features and extracted features,
    and compare their performance. Uses Optuna to tune hyperparameters.
    
    Args:
        X_train: Original training features
        X_test: Original testing features
        train_embeddings: Extracted features for training data
        test_embeddings: Extracted features for testing data
        y_train: Training labels
        y_test: Testing labels
        dataset_name: Name of the dataset
        n_trials: Number of optimization trials for Optuna
        
    Returns:
        dict: Results dictionary with accuracies and hyperparameters
    """
    results = {}
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Create concatenated features (original + SUBLIME)
    train_concat = np.hstack((X_train, train_embeddings))
    test_concat = np.hstack((X_test, test_embeddings))
    
    print(f"Original features shape: {X_train.shape}")
    print(f"SUBLIME features shape: {train_embeddings.shape}")
    print(f"Concatenated features shape: {train_concat.shape}")
    
    # Define Optuna objective for original features
    def original_objective(trial):
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
        return cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    
    # Define Optuna objective for SUBLIME features
    def sublime_objective(trial):
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
        return cross_val_score(model, train_embeddings, y_train, cv=5, scoring='accuracy').mean()
    
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
        return cross_val_score(model, train_concat, y_train, cv=5, scoring='accuracy').mean()
    
    # Tune hyperparameters for original features
    study_original = optuna.create_study(direction='maximize')
    study_original.optimize(original_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    
    best_params_original = study_original.best_params
    
    # Train XGBoost on original features with tuned hyperparameters
    original_clf = XGBClassifier(**best_params_original)
    original_clf.fit(X_train, y_train)
    
    # Evaluate on original features
    original_preds = original_clf.predict(X_test)
    original_acc = accuracy_score(y_test, original_preds)
    print(f"Original features accuracy: {original_acc:.4f}")
    
    # Tune hyperparameters for SUBLIME features
    study_sublime = optuna.create_study(direction='maximize')
    study_sublime.optimize(sublime_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    
    best_params_sublime = study_sublime.best_params
    
    # Train XGBoost on extracted features with tuned hyperparameters
    sublime_clf = XGBClassifier(**best_params_sublime)
    sublime_clf.fit(train_embeddings, y_train)
    
    # Evaluate on extracted features
    sublime_preds = sublime_clf.predict(test_embeddings)
    sublime_acc = accuracy_score(y_test, sublime_preds)
    print(f"SUBLIME features accuracy: {sublime_acc:.4f}")
    
    # Tune hyperparameters for concatenated features
    study_concat = optuna.create_study(direction='maximize')
    study_concat.optimize(concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    
    best_params_concat = study_concat.best_params
    
    # Train XGBoost on concatenated features with tuned hyperparameters
    concat_clf = XGBClassifier(**best_params_concat)
    concat_clf.fit(train_concat, y_train)
    
    # Evaluate on concatenated features
    concat_preds = concat_clf.predict(test_concat)
    concat_acc = accuracy_score(y_test, concat_preds)
    print(f"Concatenated features accuracy: {concat_acc:.4f}")
    
    # Create output directory for plots
    os.makedirs('plots', exist_ok=True)
    
    # Feature importance comparison (only for original features)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(original_clf.feature_importances_)), original_clf.feature_importances_)
    plt.title(f"Feature Importance (Original Features) - {dataset_name}")
    plt.savefig(f"plots/{dataset_name}_original_feature_importance.png")
    
    # Feature importance for SUBLIME features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sublime_clf.feature_importances_)), sublime_clf.feature_importances_)
    plt.title(f"Feature Importance (SUBLIME Features) - {dataset_name}")
    plt.savefig(f"plots/{dataset_name}_sublime_feature_importance.png")
    
    # Feature importance for concatenated features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(concat_clf.feature_importances_)), concat_clf.feature_importances_)
    plt.title(f"Feature Importance (Concatenated Features) - {dataset_name}")
    plt.savefig(f"plots/{dataset_name}_concat_feature_importance.png")
    
    # Save hyperparameter tuning visualization
    try:
        fig = optuna.visualization.plot_param_importances(study_original)
        fig.write_image(f"plots/{dataset_name}_original_param_importance.png")
        
        fig = optuna.visualization.plot_param_importances(study_sublime)
        fig.write_image(f"plots/{dataset_name}_sublime_param_importance.png")
        
        fig = optuna.visualization.plot_param_importances(study_concat)
        fig.write_image(f"plots/{dataset_name}_concat_param_importance.png")
    except:
        print("Could not generate hyperparameter importance plots. You may need to install plotly.")
    
    return {
        'dataset': dataset_name,
        'original_accuracy': original_acc,
        'sublime_accuracy': sublime_acc,
        'concat_accuracy': concat_acc,
        'original_vs_sublime_improvement': sublime_acc - original_acc,
        'concat_vs_original_improvement': concat_acc - original_acc,
        'concat_vs_sublime_improvement': concat_acc - sublime_acc,
        'best_params_original': best_params_original,
        'best_params_sublime': best_params_sublime,
        'best_params_concat': best_params_concat
    }

def process_dataset(dataset_name, dataset_loader, n_trials=50):
    """
    Process a single dataset through the complete pipeline.
    
    Args:
        dataset_name: Name of the dataset
        dataset_loader: Function to load the dataset
        n_trials: Number of optimization trials for Optuna
        
    Returns:
        dict: Results dictionary
    """
    print("="*80)
    print(f"PROCESSING DATASET: {dataset_name}")
    print("="*80)
    
    # Create timestamp for unique output directories
    timestamp = int(time.time())
    
    # 1. Prepare the data
    print("\nSTEP 1: Preparing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(dataset_name, dataset_loader)
    
    # Get number of classes to set n_clusters for SUBLIME
    n_clusters = len(np.unique(y_train))
    
    # 2. Save training data as CSV for SUBLIME
    print("\nSTEP 2: Saving training data as CSV...")
    data_file = save_data_as_csv(X_train, y_train, feature_names, dataset_name)
    
    # 3. Train SUBLIME model on the training data
    print("\nSTEP 3: Training SUBLIME model...")
    model_dir = f"sublime_model_{dataset_name}_{timestamp}"
    model_dir = train_sublime(data_file, n_clusters=n_clusters, output_dir=model_dir)
    
    # 4. Extract features using the trained SUBLIME model
    print("\nSTEP 4: Extracting features using SUBLIME...")
    train_embeddings, test_embeddings = extract_features(model_dir, X_train, X_test, dataset_name, cache_dir="cache")
    
    # 5. Evaluate the extracted features with XGBoost
    print("\nSTEP 5: Evaluating extracted features with XGBoost...")
    results = evaluate_features(X_train, X_test, train_embeddings, test_embeddings, y_train, y_test, dataset_name, n_trials=n_trials)
    
    return results

def main():
    """
    Main function to run the complete pipeline for testing SUBLIME feature extraction
    on multiple datasets.
    """
    print("="*80)
    print("TESTING SUBLIME FEATURE EXTRACTION FOR CLASSIFICATION ON MULTIPLE DATASETS")
    print("="*80)
    
    # Get available datasets
    datasets = get_available_datasets()
    
    # Store results for all datasets
    all_results = []
    
    # Process each dataset
    for dataset_name, dataset_loader in datasets.items():
        try:
            # Process the dataset
            results = process_dataset(dataset_name, dataset_loader, n_trials=30)  # Reduced trials for faster execution
            all_results.append(results)
        except Exception as e:
            logging.error(f"Error processing dataset {dataset_name}: {str(e)}", exc_info=True)
    
    # Print summary of results
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    # Create a pandas DataFrame for easy comparison
    results_df = pd.DataFrame(all_results)
    
    # Sort by improvement in accuracy
    results_df = results_df.sort_values('improvement', ascending=False)
    
    # Print table of results
    print("\nResults Table:")
    print(results_df[['dataset', 'original_accuracy', 'sublime_accuracy', 'improvement']])
    
    # Save results to CSV
    results_df.to_csv('sublime_comparison_results.csv', index=False)
    print("\nDetailed results saved to sublime_comparison_results.csv")
    
    # Print overall conclusion
    avg_improvement = results_df['improvement'].mean()
    if avg_improvement > 0:
        print(f"\nCONCLUSION: On average, SUBLIME features improved classification performance by {avg_improvement:.4f}!")
    elif avg_improvement == 0:
        print("\nCONCLUSION: On average, SUBLIME features performed the same as original features.")
    else:
        print(f"\nCONCLUSION: On average, original features performed better than SUBLIME features by {-avg_improvement:.4f}.")
    
    # Count datasets where SUBLIME improved performance
    improved_count = (results_df['improvement'] > 0).sum()
    print(f"SUBLIME improved performance on {improved_count} out of {len(results_df)} datasets.")

def optimize_sublime_hyperparameters(dataset_name='iris', n_trials=30):
    """
    Use Optuna to find the best hyperparameters for SUBLIME on the Iris dataset.
    
    Args:
        dataset_name: Name of the dataset to use
        n_trials: Number of optimization trials
        
    Returns:
        dict: Dictionary with best hyperparameters
    """
    print("="*80)
    print(f"OPTIMIZING SUBLIME HYPERPARAMETERS FOR {dataset_name.upper()} DATASET")
    print("="*80)
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Get dataset loader
    datasets = get_available_datasets()
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")
        
    dataset_loader = datasets[dataset_name]
    
    # 1. Prepare the data
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test, feature_names = prepare_data(dataset_name, dataset_loader)
    
    # Get number of classes to set n_clusters for SUBLIME
    n_clusters = len(np.unique(y_train))
    
    # 2. Save training data as CSV for SUBLIME
    print("\nSaving training data as CSV...")
    data_file = save_data_as_csv(X_train, y_train, feature_names, dataset_name)
    
    # Define Optuna objective
    def objective(trial):
        # Sample hyperparameters
        hyperparams = {
            'epochs': trial.suggest_int('epochs', 500, 2000),
            'lr': trial.suggest_float('lr', 0.001, 0.01, log=True),
            'w_decay': trial.suggest_float('w_decay', 0.0001, 0.01, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'rep_dim': trial.suggest_categorical('rep_dim', [32, 64, 128]),
            'proj_dim': trial.suggest_categorical('proj_dim', [32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'maskfeat_rate_learner': trial.suggest_float('maskfeat_rate_learner', 0.1, 0.5),
            'maskfeat_rate_anchor': trial.suggest_float('maskfeat_rate_anchor', 0.1, 0.5),
            'dropedge_rate': trial.suggest_float('dropedge_rate', 0.1, 0.5),
        }
        
        # Create a unique output directory for this trial
        timestamp = int(time.time())
        trial_id = trial.number
        model_dir = f"sublime_model_{dataset_name}_trial{trial_id}_{timestamp}"
        
        try:
            # Train SUBLIME with these hyperparameters
            print(f"\nTrial {trial_id} - Testing hyperparameters:")
            for key, value in hyperparams.items():
                print(f"  {key}: {value}")
                
            train_sublime(data_file, n_clusters=n_clusters, output_dir=model_dir, hyperparams=hyperparams)
            
            # Extract features
            train_embeddings, test_embeddings = extract_features(model_dir, X_train, X_test, dataset_name, cache_dir="cache")
            
            # Evaluate the embeddings with XGBoost (simplified evaluation)
            clf = XGBClassifier(n_estimators=100, random_state=42)
            clf.fit(train_embeddings, y_train)
            sublime_preds = clf.predict(test_embeddings)
            sublime_acc = accuracy_score(y_test, sublime_preds)
            
            print(f"Trial {trial_id} - Accuracy: {sublime_acc:.4f}")
            
            return sublime_acc
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {str(e)}")
            return 0.0  # Return low score for failed trials
    
    # Create and run Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*80)
    print(f"BEST HYPERPARAMETERS FOR {dataset_name.upper()}")
    print("="*80)
    print(f"Best accuracy: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters to file
    with open(f"best_sublime_hyperparams_{dataset_name}.txt", "w") as f:
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nBest hyperparameters saved to best_sublime_hyperparams_{dataset_name}.txt")
    
    # Visualize parameter importance if plotly is available
    try:
        importance = optuna.visualization.plot_param_importances(study)
        importance.write_image(f"sublime_param_importance_{dataset_name}.png")
        print(f"Parameter importance plot saved to sublime_param_importance_{dataset_name}.png")
    except:
        print("Could not generate parameter importance plot. You may need to install plotly.")
    
    return study.best_params

if __name__ == "__main__":
    import sys
    
    # Check if hyperparameter tuning is requested
    if len(sys.argv) > 1 and sys.argv[1] == 'iris_hyperparameter_tuning':
        # Run hyperparameter tuning for Iris dataset
        n_trials = 30  # Default number of trials
        
        if len(sys.argv) > 2:
            try:
                n_trials = int(sys.argv[2])
            except ValueError:
                print(f"Warning: Could not parse {sys.argv[2]} as integer. Using default value of 30 trials.")
        
        print(f"Running SUBLIME hyperparameter tuning for Iris dataset with {n_trials} trials")
        best_params = optimize_sublime_hyperparameters(dataset_name='iris', n_trials=n_trials)
    else:
        # Run the regular pipeline
        main()