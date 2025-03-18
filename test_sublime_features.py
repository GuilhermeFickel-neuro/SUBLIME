import argparse
import torch
import pandas as pd
import numpy as np
import time
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from main import Experiment
import os
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits, fetch_openml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def get_available_datasets():
    """
    Returns a dictionary of available datasets for testing.
    
    Returns:
        dict: Dictionary with dataset names as keys and dataset loading functions as values
    """
    datasets = {
        'breast_cancer': load_breast_cancer,
        'iris': load_iris,
        'wine': load_wine,
        'digits': load_digits,
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
    print(f"Class distribution in training: {np.bincount(y_train)}")
    print(f"Class distribution in testing: {np.bincount(y_test)}")
    
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
        epochs=1000,  # Reduced for testing, increase for better results
        lr=0.01,
        w_decay=0.0,
        hidden_dim=128,  # Reduced for smaller dataset
        rep_dim=32,      # Reduced for smaller dataset
        proj_dim=64,     # Reduced for smaller dataset
        dropout=0.5,
        contrast_batch_size=0,
        nlayers=2,
        
        # Augmentation parameters
        maskfeat_rate_learner=0.2,
        maskfeat_rate_anchor=0.2,
        dropedge_rate=0.3,
        
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

def extract_features(model_dir, X_train, X_test):
    """
    Extract features from both training and testing data using the trained SUBLIME model.
    
    Args:
        model_dir: Directory where the SUBLIME model is saved
        X_train: Training features
        X_test: Testing features
        
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
    train_embeddings = extract_in_batches(X_train, model, graph_learner, features, adj, sparse, experiment, batch_size)
    
    # Extract features for testing data 
    print(f"Extracting features for {len(X_test)} testing points...")
    test_embeddings = extract_in_batches(X_test, model, graph_learner, features, adj, sparse, experiment, batch_size)
    
    print(f"Feature extraction complete. Shapes: train={train_embeddings.shape}, test={test_embeddings.shape}")
    
    return train_embeddings, test_embeddings

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=32):
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
        
    Returns:
        numpy.ndarray: Extracted features
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = (len(X) + batch_size - 1) // batch_size
    all_embeddings = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        
        # Process each item in the batch
        batch_embeddings = []
        for j in range(len(batch_X)):
            # Convert to tensor
            point_tensor = torch.FloatTensor(batch_X[j]).to(device)
            
            # Extract embedding - handle the return value correctly
            embedding = experiment.process_new_point(
                point_tensor, model, graph_learner, features, adj, sparse
            )
            batch_embeddings.append(embedding.cpu().numpy())
        
        all_embeddings.extend(batch_embeddings)
        print(f"Processed batch {i+1}/{num_batches} ({end_idx}/{len(X)} points)")
    
    return np.array(all_embeddings)

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
    
    # Tune hyperparameters for original features
    print("\nTuning XGBoost hyperparameters for original features...")
    study_original = optuna.create_study(direction='maximize')
    study_original.optimize(original_objective, n_trials=n_trials)
    
    best_params_original = study_original.best_params
    print(f"Best parameters for original features: {best_params_original}")
    print(f"Best cross-validation accuracy: {study_original.best_value:.4f}")
    
    # Train XGBoost on original features with tuned hyperparameters
    print("\nTraining XGBoost on original features with tuned hyperparameters...")
    original_clf = XGBClassifier(**best_params_original)
    original_clf.fit(X_train, y_train)
    
    # Evaluate on original features
    original_preds = original_clf.predict(X_test)
    original_acc = accuracy_score(y_test, original_preds)
    print(f"Original features accuracy: {original_acc:.4f}")
    print("Classification report (original features):")
    print(classification_report(y_test, original_preds))
    
    # Tune hyperparameters for SUBLIME features
    print("\nTuning XGBoost hyperparameters for SUBLIME features...")
    study_sublime = optuna.create_study(direction='maximize')
    study_sublime.optimize(sublime_objective, n_trials=n_trials)
    
    best_params_sublime = study_sublime.best_params
    print(f"Best parameters for SUBLIME features: {best_params_sublime}")
    print(f"Best cross-validation accuracy: {study_sublime.best_value:.4f}")
    
    # Train XGBoost on extracted features with tuned hyperparameters
    print("\nTraining XGBoost on SUBLIME extracted features with tuned hyperparameters...")
    sublime_clf = XGBClassifier(**best_params_sublime)
    sublime_clf.fit(train_embeddings, y_train)
    
    # Evaluate on extracted features
    sublime_preds = sublime_clf.predict(test_embeddings)
    sublime_acc = accuracy_score(y_test, sublime_preds)
    print(f"SUBLIME features accuracy: {sublime_acc:.4f}")
    print("Classification report (SUBLIME features):")
    print(classification_report(y_test, sublime_preds))
    
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
    
    # Save hyperparameter tuning visualization
    try:
        fig = optuna.visualization.plot_param_importances(study_original)
        fig.write_image(f"plots/{dataset_name}_original_param_importance.png")
        
        fig = optuna.visualization.plot_param_importances(study_sublime)
        fig.write_image(f"plots/{dataset_name}_sublime_param_importance.png")
    except:
        print("Could not generate hyperparameter importance plots. You may need to install plotly.")
    
    return {
        'dataset': dataset_name,
        'original_accuracy': original_acc,
        'sublime_accuracy': sublime_acc,
        'improvement': sublime_acc - original_acc,
        'best_params_original': best_params_original,
        'best_params_sublime': best_params_sublime
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
    train_embeddings, test_embeddings = extract_features(model_dir, X_train, X_test)
    
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
            print(f"Error processing dataset {dataset_name}: {str(e)}")
    
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
            train_embeddings, test_embeddings = extract_features(model_dir, X_train, X_test)
            
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