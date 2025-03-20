import argparse
import torch
import pandas as pd
import numpy as np
import os
import joblib
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

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

def extract_in_batches(X, model, graph_learner, features, adj, sparse, experiment, batch_size=16):
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
        
    Returns:
        numpy.ndarray: Extracted features
    """
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
    
    return np.array(all_embeddings)

def evaluate_features(X_test, test_embeddings, y_test, dataset_name, n_trials=50):
    """
    Train XGBoost classifiers on three different feature sets and compare performance
    
    Args:
        X_test: Original features
        test_embeddings: SUBLIME extracted features 
        y_test: Target labels
        dataset_name: Name of the dataset
        n_trials: Number of optimization trials for Optuna
        
    Returns:
        dict: Results dictionary with accuracies and hyperparameters
    """
    results = {}
    
    # Create concatenated features (original + SUBLIME)
    test_concat = np.hstack((X_test, test_embeddings))
    
    print(f"Original features shape: {X_test.shape}")
    print(f"SUBLIME features shape: {test_embeddings.shape}")
    print(f"Concatenated features shape: {test_concat.shape}")
    
    # Split the test data into training and validation sets (70/30 split)
    # We're using the test data for both training and testing since we're just evaluating the features
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=42, stratify=y_test)
    
    # Split the embeddings the same way
    train_embeddings = test_embeddings[:len(X_train)]
    val_embeddings = test_embeddings[len(X_train):]
    
    # Split the concatenated features
    train_concat = test_concat[:len(X_train)]
    val_concat = test_concat[len(X_train):]
    
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
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)
    
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
        model.fit(train_embeddings, y_train)
        preds = model.predict(val_embeddings)
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
        model.fit(train_concat, y_train)
        preds = model.predict(val_concat)
        return accuracy_score(y_val, preds)
    
    # Tune hyperparameters for original features
    print("\nOptimizing model for original features...")
    study_original = optuna.create_study(direction='maximize')
    study_original.optimize(original_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_original = study_original.best_params
    
    # Train XGBoost on original features with tuned hyperparameters
    print("\nTraining model on original features with best hyperparameters...")
    original_clf = XGBClassifier(**best_params_original)
    original_clf.fit(X_train, y_train)
    
    # Evaluate on original features
    original_preds = original_clf.predict(X_val)
    original_acc = accuracy_score(y_val, original_preds)
    print(f"Original features accuracy: {original_acc:.4f}")
    print(classification_report(y_val, original_preds))
    
    # Tune hyperparameters for SUBLIME features
    print("\nOptimizing model for SUBLIME features...")
    study_sublime = optuna.create_study(direction='maximize')
    study_sublime.optimize(sublime_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_sublime = study_sublime.best_params
    
    # Train XGBoost on extracted features with tuned hyperparameters
    print("\nTraining model on SUBLIME features with best hyperparameters...")
    sublime_clf = XGBClassifier(**best_params_sublime)
    sublime_clf.fit(train_embeddings, y_train)
    
    # Evaluate on extracted features
    sublime_preds = sublime_clf.predict(val_embeddings)
    sublime_acc = accuracy_score(y_val, sublime_preds)
    print(f"SUBLIME features accuracy: {sublime_acc:.4f}")
    print(classification_report(y_val, sublime_preds))
    
    # Tune hyperparameters for concatenated features
    print("\nOptimizing model for concatenated features...")
    study_concat = optuna.create_study(direction='maximize')
    study_concat.optimize(concat_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)
    
    best_params_concat = study_concat.best_params
    
    # Train XGBoost on concatenated features with tuned hyperparameters
    print("\nTraining model on concatenated features with best hyperparameters...")
    concat_clf = XGBClassifier(**best_params_concat)
    concat_clf.fit(train_concat, y_train)
    
    # Evaluate on concatenated features
    concat_preds = concat_clf.predict(val_concat)
    concat_acc = accuracy_score(y_val, concat_preds)
    print(f"Concatenated features accuracy: {concat_acc:.4f}")
    print(classification_report(y_val, concat_preds))
    
    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Feature importance comparison
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(original_clf.feature_importances_)), original_clf.feature_importances_)
    plt.title(f"Feature Importance (Original Features) - {dataset_name}")
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_original_feature_importance.png"))
    
    # Feature importance for SUBLIME features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sublime_clf.feature_importances_)), sublime_clf.feature_importances_)
    plt.title(f"Feature Importance (SUBLIME Features) - {dataset_name}")
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_sublime_feature_importance.png"))
    
    # Feature importance for concatenated features
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(concat_clf.feature_importances_)), concat_clf.feature_importances_)
    plt.title(f"Feature Importance (Concatenated Features) - {dataset_name}")
    plt.savefig(os.path.join(plots_dir, f"{dataset_name}_concat_feature_importance.png"))
    
    # Save results
    results = {
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
    
    # Save the detailed results to a CSV file
    results_df = pd.DataFrame([results])
    results_path = os.path.join(args.output_dir, f"{dataset_name}_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    return results

def load_and_process_test_data(test_csv, model_dir, target_column):
    """
    Load test data from CSV and preprocess it
    
    Args:
        test_csv: Path to the test CSV
        model_dir: Directory where the SUBLIME model is stored (for transformer)
        target_column: Name of the target column
        
    Returns:
        tuple: (X_test, y_test)
    """
    print(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv, delimiter='\t')
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset")
    
    y_test = df[target_column].values
    features_df = df.drop(columns=[target_column])
    
    # Preprocess the data using the same transformer
    X_test = preprocess_mixed_data(features_df, model_dir=model_dir, load_transformer=True)
    
    print(f"Test data loaded: {len(X_test)} samples with {X_test.shape[1]} features")
    print(f"Target distribution: {np.unique(y_test, return_counts=True)}")
    
    return X_test, y_test

def main(args):
    """Main function to run the evaluation pipeline"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load the test data
    X_test, y_test = load_and_process_test_data(args.test_csv, args.model_dir, args.target_column)
    
    # 2. Load the SUBLIME model
    print(f"\nLoading SUBLIME model from {args.model_dir}")
    experiment = Experiment(device)
    model, graph_learner, features, adj, sparse = experiment.load_model(input_dir=args.model_dir)
    print("Model loaded successfully!")
    
    # 3. Extract SUBLIME features for the test data
    print("\nExtracting SUBLIME features for test data...")
    test_embeddings = extract_in_batches(
        X_test, model, graph_learner, features, adj, sparse, experiment, batch_size=args.batch_size
    )
    print(f"Feature extraction complete. Extracted shape: {test_embeddings.shape}")
    
    # 4. Evaluate the features with XGBoost
    print("\nEvaluating features with XGBoost...")
    dataset_name = os.path.basename(args.test_csv).split('.')[0]
    results = evaluate_features(X_test, test_embeddings, y_test, dataset_name, n_trials=args.n_trials)
    
    # 5. Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Original features accuracy: {results['original_accuracy']:.4f}")
    print(f"SUBLIME features accuracy: {results['sublime_accuracy']:.4f}")
    print(f"Concatenated features accuracy: {results['concat_accuracy']:.4f}")
    print(f"SUBLIME vs Original improvement: {results['original_vs_sublime_improvement']:.4f}")
    print(f"Concatenated vs Original improvement: {results['concat_vs_original_improvement']:.4f}")
    print(f"Concatenated vs SUBLIME improvement: {results['concat_vs_sublime_improvement']:.4f}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SUBLIME features on a test dataset with a target")
    parser.add_argument('--test-csv', type=str, required=True, help='Path to the test CSV file')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory where the SUBLIME model is saved')
    parser.add_argument('--target-column', type=str, required=True, help='Name of the target column in the CSV')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for feature extraction')
    
    args = parser.parse_args()
    main(args) 