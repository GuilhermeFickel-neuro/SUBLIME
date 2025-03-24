import argparse
import os
import optuna
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Import functions from existing scripts
from main import Experiment
from train_person_data import preprocess_mixed_data, load_person_data
from evaluate_sublime_on_test_data import extract_in_batches, preprocess_dataset_features
from xgboost import XGBClassifier

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial, args):
    """Optuna objective function to optimize SUBLIME hyperparameters"""
    
    # Define hyperparameters to tune
    params = {
        'lr': trial.suggest_float('lr', 0.001, 0.1, log=True),
        'hidden_dim': trial.suggest_int('hidden_dim', 32, 512),
        'rep_dim': trial.suggest_int('rep_dim', 16, 128),
        'proj_dim': trial.suggest_int('proj_dim', 16, 256),
        'dropout': trial.suggest_float('dropout', 0.01, 0.7),
        'nlayers': trial.suggest_int('nlayers', 1, 4),
        'maskfeat_rate_learner': trial.suggest_float('maskfeat_rate_learner', 0.1, 0.5),
        'maskfeat_rate_anchor': trial.suggest_float('maskfeat_rate_anchor', 0.1, 0.5),
        'dropedge_rate': trial.suggest_float('dropedge_rate', 0.1, 0.7),
        'k': trial.suggest_int('k', 10, 50),
        'gamma': trial.suggest_float('gamma', 0.5, 0.99),
        'w_decay': trial.suggest_float('w_decay', 0.0, 0.01),
        'type_learner': trial.suggest_categorical('type_learner', ['fgp', 'mlp', 'gnn', 'att']),
        'epochs': trial.suggest_int('epochs', 750, 3000),
    }
    
    # Create a unique output directory for this trial
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
    
    # Create a copy of the args to avoid modifying the original
    trial_args = argparse.Namespace(**vars(args))
    
    # Update trial_args with trial parameters
    for param, value in params.items():
        setattr(trial_args, param, value)
    trial_args.output_dir = trial_dir
    
    # Train the SUBLIME model
    experiment = Experiment(device)
    try:
        experiment.train(trial_args, load_data_fn=load_person_data)
    except Exception as e:
        print(f"Error during training: {e}")
        return float('-inf')  # Return a bad score if training fails
    
    # Evaluate using classification performance
    try:
        # Load the trained model
        model, graph_learner, features, adj, sparse = experiment.load_model(input_dir=trial_dir)
        
        # Load and preprocess neurolake data
        neurolake_df = pd.read_csv(args.neurolake_csv, delimiter='\t')
        
        # Load and preprocess dataset features for evaluation
        dataset_df = pd.read_csv(args.dataset_features_csv, delimiter='\t')
        
        # Ensure the datasets are aligned
        if len(neurolake_df) != len(dataset_df):
            raise ValueError("Neurolake data and dataset features must have the same number of rows!")
        
        # Filter out rows where target values are not 0 or 1
        if args.target_column in dataset_df.columns:
            valid_mask = dataset_df[args.target_column].isin([0, 1])
            dataset_df = dataset_df[valid_mask].reset_index(drop=True)
            neurolake_df = neurolake_df[valid_mask].reset_index(drop=True)
        
        # Process data
        X_neurolake = preprocess_mixed_data(neurolake_df, model_dir='sublime_models/', load_transformer=True)
        X_dataset, _, y = preprocess_dataset_features(dataset_df, args.target_column, fit_transform=True)
        
        # Extract SUBLIME features
        sublime_embeddings = extract_in_batches(
            X_neurolake, model, graph_learner, features, adj, sparse, experiment, 
            batch_size=args.batch_size
        )
        
        # Split data for evaluation
        X_train, X_val, sublime_train, sublime_val, y_train, y_val = train_test_split(
            X_dataset, sublime_embeddings, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Create concatenated features (dataset features + SUBLIME)
        concat_train = np.hstack((X_train, sublime_train))
        concat_val = np.hstack((X_val, sublime_val))
        
        # Define nested XGBoost optimization function
        def xgb_objective(xgb_trial):
            xgb_params = {
                'n_estimators': xgb_trial.suggest_int('n_estimators', 50, 500),
                'max_depth': xgb_trial.suggest_int('max_depth', 3, 10),
                'learning_rate': xgb_trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': xgb_trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': xgb_trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': xgb_trial.suggest_int('min_child_weight', 1, 10),
                'gamma': xgb_trial.suggest_float('gamma', 0, 5),
                'random_state': 42
            }
            
            # Train XGBoost model
            clf = XGBClassifier(**xgb_params)
            clf.fit(concat_train, y_train)
            
            # Evaluate using AUC-ROC
            preds_proba = clf.predict_proba(concat_val)[:, 1]
            return roc_auc_score(y_val, preds_proba)
        
        # Create nested Optuna study for XGBoost
        xgb_study = optuna.create_study(direction='maximize')
        # Run fewer trials for the nested optimization to save time
        xgb_study.optimize(xgb_objective, n_trials=args.xgb_n_trials)
        
        # Get best XGBoost parameters and score
        best_xgb_params = xgb_study.best_params
        best_xgb_auc = xgb_study.best_value
        
        # Train final XGBoost model with best parameters
        best_clf = XGBClassifier(**best_xgb_params)
        best_clf.fit(concat_train, y_train)
        
        # Calculate KS statistic for the best model
        best_preds_proba = best_clf.predict_proba(concat_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, best_preds_proba)
        best_ks = max(tpr - fpr)
        
        # Save the best XGBoost model
        joblib.dump(best_clf, os.path.join(trial_dir, 'best_xgb_model.joblib'))
        
        # Save XGBoost hyperparameters
        with open(os.path.join(trial_dir, 'best_xgb_params.txt'), 'w') as f:
            f.write(f"Best XGBoost AUC-ROC: {best_xgb_auc}\n")
            f.write(f"Best XGBoost KS: {best_ks}\n")
            f.write("Best XGBoost hyperparameters:\n")
            for key, value in best_xgb_params.items():
                f.write(f"  {key}: {value}\n")
        
        # Save the score for this trial
        with open(os.path.join(trial_dir, 'score.txt'), 'w') as f:
            f.write(f"SUBLIME + XGBoost AUC-ROC: {best_xgb_auc}\n")
            f.write(f"SUBLIME + XGBoost KS: {best_ks}")
        
        return best_xgb_auc
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return float('-inf')  # Return a bad score if evaluation fails

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Tune SUBLIME hyperparameters with Optuna")
    
    # Basic settings
    parser.add_argument('--dataset', type=str, required=True, help='Path to the person data CSV for training SUBLIME')
    parser.add_argument('--neurolake-csv', type=str, required=True, help='Path to the neurolake CSV for evaluation')
    parser.add_argument('--dataset-features-csv', type=str, required=True, help='Path to the dataset features CSV for evaluation')
    parser.add_argument('--target-column', type=str, default='alvo', help='Name of the target column in dataset features')
    parser.add_argument('--output-dir', type=str, default='sublime_tuning_results', help='Directory to save results')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials for SUBLIME')
    parser.add_argument('--xgb-n-trials', type=int, default=30, help='Number of Optuna trials for XGBoost per SUBLIME model')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for feature extraction')
    
    # Fixed SUBLIME parameters (can be overridden by Optuna)
    parser.add_argument('-ntrials', type=int, default=20)
    parser.add_argument('-sparse', type=int, default=1)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference")
    parser.add_argument('-eval_freq', type=int, default=500)
    parser.add_argument('-downstream_task', type=str, default='clustering')
    parser.add_argument('-n_clusters', type=int, default=5)
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-contrast_batch_size', type=int, default=10000)
    parser.add_argument('-sim_function', type=str, default='cosine')
    parser.add_argument('-activation_learner', type=str, default='relu')
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)
    parser.add_argument('-verbose', type=int, default=0)
    parser.add_argument('-save_model', type=int, default=1)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the study
    study = optuna.create_study(direction='maximize', 
                               study_name='sublime_hyperparameter_tuning')
    
    # Run the optimization
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    # Get the best trial
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best AUC-ROC: {best_trial.value}")
    print("Best SUBLIME hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save the best hyperparameters
    best_params_path = os.path.join(args.output_dir, 'best_hyperparameters.txt')
    with open(best_params_path, 'w') as f:
        f.write(f"Best trial: {best_trial.number}\n")
        f.write(f"Best AUC-ROC: {best_trial.value}\n")
        f.write("Best SUBLIME hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    # Save the study for later analysis
    joblib.dump(study, os.path.join(args.output_dir, 'optuna_study.pkl'))
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"Best hyperparameters saved to {best_params_path}")
    print(f"To use the best model, check the directory: {os.path.join(args.output_dir, f'trial_{best_trial.number}')}")

if __name__ == "__main__":
    main() 