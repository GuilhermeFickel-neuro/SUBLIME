import argparse
import torch
import pandas as pd
import numpy as np
import time
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from experiment import Experiment
from utils import get_available_datasets, prepare_data, save_data_as_csv, extract_features

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
        lr=0.005,
        w_decay=0.005,
        hidden_dim=512,  # Reduced for smaller dataset
        rep_dim=128,      # Reduced for smaller dataset
        proj_dim=128,     # Reduced for smaller dataset
        dropout=0.3,
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
        output_dir=output_dir
    )
    
    # Override with provided hyperparameters if any
    if hyperparams:
        for key, value in hyperparams.items():
            setattr(args, key, value)
    
    # Create experiment and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(device)
    
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
    print(f"Using hyperparameters: epochs={args.epochs}, lr={args.lr}, w_decay={args.w_decay}, "
          f"hidden_dim={args.hidden_dim}, rep_dim={args.rep_dim}, proj_dim={args.proj_dim}, "
          f"dropout={args.dropout}, maskfeat_rate_learner={args.maskfeat_rate_learner}, "
          f"maskfeat_rate_anchor={args.maskfeat_rate_anchor}, dropedge_rate={args.dropedge_rate}")
    
    experiment.train(args, load_data_fn=load_csv_data)
    print(f"SUBLIME model trained and saved to {output_dir}")
    
    return output_dir

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
    
    datasets = get_available_datasets()
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
            'w_decay': trial.suggest_float('w_decay', 0.001, 0.01, log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'rep_dim': trial.suggest_categorical('rep_dim', [64, 128, 256]),
            'proj_dim': trial.suggest_categorical('proj_dim', [64, 128, 256]),
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
    # Check if 'iris_hyperparameter_tuning' is specified as a command line arg
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'iris_hyperparameter_tuning':
        # Run hyperparameter tuning for Iris dataset
        n_trials = 30
        if len(sys.argv) > 2:
            try:
                n_trials = int(sys.argv[2])
            except ValueError:
                pass
        
        print(f"Running hyperparameter tuning for Iris dataset with {n_trials} trials")
        best_params = optimize_sublime_hyperparameters(dataset_name='iris', n_trials=n_trials)
    else:
        # Run the original main function
        main()