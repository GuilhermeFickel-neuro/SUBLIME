import argparse
import os
import numpy as np
import pandas as pd
import torch
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from main import Experiment
from model import GCL
import torch.nn as nn

def load_dataset(dataset_name='iris'):
    """Load the specified dataset
    
    Args:
        dataset_name: Name of the dataset to load ('iris', 'breast_cancer', 'wine', 'adult')
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    # Define column names and file path based on dataset
    if dataset_name == 'iris':
        feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        file_path = 'data/iris_train.csv'
    elif dataset_name == 'breast_cancer':
        file_path = 'data/breast_cancer_train.csv'
        feature_names = None  # Will be extracted from the data
    elif dataset_name == 'wine':
        file_path = 'data/wine_train.csv'
        feature_names = None  # Will be extracted from the data
    elif dataset_name == 'adult':
        # Adult Census Income dataset
        feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                         'marital-status', 'occupation', 'relationship', 'race', 'sex',
                         'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
        file_path = 'adult.data'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load the dataset
    print(f"Loading {dataset_name} dataset...")
    if dataset_name == 'adult':
        # Adult dataset has specific format with space after comma
        data = pd.read_csv(file_path, header=None, sep=', ', engine='python')
        # Assign column names
        data.columns = feature_names + ['income']
    else:
        data = pd.read_csv(file_path)
    
    # Extract feature names if not provided
    if feature_names is None:
        if dataset_name in ['breast_cancer', 'wine']:
            # Assuming the last column is the target
            feature_names = data.columns[:-1].tolist()
    
    # Create labels based on dataset
    if dataset_name == 'iris':
        # For Iris, use first column for binary classification
        labels = (data.iloc[:, 0] > 0).astype(int).values
        # Features are all columns except the first
        X = data.iloc[:, 1:].values
    elif dataset_name == 'breast_cancer':
        # For breast cancer, assume last column is binary target
        labels = data.iloc[:, -1].values
        # Ensure binary labels are 0 and 1
        labels = (labels > 0).astype(int)
        # Features are all columns except the last
        X = data.iloc[:, :-1].values
    elif dataset_name == 'wine':
        # For wine, convert multiclass to binary (class 1 vs rest)
        labels = data.iloc[:, -1].values
        labels = (labels == 1).astype(int)
        # Features are all columns except the last
        X = data.iloc[:, :-1].values
    elif dataset_name == 'adult':
        # For adult dataset, convert income to binary target (>50K = 1, <=50K = 0)
        labels = (data['income'] == '>50K').astype(int).values
        
        # Handle categorical features
        # Select only numerical features for simplicity
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                             'capital-loss', 'hours-per-week']
        X = data[numerical_features].values
    
    # Normalize the features to prevent extreme values
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data into train, val, test (same as original)
    X_train, X_temp, y_train, y_temp = train_test_split(X, labels, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Validate labels to ensure they're properly encoded as 0 and 1
    for label_set in [y_train, y_val, y_test]:
        unique_values = np.unique(label_set)
        print(f"Unique label values: {unique_values}")
        if not all(val in [0, 1] for val in unique_values):
            print("Warning: Labels contain values other than 0 and 1, converting...")
            label_set = (label_set > 0).astype(int)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# Helper function to convert matrices to torch tensors
def convert_to_torch_tensor(data):
    """Convert to torch tensors"""
    return torch.FloatTensor(data)

def train_sublime_model(X_train, use_arcface=False, arcface_weight=1.0, output_dir='sublime_models'):
    """Train a SUBLIME model with or without ArcFace
    
    Args:
        X_train: Training data
        use_arcface: Whether to use ArcFace loss
        arcface_weight: Weight for ArcFace loss
        output_dir: Directory to save the model
        
    Returns:
        Path to the saved model
    """
    # Debug information about input data
    print(f"X_train type: {type(X_train)}")
    print(f"X_train shape: {X_train.shape}")
    
    # Create output directory with a unique name based on configuration
    if use_arcface:
        model_dir = f"{output_dir}/arcface_w{arcface_weight}"
    else:
        model_dir = f"{output_dir}/no_arcface"
    
    os.makedirs(model_dir, exist_ok=True)
    
    # Convert to dense tensor format
    print("Converting to dense PyTorch tensor format")
    features = convert_to_torch_tensor(X_train)
    
    # Create an identity adjacency matrix
    n_samples = features.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup argument parser with default values
    parser = argparse.ArgumentParser()
    # Core parameters
    parser.add_argument('-ntrials', type=int, default=1)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference")
    parser.add_argument('-eval_freq', type=int, default=1000000)
    parser.add_argument('-downstream_task', type=str, default='clustering')
    parser.add_argument('-n_clusters', type=int, default=2)  # Binary classification

    parser.add_argument('-use_one_cycle', type=int, default=1,
                        help='Whether to use OneCycleLR scheduler (0=disabled, 1=enabled)')
    parser.add_argument('-one_cycle_pct_start', type=float, default=0.3,
                        help='Percentage of cycle spent increasing learning rate (default: 0.3)')
    parser.add_argument('-one_cycle_div_factor', type=float, default=25.0,
                        help='Initial learning rate is max_lr/div_factor (default: 25.0)')
    parser.add_argument('-one_cycle_final_div_factor', type=float, default=10000.0,
                        help='Final learning rate is max_lr/(div_factor*final_div_factor) (default: 10000.0)')
    
    # GCL Module parameters
    parser.add_argument('-epochs', type=int, default=5000)  # Reduced epochs for faster training
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0001)
    parser.add_argument('-hidden_dim', type=int, default=64)  # Reduced for smaller dataset
    parser.add_argument('-rep_dim', type=int, default=32)  # Reduced for smaller dataset
    parser.add_argument('-proj_dim', type=int, default=32)  # Reduced for smaller dataset
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    
    # Augmentation parameters
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    
    # GSL Module parameters
    parser.add_argument('-type_learner', type=str, default='fgp')
    parser.add_argument('-k', type=int, default=10)  # Reduced for smaller dataset
    parser.add_argument('-sim_function', type=str, default='cosine')
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu')
    
    # Structure bootstrapping parameters
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)
    
    # ArcFace parameters
    parser.add_argument('-use_arcface', type=int, default=1 if use_arcface else 0)
    parser.add_argument('-arcface_scale', type=float, default=30.0)
    parser.add_argument('-arcface_margin', type=float, default=0.5)
    parser.add_argument('-arcface_weight', type=float, default=arcface_weight)
    
    # Other parameters
    parser.add_argument('-verbose', type=int, default=1)
    parser.add_argument('-save_model', type=int, default=1)
    parser.add_argument('-output_dir', type=str, default=model_dir)
    
    args = parser.parse_args([])  # Empty list to avoid reading command line args
    
    # Custom data loading function for our specific case
    def load_data_fn(args):
        # Create empty labels (indices will be created automatically for ArcFace)
        labels = None
        # Create placeholder masks (not used)
        train_mask = torch.zeros(n_samples, dtype=torch.bool)
        val_mask = torch.zeros(n_samples, dtype=torch.bool)
        test_mask = torch.zeros(n_samples, dtype=torch.bool)
        # Create adjacency matrix
        adj = torch.eye(n_samples).to(device)
        # Get feature dimension
        nfeats = features.shape[1]
        # Return in expected format
        return features, nfeats, labels, args.n_clusters, train_mask, val_mask, test_mask, adj
    
    # Train the model
    experiment = Experiment(device=device)
    experiment.train(args, load_data_fn=load_data_fn)
    
    return model_dir

def extract_features_using_process_new_point(model_path, X_samples):
    """Extract features for a set of samples using process_new_point
    
    Args:
        model_path (str): Path to the model directory
        X_samples (numpy.ndarray): Samples to extract features for
        
    Returns:
        numpy.ndarray: Normalized embeddings for the samples
    """
    # Initialize experiment
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    experiment = Experiment(device=device)
    
    # Special handling for ArcFace models - modify the model class directly
    # to avoid the 'arcface.weight' error during loading
    if 'arcface' in model_path:
        # Patch GCL class temporarily to ignore arcface parameters during loading
        original_load_state_dict = torch.nn.Module.load_state_dict
        
        def patched_load_state_dict(self, state_dict, strict=True):
            # Filter out arcface parameters if they're not in the model
            if not hasattr(self, 'arcface') and any(k.startswith('arcface.') for k in state_dict.keys()):
                filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('arcface.')}
                return original_load_state_dict(self, filtered_state_dict, False)
            return original_load_state_dict(self, state_dict, strict)
        
        # Apply the patch
        torch.nn.Module.load_state_dict = patched_load_state_dict
    
    try:
        # Load the model
        model, graph_learner, features, adj, sparse = experiment.load_model(model_path)
        model.eval()
        graph_learner.eval()
        
        # Convert samples to tensor
        X_tensor = convert_to_torch_tensor(X_samples)
        
        # Performance optimization: use torch.no_grad() to reduce memory usage during inference
        embeddings = []
        with torch.no_grad():
            for i in range(X_samples.shape[0]):
                if i % 10 == 0:
                    print(f"Processing sample {i+1}/{X_samples.shape[0]}")
                    
                try:
                    # Get the embedding for the current sample
                    embedding = experiment.process_new_point(
                        X_tensor[i], model, graph_learner, features, adj, sparse, 0
                    )
                    
                    # Error handling: Check for NaN values or zero-length embeddings
                    if torch.isnan(embedding).any() or torch.all(embedding == 0):
                        print(f"Warning: Invalid embedding detected for sample {i}. Using zeros instead.")
                        # Create a zero tensor with the same shape as valid embeddings
                        if len(embeddings) > 0:
                            embedding = torch.zeros_like(embeddings[0])
                        else:
                            # If this is the first sample, we'll use it but print a warning
                            print("Warning: First sample has invalid embedding. Results may be affected.")
                    
                    # Normalize the embedding using L2 normalization
                    # This ensures all vectors have unit norm for better downstream model comparison
                    # Adding a small epsilon to prevent division by zero for near-zero vectors
                    normalized_embedding = torch.nn.functional.normalize(embedding, p=2, dim=0, eps=1e-12)
                    
                    # Convert to NumPy and append to results
                    embeddings.append(normalized_embedding.cpu().numpy())
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    # If there are any embeddings already, use zeros with matching dimensions
                    if len(embeddings) > 0:
                        embeddings.append(np.zeros_like(embeddings[0]))
                    else:
                        # If this is the first sample and it failed, raise the exception
                        raise RuntimeError(f"Failed to process first sample: {e}")
        
        # Convert list of embeddings to a numpy array
        if not embeddings:
            raise ValueError("No valid embeddings were generated")
            
        return np.array(embeddings)
    
    finally:
        # Restore original load_state_dict if we patched it
        if 'arcface' in model_path:
            torch.nn.Module.load_state_dict = original_load_state_dict

def objective(trial, X_train, X_val, y_train, y_val, embeddings=None):
    """Optuna objective function for XGBoost hyperparameter tuning
    
    Args:
        trial: Optuna trial
        X_train, X_val, y_train, y_val: Training and validation data
        embeddings: Optional embeddings to append to features
        
    Returns:
        Validation accuracy
    """
    # Combine features with embeddings if provided
    if embeddings is not None:
        X_train_with_emb = np.hstack([X_train, embeddings['train']])
        X_val_with_emb = np.hstack([X_val, embeddings['val']])
    else:
        X_train_with_emb = X_train
        X_val_with_emb = X_val
    
    # Ensure labels are binary (0, 1)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)
    
    # Define hyperparameters to optimize with safer values
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'base_score': 0.5,  # Set explicitly to avoid issues
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'lambda': trial.suggest_float('lambda', 0.01, 0.5, log=True),
        'alpha': trial.suggest_float('alpha', 0.01, 0.5, log=True),
    }
    
    # Add different hyperparameters depending on the booster
    if param['booster'] == 'gbtree':
        param['max_depth'] = trial.suggest_int('max_depth', 3, 6)
        param['eta'] = trial.suggest_float('eta', 0.03, 0.2, log=True)
        param['gamma'] = trial.suggest_float('gamma', 0.01, 0.5, log=True)
    
    # Print distribution of labels for debugging
    print(f"Train labels distribution: {np.bincount(y_train_binary)}")
    print(f"Validation labels distribution: {np.bincount(y_val_binary)}")
    
    # Create and train the model
    dtrain = xgb.DMatrix(X_train_with_emb, label=y_train_binary)
    dval = xgb.DMatrix(X_val_with_emb, label=y_val_binary)
    
    # Use safer hyperparameters for initial training
    try:
        model = xgb.train(param, dtrain, num_boost_round=30, evals=[(dval, 'validation')],
                         early_stopping_rounds=5, verbose_eval=False)
        
        # Predict and calculate accuracy
        y_pred = model.predict(dval)
        y_pred_binary = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_val_binary, y_pred_binary)
        
        return accuracy
    except Exception as e:
        print(f"Error in XGBoost training: {e}")
        # Return a very low score to deprioritize this parameter set
        return 0.0

def train_xgboost_models(X_train, X_val, X_test, y_train, y_val, y_test,
                         unsupervised_embeddings=None, arcface_embeddings=None,
                         n_trials=10):  # Reduced trials for faster execution
    """Train XGBoost models with different feature sets and evaluate them
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target vectors
        unsupervised_embeddings: Dict with 'train', 'val', 'test' embeddings from unsupervised model
        arcface_embeddings: Dict with 'train', 'val', 'test' embeddings from ArcFace model
        n_trials: Number of Optuna trials for hyperparameter tuning
        
    Returns:
        Dictionary with evaluation results
    """
    # Ensure labels are binary (0, 1)
    y_train_binary = (y_train > 0).astype(int)
    y_val_binary = (y_val > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)
    
    print(f"Label distributions - Train: {np.bincount(y_train_binary)}, Val: {np.bincount(y_val_binary)}, Test: {np.bincount(y_test_binary)}")
    
    results = {}
    
    # Helper function for XGBoost model training with error handling
    def train_and_evaluate_xgboost(X_train_data, X_val_data, X_test_data,
                                  y_train_data, y_val_data, y_test_data,
                                  params, model_name):
        try:
            # Always set base_score to avoid issues
            params['base_score'] = 0.5
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'auc'
            
            dtrain = xgb.DMatrix(X_train_data, label=y_train_data)
            dval = xgb.DMatrix(X_val_data, label=y_val_data)
            dtest = xgb.DMatrix(X_test_data, label=y_test_data)
            
            model = xgb.train(params, dtrain, num_boost_round=30,
                            evals=[(dval, 'validation')], early_stopping_rounds=5, verbose_eval=False)
            
            # Evaluate
            y_pred = model.predict(dtest)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            model_results = {
                'accuracy': accuracy_score(y_test_data, y_pred_binary),
                'precision': precision_score(y_test_data, y_pred_binary),
                'recall': recall_score(y_test_data, y_pred_binary),
                'f1': f1_score(y_test_data, y_pred_binary),
                'auc': roc_auc_score(y_test_data, y_pred),
                'model': model,
                'best_params': params
            }
            
            print(f"Accuracy with {model_name}: {model_results['accuracy']:.4f}")
            return model_results
            
        except Exception as e:
            print(f"Error training XGBoost with {model_name}: {e}")
            # Return a default result with zeros for metrics
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'auc': 0.0,
                'model': None,
                'best_params': params,
                'error': str(e)
            }
    
    # 1. Train with original features only
    print("\nTraining XGBoost with original features only:")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, X_val, y_train_binary, y_val_binary), n_trials=n_trials)
    
    # Train final model with best parameters
    if hasattr(study, 'best_params'):
        best_params = study.best_params
        results['original'] = train_and_evaluate_xgboost(
            X_train, X_val, X_test,
            y_train_binary, y_val_binary, y_test_binary,
            best_params, "original features"
        )
    else:
        print("Failed to find best parameters for original features model")
        results['original'] = {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
            'model': None, 'best_params': {}, 'error': "Failed to find best parameters"
        }
    
    # 2. Train with original features + unsupervised embeddings
    if unsupervised_embeddings is not None:
        print("\nTraining XGBoost with original features + unsupervised embeddings:")
        embeddings = {
            'train': unsupervised_embeddings['train'],
            'val': unsupervised_embeddings['val']
        }
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train_binary, y_val_binary, embeddings), n_trials=n_trials)
        
        # Train final model with best parameters if found
        if hasattr(study, 'best_params'):
            best_params = study.best_params
            
            X_train_with_emb = np.hstack([X_train, unsupervised_embeddings['train']])
            X_val_with_emb = np.hstack([X_val, unsupervised_embeddings['val']])
            X_test_with_emb = np.hstack([X_test, unsupervised_embeddings['test']])
            
            results['unsupervised'] = train_and_evaluate_xgboost(
                X_train_with_emb, X_val_with_emb, X_test_with_emb,
                y_train_binary, y_val_binary, y_test_binary,
                best_params, "original features + unsupervised embeddings"
            )
        else:
            print("Failed to find best parameters for unsupervised embeddings model")
            results['unsupervised'] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
                'model': None, 'best_params': {}, 'error': "Failed to find best parameters"
            }
    
    # 3. Train with original features + arcface embeddings
    if arcface_embeddings is not None:
        print("\nTraining XGBoost with original features + ArcFace embeddings:")
        embeddings = {
            'train': arcface_embeddings['train'],
            'val': arcface_embeddings['val']
        }
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train_binary, y_val_binary, embeddings), n_trials=n_trials)
        
        # Train final model with best parameters if found
        if hasattr(study, 'best_params'):
            best_params = study.best_params
            
            X_train_with_emb = np.hstack([X_train, arcface_embeddings['train']])
            X_val_with_emb = np.hstack([X_val, arcface_embeddings['val']])
            X_test_with_emb = np.hstack([X_test, arcface_embeddings['test']])
            
            results['arcface'] = train_and_evaluate_xgboost(
                X_train_with_emb, X_val_with_emb, X_test_with_emb,
                y_train_binary, y_val_binary, y_test_binary,
                best_params, "original features + ArcFace embeddings"
            )
        else:
            print("Failed to find best parameters for ArcFace embeddings model")
            results['arcface'] = {
                'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0,
                'model': None, 'best_params': {}, 'error': "Failed to find best parameters"
            }
    
    return results

def plot_results(results, dataset_name='iris'):
    """Plot comparison of model performance
    
    Args:
        results: Dictionary with evaluation results
        dataset_name: Name of the dataset used
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    models = list(results.keys())
    
    # Filter out models that failed (those with None or 0.0 in all metrics)
    valid_models = []
    for model_name in models:
        if results[model_name]['model'] is not None or any(results[model_name][metric] > 0 for metric in metrics):
            valid_models.append(model_name)
    
    if not valid_models:
        print("No valid models to plot")
        return f"No valid results for {dataset_name}"
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bar positions
    bar_width = 0.2
    positions = np.arange(len(metrics))
    
    # Plot bars for each valid model
    for i, model_name in enumerate(valid_models):
        values = [results[model_name][metric] for metric in metrics]
        ax.bar(positions + i * bar_width, values, bar_width, label=model_name)
    
    # Set labels and title
    ax.set_ylabel('Score')
    ax.set_title(f'Performance Comparison of Models on {dataset_name.title()} Dataset')
    ax.set_xticks(positions + bar_width * (len(valid_models) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add dataset information to plot if available
    if 'feature_count' in results.get(valid_models[0], {}) and 'sample_count' in results.get(valid_models[0], {}):
        ax.text(0.02, 0.02, f"Dataset: {dataset_name}\nFeatures: {results[valid_models[0]]['feature_count']}\nSamples: {results[valid_models[0]]['sample_count']}",
                transform=ax.transAxes, fontsize=9, verticalalignment='bottom')
    
    # Save the figure
    plt.tight_layout()
    filename = f'model_comparison_{dataset_name}.png'
    plt.savefig(filename)
    plt.close()
    
    return filename

def main(dataset_name='iris'):
    """Main function to run the experiment
    
    Args:
        dataset_name: Name of the dataset to use ('iris', 'breast_cancer', 'wine')
    """
    # Load dataset
    print(f"Loading {dataset_name} dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_dataset(dataset_name)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Debug info about data
    print("\nDebug information about training data:")
    print(f"X_train type: {type(X_train)}")
    print(f"X_train shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Number of samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
    
    # Track feature and sample counts for reporting
    total_samples = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    feature_count = X_train.shape[1]
    
    # Create output directory with dataset name
    output_dir = f'sublime_models_{dataset_name}'
    
    # Initialize embedding dictionaries
    unsupervised_embeddings = None
    arcface_embeddings = None
    
    # Train SUBLIME models with error handling
    try:
        # Train unsupervised model
        print(f"\nTraining unsupervised SUBLIME model for {dataset_name} dataset...")
        unsupervised_model_dir = train_sublime_model(X_train, use_arcface=False, output_dir=output_dir)
        
        # Extract embeddings
        print("\nExtracting embeddings from unsupervised model...")
        unsupervised_train_embeddings = extract_features_using_process_new_point(unsupervised_model_dir, X_train)
        unsupervised_val_embeddings = extract_features_using_process_new_point(unsupervised_model_dir, X_val)
        unsupervised_test_embeddings = extract_features_using_process_new_point(unsupervised_model_dir, X_test)
        
        unsupervised_embeddings = {
            'train': unsupervised_train_embeddings,
            'val': unsupervised_val_embeddings,
            'test': unsupervised_test_embeddings
        }
        
        # Train ArcFace model
        print(f"\nTraining SUBLIME model with ArcFace for {dataset_name} dataset...")
        arcface_model_dir = train_sublime_model(X_train, use_arcface=True, arcface_weight=1.0, output_dir=output_dir)
        
        # Extract ArcFace embeddings
        print("\nExtracting embeddings from ArcFace model...")
        arcface_train_embeddings = extract_features_using_process_new_point(arcface_model_dir, X_train)
        arcface_val_embeddings = extract_features_using_process_new_point(arcface_model_dir, X_val)
        arcface_test_embeddings = extract_features_using_process_new_point(arcface_model_dir, X_test)
        
        arcface_embeddings = {
            'train': arcface_train_embeddings,
            'val': arcface_val_embeddings,
            'test': arcface_test_embeddings
        }
        
    except Exception as e:
        print(f"Error in training SUBLIME models: {e}")
        import traceback
        traceback.print_exc()
        print("Continuing with XGBoost models using original features only...")
    
    # Train XGBoost models and evaluate
    print("\nTraining and evaluating XGBoost models...")
    results = train_xgboost_models(
        X_train, X_val, X_test, y_train, y_val, y_test,
        unsupervised_embeddings, arcface_embeddings,
        n_trials=10  # Reduced for faster results
    )
    
    # Add dataset information to results for plotting
    for model_name in results:
        results[model_name]['feature_count'] = feature_count
        results[model_name]['sample_count'] = total_samples
    
    # Plot results
    output_file = plot_results(results, dataset_name)
    
    # Print summary
    print("\nPerformance Summary:")
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    models = list(results.keys())
    
    # Create a table
    print("\n" + "-" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Features: {feature_count}")
    print(f"Total Samples: {total_samples}")
    print("-" * 80)
    header = f"{'Metric':<10}"
    for model in models:
        header += f"{model:<25}"
    print(header)
    print("-" * 80)
    
    for metric in metrics:
        row = f"{metric:<10}"
        for model in models:
            # Check if the model has valid metrics
            if 'error' in results[model] and results[model]['model'] is None:
                row += f"{'ERROR':25}"
            else:
                row += f"{results[model][metric]:.4f}{'':20}"
        print(row)
    
    print("-" * 80)
    print(f"\nResults have been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SUBLIME experiments on different datasets.')
    parser.add_argument('--dataset', type=str, default='iris',
                        choices=['iris', 'breast_cancer', 'wine', 'adult'],
                        help='Dataset to use (iris, breast_cancer, wine, or adult)')
    args = parser.parse_args()
    
    try:
        print(f"Running experiment with {args.dataset} dataset...")
        main(args.dataset)
    except Exception as e:
        print(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()