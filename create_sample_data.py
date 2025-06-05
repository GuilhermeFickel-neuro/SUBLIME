import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

def generate_sample_data(n_samples=1000, n_features=10, seed=42):
    """
    Generate sample person data with features (no labels).
    
    Args:
        n_samples: Number of persons/rows
        n_features: Number of features per person
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with person features and saves it to CSV
    """
    np.random.seed(seed)
    
    # Generate random features
    features = np.random.randn(n_samples, n_features)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Create DataFrame
    df = pd.DataFrame(features, columns=feature_names)
    
    # Normalize features
    scaler = StandardScaler()
    df[feature_names] = scaler.fit_transform(df[feature_names])
    
    # Save to CSV
    df.to_csv('person_data.csv', index=False)
    return df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate sample person data')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Number of samples to generate')
    parser.add_argument('--n_features', type=int, default=10,
                        help='Number of features per sample')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Generate sample data
    df = generate_sample_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed
    )
    # print(f"Generated sample data with shape: {df.shape}")
    # print("\nFirst few rows:")
    # print(df.head()) 