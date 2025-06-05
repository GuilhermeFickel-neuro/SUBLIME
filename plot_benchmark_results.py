import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
from matplotlib.ticker import ScalarFormatter

def load_data(csv_file):
    """Load benchmark data from CSV file"""
    df = pd.read_csv(csv_file)
    
    # Check for error messages in any column that should be numeric
    numeric_cols = ['n_samples', 'n_features', 'sparse', 'n_clusters', 
                   'data_generation_time_s', 'training_time_s', 
                   'peak_cpu_memory_mb', 'peak_gpu_memory_mb']
    
    # First convert columns to appropriate types
    for col in numeric_cols:
        if col in df.columns:
            # Check if the column contains any string that starts with "Command"
            # which indicates an error message
            if df[col].astype(str).str.contains('Command').any():
                # Create a mask for rows with error messages
                error_mask = df[col].astype(str).str.contains('Command')
                print(f"Found {error_mask.sum()} rows with error messages in column '{col}'")
                # Filter out these rows
                df = df[~error_mask]
            
            # Now convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in essential columns
    essential_cols = ['training_time_s', 'peak_cpu_memory_mb']
    df = df.dropna(subset=essential_cols)
    
    print(f"After filtering, {len(df)} valid benchmark results remain.")
    
    return df

def create_output_dir(output_dir):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_runtime_vs_samples(df, output_dir):
    """Plot runtime vs number of samples for different feature sizes"""
    plt.figure(figsize=(12, 8))
    
    for n_features in df['n_features'].unique():
        subset = df[df['n_features'] == n_features]
        plt.plot(subset['n_samples'], subset['training_time_s'], 
                 marker='o', label=f'Features: {n_features}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Number of Samples')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    # Format axes with actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_samples.png'), dpi=300)
    plt.close()

def plot_memory_vs_samples(df, output_dir):
    """Plot memory usage vs number of samples for different feature sizes"""
    # CPU Memory
    plt.figure(figsize=(12, 8))
    
    for n_features in df['n_features'].unique():
        subset = df[df['n_features'] == n_features]
        plt.plot(subset['n_samples'], subset['peak_cpu_memory_mb'], 
                 marker='o', label=f'Features: {n_features}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Peak CPU Memory (MB)')
    plt.title('CPU Memory Usage vs Number of Samples')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    # Format axes with actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_memory_vs_samples.png'), dpi=300)
    plt.close()
    
    # GPU Memory (if available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:  # Only if GPU was used
        plt.figure(figsize=(12, 8))
        
        for n_features in df['n_features'].unique():
            subset = df[df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset['peak_gpu_memory_mb'], 
                     marker='o', label=f'Features: {n_features}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Samples')
        plt.ylabel('Peak GPU Memory (MB)')
        plt.title('GPU Memory Usage vs Number of Samples')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        
        # Format axes with actual values instead of powers
        for axis in [plt.gca().xaxis, plt.gca().yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_memory_vs_samples.png'), dpi=300)
        plt.close()

def plot_memory_vs_features(df, output_dir):
    """Plot memory usage vs number of features for different sample sizes"""
    # CPU Memory
    plt.figure(figsize=(12, 8))
    
    for n_samples in df['n_samples'].unique():
        subset = df[df['n_samples'] == n_samples]
        plt.plot(subset['n_features'], subset['peak_cpu_memory_mb'], 
                 marker='o', label=f'Samples: {n_samples}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Features')
    plt.ylabel('Peak CPU Memory (MB)')
    plt.title('CPU Memory Usage vs Number of Features')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    # Format axes with actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_memory_vs_features.png'), dpi=300)
    plt.close()
    
    # GPU Memory (if available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        plt.figure(figsize=(12, 8))
        
        for n_samples in df['n_samples'].unique():
            subset = df[df['n_samples'] == n_samples]
            plt.plot(subset['n_features'], subset['peak_gpu_memory_mb'], 
                     marker='o', label=f'Samples: {n_samples}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Features')
        plt.ylabel('Peak GPU Memory (MB)')
        plt.title('GPU Memory Usage vs Number of Features')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        
        # Format axes with actual values instead of powers
        for axis in [plt.gca().xaxis, plt.gca().yaxis]:
            axis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_memory_vs_features.png'), dpi=300)
        plt.close()

def plot_comprehensive_memory_analysis(df, output_dir):
    """Create comprehensive memory analysis plots to show how different settings affect memory usage"""
    # Create a dedicated directory for memory analysis
    memory_dir = os.path.join(output_dir, 'memory_analysis')
    if not os.path.exists(memory_dir):
        os.makedirs(memory_dir)
    
    # 1. Memory usage by n_clusters (boxplots)
    plt.figure(figsize=(14, 8))
    
    # CPU Memory by n_clusters
    plt.subplot(1, 2, 1)
    sns.boxplot(x='n_clusters', y='peak_cpu_memory_mb', data=df)
    plt.title('CPU Memory Usage by Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Peak CPU Memory (MB)')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # GPU Memory by n_clusters (if available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        plt.subplot(1, 2, 2)
        sns.boxplot(x='n_clusters', y='peak_gpu_memory_mb', data=df)
        plt.title('GPU Memory Usage by Number of Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Peak GPU Memory (MB)')
        plt.grid(True, which="both", ls="--", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(memory_dir, 'memory_by_clusters.png'), dpi=300)
    plt.close()
    
    # 2. Memory usage by sparsity (if applicable)
    if 'sparse' in df.columns and len(df['sparse'].unique()) > 1:
        plt.figure(figsize=(14, 8))
        
        # CPU Memory by sparsity
        plt.subplot(1, 2, 1)
        sns.boxplot(x='sparse', y='peak_cpu_memory_mb', data=df)
        plt.title('CPU Memory Usage by Data Sparsity')
        plt.xlabel('Sparse Data')
        plt.ylabel('Peak CPU Memory (MB)')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # GPU Memory by sparsity (if available)
        if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
            plt.subplot(1, 2, 2)
            sns.boxplot(x='sparse', y='peak_gpu_memory_mb', data=df)
            plt.title('GPU Memory Usage by Data Sparsity')
            plt.xlabel('Sparse Data')
            plt.ylabel('Peak GPU Memory (MB)')
            plt.grid(True, which="both", ls="--", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(memory_dir, 'memory_by_sparsity.png'), dpi=300)
        plt.close()
    
    # 3. Memory scaling with dataset size
    plt.figure(figsize=(12, 8))
    
    # Calculate dataset size (samples * features)
    df['dataset_size'] = df['n_samples'] * df['n_features']
    
    # CPU Memory vs dataset size
    plt.subplot(1, 2, 1)
    plt.scatter(df['dataset_size'], df['peak_cpu_memory_mb'], alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size (samples × features)')
    plt.ylabel('Peak CPU Memory (MB)')
    plt.title('CPU Memory Scaling with Dataset Size')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Add regression line
    x = np.log10(df['dataset_size'])
    y = np.log10(df['peak_cpu_memory_mb'])
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) > 1:  # Need at least 2 points for regression
        z = np.polyfit(x[mask], y[mask], 1)
        p = np.poly1d(z)
        plt.plot(df['dataset_size'], 10**p(np.log10(df['dataset_size'])), 
                 "r--", alpha=0.8, label=f"Slope: {z[0]:.2f}")
        plt.legend()
    
    # GPU Memory vs dataset size (if available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        plt.subplot(1, 2, 2)
        plt.scatter(df['dataset_size'], df['peak_gpu_memory_mb'], alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Dataset Size (samples × features)')
        plt.ylabel('Peak GPU Memory (MB)')
        plt.title('GPU Memory Scaling with Dataset Size')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        
        # Add regression line
        x = np.log10(df['dataset_size'])
        y = np.log10(df['peak_gpu_memory_mb'])
        mask = ~np.isnan(x) & ~np.isnan(y) & (df['peak_gpu_memory_mb'] > 0)
        if np.sum(mask) > 1:  # Need at least 2 points for regression
            z = np.polyfit(x[mask], y[mask], 1)
            p = np.poly1d(z)
            plt.plot(df['dataset_size'][mask], 10**p(np.log10(df['dataset_size'][mask])), 
                     "r--", alpha=0.8, label=f"Slope: {z[0]:.2f}")
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(memory_dir, 'memory_scaling.png'), dpi=300)
    plt.close()
    
    # 4. Memory usage heatmap by samples and features
    for n_clusters in df['n_clusters'].unique():
        subset = df[df['n_clusters'] == n_clusters]
        
        # Create pivot tables
        cpu_pivot = subset.pivot_table(
            index='n_samples', 
            columns='n_features',
            values='peak_cpu_memory_mb',
            aggfunc='mean'
        )
        
        # Plot CPU memory heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cpu_pivot, annot=True, fmt='.1f', cmap='viridis', linewidths=.5, cbar_kws={'label': 'MB'})
        plt.title(f'CPU Memory Usage (MB) with {n_clusters} Clusters')
        plt.tight_layout()
        plt.savefig(os.path.join(memory_dir, f'cpu_memory_heatmap_{n_clusters}_clusters.png'), dpi=300)
        plt.close()
        
        # Plot GPU memory heatmap (if available)
        if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
            gpu_pivot = subset.pivot_table(
                index='n_samples', 
                columns='n_features',
                values='peak_gpu_memory_mb',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(gpu_pivot, annot=True, fmt='.1f', cmap='plasma', linewidths=.5, cbar_kws={'label': 'MB'})
            plt.title(f'GPU Memory Usage (MB) with {n_clusters} Clusters')
            plt.tight_layout()
            plt.savefig(os.path.join(memory_dir, f'gpu_memory_heatmap_{n_clusters}_clusters.png'), dpi=300)
            plt.close()
    
    # 5. Memory efficiency (memory per sample) by feature size
    plt.figure(figsize=(14, 8))
    
    # Calculate memory per sample
    df['cpu_memory_per_sample_kb'] = df['peak_cpu_memory_mb'] * 1024 / df['n_samples']
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        df['gpu_memory_per_sample_kb'] = df['peak_gpu_memory_mb'] * 1024 / df['n_samples']
    
    # CPU Memory efficiency
    plt.subplot(1, 2, 1)
    for n_clusters in df['n_clusters'].unique():
        subset = df[df['n_clusters'] == n_clusters]
        plt.plot(subset['n_features'], subset['cpu_memory_per_sample_kb'], 
                 marker='o', label=f'Clusters: {n_clusters}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Features')
    plt.ylabel('CPU Memory per Sample (KB)')
    plt.title('CPU Memory Efficiency by Feature Size')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    # GPU Memory efficiency (if available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        plt.subplot(1, 2, 2)
        for n_clusters in df['n_clusters'].unique():
            subset = df[df['n_clusters'] == n_clusters]
            plt.plot(subset['n_features'], subset['gpu_memory_per_sample_kb'], 
                     marker='o', label=f'Clusters: {n_clusters}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Features')
        plt.ylabel('GPU Memory per Sample (KB)')
        plt.title('GPU Memory Efficiency by Feature Size')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(memory_dir, 'memory_efficiency.png'), dpi=300)
    plt.close()
    
    # 6. Memory usage comparison: CPU vs GPU (if GPU data available)
    if 'peak_gpu_memory_mb' in df.columns and df['peak_gpu_memory_mb'].sum() > 0:
        plt.figure(figsize=(12, 8))
        
        # Filter out rows with zero GPU memory
        gpu_df = df[df['peak_gpu_memory_mb'] > 0]
        
        if len(gpu_df) > 0:
            plt.scatter(gpu_df['peak_cpu_memory_mb'], gpu_df['peak_gpu_memory_mb'], 
                        alpha=0.7, c=gpu_df['n_clusters'], cmap='viridis')
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Peak CPU Memory (MB)')
            plt.ylabel('Peak GPU Memory (MB)')
            plt.title('CPU vs GPU Memory Usage Comparison')
            plt.grid(True, which="both", ls="--", alpha=0.3)
            plt.colorbar(label='Number of Clusters')
            
            # Add diagonal line (equal CPU and GPU memory)
            min_val = min(gpu_df['peak_cpu_memory_mb'].min(), gpu_df['peak_gpu_memory_mb'].min())
            max_val = max(gpu_df['peak_cpu_memory_mb'].max(), gpu_df['peak_gpu_memory_mb'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='CPU = GPU')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(memory_dir, 'cpu_vs_gpu_memory.png'), dpi=300)
            plt.close()

def plot_runtime_vs_features(df, output_dir):
    """Plot runtime vs number of features for different sample sizes"""
    plt.figure(figsize=(12, 8))
    
    for n_samples in df['n_samples'].unique():
        subset = df[df['n_samples'] == n_samples]
        plt.plot(subset['n_features'], subset['training_time_s'], 
                 marker='o', label=f'Samples: {n_samples}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Features')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Number of Features')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    # Format axes with actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_vs_features.png'), dpi=300)
    plt.close()

def plot_heatmap(df, output_dir):
    """Plot heatmaps of runtime and memory usage"""
    # Prepare data for heatmap
    metrics = ['training_time_s', 'peak_cpu_memory_mb']
    if df['peak_gpu_memory_mb'].sum() > 0:
        metrics.append('peak_gpu_memory_mb')
    
    for metric in metrics:
        # Create pivot table for each n_clusters value
        for n_clusters in df['n_clusters'].unique():
            subset = df[df['n_clusters'] == n_clusters]
            
            # Create pivot table
            pivot = subset.pivot_table(
                index='n_samples', 
                columns='n_features',
                values=metric,
                aggfunc='mean'
            )
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='viridis', linewidths=.5)
            
            metric_name = metric.replace('_', ' ').title()
            plt.title(f'{metric_name} (Clusters: {n_clusters})')
            plt.tight_layout()
            
            # Save figure
            filename = f'heatmap_{metric}_{n_clusters}_clusters.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()

def plot_cluster_comparison(df, output_dir):
    """Plot comparison of different n_clusters values"""
    # Ensure we're only working with numeric data
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Add back the grouping columns if they're not already in numeric_df
    for col in ['n_samples', 'n_features', 'n_clusters']:
        if col not in numeric_df.columns and col in df.columns:
            numeric_df[col] = df[col]
    
    # Group by n_samples, n_features, and n_clusters
    try:
        grouped = numeric_df.groupby(['n_samples', 'n_features', 'n_clusters']).mean().reset_index()
    except TypeError as e:
        print(f"Warning: Error in grouping data: {e}")
        print("Skipping cluster comparison plots due to data issues.")
        return
    
    # Plot for each n_samples and n_features combination
    for n_samples in grouped['n_samples'].unique():
        for n_features in grouped['n_features'].unique():
            subset = grouped[(grouped['n_samples'] == n_samples) & 
                             (grouped['n_features'] == n_features)]
            
            if len(subset) > 1:  # Only if we have multiple n_clusters values
                # Create bar plot
                plt.figure(figsize=(12, 6))
                
                # Set up metrics to plot
                metrics = ['training_time_s', 'peak_cpu_memory_mb']
                if 'peak_gpu_memory_mb' in subset.columns and subset['peak_gpu_memory_mb'].sum() > 0:
                    metrics.append('peak_gpu_memory_mb')
                
                # Create subplots
                fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
                if len(metrics) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(metrics):
                    if metric in subset.columns:  # Check if the metric exists in the subset
                        ax = axes[i]
                        ax.bar(subset['n_clusters'].astype(str), subset[metric])
                        ax.set_title(metric.replace('_', ' ').title())
                        ax.set_xlabel('Number of Clusters')
                        
                        # Add values on top of bars
                        for j, v in enumerate(subset[metric]):
                            ax.text(j, v * 1.01, f'{v:.1f}', ha='center')
                
                plt.suptitle(f'Comparison for {n_samples} Samples, {n_features} Features')
                plt.tight_layout()
                
                # Save figure
                filename = f'cluster_comparison_{n_samples}samples_{n_features}features.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=300)
                plt.close()

def plot_data_generation_time(df, output_dir):
    """Plot data generation time vs dataset size"""
    plt.figure(figsize=(12, 8))
    
    # Calculate dataset size (samples * features)
    df['dataset_size'] = df['n_samples'] * df['n_features']
    
    # Group by dataset size
    grouped = df.groupby('dataset_size')['data_generation_time_s'].mean().reset_index()
    
    # Sort by dataset size
    grouped = grouped.sort_values('dataset_size')
    
    plt.plot(grouped['dataset_size'], grouped['data_generation_time_s'], marker='o')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Dataset Size (samples × features)')
    plt.ylabel('Data Generation Time (seconds)')
    plt.title('Data Generation Time vs Dataset Size')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    
    # Format axes with actual values instead of powers
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_generation_time.png'), dpi=300)
    plt.close()

def plot_efficiency_metrics(df, output_dir):
    """Plot efficiency metrics (time per sample, memory per sample)"""
    # Calculate efficiency metrics
    df['time_per_sample_ms'] = df['training_time_s'] * 1000 / df['n_samples']
    df['cpu_memory_per_sample_kb'] = df['peak_cpu_memory_mb'] * 1024 / df['n_samples']
    
    if df['peak_gpu_memory_mb'].sum() > 0:
        df['gpu_memory_per_sample_kb'] = df['peak_gpu_memory_mb'] * 1024 / df['n_samples']
    
    # Plot time per sample
    plt.figure(figsize=(12, 8))
    
    for n_features in df['n_features'].unique():
        subset = df[df['n_features'] == n_features]
        plt.plot(subset['n_samples'], subset['time_per_sample_ms'], 
                 marker='o', label=f'Features: {n_features}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('Training Time per Sample (ms)')
    plt.title('Training Efficiency: Time per Sample')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_per_sample.png'), dpi=300)
    plt.close()
    
    # Plot CPU memory per sample
    plt.figure(figsize=(12, 8))
    
    for n_features in df['n_features'].unique():
        subset = df[df['n_features'] == n_features]
        plt.plot(subset['n_samples'], subset['cpu_memory_per_sample_kb'], 
                 marker='o', label=f'Features: {n_features}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Samples')
    plt.ylabel('CPU Memory per Sample (KB)')
    plt.title('Memory Efficiency: CPU Memory per Sample')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_memory_per_sample.png'), dpi=300)
    plt.close()
    
    # Plot GPU memory per sample (if available)
    if df['peak_gpu_memory_mb'].sum() > 0:
        plt.figure(figsize=(12, 8))
        
        for n_features in df['n_features'].unique():
            subset = df[df['n_features'] == n_features]
            plt.plot(subset['n_samples'], subset['gpu_memory_per_sample_kb'], 
                     marker='o', label=f'Features: {n_features}')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of Samples')
        plt.ylabel('GPU Memory per Sample (KB)')
        plt.title('Memory Efficiency: GPU Memory per Sample')
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gpu_memory_per_sample.png'), dpi=300)
        plt.close()

def create_summary_table(df, output_dir):
    """Create a summary table of the benchmark results"""
    # Ensure we're only working with numeric data
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Add back the grouping columns if they're not already in numeric_df
    for col in ['n_samples', 'n_features', 'sparse', 'n_clusters']:
        if col not in numeric_df.columns and col in df.columns:
            numeric_df[col] = df[col]
    
    try:
        # Group by configuration
        grouped = numeric_df.groupby(['n_samples', 'n_features', 'sparse', 'n_clusters']).agg({
            'training_time_s': ['mean', 'min', 'max'],
            'peak_cpu_memory_mb': ['mean', 'min', 'max'],
            'peak_gpu_memory_mb': ['mean', 'min', 'max'] if 'peak_gpu_memory_mb' in numeric_df.columns else [],
            'data_generation_time_s': ['mean'] if 'data_generation_time_s' in numeric_df.columns else []
        }).reset_index()
        
        # Flatten the column names
        grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
        
        # Sort by dataset size and then by n_clusters
        grouped['dataset_size'] = grouped['n_samples'] * grouped['n_features']
        grouped = grouped.sort_values(['dataset_size', 'n_clusters'])
        
        # Save to CSV
        grouped.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
        
        # Create a more readable HTML version
        html = grouped.to_html(
            float_format=lambda x: f'{x:.2f}',
            classes='table table-striped table-hover',
            border=0
        )
        
        with open(os.path.join(output_dir, 'summary_table.html'), 'w') as f:
            f.write('''
            <html>
            <head>
                <title>Benchmark Summary</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .table { border-collapse: collapse; width: 100%; }
                    .table th, .table td { padding: 8px; text-align: right; }
                    .table th { background-color: #f2f2f2; }
                    .table-striped tbody tr:nth-of-type(odd) { background-color: rgba(0,0,0,.05); }
                    .table-hover tbody tr:hover { background-color: rgba(0,0,0,.075); }
                </style>
            </head>
            <body>
                <h1>Benchmark Summary</h1>
            ''')
            f.write(html)
            f.write('''
            </body>
            </html>
            ''')
    except Exception as e:
        print(f"Warning: Error creating summary table: {e}")
        print("Skipping summary table creation due to data issues.")

def main():
    parser = argparse.ArgumentParser(description='Generate plots from benchmark results')
    parser.add_argument('--input', type=str, default='benchmark_results.csv',
                        help='Input CSV file with benchmark results')
    parser.add_argument('--output-dir', type=str, default='benchmark_plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    # Create output directory
    create_output_dir(args.output_dir)
    
    # Load data
    df = load_data(args.input)
    
    if len(df) == 0:
        print("No valid benchmark data found in the CSV file.")
        return
    
    print(f"Proceeding with {len(df)} valid benchmark results.")
    
    # Generate plots
    print("Generating plots...")
    
    try:
        plot_runtime_vs_samples(df, args.output_dir)
        print("✓ Runtime vs samples plot generated")
    except Exception as e:
        print(f"✗ Error generating runtime vs samples plot: {e}")
    
    try:
        plot_memory_vs_samples(df, args.output_dir)
        print("✓ Memory vs samples plot generated")
    except Exception as e:
        print(f"✗ Error generating memory vs samples plot: {e}")
    
    try:
        plot_memory_vs_features(df, args.output_dir)
        print("✓ Memory vs features plot generated")
    except Exception as e:
        print(f"✗ Error generating memory vs features plot: {e}")
    
    try:
        plot_comprehensive_memory_analysis(df, args.output_dir)
        print("✓ Comprehensive memory analysis plots generated")
    except Exception as e:
        print(f"✗ Error generating comprehensive memory analysis plots: {e}")
    
    try:
        plot_runtime_vs_features(df, args.output_dir)
        print("✓ Runtime vs features plot generated")
    except Exception as e:
        print(f"✗ Error generating runtime vs features plot: {e}")
    
    try:
        plot_heatmap(df, args.output_dir)
        print("✓ Heatmap plot generated")
    except Exception as e:
        print(f"✗ Error generating heatmap plot: {e}")
    
    try:
        plot_cluster_comparison(df, args.output_dir)
        print("✓ Cluster comparison plot generated")
    except Exception as e:
        print(f"✗ Error generating cluster comparison plot: {e}")
    
    try:
        plot_data_generation_time(df, args.output_dir)
        print("✓ Data generation time plot generated")
    except Exception as e:
        print(f"✗ Error generating data generation time plot: {e}")
    
    try:
        plot_efficiency_metrics(df, args.output_dir)
        print("✓ Efficiency metrics plot generated")
    except Exception as e:
        print(f"✗ Error generating efficiency metrics plot: {e}")
    
    # Create summary table
    try:
        print("Creating summary table...")
        create_summary_table(df, args.output_dir)
        print("✓ Summary table created")
    except Exception as e:
        print(f"✗ Error creating summary table: {e}")
    
    print(f"All successfully generated plots and summary tables saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 