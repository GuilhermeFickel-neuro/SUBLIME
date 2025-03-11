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
    
    # Filter out rows with errors (if any)
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    
    # Convert columns to appropriate types
    numeric_cols = ['n_samples', 'n_features', 'sparse', 'n_clusters', 
                   'data_generation_time_s', 'training_time_s', 
                   'peak_cpu_memory_mb', 'peak_gpu_memory_mb']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
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
    if df['peak_gpu_memory_mb'].sum() > 0:  # Only if GPU was used
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
    # Group by n_samples, n_features, and n_clusters
    grouped = df.groupby(['n_samples', 'n_features', 'n_clusters']).mean().reset_index()
    
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
                if subset['peak_gpu_memory_mb'].sum() > 0:
                    metrics.append('peak_gpu_memory_mb')
                
                # Create subplots
                fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
                if len(metrics) == 1:
                    axes = [axes]
                
                for i, metric in enumerate(metrics):
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
    # Group by configuration
    grouped = df.groupby(['n_samples', 'n_features', 'sparse', 'n_clusters']).agg({
        'training_time_s': ['mean', 'min', 'max'],
        'peak_cpu_memory_mb': ['mean', 'min', 'max'],
        'peak_gpu_memory_mb': ['mean', 'min', 'max'],
        'data_generation_time_s': ['mean']
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
    
    print(f"Loaded {len(df)} benchmark results.")
    
    # Generate plots
    print("Generating plots...")
    plot_runtime_vs_samples(df, args.output_dir)
    plot_memory_vs_samples(df, args.output_dir)
    plot_runtime_vs_features(df, args.output_dir)
    plot_heatmap(df, args.output_dir)
    plot_cluster_comparison(df, args.output_dir)
    plot_data_generation_time(df, args.output_dir)
    plot_efficiency_metrics(df, args.output_dir)
    
    # Create summary table
    print("Creating summary table...")
    create_summary_table(df, args.output_dir)
    
    print(f"All plots and summary tables saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 