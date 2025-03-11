import subprocess
import time
import os
import csv
import psutil
import argparse
import itertools
from datetime import datetime

# Try to import GPU monitoring - if not available, use dummy values
try:
    import torch
    import nvidia_smi
    HAS_GPU = torch.cuda.is_available()
    if HAS_GPU:
        nvidia_smi.nvmlInit()
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False

def get_gpu_memory_usage():
    """Get GPU memory usage in MB"""
    if not HAS_GPU:
        return 0
    
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024 / 1024  # Convert to MB
    except:
        return 0

def run_benchmark(n_samples, n_features, sparse, n_clusters, output_file):
    """Run a single benchmark with the given parameters"""
    
    # First generate the sample data
    start_time = time.time()
    subprocess.run([
        "python", "create_sample_data.py",
        "--n_samples", str(n_samples),
        "--n_features", str(n_features)
    ], check=True)
    data_gen_time = time.time() - start_time
    
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_cpu_mem = process.memory_info().rss / (1024 * 1024)  # MB
    initial_gpu_mem = get_gpu_memory_usage()
    
    # Run the training script
    start_time = time.time()
    subprocess.run([
        "python", "train_person_data.py",
        "-dataset", "person_data.csv",
        "-sparse", str(sparse),
        "-n_clusters", str(n_clusters)
    ], check=True)
    training_time = time.time() - start_time
    
    # Get peak memory usage
    peak_cpu_mem = process.memory_info().rss / (1024 * 1024) - initial_cpu_mem  # MB
    peak_gpu_mem = get_gpu_memory_usage() - initial_gpu_mem  # MB
    
    # Record results
    result = {
        'n_samples': n_samples,
        'n_features': n_features,
        'sparse': sparse,
        'n_clusters': n_clusters,
        'data_generation_time_s': round(data_gen_time, 2),
        'training_time_s': round(training_time, 2),
        'peak_cpu_memory_mb': round(peak_cpu_mem, 2),
        'peak_gpu_memory_mb': round(peak_gpu_mem, 2),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Write to CSV
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks for SUBLIME on person data')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV file for benchmark results')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip configurations that are already in the output file')
    args = parser.parse_args()
    
    # Define parameter grid
    n_samples_list = [10000, 100000, 500000, 1000000]
    n_features_list = [100, 200, 500, 1000]
    sparse_list = [0]
    n_clusters_list = [100, 1000]
    
    # Generate all combinations
    configurations = list(itertools.product(
        n_samples_list, n_features_list, sparse_list, n_clusters_list
    ))
    
    # Skip existing configurations if requested
    existing_configs = set()
    if args.skip_existing and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config = (
                    int(row['n_samples']),
                    int(row['n_features']),
                    int(row['sparse']),
                    int(row['n_clusters'])
                )
                existing_configs.add(config)
        
        configurations = [c for c in configurations if c not in existing_configs]
    
    # Run benchmarks
    total_configs = len(configurations)
    for i, (n_samples, n_features, sparse, n_clusters) in enumerate(configurations, 1):
        print(f"Running benchmark {i}/{total_configs}:")
        print(f"  n_samples={n_samples}, n_features={n_features}, sparse={sparse}, n_clusters={n_clusters}")
        
        try:
            result = run_benchmark(n_samples, n_features, sparse, n_clusters, args.output)
            print(f"  Completed in {result['training_time_s']}s, "
                  f"CPU: {result['peak_cpu_memory_mb']}MB, "
                  f"GPU: {result['peak_gpu_memory_mb']}MB")
        except Exception as e:
            print(f"  Error: {e}")
            # Log the error to the CSV
            with open(args.output, 'a', newline='') as f:
                writer = csv.writer(f)
                if not os.path.isfile(args.output):
                    writer.writerow(['n_samples', 'n_features', 'sparse', 'n_clusters', 
                                     'error', 'timestamp'])
                writer.writerow([
                    n_samples, n_features, sparse, n_clusters,
                    str(e), datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
        
        print()

if __name__ == "__main__":
    main() 