import subprocess
import time
import os
import csv
import psutil
import argparse
import itertools
import threading
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

# Global variables for memory tracking
peak_cpu_memory = 0
peak_gpu_memory = 0
stop_monitoring = False

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

def memory_monitor(process_id):
    """Monitor memory usage in a separate thread"""
    global peak_cpu_memory, peak_gpu_memory, stop_monitoring
    
    while not stop_monitoring:
        try:
            # Get current system-wide memory usage instead of just the process
            system_memory = psutil.virtual_memory()
            current_cpu_mem = system_memory.used / (1024 * 1024)  # MB
            peak_cpu_memory = max(peak_cpu_memory, current_cpu_mem)
            
            # Get current GPU memory usage
            if HAS_GPU:
                current_gpu_mem = get_gpu_memory_usage()
                peak_gpu_memory = max(peak_gpu_memory, current_gpu_mem)
                
            # Sleep to avoid excessive CPU usage
            time.sleep(1)
        except:
            # Process might have ended
            break

def run_benchmark(n_samples, n_features, sparse, n_clusters, output_file):
    """Run a single benchmark with the given parameters"""
    global peak_cpu_memory, peak_gpu_memory, stop_monitoring
    
    # Reset peak memory values
    peak_cpu_memory = 0
    peak_gpu_memory = 0
    stop_monitoring = False
    
    # Start memory monitoring thread
    monitor_thread = threading.Thread(
        target=memory_monitor, 
        args=(os.getpid(),),
        daemon=True
    )
    monitor_thread.start()
    
    try:
        # First generate the sample data
        start_time = time.time()
        subprocess.run([
            "python", "create_sample_data.py",
            "--n_samples", str(n_samples),
            "--n_features", str(n_features)
        ], check=True)
        data_gen_time = time.time() - start_time
        
        # Run the training script
        start_time = time.time()
        subprocess.run([
            "python", "train_person_data.py",
            "-dataset", "person_data.csv",
            "-sparse", str(sparse),
            "-n_clusters", str(n_clusters)
        ], check=True)
        training_time = time.time() - start_time
        
        # Stop memory monitoring
        stop_monitoring = True
        monitor_thread.join(timeout=1.0)
        
        # Record results
        result = {
            'n_samples': n_samples,
            'n_features': n_features,
            'sparse': sparse,
            'n_clusters': n_clusters,
            'data_generation_time_s': round(data_gen_time, 2),
            'training_time_s': round(training_time, 2),
            'peak_cpu_memory_mb': round(peak_cpu_memory, 2),
            'peak_gpu_memory_mb': round(peak_gpu_memory, 2),
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
    
    finally:
        # Ensure monitoring is stopped even if an exception occurs
        stop_monitoring = True
        monitor_thread.join(timeout=1.0)

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
    sparse_list = [0, 1]
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