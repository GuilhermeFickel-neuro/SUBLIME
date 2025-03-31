import torch
import time

def profile_arcface_memory(batch_size, emb_dim, num_classes):
    """
    Estimate memory usage for ArcFace with given dimensions.
    
    Args:
        batch_size: Number of samples in batch
        emb_dim: Embedding dimension
        num_classes: Number of classes (typically equal to num_rows)
        
    Returns:
        Total estimated memory in GB
    """
    weight_size = num_classes * emb_dim * 4 / (1024**3)  # GB
    cosine_size = batch_size * num_classes * 4 / (1024**3)  # GB
    onehot_size = batch_size * num_classes * 4 / (1024**3)  # GB
    
    total = weight_size + 5 * cosine_size  # Including intermediate tensors
    print(f"ArcFace estimated memory usage:")
    print(f"  Weight matrix: {weight_size:.2f} GB")
    print(f"  Cosine matrix: {cosine_size:.2f} GB")
    print(f"  One-hot matrix: {onehot_size:.2f} GB")
    print(f"  Total estimated: {total:.2f} GB")
    return total

def analyze_forward_pass(model, features, adj, labels, max_samples=10000):
    """
    Run forward pass with increasing numbers of samples to find threshold.
    
    Args:
        model: The GCL model
        features: Input features
        adj: Adjacency matrix
        labels: Class labels
        max_samples: Maximum number of samples to test
        
    Returns:
        Dictionary with results for each sample size
    """
    results = {}
    sample_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    
    for num_samples in sample_sizes:
        if num_samples > max_samples or num_samples > features.shape[0]:
            break
            
        torch.cuda.empty_cache()
        try:
            sample_idx = torch.randperm(features.shape[0])[:num_samples]
            sample_feat = features[sample_idx]
            sample_labels = labels[sample_idx]
            
            # Record start memory
            start_mem = torch.cuda.memory_allocated() / (1024**3)
            
            # Measure time
            start_time = time.time()
            
            # Try forward pass
            result = model(sample_feat, adj, 'learner', sample_labels)
            
            # Record time and peak memory
            end_time = time.time()
            end_mem = torch.cuda.memory_allocated() / (1024**3)
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"✓ Forward pass successful with {num_samples} samples")
            print(f"  Time: {end_time - start_time:.4f} seconds")
            print(f"  Memory before: {start_mem:.2f} GB")
            print(f"  Memory after: {end_mem:.2f} GB")
            print(f"  Peak memory: {peak_mem:.2f} GB")
            
            results[num_samples] = {
                'success': True,
                'time': end_time - start_time,
                'memory_before': start_mem,
                'memory_after': end_mem,
                'peak_memory': peak_mem
            }
        except RuntimeError as e:
            print(f"✗ Failed with {num_samples} samples: {e}")
            results[num_samples] = {
                'success': False,
                'error': str(e)
            }
            break
    
    return results

def measure_arcface_memory_usage(arcface_layer, embedding, labels):
    """
    Measure memory used specifically by ArcFace layer.
    
    Args:
        arcface_layer: The ArcFace layer instance
        embedding: Input embeddings
        labels: Class labels
        
    Returns:
        Memory used in bytes
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Record start memory
    start_mem = torch.cuda.memory_allocated()
    
    # Run ArcFace
    _ = arcface_layer(embedding, labels)
    
    # Record peak memory
    peak_mem = torch.cuda.max_memory_allocated()
    
    used_mem = peak_mem - start_mem
    print(f"ArcFace memory usage: {used_mem / (1024**3):.2f} GB")
    return used_mem