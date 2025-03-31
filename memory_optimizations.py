import torch
import torch.nn as nn
import math
import copy

class GradientAccumulation:
    """
    Utility class to perform gradient accumulation for memory efficiency
    """
    def __init__(self, model, optimizer, accumulation_steps=8):
        """
        Args:
            model: PyTorch model
            optimizer: PyTorch optimizer
            accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def zero_gradients(self):
        """Zero gradients at the beginning of accumulation cycle"""
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
    
    def backward(self, loss):
        """Perform backward pass with scaling"""
        # Scale the loss to avoid numerical issues
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        self.current_step += 1
    
    def step(self):
        """Update weights if we've accumulated enough gradients"""
        if self.current_step % self.accumulation_steps == 0:
            self.optimizer.step()
            return True
        return False

def enable_mixed_precision():
    """
    Enable mixed precision training
    
    Returns:
        Gradient scaler for mixed precision
    """
    # Only enable if CUDA is available
    if torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        return scaler
    return None

def train_with_memory_optimization(experiment, model, graph_learner, features, anchor_adj, 
                                  labels, args, optimizer_cl, optimizer_learner):
    """
    Memory-efficient training function that uses both gradient accumulation and mixed precision
    
    Args:
        experiment: Experiment instance
        model: GCL model
        graph_learner: Graph learner model
        features: Input features
        anchor_adj: Anchor adjacency matrix
        labels: Class labels
        args: Training arguments
        optimizer_cl: Optimizer for GCL model
        optimizer_learner: Optimizer for graph learner model
        
    Returns:
        loss: Average loss value
    """
    # Enable mixed precision if requested
    scaler = None
    if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision:
        scaler = enable_mixed_precision()
        if args.verbose:
            print("Using mixed precision training")
    
    # Setup gradient accumulation
    accumulation_steps = args.grad_accumulation_steps if hasattr(args, 'grad_accumulation_steps') else 1
    if accumulation_steps > 1 and args.verbose:
        print(f"Using gradient accumulation with {accumulation_steps} steps")
    
    # Initialize gradient accumulation wrappers
    grad_accum_cl = GradientAccumulation(model, optimizer_cl, accumulation_steps)
    grad_accum_learner = GradientAccumulation(graph_learner, optimizer_learner, accumulation_steps)
    
    # Set models to training mode
    model.train()
    graph_learner.train()
    
    # Zero gradients before training
    grad_accum_cl.zero_gradients()
    grad_accum_learner.zero_gradients()
    
    # Split dataset into chunks for gradient accumulation
    n_samples = features.shape[0]
    chunk_size = math.ceil(n_samples / accumulation_steps)
    total_loss = 0
    
    # Process each chunk
    for i in range(0, accumulation_steps):
        # Get chunk indices
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        
        # Skip empty chunks
        if start_idx >= n_samples:
            break
            
        if args.verbose and accumulation_steps > 1:
            print(f"Processing chunk {i+1}/{accumulation_steps} (samples {start_idx}-{end_idx})")
        
        # Get chunk data
        chunk_indices = torch.arange(start_idx, end_idx, device=features.device)
        chunk_features = features[chunk_indices] if features.shape[0] > chunk_size else features
        chunk_labels = labels[chunk_indices] if labels.shape[0] > chunk_size else labels
        
        # For adjacency, we'll still need to use the full matrix
        # but we're processing fewer samples so memory impact is reduced
        
        # Forward pass with mixed precision if enabled
        if scaler:
            with torch.cuda.amp.autocast():
                if args.use_arcface:
                    chunk_loss, _ = experiment.loss_arcface(model, graph_learner, chunk_features, anchor_adj, chunk_labels, args)
                else:
                    chunk_loss, _ = experiment.loss_gcl(model, graph_learner, chunk_features, anchor_adj, args)
                
                # Scale loss based on chunk size
                if chunk_size < n_samples:
                    chunk_loss = chunk_loss * (end_idx - start_idx) / n_samples
                
                total_loss += chunk_loss.item() * (end_idx - start_idx) / n_samples
                
            # Backward pass with scaler
            scaler.scale(chunk_loss).backward()
        else:
            # Standard forward pass
            if args.use_arcface:
                chunk_loss, _ = experiment.loss_arcface(model, graph_learner, chunk_features, anchor_adj, chunk_labels, args)
            else:
                chunk_loss, _ = experiment.loss_gcl(model, graph_learner, chunk_features, anchor_adj, args)
                
            # Scale loss based on chunk size
            if chunk_size < n_samples:
                chunk_loss = chunk_loss * (end_idx - start_idx) / n_samples
                
            total_loss += chunk_loss.item() * (end_idx - start_idx) / n_samples
                
            # Backward pass
            grad_accum_cl.backward(chunk_loss)
            grad_accum_learner.backward(chunk_loss)
    
    # Step with scaler if using mixed precision
    if scaler:
        scaler.step(optimizer_cl)
        scaler.step(optimizer_learner)
        scaler.update()
    else:
        # Otherwise use gradient accumulation step
        optimizer_cl.step()
        optimizer_learner.step()
    
    # Forward pass on full data to get the learned adjacency
    model.eval()
    graph_learner.eval()
    with torch.no_grad():
        if args.use_arcface:
            _, learned_adj = experiment.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
        else:
            _, learned_adj = experiment.loss_gcl(model, graph_learner, features, anchor_adj, args)
    
    return total_loss, learned_adj