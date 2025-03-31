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
    Memory-efficient training function that uses gradient accumulation
    
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
    # Setup gradient accumulation
    accumulation_steps = args.grad_accumulation_steps if hasattr(args, 'grad_accumulation_steps') else 1
    if accumulation_steps > 1 and args.verbose:
        print(f"Using gradient accumulation with {accumulation_steps} steps")
    
    # Set models to training mode
    model.train()
    graph_learner.train()
    
    # Zero gradients before training
    optimizer_cl.zero_grad()
    optimizer_learner.zero_grad()
    
    # Keep track of total loss
    total_loss = 0
    
    # Process in accumulation steps
    for i in range(accumulation_steps):
        # Forward pass
        if args.use_arcface:
            loss, Adj = experiment.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
        else:
            loss, Adj = experiment.loss_gcl(model, graph_learner, features, anchor_adj, args)
        
        # Scale loss for accumulation
        scaled_loss = loss / accumulation_steps
        total_loss += loss.item() / accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        if args.verbose and i == 0:
            print(f"Processing step {i+1}/{accumulation_steps}, loss: {loss.item():.4f}")
    
    # Update weights after accumulation
    optimizer_cl.step()
    optimizer_learner.step()
    
    # Forward pass on full data to get the learned adjacency (for return value)
    model.eval()
    graph_learner.eval()
    with torch.no_grad():
        if args.use_arcface:
            _, learned_adj = experiment.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
        else:
            _, learned_adj = experiment.loss_gcl(model, graph_learner, features, anchor_adj, args)
    
    return total_loss, learned_adj