import torch
import torch.nn as nn
import math
import copy
import contextlib # Added for nullcontext

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

    # --- Optimization 2 Implementation ---
    # Check if mixed precision training is enabled via args
    scaler = None
    if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision and torch.cuda.is_available():
         # Initialize GradScaler only if mixed precision is enabled and CUDA is available
         from torch.cuda.amp import GradScaler
         scaler = GradScaler()
         if args.verbose and accumulation_steps > 1:
             print(f"Using gradient accumulation ({accumulation_steps} steps) with Mixed Precision")
         elif args.verbose:
             print(f"Using Mixed Precision in memory_efficient_training path")
    elif accumulation_steps > 1 and args.verbose:
         # Print message for gradient accumulation without mixed precision
         print(f"Using gradient accumulation with {accumulation_steps} steps")
    # --- End Optimization 2 (Scaler Setup) ---

    # Set models to training mode
    model.train()
    graph_learner.train()

    # Zero gradients before starting the accumulation loop
    optimizer_cl.zero_grad()
    optimizer_learner.zero_grad()

    # Keep track of total accumulated loss and the last computed adjacency matrix
    total_loss = 0
    last_Adj = None # Initialize variable to store the last adjacency matrix

    # Process in accumulation steps
    for i in range(accumulation_steps):
        # --- Optimization 2 (Forward Pass) ---
        # Use torch.cuda.amp.autocast for the forward pass if scaler is enabled (mixed precision)
        # Otherwise, use a null context (no change in behavior)
        with torch.cuda.amp.autocast() if scaler else contextlib.nullcontext():
             # Perform the forward pass to calculate loss and the learned adjacency matrix
             if args.use_arcface:
                 loss, Adj = experiment.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
             else:
                 loss, Adj = experiment.loss_gcl(model, graph_learner, features, anchor_adj, args)
        # --- End Optimization 2 (Forward Pass) ---

        # --- Optimization 1 (Store last Adj) ---
        # Store the adjacency matrix computed in this step.
        # After the loop, the 'last_Adj' will hold the one from the final step.
        last_Adj = Adj
        # --- End Optimization 1 ---

        # Scale loss for accumulation (average loss over accumulation steps)
        scaled_loss = loss / accumulation_steps

        # Accumulate the loss *value* (use .item() to detach from computation graph and save memory)
        total_loss += scaled_loss.item()

        # --- Optimization 2 (Backward Pass) ---
        # Perform backward pass.
        # If scaler is enabled, use scaler.scale() to scale the gradients.
        if scaler:
             scaler.scale(scaled_loss).backward()
        else:
             # Standard backward pass if not using mixed precision
             scaled_loss.backward()
        # --- End Optimization 2 (Backward Pass) ---

        # Optional: Print loss for the first accumulation step for monitoring
        if args.verbose and i == 0:
            print(f"Accumulation step {i+1}/{accumulation_steps}, current mini-batch loss: {loss.item():.4f}")

    # --- Optimization 2 (Optimizer Step) ---
    # Update model weights after accumulating gradients for 'accumulation_steps' steps.
    # If scaler is enabled, use scaler.step() which also unscales gradients.
    if scaler:
         scaler.step(optimizer_cl)
         scaler.step(optimizer_learner)
         # Update the scaler for the next iteration
         scaler.update()
    else:
         # Standard optimizer step if not using mixed precision
         optimizer_cl.step()
         optimizer_learner.step()
    # --- End Optimization 2 (Optimizer Step) ---

    # --- Optimization 1 (Return Value) ---
    # Return the accumulated loss and the adjacency matrix from the LAST accumulation step.
    # No need for an extra forward pass after the loop.
    return total_loss, last_Adj
    # --- End Optimization 1 ---