import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class SampledArcFaceLayer(nn.Module):
    """
    ArcFace layer that uses a fixed subset of classes to significantly reduce memory usage.
    
    This implementation reduces the memory requirements by:
    1. Only computing loss for a fixed subset of classes (e.g., 5,000 out of 35,000)
    2. Using a smaller weight matrix
    3. Generating smaller cosine similarity and one-hot matrices
    
    The subset of classes is chosen at initialization and remains fixed throughout training,
    making this more like a semi-supervised learning approach.
    """
    def __init__(self, in_features, max_classes, num_samples=5000,
                 scale=30.0, margin=0.5, easy_margin=False, seed=42):
        """
        Args:
            in_features: Size of input features
            max_classes: Maximum number of classes (total unique classes)
            num_samples: Number of classes to sample (e.g., 5000 instead of all classes)
            scale: Scale factor for cosine values (default: 30.0)
            margin: Margin to separate classes (default: 0.5)
            easy_margin: Use easy margin (default: False)
            seed: Random seed for reproducible sampling (default: 42)
        """
        super(SampledArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.max_classes = max_classes
        self.num_samples = min(num_samples, max_classes)  # Can't sample more than max classes
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        
        # Fix the random seed for reproducible sampling
        torch.manual_seed(seed)
        
        # Sample a fixed subset of classes that will be used throughout training
        self.sampled_classes = self._sample_fixed_classes(max_classes, num_samples)
        
        # Create mappings between original class indices and sampled indices
        self.class_mapping = {int(original_idx): i for i, original_idx in enumerate(self.sampled_classes)}
        self.inverse_mapping = {i: int(original_idx) for i, original_idx in enumerate(self.sampled_classes)}
        
        # Create a weight matrix only for the sampled classes
        # This significantly reduces memory usage
        self.weight = nn.Parameter(torch.Tensor(num_samples, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Compatibility attributes to match standard ArcFace interface
        self.out_features = num_samples  # This was missing and causing the error
        self.s = scale  # Alternative name used in some code
        self.m = margin  # Alternative name used in some code
        
        # Restore the random seed
        torch.manual_seed(torch.initial_seed())
        
        print(f"Fixed Sampled ArcFace initialized with {num_samples} out of {max_classes} classes")
    
    def _sample_fixed_classes(self, max_classes, num_samples):
        """Sample a fixed subset of classes that will be used throughout training"""
        # Generate indices for all possible classes
        all_classes = torch.arange(max_classes)
        
        # Randomly select a subset
        perm = torch.randperm(max_classes)
        sampled_classes = all_classes[perm[:num_samples]]
        
        return sampled_classes
        
    def forward(self, embeddings, labels):
        """
        Forward pass with fixed subset of classes
        
        Args:
            embeddings: Feature embeddings (batch_size, in_features)
            labels: Target labels (batch_size,)
            
        Returns:
            logits: Output logits (batch_size, num_samples)
            remapped_labels: Remapped labels for the sampled classes
        """
        # Remap original labels to the sampled class indices
        remapped_labels = torch.full_like(labels, -1)  # Initialize with -1 (invalid)
        
        # For each label in the batch
        for i, label in enumerate(labels):
            label_int = int(label.item())
            
            # If the label is in our sampled classes, use the mapped index
            if label_int in self.class_mapping:
                remapped_labels[i] = self.class_mapping[label_int]
        
        # Normalize features and weights
        embeddings_norm = F.normalize(embeddings, dim=1)
        weights_norm = F.normalize(self.weight, dim=1)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings_norm, weights_norm)
        
        # Apply ArcFace formula
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute phi (add angular margin)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding for valid labels
        one_hot = torch.zeros_like(cosine)
        valid_indices = (remapped_labels != -1)
        
        # Only set one_hot to 1 for valid labels
        for i, label in enumerate(remapped_labels):
            if label.item() != -1:
                one_hot[i, label] = 1
        
        # Apply one-hot encoding
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by s
        output = output * self.scale
        
        return output, remapped_labels

def arcface_loss_with_sampling(outputs, labels):
    """
    Compute CrossEntropyLoss for ArcFace outputs but only for valid labels
    
    Args:
        outputs: Output logits from SampledArcFaceLayer
        labels: Remapped labels for the sampled classes
        
    Returns:
        loss: CrossEntropyLoss
    """
    # Filter out invalid labels (-1)
    valid_mask = (labels != -1)
    
    if not torch.any(valid_mask):
        # If no valid labels, return zero loss
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)
    
    # Only compute loss for valid samples
    valid_outputs = outputs[valid_mask]
    valid_labels = labels[valid_mask]
    
    # Apply standard cross entropy loss
    loss = F.cross_entropy(valid_outputs, valid_labels)
    
    return loss