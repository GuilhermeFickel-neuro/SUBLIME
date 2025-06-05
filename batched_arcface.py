import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchedArcFaceLayer(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5, easy_margin=False, batch_size=1000):
        """
        Memory-efficient ArcFace layer that processes data in batches
        
        Args:
            in_features: Size of input features
            out_features: Number of classes (rows in dataset)
            scale: Scale factor for cosine values (s in the paper)
            margin: Angular margin to enforce separation (m in the paper)
            easy_margin: Use the "easy margin" version
            batch_size: Number of classes to process at once
        """
        super(BatchedArcFaceLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.easy_margin = easy_margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(math.pi - margin))
        self.mm = torch.sin(torch.tensor(math.pi - margin)) * margin
        self.batch_size = batch_size
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, input, label=None):
        """Forward pass with batched computation for memory efficiency"""
        # Normalize input features
        normalized_input = F.normalize(input)
        
        # For inference mode (no labels)
        if label is None:
            # Even for inference, we'll use batching for large class counts
            if self.out_features <= self.batch_size:
                # If we have fewer classes than batch size, just do regular computation
                cosine = F.linear(normalized_input, F.normalize(self.weight))
                return cosine * self.scale
            else:
                # Process in batches
                n_samples = input.shape[0]
                output = torch.zeros(n_samples, self.out_features, device=input.device)
                
                for i in range(0, self.out_features, self.batch_size):
                    end_idx = min(i + self.batch_size, self.out_features)
                    # Get weight batch
                    weight_batch = F.normalize(self.weight[i:end_idx])
                    # Compute partial cosine similarity
                    batch_output = F.linear(normalized_input, weight_batch)
                    # Store in output
                    output[:, i:end_idx] = batch_output
                
                return output * self.scale
        
        # For training with labels
        n_samples = input.shape[0]
        output = torch.zeros(n_samples, self.out_features, device=input.device)
        
        # Normalize all weights at once to ensure consistency
        normalized_weight = F.normalize(self.weight)
        
        # Process in batches for the weight matrix
        for i in range(0, self.out_features, self.batch_size):
            end_idx = min(i + self.batch_size, self.out_features)
            
            # Get weight batch
            weight_batch = normalized_weight[i:end_idx]
            
            # Compute cosine similarity for this batch
            batch_cosine = F.linear(normalized_input, weight_batch)
            
            # Apply margin only for the classes that match the labels
            # First create a mask for relevant labels in this batch
            label_in_batch = (label >= i) & (label < end_idx)
            
            # Process samples with labels in current batch
            if label_in_batch.any():
                # Get samples with labels in this batch
                batch_samples = torch.where(label_in_batch)[0]
                
                # Process each sample in this mini-batch
                for sample_idx in batch_samples:
                    # Get the true class index relative to batch
                    true_cls_idx_in_batch = label[sample_idx].item() - i
                    
                    # Get cosine value for true class
                    cos_theta = batch_cosine[sample_idx, true_cls_idx_in_batch].item()
                    
                    # Calculate sin value 
                    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
                    
                    # Apply angular margin
                    phi = cos_theta * self.cos_m.item() - sin_theta * self.sin_m.item()
                    
                    # Apply easy margin if specified
                    if self.easy_margin:
                        phi = cos_theta if cos_theta > 0 else phi
                    else:
                        phi = phi if cos_theta > self.th.item() else cos_theta - self.mm.item()
                    
                    # Replace original cosine with phi for true class
                    batch_cosine[sample_idx, true_cls_idx_in_batch] = phi
            
            # Store in output
            output[:, i:end_idx] = batch_cosine
        
        # Apply scaling
        output = output * self.scale
        
        return output

def batched_arcface_loss(self, model, graph_learner, features, anchor_adj, labels, args):
    """
    Compute ArcFace loss using batched processing to reduce memory usage
    
    Args:
        self: Experiment instance
        model: The GCL model with ArcFace
        graph_learner: The graph learner model
        features: Input features
        anchor_adj: Anchor adjacency matrix
        labels: Class labels
        args: Arguments containing additional parameters
        
    Returns:
        Tuple of (loss, learned_adjacency_matrix)
    """
    # First calculate the original contrastive loss
    # This part doesn't use ArcFace so it should be memory-efficient
    if hasattr(self, 'loss_gcl'):
        contrastive_loss, learned_adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)
    else:
        # Fallback in case loss_gcl is not available
        print("Warning: loss_gcl method not found. Using default contrastive loss calculation.")
        z1, _ = model(features, anchor_adj, 'anchor')
        
        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)
            
        z2, _ = model(features, learned_adj, 'learner')
        contrastive_loss = model.calc_loss(z1, z2)
    
    # Get arcface weight or use default
    if not hasattr(args, 'arcface_weight'):
        arcface_weight = 1.0  # Default weight
    else:
        arcface_weight = args.arcface_weight
    
    # For ArcFace, we need a separate forward pass to get the embeddings
    z, embedding = model.encoder(features, learned_adj, 'learner')
    
    # Use the batched ArcFace layer
    arcface_output = model.arcface(embedding, labels)
    
    # Calculate cross-entropy loss
    criterion = nn.CrossEntropyLoss()
    arcface_loss = criterion(arcface_output, labels)
    
    # Combine both losses
    combined_loss = contrastive_loss + arcface_weight * arcface_loss
    
    if args.verbose and (not hasattr(args, '_loss_printed') or not args._loss_printed):
        print(f"Combined loss: contrastive_loss={contrastive_loss.item():.4f}, arcface_loss={arcface_loss.item():.4f}, weight={arcface_weight}")
        args._loss_printed = True
        
    return combined_loss, learned_adj