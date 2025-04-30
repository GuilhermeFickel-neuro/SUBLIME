import argparse
import copy
from datetime import datetime
import math
import wandb # Add wandb import

from tqdm import tqdm # Keep import for now, might be used elsewhere
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dgl
from data_loader import load_data

import random
import os

# Add sampled ArcFace imports
from sampled_arcface import SampledArcFaceLayer, arcface_loss_with_sampling
# Import memory optimization utilities (only gradient accumulation remains relevant)
from memory_optimizations import GradientAccumulation # Keep if grad_accumulation is used

EOS = 1e-10

class Experiment:
    def __init__(self, device=None):
        super(Experiment, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_cuda_memory_tracking = torch.cuda.is_available() and self.device.type == 'cuda'

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def _log_vram(self, label=""):
        """Helper function to log VRAM usage using tqdm.write if CUDA is active."""
        if self.use_cuda_memory_tracking:
            # Reset peak stats before logging to get current usage accurately for the label
            # torch.cuda.reset_peak_memory_stats(self.device) # Resetting might obscure peak usage between calls
            allocated_mem = torch.cuda.memory_allocated(self.device)
            max_allocated_mem = torch.cuda.max_memory_allocated(self.device) # Peak since last reset
            reserved_mem = torch.cuda.memory_reserved(self.device)
            max_reserved_mem = torch.cuda.max_memory_reserved(self.device) # Peak since last reset
            # Use standard print instead of tqdm.write
            # print(
            #     f"[VRAM Log @ {label}] Allocated: {allocated_mem / (1024**2):.2f} MB "
            #     f"(Peak Allocated: {max_allocated_mem / (1024**2):.2f} MB) | "
            #     f"Reserved: {reserved_mem / (1024**2):.2f} MB "
            #     f"(Peak Reserved: {max_reserved_mem / (1024**2):.2f} MB)"
            # )
            # Optionally reset peak stats *after* logging if you want peaks *between* logs
            # torch.cuda.reset_peak_memory_stats(self.device)

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def loss_binary_cls(self, cls_output, binary_labels):
        """
        Calculate binary classification loss
        
        Args:
            cls_output: Binary classification logits (N, 1)
            binary_labels: Binary labels (N,) with values 0 or 1
        
        Returns:
            loss: Binary cross entropy loss
            accuracy: Classification accuracy
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(cls_output.squeeze())
        
        # Calculate binary cross entropy loss
        loss = F.binary_cross_entropy(probs, binary_labels.float())
        
        # Calculate accuracy
        predictions = (probs >= 0.5).float()
        correct = (predictions == binary_labels.float()).float().sum()
        accuracy = correct / binary_labels.size(0)
        
        return loss, accuracy

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args):
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        # Get projection (z1) and embedding (emb1) from anchor view
        z1, emb1 = model(features_v1, anchor_adj, 'anchor') # Modified to get embedding

        # view 2: learned graph
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        # Get projection (z2) and embedding (emb2) from learner view
        z2, emb2 = model(features_v2, learned_adj, 'learner') # Modified to get embedding

        # compute loss (contrastive)
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        # Return contrastive loss, learned adj, and the learner embedding
        # The learner embedding (emb2) will be used by loss_arcface_batched if called
        return loss, learned_adj, emb2

    def loss_gcl_with_classification(self, model, graph_learner, features, anchor_adj, combined_labels, args):
        """
        Combined loss function: Contrastive on ALL data, Classification on subset.

        Args:
            model: The GCL model with classification head
            graph_learner: Graph structure learning module
            features: COMBINED node features (e.g., 57k)
            anchor_adj: COMBINED anchor adjacency matrix
            combined_labels: COMBINED labels (Tensor, -1 for main, 0/1 for annotated)
            args: Training arguments

        Returns:
            total_loss: Combined loss from contrastive learning and masked classification
            learned_adj: Learned adjacency matrix from the COMBINED data
            cls_accuracy: Accuracy of binary classification on the annotated subset
        """
        # --- Contrastive Part (using ALL features and anchor_adj) ---
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)
        # Get projection (z1) from anchor view
        z1, _, _, _ = model(features_v1, anchor_adj, 'anchor', include_features=True)

        # view 2: learned graph (from ALL features)
        if args.maskfeat_rate_learner:
            mask, _ = get_feat_mask(features, args.maskfeat_rate_learner)
            features_v2 = features * (1 - mask)
        else:
            features_v2 = copy.deepcopy(features)

        # Learn graph structure from ALL features (e.g., 57k graph)
        # Note: FGP learner uses its stored initial graph, others compute dynamically
        learned_adj = graph_learner(features)
        if not args.sparse:
            learned_adj = symmetrize(learned_adj)
            learned_adj = normalize(learned_adj, 'sym', args.sparse)

        # Get projection (z2) and classification output from learner view
        # This forward pass processes ALL nodes
        z2, _, _, cls_output = model(features_v2, learned_adj, 'learner', include_features=True)

        # Compute contrastive loss (using z1, z2 from ALL features)
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, args.contrast_batch_size)
            contrastive_loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            contrastive_loss = model.calc_loss(z1, z2)


        # --- Masked Classification Part (using subset of outputs based on combined_labels) ---
        cls_loss = torch.tensor(0.0, device=features.device) # Default to 0 if no valid labels
        cls_accuracy = torch.tensor(0.0, device=features.device)

        if cls_output is not None and combined_labels is not None:
            # Create a mask for nodes that have valid binary labels (0 or 1)
            classification_mask = (combined_labels != -1)

            if classification_mask.any():
                # Select the logits and labels for the annotated nodes
                masked_logits = cls_output[classification_mask]
                masked_labels = combined_labels[classification_mask]

                # Compute classification loss ONLY on the masked subset
                cls_loss, cls_accuracy = self.loss_binary_cls(masked_logits, masked_labels)
            else:
                if args.verbose:
                    print("Warning: Classification head is active, but no valid binary labels (0 or 1) found in combined_labels.")


        # --- Combine losses ---
        # Make sure cls_loss requires grad if it contributed to the graph
        if not cls_loss.requires_grad and contrastive_loss.requires_grad:
            cls_loss = cls_loss.detach().requires_grad_(False) # Ensure it doesn't affect contrastive grads if it was 0

        total_loss = contrastive_loss + args.annotation_loss_weight * cls_loss

        print(f"Combined loss: contrastive={contrastive_loss.item():.4f} (all_data), classification={cls_loss.item():.4f} (annotated_subset), "
                f"weight={args.annotation_loss_weight}, cls_accuracy={cls_accuracy.item():.4f}")

        # Return total loss, the LEARNED adj from COMBINED data, and cls accuracy on subset
        return total_loss, learned_adj, cls_accuracy

    def loss_arcface_sampled(self, model, graph_learner, features, anchor_adj, labels, args):
        """
        Memory-efficient implementation of ArcFace loss using class sampling.
        This computes contrastive loss first, then computes ArcFace loss on the
        learned graph embedding using a subset of classes.
        """
        # 1. Calculate contrastive loss and get learned adj/embedding
        # We need the embedding from the learned graph view for ArcFace
        contrastive_loss, learned_adj, learner_embedding = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

        # 2. Ensure the model is configured for sampled ArcFace
        if not (hasattr(model, 'use_arcface') and model.use_arcface and 
                hasattr(model, 'sampled_arcface') and model.sampled_arcface and 
                isinstance(model.arcface, SampledArcFaceLayer)):
            raise ValueError("Model is not correctly configured for Sampled ArcFace loss.")

        # 3. Get ArcFace weight
        arcface_weight = args.arcface_weight if hasattr(args, 'arcface_weight') else 1.0

        # 4. Forward pass through the SampledArcFace layer
        # The layer itself handles the sampling based on its initialization
        arcface_output, sampled_labels = model.arcface(learner_embedding, labels)

        # 5. Calculate the sampled ArcFace loss
        arcface_loss = arcface_loss_with_sampling(arcface_output, sampled_labels)

        # 6. Combine losses
        # Ensure tensors before combining if necessary (might depend on loss_gcl output type)
        if not isinstance(contrastive_loss, torch.Tensor):
            contrastive_loss = torch.tensor(contrastive_loss, device=arcface_output.device, requires_grad=True)
        if not isinstance(arcface_loss, torch.Tensor):
             # arcface_loss_with_sampling should return a tensor, but check just in case
             arcface_loss = torch.tensor(arcface_loss, device=arcface_output.device, requires_grad=True)

        combined_loss = contrastive_loss + arcface_weight * arcface_loss
        
        print(f"Sampled ArcFace Combined loss: contrastive={contrastive_loss.item():.4f}, arcface_sampled={arcface_loss.item():.4f}, weight={arcface_weight}")
        
        return combined_loss, learned_adj

    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse,
                    use_layer_norm=bool(args.use_layer_norm), use_residual=bool(args.use_residual))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_cls, weight_decay=args.w_decay_cls)

        bad_counter = 0
        best_val = 0
        best_model = None

        model = model.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)

        for epoch in range(1, args.epochs_cls + 1):
            model.train()
            loss, accu = self.loss_cls(model, train_mask, features, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                val_loss, accu = self.loss_cls(model, val_mask, features, labels)
                if accu > best_val:
                    bad_counter = 0
                    best_val = accu
                    best_model = copy.deepcopy(model)
                else:
                    bad_counter += 1

                if bad_counter >= args.patience_cls:
                    break

        best_model.eval()
        test_loss, test_accu = self.loss_cls(best_model, test_mask, features, labels)
        return best_val, test_accu, best_model

    def save_model(self, model, graph_learner, features, adj, sparse, args=None, output_dir='saved_models'):
        """
        Save model, graph learner, features and adjacency matrix
        
        Args:
            model: The GCL model
            graph_learner: The graph learner model
            features: Node features
            adj: Adjacency matrix
            sparse: Whether the adjacency matrix is sparse
            args: Arguments from the train function
            output_dir: Directory to save models
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
        torch.save(graph_learner.state_dict(), os.path.join(output_dir, 'graph_learner.pt'))
        
        # Save features
        torch.save(features.cpu(), os.path.join(output_dir, 'features.pt'))
        
        # Save adjacency matrix based on sparse/dense format
        if sparse:
            if isinstance(adj, dgl.DGLGraph):
                # For DGL graph, save edges and weights
                torch.save({
                    'edges': adj.edges(),
                    'weights': adj.edata['w'],
                    'num_nodes': adj.num_nodes()
                }, os.path.join(output_dir, 'adjacency.pt'))
            else:
                # For torch sparse tensor
                torch.save(adj, os.path.join(output_dir, 'adjacency.pt'))
        else:
            # For dense adjacency
            torch.save(adj.cpu(), os.path.join(output_dir, 'adjacency.pt'))
        
        # Save config using the dedicated function
        with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
            self.save_model_config(f, args, model, graph_learner, features, sparse, adj)

    def save_model_config(self, f, args, model, graph_learner, features, sparse, initial_graph_data):
        # Save all args parameters if provided
        if args:
            for arg_name, arg_value in vars(args).items():
                f.write(f'{arg_name}: {arg_value}\n')
        # Basic parameters
        f.write(f'sparse: {sparse}\n')
        f.write(f'feature_dim: {features.shape[1]}\n')
        f.write(f'num_nodes: {features.shape[0]}\n')
        # Get model architecture parameters by inspecting the encoder
        encoder = model.encoder
        f.write(f'nlayers: {len(encoder.gnn_encoder_layers)}\n')
        # Get hidden_dim from the first layer's linear out_features
        if hasattr(encoder.gnn_encoder_layers[0], 'linear'):
            hidden_dim = encoder.gnn_encoder_layers[0].linear.out_features
            emb_dim = encoder.gnn_encoder_layers[-1].linear.out_features
        else: # Fallback for potentially different layer types
            hidden_dim = args.hidden_dim if args else 128 # Example fallback
            emb_dim = args.rep_dim if args else 32 # Example fallback
        f.write(f'hidden_dim: {hidden_dim}\n')
        f.write(f'emb_dim: {emb_dim}\n')
        # Get proj_dim from the first layer of the projection head
        if hasattr(encoder.proj_head[0], 'out_features'):
            proj_dim = encoder.proj_head[0].out_features
        else: # Fallback
            proj_dim = args.proj_dim if args else 32
        f.write(f'proj_dim: {proj_dim}\n')
        f.write(f'dropout: {encoder.dropout}\n')
        f.write(f'dropout_adj: {encoder.dropout_adj_p}\n')
        # Add flags to config
        f.write(f'use_layer_norm: {encoder.use_layer_norm}\n')
        f.write(f'use_residual: {encoder.use_residual}\n')
        # Graph learner parameters
        if isinstance(graph_learner, FGP_learner):
            learner_type = 'fgp'
        elif isinstance(graph_learner, MLP_learner):
            learner_type = 'mlp'
        elif isinstance(graph_learner, ATT_learner):
            learner_type = 'att'
        elif isinstance(graph_learner, GNN_learner):
            learner_type = 'gnn'
        else:
            learner_type = 'unknown'
        f.write(f'type_learner: {learner_type}\n')
        if hasattr(graph_learner, 'k'):
            f.write(f'k: {graph_learner.k}\n')
        if hasattr(graph_learner, 'knn_metric'):
            f.write(f'sim_function: {graph_learner.knn_metric}\n')
        if hasattr(graph_learner, 'mlp_act'):
            f.write(f'activation_learner: {graph_learner.mlp_act}\n')
        # Save ArcFace parameters if the model has ArcFace support
        if hasattr(model, 'use_arcface') and model.use_arcface:
            f.write(f'use_arcface: True\n')
            f.write(f'num_classes: {model.arcface.weight.shape[0]}\n')
            f.write(f'arcface_scale: {model.arcface.s}\n')
            f.write(f'arcface_margin: {model.arcface.m}\n')
            # Add sampled arcface config saving
            if hasattr(model, 'sampled_arcface') and model.sampled_arcface:
                f.write(f'use_sampled_arcface: True\n')
                if hasattr(model.arcface, 'num_samples'):
                    f.write(f'arcface_num_samples: {model.arcface.num_samples}\n')
        
        # Save classification head parameters if present
        if hasattr(model, 'use_classification_head') and model.use_classification_head:
            f.write(f'use_classification_head: True\n')
            if hasattr(model, 'classification_dropout') and hasattr(model.classification_dropout, 'p'):
                f.write(f'classification_dropout: {model.classification_dropout.p}\n')
            if hasattr(model, 'classification_head_layers'):
                f.write(f'classification_head_layers: {model.classification_head_layers}\n')

        # Add 'i' parameter for FGP learner if applicable
        if isinstance(graph_learner, FGP_learner) and hasattr(graph_learner, 'i'):
            f.write(f'fgp_elu_alpha: {graph_learner.i}\n')

    def process_new_point(self, new_point, model, graph_learner, features, adj, sparse, replace_idx=None, faiss_index=None):
        """
        Process a new data point by replacing an existing point and extract its embedding features
        
        Args:
            new_point: Tensor of shape [feature_dim] containing the new point features
            model: The trained GCL model
            graph_learner: The trained graph learner
            features: The existing node features
            adj: The current adjacency matrix
            sparse: Whether the adjacency is sparse
            replace_idx: Index of the point to replace. If None, will find the most similar point.
            faiss_index: Optional pre-built FAISS index for the graph_learner to use
            
        Returns:
            dict: A dictionary containing:
                - 'embedding': The node embedding for the new point
                - 'classification': Binary classification prediction (if model has classification head)
        """
        # Ensure model and graph_learner are in evaluation mode
        model.eval()
        graph_learner.eval()
        
        # Reshape new point if needed
        if len(new_point.shape) == 1:
            new_point = new_point.unsqueeze(0)  # Add batch dimension
        elif new_point.shape[0] > 1:
            # If multiple points were passed, only use the first one
            new_point = new_point[0].unsqueeze(0)
            
        # Move to appropriate device
        new_point = new_point.to(self.device)
        
        # If no specific index provided, find the most similar point to replace
        if replace_idx is None:
            # Calculate cosine similarity between new point and all existing points
            normalized_features = F.normalize(features, p=2, dim=1)
            normalized_new_point = F.normalize(new_point, p=2, dim=1)
            similarities = torch.mm(normalized_new_point, normalized_features.t())
            
            # Get the index of the most similar point
            replace_idx = torch.argmax(similarities).item()
        
        # Create a copy of features and replace the selected point
        modified_features = features.clone()
        modified_features[replace_idx] = new_point
        
        # Generate new adjacency matrix using the graph learner
        with torch.no_grad():
            new_adj = graph_learner(modified_features, faiss_index=faiss_index)
            
            # Process adjacency matrix based on sparse flag
            if not sparse:
                new_adj = symmetrize(new_adj)
                new_adj = normalize(new_adj, 'sym', sparse)
            
            # Get embeddings for all nodes including the new one
            # Check if model has classification head to determine the right forward call
            has_classification_head = hasattr(model, 'use_classification_head') and model.use_classification_head
            
            if has_classification_head:
                # Call with include_features=True to get classification output
                z, embedding, _, classification_output = model(modified_features, new_adj, include_features=True)
                
                # Get binary classification prediction if available
                if classification_output is not None:
                    classification_prob = torch.sigmoid(classification_output[replace_idx]).item()
                    classification_pred = 1 if classification_prob >= 0.5 else 0
                else:
                    classification_prob = None
                    classification_pred = None
            else:
                # Standard call for models without classification head
                z, embedding = model(modified_features, new_adj)
                classification_prob = None
                classification_pred = None
            
            # Create return dictionary with embedding and optional classification results
            result = {
                'embedding': embedding[replace_idx].detach(),
                'embedding_vector': embedding[replace_idx].detach().cpu().numpy()
            }
            
            # Add classification results if available
            if classification_prob is not None:
                result['classification_probability'] = classification_prob
                result['classification_prediction'] = classification_pred
            
            return result

    def train(self, args, load_data_fn=None):
        """
        Train the model
        
        Args:
            args: Arguments for training
            load_data_fn: Function to load the dataset. Should return:
                (features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, initial_graph)
        """
        self._log_vram("Train Start") # Log at the very beginning

        if load_data_fn is None:
            raise ValueError("Must provide a data loading function")
            
        # Initialize wandb
        wandb.init(
            project="SUBLIME_GCL", # Or your preferred project name
            config=args,
            # mode="offline" # Uncomment this line to run offline
        )
            
        # Load data - Expected return: features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, initial_graph_data
        # Note: nclasses might be n_clusters from args in clustering mode
        features, nfeats, labels, nclasses_or_clusters, train_mask, val_mask, test_mask, initial_graph_data = load_data_fn(args)
        
        # Assign nclasses based on downstream task
        if args.downstream_task == 'clustering':
            print("Warning: Labels provided but downstream task is clustering. Labels will be used for evaluation only.")
        elif labels is not None:
            # 'labels' now contains combined labels (-1 for main, 0/1 for annotated)
            # If annotated data was provided, nclasses_or_clusters should be 2 (binary task)
            # If only main data (all -1s), nclasses_or_clusters should be n_clusters or n_nodes
            # The load_person_data function now correctly sets nclasses_or_clusters based on this.
            nclasses = nclasses_or_clusters # Use value determined by load_person_data
        else:
            # This case should ideally not happen if load_person_data returns correctly
            # Handle case where labels are None (e.g., only main data, no annotation requested)
            # Or if ArcFace is used without any real labels or annotation
            nclasses = nclasses_or_clusters # Should be args.n_clusters or features.shape[0]
            if args.use_arcface and labels is None: # labels SHOULD NOT be None if we combine
                nclasses = features.shape[0] # ArcFace uses row indices over the combined set
            elif args.verbose:
                print("Warning: Labels are None or downstream task needs clarification. Using nclasses/clusters = ", nclasses_or_clusters)
            nclasses = nclasses_or_clusters

        # Labels processing for ArcFace (if needed)
        # 'labels' here refers to the combined labels tensor potentially containing -1
        if args.use_arcface:
            # If using ArcFace, typically requires class indices for *all* samples.
            # The common practice is to use row indices if no ground truth classes exist
            # for the full dataset (main + annotated).
            if args.verbose:
                print("ArcFace is enabled. Using combined row indices (0 to N-1) as labels for ArcFace loss.")
            # Create row indices for the *combined* feature set
            arcface_labels = torch.arange(features.shape[0], device=self.device)
            # nclasses for ArcFace layer should be the total number of nodes
            nclasses = features.shape[0]
        else:
            # If not using ArcFace, we don't need specific labels for it.
            # The 'labels' variable holds the combined labels (-1, 0, 1) for the classification head.
            arcface_labels = None # Explicitly set to None
            # nclasses remains what was determined earlier (e.g., 2 for binary classification eval, or n_clusters)

        # The variable `labels` now holds the combined labels (-1, 0, 1) from load_person_data
        # The variable `arcface_labels` holds the row indices (0..N-1) if ArcFace is used.
        # The variable `nclasses` holds the dimension for ArcFace OR the number for downstream evaluation.

        # Downstream task setup (remains the same)
        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):
            self.setup_seed(trial)
            self._log_vram(f"Trial {trial+1} Start") # Log at start of trial

            # --- Initialize Anchor Adjacency --- 
            # The initial_graph_data IS the raw anchor adjacency before normalization.
            # It might be a KNN graph (from load_person_data) or an identity/original graph (from other loaders).
            anchor_adj_raw = initial_graph_data
            # Ensure it's on the correct device (load_data should handle this, but double-check)
            if isinstance(anchor_adj_raw, torch.Tensor):
                 anchor_adj_raw = anchor_adj_raw.to(self.device)
            elif isinstance(anchor_adj_raw, dgl.DGLGraph):
                 anchor_adj_raw = anchor_adj_raw.to(self.device)
            else:
                 # Handle potential sparse scipy matrix if loader didn't convert fully
                 try:
                      if args.sparse:
                           anchor_adj_raw = sparse_mx_to_torch_sparse_tensor(anchor_adj_raw).to(self.device)
                      else:
                           anchor_adj_raw = torch.FloatTensor(anchor_adj_raw.todense()).to(self.device)
                 except AttributeError:
                      raise TypeError(f"Unsupported type for initial_graph_data: {type(anchor_adj_raw)}")

            # Normalize the combined initial graph
            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            # Convert to DGL graph if sparse (<<< THIS WAS THE MISSING STEP)
            if args.sparse:
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            # --- Initialize Graph Learner ---
            # The graph learner type determines how the learned graph is generated.
            # FGP uses the initial graph data directly, others compute dynamically.
            if args.type_learner == 'fgp':
                # FGP requires the raw (pre-normalized, pre-DGL conversion) initial data
                # It expects a sparse tensor if args.sparse is True
                fgp_initial_data = anchor_adj_raw # Use the raw data before normalization/DGL
                if args.sparse and not isinstance(fgp_initial_data, torch.Tensor) or not fgp_initial_data.is_sparse:
                     # Ensure it's a sparse tensor if needed by FGP
                     # This should be guaranteed by load_person_data returning sparse tensor when args.sparse=1
                     print("Warning: Initial data for FGP sparse mode is not a sparse tensor. Attempting conversion.")
                     try:
                          # Example conversion, might need adjustment based on actual type
                          if isinstance(fgp_initial_data, dgl.DGLGraph): # If it somehow became DGL
                              fgp_initial_data = dgl_graph_to_torch_sparse(fgp_initial_data)
                          elif isinstance(fgp_initial_data, np.ndarray):
                              fgp_initial_data = sparse_mx_to_torch_sparse_tensor(sp.csr_matrix(fgp_initial_data))
                          # Add other potential conversions if necessary
                          if not fgp_initial_data.is_sparse:
                              raise TypeError("Could not convert FGP initial data to sparse tensor.")
                     except Exception as e:
                          raise TypeError(f"Failed to ensure FGP initial data is sparse: {e}")

                graph_learner = FGP_learner(
                    k=args.k, knn_metric=args.sim_function, i=6,
                    sparse=args.sparse, initial_graph_data=fgp_initial_data # Pass raw/sparse tensor
                )
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, args.knn_threshold_type, args.knn_std_dev_factor,
                                     chunk_size=args.graph_learner_chunk_size) # Pass chunk size
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                 # GNN Learner needs the *normalized DGL graph* anchor_adj
                 if args.sparse and not isinstance(anchor_adj, dgl.DGLGraph):
                      raise TypeError("GNN Learner in sparse mode requires a DGL graph as anchor_adj.")
                 elif not args.sparse and not isinstance(anchor_adj, torch.Tensor):
                     raise TypeError("GNN Learner in dense mode requires a Tensor as anchor_adj.")

                 graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                      args.activation_learner, anchor_adj)

            # Determine if we should use classification head based on whether combined labels contain valid binary labels
            use_classification_head = labels is not None and (labels != -1).any().item()

            # Shared GCL parameters
            gcl_params = {
                'nlayers': args.nlayers,
                'in_dim': nfeats,
                'hidden_dim': args.hidden_dim,
                'emb_dim': args.rep_dim,
                'proj_dim': args.proj_dim,
                'dropout': args.dropout,
                'dropout_adj': args.dropedge_rate,
                'sparse': args.sparse,
                'use_layer_norm': bool(args.use_layer_norm),
                'use_residual': bool(args.use_residual),
                'use_arcface': args.use_arcface,
                'use_classification_head': use_classification_head
            }

            if args.use_arcface:
                gcl_params.update({
                    'num_classes': nclasses, # nclasses is N (total nodes) if using ArcFace
                    'arcface_scale': args.arcface_scale,
                    'arcface_margin': args.arcface_margin,
                    'use_sampled_arcface': args.use_sampled_arcface if hasattr(args, 'use_sampled_arcface') else False,
                    'arcface_num_samples': args.arcface_num_samples if hasattr(args, 'arcface_num_samples') else None
                })
                if args.verbose:
                    arcface_msg = "Using Sampled ArcFace" if gcl_params['use_sampled_arcface'] else "Using Standard ArcFace"
                    print(f"{arcface_msg} loss with scale={args.arcface_scale}, margin={args.arcface_margin}")
            
            if use_classification_head:
                gcl_params.update({
                    'classification_dropout': args.classification_dropout,
                    'classification_head_layers': args.classification_head_layers
                })

            # Initialize the model
            model = GCL(**gcl_params)

            # --- Load Pre-trained Weights (if specified) ---
            if args.load_pretrained_path and os.path.isdir(args.load_pretrained_path):
                pretrained_model_path = os.path.join(args.load_pretrained_path, 'model.pt')
                pretrained_learner_path = os.path.join(args.load_pretrained_path, 'graph_learner.pt')
                if os.path.exists(pretrained_model_path) and os.path.exists(pretrained_learner_path):
                    try:
                        print(f"Loading pre-trained model weights from {pretrained_model_path}")
                        model.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
                        print(f"Loading pre-trained graph learner weights from {pretrained_learner_path}")
                        graph_learner.load_state_dict(torch.load(pretrained_learner_path, map_location=self.device))
                        print("Successfully loaded pre-trained weights.")
                    except Exception as e:
                        print(f"Warning: Failed to load pre-trained weights from {args.load_pretrained_path}: {e}")
                else:
                    print(f"Warning: Pre-trained model path specified ({args.load_pretrained_path}), but model.pt or graph_learner.pt not found.")

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr_learner, weight_decay=args.w_decay)
            
            # --- Checkpoint Setup --- 
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            start_epoch = 0
            last_checkpoint_path = None

            # Load latest checkpoint if available
            model, graph_learner, optimizer_cl, optimizer_learner, start_epoch, loaded_anchor_adj = \
                self.load_checkpoint(model, graph_learner, optimizer_cl, optimizer_learner, args)
            
            # If checkpoint loaded an anchor_adj, use it (important for resuming bootstrapping)
            if loaded_anchor_adj is not None:
                anchor_adj = loaded_anchor_adj
            # --- End Checkpoint Setup --- 


            model = model.to(self.device)
            graph_learner = graph_learner.to(self.device)
            features = features.to(self.device) # Combined features
            self._log_vram(f"Trial {trial+1}: Data Moved to Device") # Log after moving data
            # Move combined labels and arcface_labels (if used) to device
            if labels is not None:
                labels = labels.to(self.device) # Combined labels (-1, 0, 1)
            if arcface_labels is not None:
                arcface_labels = arcface_labels.to(self.device) # Row indices (0..N-1)

            if args.downstream_task == 'classification':
                best_val = 0.0 # Initialize as float
                best_val_test = 0
                best_epoch = 0
                
            # For tracking classification performance if using annotated dataset
            if use_classification_head:
                best_cls_accuracy = 0.0
                cls_accuracies = []

            # Initialize gradient accumulation steps
            grad_accumulation_steps = args.grad_accumulation_steps if hasattr(args, 'grad_accumulation_steps') else 1
            if grad_accumulation_steps > 1 and args.verbose:
                print(f"Using gradient accumulation with {grad_accumulation_steps} steps")
                
            # --- Initialize Schedulers --- 
            scheduler_cl = None
            scheduler_learner = None
            one_cycle_defaults = {
                'pct_start': args.one_cycle_pct_start if hasattr(args, 'one_cycle_pct_start') else 0.3,
                'div_factor': args.one_cycle_div_factor if hasattr(args, 'one_cycle_div_factor') else 25.0,
                'final_div_factor': args.one_cycle_final_div_factor if hasattr(args, 'one_cycle_final_div_factor') else 10000.0
            }

            # Calculate phase durations
            phase1_epochs = args.embedding_only_epochs
            phase2_epochs = args.graph_learner_only_epochs
            phase3_epochs = args.epochs - phase1_epochs - phase2_epochs

            if phase3_epochs < 0:
                print("Warning: embedding_only_epochs + graph_learner_only_epochs exceeds total epochs. Adjusting phase durations.")
                if phase1_epochs >= args.epochs:
                    phase1_epochs = args.epochs
                    phase2_epochs = 0
                    phase3_epochs = 0
                elif phase1_epochs + phase2_epochs > args.epochs:
                    phase2_epochs = args.epochs - phase1_epochs
                    phase3_epochs = 0

            # Get tqdm iterator --- REMOVED TQDM WRAPPER
            from tqdm import tqdm # Explicit import before use
            epoch_iterator = tqdm(range(start_epoch, args.epochs), desc="Training", initial=start_epoch, total=args.epochs)
            for epoch in epoch_iterator: # Iterate over tqdm iterator
                self._log_vram(f"Epoch {epoch} Start") # Log start of epoch
                # Determine current training phase
                phase1_end = args.embedding_only_epochs
                phase2_end = phase1_end + args.graph_learner_only_epochs
                
                is_embedding_phase = epoch < phase1_end
                is_graph_learner_phase = phase1_end <= epoch < phase2_end
                is_joint_phase = epoch >= phase2_end

                # --- Initialize/Re-initialize Schedulers at Phase Start ---
                if args.use_one_cycle:
                    if epoch == 0 and phase1_epochs > 0: # Start of Phase 1
                        print(f"Initializing OneCycleLR for Model (Phase 1: {phase1_epochs} steps)")
                        scheduler_cl = lr_scheduler.OneCycleLR(
                            optimizer_cl, max_lr=args.lr, total_steps=phase1_epochs, **one_cycle_defaults
                        )
                    elif epoch == phase1_end and phase2_epochs > 0: # Start of Phase 2
                         print(f"Initializing OneCycleLR for Learner (Phase 2: {phase2_epochs} steps)")
                         scheduler_learner = lr_scheduler.OneCycleLR(
                             # Use the new lr_learner argument for the graph learner scheduler
                             optimizer_learner, max_lr=args.lr_learner, total_steps=phase2_epochs, **one_cycle_defaults
                         )
                    elif epoch == phase2_end and phase3_epochs > 0: # Start of Phase 3
                        print(f"Re-initializing OneCycleLR for Model (Phase 3: {phase3_epochs} steps)")
                        scheduler_cl = lr_scheduler.OneCycleLR(
                            optimizer_cl, max_lr=args.lr, total_steps=phase3_epochs, **one_cycle_defaults
                        )
                        print(f"Re-initializing OneCycleLR for Learner (Phase 3: {phase3_epochs} steps)")
                        scheduler_learner = lr_scheduler.OneCycleLR(
                            # Use the new lr_learner argument for the graph learner scheduler
                            optimizer_learner, max_lr=args.lr_learner, total_steps=phase3_epochs, **one_cycle_defaults
                        )

                # --- Set Model Modes and Print Phase Info ---
                if is_embedding_phase:
                    # Phase 1: Train GCL model, freeze learner
                    model.train()
                    graph_learner.eval()
                    if args.verbose and epoch == 0 and phase1_end > 0:
                        print(f"\n--- Starting Phase 1: Training Embeddings Only (Epochs 0-{phase1_end-1}) ---")
                elif is_graph_learner_phase:
                    # Phase 2: Freeze GCL model, train learner
                    model.eval() # Freeze GCL model
                    graph_learner.train()
                    if args.verbose and epoch == phase1_end:
                        print(f"\n--- Starting Phase 2: Training Graph Learner Only (Epochs {phase1_end}-{phase2_end - 1}) ---")
                else: # is_joint_phase
                    # Phase 3: Train both
                    model.train()
                    graph_learner.train()
                    if args.verbose and epoch == phase2_end:
                        print(f"\n--- Starting Phase 3: Training Model and Learner Jointly (Epochs {phase2_end}-{args.epochs-1}) ---")

                # Simplified training loop with combined loss calculation
                if grad_accumulation_steps > 1:
                    # --- Gradient Accumulation Path ---
                    accumulated_loss = 0
                    accumulated_contrastive_loss = 0.0
                    accumulated_arcface_loss = 0.0
                    accumulated_cls_loss = 0.0
                    
                    # Zero gradients based on phase
                    if is_embedding_phase or is_joint_phase: # Model trained in Phase 1 & 3
                        optimizer_cl.zero_grad()
                    if is_graph_learner_phase or is_joint_phase: # Learner trained in Phase 2 & 3
                        optimizer_learner.zero_grad()
                    # optimizer_cl.zero_grad()
                    # if not is_embedding_phase: # Only zero learner grad if training it # Old logic
                        # optimizer_learner.zero_grad()
                    self._log_vram(f"Epoch {epoch} GradAccum: Grads Zeroed (Step {i+1}/{grad_accumulation_steps})")

                    for i in range(grad_accumulation_steps):
                        # --- Forward passes ---
                        contrastive_loss = torch.tensor(0.0, device=features.device)
                        arcface_loss = torch.tensor(0.0, device=features.device)
                        cls_loss = torch.tensor(0.0, device=features.device)
                        current_cls_accuracy = torch.tensor(0.0, device=features.device)
                        emb1, cls_output1 = None, None
                        emb2, cls_output2 = None, None
                        Adj = None # Learned adj

                        # --- Phase 1: Only Anchor Graph (Train Model) ---
                        if is_embedding_phase:
                            if args.maskfeat_rate_anchor:
                                mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                                features_v1 = features * (1 - mask_v1)
                            else:
                                features_v1 = copy.deepcopy(features)
                            # Only need anchor view outputs
                            _, emb1, _, cls_output1 = model(features_v1, anchor_adj, 'anchor', include_features=True)
                            # Contrastive loss is 0
                            self._log_vram(f"Epoch {epoch} GradAccum: Phase 1 Forward Done (Step {i+1})")

                        # --- Phase 2: Anchor + Learned Graph (Train Learner, Frozen Model) ---
                        elif is_graph_learner_phase:
                            # view 1: anchor graph (Frozen Model)
                            if args.maskfeat_rate_anchor:
                               mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                               features_v1 = features * (1 - mask_v1)
                            else:
                               features_v1 = copy.deepcopy(features)
                            with torch.no_grad(): # Model is frozen
                               z1, _, _, _ = model(features_v1, anchor_adj, 'anchor', include_features=True)

                            # view 2: learned graph (Train Learner, Frozen Model)
                            if args.maskfeat_rate_learner:
                               mask_v2, _ = get_feat_mask(features, args.maskfeat_rate_learner)
                               features_v2 = features * (1 - mask_v2)
                            else:
                               features_v2 = copy.deepcopy(features)
                            # Learn graph
                            learned_adj = graph_learner(features) # Should return DGLGraph if sparse
                            if not args.sparse:
                               # Dense case (remains the same)
                               learned_adj = symmetrize(learned_adj)
                               learned_adj = normalize(learned_adj, 'sym', args.sparse)
                               Adj = learned_adj # Adj is a Tensor
                            else:
                               # Sparse case (DGL Graph)
                               Adj = learned_adj # Adj is a DGLGraph
                               # --- Check if edge weights require grad ---
                               if 'w' in Adj.edata:
                                   # Ensure the weights tensor requires gradients. It should already if
                                   # the learner calculated them using its parameters.
                                   if not Adj.edata['w'].requires_grad:
                                       # This case is unexpected if learner is trainable & creates weights.
                                       # Maybe try setting it? Or log a warning.
                                       # Let's log a warning first, as forcing requires_grad might mask issues.
                                       if args.verbose:
                                            tqdm.write(f"Warning: Epoch {epoch} [Phase 2] - Learned graph edge weights (Adj.edata['w']) do not require gradients. Grad flow to learner might be broken.") # Changed to tqdm.write
                                       # Optionally, try forcing it:
                                       # Adj.edata['w'].requires_grad_(True) # Use with caution
                               else:
                                   # If no 'w', gradient must flow via structure changes implicitly handled by DGL/learner.
                                   pass
                               # --- End check ---

                            # Get learner outputs (Frozen Model, allow graph connection)
                            # Pass the DGL graph Adj
                            z2, _, _, _ = model(features_v2, Adj, 'learner', include_features=True) # Model params frozen via eval()
                            self._log_vram(f"Epoch {epoch} GradAccum: Phase 2 Forward Done (Step {i+1})")

                            # Calculate Contrastive Loss
                            if args.contrast_batch_size:
                                node_idxs = list(range(features.shape[0]))
                                batches = split_batch(node_idxs, args.contrast_batch_size)
                                contrastive_loss = 0
                                for batch in batches:
                                    weight = len(batch) / features.shape[0]
                                    # Use z1, z2 calculated under no_grad
                                    contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
                            else:
                                contrastive_loss = model.calc_loss(z1, z2)
                            # ArcFace and Cls loss are 0 (not relevant for learner training)

                        # --- Phase 3: Anchor + Learned Graph (Train Both) ---
                        else: # is_joint_phase
                            # view 1: anchor graph
                            if args.maskfeat_rate_anchor:
                                mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                                features_v1 = features * (1 - mask_v1)
                            else:
                                features_v1 = copy.deepcopy(features)
                            z1, emb1, _, cls_output1 = model(features_v1, anchor_adj, 'anchor', include_features=True)

                            # view 2: learned graph
                            if args.maskfeat_rate_learner:
                                mask_v2, _ = get_feat_mask(features, args.maskfeat_rate_learner)
                                features_v2 = features * (1 - mask_v2)
                            else:
                                features_v2 = copy.deepcopy(features)
                            # Learn graph
                            learned_adj = graph_learner(features)
                            if not args.sparse:
                                learned_adj = symmetrize(learned_adj)
                                learned_adj = normalize(learned_adj, 'sym', args.sparse)
                            Adj = learned_adj # Store for potential use later
                            # Get learner outputs
                            z2, emb2, _, cls_output2 = model(features_v2, learned_adj, 'learner', include_features=True)
                            self._log_vram(f"Epoch {epoch} GradAccum: Phase 3 Forward Done (Step {i+1})")

                            # Calculate Contrastive Loss
                            if args.contrast_batch_size:
                                node_idxs = list(range(features.shape[0]))
                                batches = split_batch(node_idxs, args.contrast_batch_size)
                                contrastive_loss = 0
                                for batch in batches:
                                    weight = len(batch) / features.shape[0]
                                    contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
                            else:
                                contrastive_loss = model.calc_loss(z1, z2)


                        # --- Loss Calculation (Common part, adapted for phases) ---
                        # 1. Initialize total loss for this mini-step
                        mini_total_loss = torch.tensor(0.0, device=features.device)
                        self._log_vram(f"Epoch {epoch} GradAccum: Before Loss Calc (Step {i+1})")

                        # 2. Add Contrastive Loss (only in Phase 2 and 3)
                        if is_graph_learner_phase or is_joint_phase:
                            mini_total_loss += contrastive_loss
                            accumulated_contrastive_loss += contrastive_loss.item() # Accumulate for reporting

                        # 3. ArcFace Loss (if enabled, only in Phase 1 and 3)
                        current_arcface_loss = torch.tensor(0.0, device=features.device)
                        if args.use_arcface and (is_embedding_phase or is_joint_phase):
                            if hasattr(model, 'arcface') and isinstance(model.arcface, SampledArcFaceLayer):
                                # Use anchor embedding (emb1) in Phase 1, learner embedding (emb2) in Phase 3
                                embedding_for_arcface = emb1 if is_embedding_phase else emb2
                                try:
                                    # Ensure the embedding is valid before passing
                                    if embedding_for_arcface is not None:
                                        arcface_output, sampled_labels = model.arcface(embedding_for_arcface, arcface_labels)
                                        current_arcface_loss = arcface_loss_with_sampling(arcface_output, sampled_labels)
                                        mini_total_loss += args.arcface_weight * current_arcface_loss
                                        accumulated_arcface_loss += current_arcface_loss.item() # Accumulate for reporting
                                    else:
                                        if args.verbose and i == 0: print(f"Warning: Embedding for ArcFace is None in epoch {epoch}, step {i} (Phase {'1' if is_embedding_phase else '3'}). Skipping ArcFace.")
                                except Exception as e:
                                    print(f"Warning: Error during ArcFace calculation step {i} (Phase {'1' if is_embedding_phase else '3'}): {e}")
                                    current_arcface_loss = torch.tensor(0.0, device=features.device)
                            else:
                                if args.verbose and i == 0: print("Warning: use_arcface=True but SampledArcFaceLayer not found/configured correctly.")

                        # 4. Classification Loss (if enabled, only in Phase 1 and 3)
                        current_cls_loss = torch.tensor(0.0, device=features.device)
                        current_cls_accuracy = torch.tensor(0.0, device=features.device)
                        if use_classification_head and (is_embedding_phase or is_joint_phase):
                            # Use anchor cls output (cls_output1) in Phase 1, learner cls output (cls_output2) in Phase 3
                            cls_output_for_loss = cls_output1 if is_embedding_phase else cls_output2
                            if cls_output_for_loss is not None and labels is not None:
                                classification_mask = (labels != -1)
                                if classification_mask.any():
                                    masked_logits = cls_output_for_loss[classification_mask]
                                    masked_labels = labels[classification_mask]
                                    current_cls_loss, current_cls_accuracy = self.loss_binary_cls(masked_logits, masked_labels)
                                    mini_total_loss += args.annotation_loss_weight * current_cls_loss
                                    accumulated_cls_loss += current_cls_loss.item() # Accumulate for reporting
                                else:
                                    # No valid labels found this step
                                    pass
                            else:
                                # Should not happen if use_classification_head is True
                                if args.verbose and i == 0: print(f"Warning: use_classification_head=True but cls_output or labels are None (Phase {'1' if is_embedding_phase else '3'}).")


                        # Scale loss for accumulation
                        scaled_mini_loss = mini_total_loss / grad_accumulation_steps
                        self._log_vram(f"Epoch {epoch} GradAccum: After Loss Calc (Step {i+1})")

                        # Backward pass for this step
                        self._log_vram(f"Epoch {epoch} GradAccum: Before Backward (Step {i+1})")
                        scaled_mini_loss.backward()
                        self._log_vram(f"Epoch {epoch} GradAccum: After Backward (Step {i+1})")

                        accumulated_loss += mini_total_loss.item() # Accumulate total unscaled loss for reporting average

                        # Store accuracy from first step for reporting
                        if i == 0:
                           first_step_cls_accuracy = current_cls_accuracy

                    # --- After Accumulation Steps ---
                    # Gradient Clipping (apply to accumulated grads based on phase)
                    if args.clip_norm > 0:
                        if is_embedding_phase or is_joint_phase: # Clip model grads if model was trained
                             torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        if is_graph_learner_phase or is_joint_phase: # Clip learner grads if learner was trained
                            torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)

                    # Update weights based on phase
                    if is_embedding_phase or is_joint_phase: # Update model weights if model was trained
                        optimizer_cl.step()
                    if is_graph_learner_phase or is_joint_phase: # Update learner weights if learner was trained
                        optimizer_learner.step()
                    self._log_vram(f"Epoch {epoch} GradAccum: After Optimizer Step")


                    # Calculate average losses for reporting
                    loss = accumulated_loss / grad_accumulation_steps
                    avg_contrastive_loss = accumulated_contrastive_loss / grad_accumulation_steps
                    avg_arcface_loss = accumulated_arcface_loss / grad_accumulation_steps
                    avg_cls_loss = accumulated_cls_loss / grad_accumulation_steps
                    # Use accuracy from the first mini-batch for reporting consistency
                    if use_classification_head:
                        current_cls_accuracy = first_step_cls_accuracy

                else:
                    # --- Standard Training Path (no gradient accumulation) ---
                    # Zero gradients based on phase
                    if is_embedding_phase or is_joint_phase: # Model trained in Phase 1 & 3
                        optimizer_cl.zero_grad()
                    if is_graph_learner_phase or is_joint_phase: # Learner trained in Phase 2 & 3
                        optimizer_learner.zero_grad()
                    # optimizer_cl.zero_grad()
                    # if not is_embedding_phase: # Only zero learner grad if training it # Old logic
                        # optimizer_learner.zero_grad()
                    self._log_vram(f"Epoch {epoch} Standard: Grads Zeroed")

                    # --- Forward passes (Select based on phase) ---
                    contrastive_loss = torch.tensor(0.0, device=features.device)
                    arcface_loss = torch.tensor(0.0, device=features.device)
                    cls_loss = torch.tensor(0.0, device=features.device)
                    current_cls_accuracy = torch.tensor(0.0, device=features.device)
                    emb1, cls_output1 = None, None
                    emb2, cls_output2 = None, None
                    Adj = None # Learned adj

                    # --- Phase 1: Only Anchor Graph (Train Model) ---
                    if is_embedding_phase:
                       if args.maskfeat_rate_anchor:
                           mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                           features_v1 = features * (1 - mask_v1)
                       else:
                           features_v1 = copy.deepcopy(features)
                       # Only need anchor view outputs
                       _, emb1, _, cls_output1 = model(features_v1, anchor_adj, 'anchor', include_features=True)
                       Adj = None # No learned adj in Phase 1
                       self._log_vram(f"Epoch {epoch} Standard: Phase 1 Forward Done")

                    # --- Phase 2: Anchor + Learned Graph (Train Learner, Frozen Model) ---
                    elif is_graph_learner_phase:
                        # view 1: anchor graph (Frozen Model)
                        if args.maskfeat_rate_anchor:
                           mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                           features_v1 = features * (1 - mask_v1)
                        else:
                           features_v1 = copy.deepcopy(features)
                        with torch.no_grad(): # Model is frozen
                           z1, _, _, _ = model(features_v1, anchor_adj, 'anchor', include_features=True)

                        # view 2: learned graph (Train Learner, Frozen Model)
                        if args.maskfeat_rate_learner:
                           mask_v2, _ = get_feat_mask(features, args.maskfeat_rate_learner)
                           features_v2 = features * (1 - mask_v2)
                        else:
                           features_v2 = copy.deepcopy(features)
                        # Learn graph
                        learned_adj = graph_learner(features) # Should return DGLGraph if sparse
                        if not args.sparse:
                           # Dense case (remains the same)
                           learned_adj = symmetrize(learned_adj)
                           learned_adj = normalize(learned_adj, 'sym', args.sparse)
                           Adj = learned_adj # Adj is a Tensor
                        else:
                           # Sparse case (DGL Graph)
                           Adj = learned_adj # Adj is a DGLGraph
                           # --- Check if edge weights require grad ---
                           if 'w' in Adj.edata:
                               # Ensure the weights tensor requires gradients. It should already if
                               # the learner calculated them using its parameters.
                               if not Adj.edata['w'].requires_grad:
                                   # This case is unexpected if learner is trainable & creates weights.
                                   # Maybe try setting it? Or log a warning.
                                   # Let's log a warning first, as forcing requires_grad might mask issues.
                                   if args.verbose:
                                        tqdm.write(f"Warning: Epoch {epoch} [Phase 2] - Learned graph edge weights (Adj.edata['w']) do not require gradients. Grad flow to learner might be broken.") # Changed to tqdm.write
                                   # Optionally, try forcing it:
                                   # Adj.edata['w'].requires_grad_(True) # Use with caution
                           else:
                               # If no 'w', gradient must flow via structure changes implicitly handled by DGL/learner.
                               pass
                           # --- End check ---

                        # Get learner outputs (Frozen Model, allow graph connection)
                        # Pass the DGL graph Adj
                        z2, _, _, _ = model(features_v2, Adj, 'learner', include_features=True) # Model params frozen via eval()
                        self._log_vram(f"Epoch {epoch} Standard: Phase 2 Forward Done")

                        # Calculate Contrastive Loss
                        if args.contrast_batch_size:
                           node_idxs = list(range(features.shape[0]))
                           batches = split_batch(node_idxs, args.contrast_batch_size)
                           contrastive_loss = 0
                           for batch in batches:
                               weight = len(batch) / features.shape[0]
                               contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
                        else:
                           contrastive_loss = model.calc_loss(z1, z2)
                        # Assign None to outputs not used for loss in this phase
                        emb1, cls_output1, emb2, cls_output2 = None, None, None, None


                    # --- Phase 3: Anchor + Learned Graph (Train Both) ---
                    else: # is_joint_phase
                        # view 1: anchor graph
                        if args.maskfeat_rate_anchor:
                            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
                            features_v1 = features * (1 - mask_v1)
                        else:
                            features_v1 = copy.deepcopy(features)
                        z1, emb1, _, cls_output1 = model(features_v1, anchor_adj, 'anchor', include_features=True)

                        # view 2: learned graph
                        if args.maskfeat_rate_learner:
                            mask_v2, _ = get_feat_mask(features, args.maskfeat_rate_learner)
                            features_v2 = features * (1 - mask_v2)
                        else:
                            features_v2 = copy.deepcopy(features)
                        # Learn graph
                        learned_adj = graph_learner(features)
                        if not args.sparse:
                           learned_adj = symmetrize(learned_adj)
                           learned_adj = normalize(learned_adj, 'sym', args.sparse)
                        Adj = learned_adj # Store for potential use later
                        # Get learner outputs
                        z2, emb2, _, cls_output2 = model(features_v2, learned_adj, 'learner', include_features=True)
                        self._log_vram(f"Epoch {epoch} Standard: Phase 3 Forward Done")

                        # Calculate Contrastive Loss
                        if args.contrast_batch_size:
                           node_idxs = list(range(features.shape[0]))
                           batches = split_batch(node_idxs, args.contrast_batch_size)
                           contrastive_loss = 0
                           for batch in batches:
                               weight = len(batch) / features.shape[0]
                               contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
                        else:
                           contrastive_loss = model.calc_loss(z1, z2)


                    # --- Loss Calculation ---
                    # 1. Initialize total loss
                    total_loss = torch.tensor(0.0, device=features.device)
                    print_loss_components = {} # Reset components dict
                    self._log_vram(f"Epoch {epoch} Standard: Before Loss Calc")

                    # 2. Add Contrastive Loss (Phase 2 & 3)
                    if is_graph_learner_phase or is_joint_phase:
                        total_loss += contrastive_loss
                        print_loss_components['contrastive'] = contrastive_loss.item()
                    else:
                        print_loss_components['contrastive'] = 0.0 # Show 0 for contrastive in Phase 1 report

                    # 3. ArcFace Loss (Phase 1 & 3, if enabled)
                    arcface_loss = torch.tensor(0.0, device=features.device)
                    if args.use_arcface and (is_embedding_phase or is_joint_phase):
                        if hasattr(model, 'arcface') and isinstance(model.arcface, SampledArcFaceLayer):
                            # Use anchor embedding (emb1) in Phase 1, learner embedding (emb2) in Phase 3
                            embedding_for_arcface = emb1 if is_embedding_phase else emb2
                            try:
                                if embedding_for_arcface is not None:
                                    arcface_output, sampled_labels = model.arcface(embedding_for_arcface, arcface_labels)
                                    arcface_loss = arcface_loss_with_sampling(arcface_output, sampled_labels)
                                    total_loss += args.arcface_weight * arcface_loss
                                    print_loss_components['arcface_sampled'] = arcface_loss.item()
                                else:
                                     if args.verbose: print(f"Warning: Embedding for ArcFace is None in epoch {epoch} (Phase {'1' if is_embedding_phase else '3'}). Skipping ArcFace.")
                                     print_loss_components['arcface_sampled'] = 0.0
                            except Exception as e:
                                print(f"Warning: Error during ArcFace calculation (Phase {'1' if is_embedding_phase else '3'}): {e}")
                                arcface_loss = torch.tensor(0.0, device=features.device)
                                print_loss_components['arcface_sampled'] = 0.0
                        else:
                            print("Warning: use_arcface=True but SampledArcFaceLayer not found/configured correctly.")
                            print_loss_components['arcface_sampled'] = 0.0
                    else:
                        print_loss_components['arcface_sampled'] = 0.0 # Ensure key exists


                    # 4. Classification Loss (Phase 1 & 3, if enabled)
                    cls_loss = torch.tensor(0.0, device=features.device)
                    current_cls_accuracy = torch.tensor(0.0, device=features.device)
                    if use_classification_head and (is_embedding_phase or is_joint_phase):
                        # Use anchor cls output (cls_output1) in Phase 1, learner cls output (cls_output2) in Phase 3
                        cls_output_for_loss = cls_output1 if is_embedding_phase else cls_output2
                        if cls_output_for_loss is not None and labels is not None:
                            classification_mask = (labels != -1)
                            if classification_mask.any():
                                masked_logits = cls_output_for_loss[classification_mask]
                                masked_labels = labels[classification_mask]
                                cls_loss, current_cls_accuracy = self.loss_binary_cls(masked_logits, masked_labels)
                                total_loss += args.annotation_loss_weight * cls_loss
                                print_loss_components['classification'] = cls_loss.item()

                                # Track best classification accuracy during training (only when model is trained)
                                if current_cls_accuracy > best_cls_accuracy:
                                    best_cls_accuracy = current_cls_accuracy
                            else:
                                if args.verbose: print("Warning: Classification head active, but no valid labels (0 or 1) found in this epoch.")
                                print_loss_components['classification'] = 0.0
                        else:
                            print(f"Warning: use_classification_head=True but cls_output or labels are None (Phase {'1' if is_embedding_phase else '3'}).")
                            print_loss_components['classification'] = 0.0
                    else:
                         # Ensure keys exist even if classification head is off
                         print_loss_components['classification'] = 0.0

                    # --- Combined Loss Backward Pass ---
                    loss = total_loss # Assign to 'loss' variable for consistency
                    self._log_vram(f"Epoch {epoch} Standard: After Loss Calc")

                    # NaN Loss Check (remains the same)
                    if torch.isnan(loss):
                        print(f"\nWarning: NaN loss detected at epoch {epoch}. Attempting to reload last checkpoint.")
                        if last_checkpoint_path:
                            try:
                                # Try reloading the last known good state
                                model, graph_learner, optimizer_cl, optimizer_learner, _, loaded_anchor_adj = \
                                    self.load_checkpoint(model, graph_learner, optimizer_cl, optimizer_learner, args)
                                # Restore anchor_adj if it was loaded
                                if loaded_anchor_adj is not None:
                                     anchor_adj = loaded_anchor_adj
                                print(f"Successfully reloaded checkpoint from {last_checkpoint_path}. Skipping optimizer step for epoch {epoch}.")
                                # Reset tqdm iterator description if needed after reload? Or assume it continues
                                epoch_iterator.set_description(f"Training (Resumed from {last_checkpoint_path})")
                                continue # Skip the rest of the loop for this epoch
                            except Exception as e:
                                print(f"Error reloading checkpoint {last_checkpoint_path} after NaN loss: {e}")
                                print("Stopping training trial due to unrecoverable NaN loss.")
                                break # Exit the epoch loop for this trial
                        else:
                            print("No previous checkpoint found to revert to after NaN loss. Stopping training trial.")
                            break # Exit the epoch loop for this trial

                    # Backward and optimize based on phase
                    self._log_vram(f"Epoch {epoch} Standard: Before Backward")
                    loss.backward()
                    self._log_vram(f"Epoch {epoch} Standard: After Backward")

                    # --- Check Learner Gradients (Diagnostic) ---
                    if is_graph_learner_phase: # Only check during Phase 2
                        learner_grad_norm = 0.0
                        params_with_grad = 0
                        for p in graph_learner.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                learner_grad_norm += param_norm.item() ** 2
                                params_with_grad += 1
                        learner_grad_norm = learner_grad_norm ** 0.5
                        # Only print if verbose and gradient is calculated
                        if args.verbose and params_with_grad > 0 and epoch % 5 == 0: # Print every 5 epochs in phase 2
                            tqdm.write(f"Epoch {epoch} [Phase 2] - Learner Grad Norm: {learner_grad_norm:.6f}") # Changed to tqdm.write
                        elif args.verbose and params_with_grad == 0 and epoch % 5 == 0:
                             tqdm.write(f"Epoch {epoch} [Phase 2] - Learner Grad Norm: N/A (No Grads Found)") # Changed to tqdm.write


                    # Gradient Clipping based on phase
                    if args.clip_norm > 0:
                        if is_embedding_phase or is_joint_phase: # Clip model grads if model was trained
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        if is_graph_learner_phase or is_joint_phase: # Clip learner grads if learner was trained
                            torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)

                    # Optimizer step based on phase
                    self._log_vram(f"Epoch {epoch} Standard: Before Optimizer Step")
                    if is_embedding_phase or is_joint_phase: # Step model optimizer if model was trained
                        optimizer_cl.step()
                    if is_graph_learner_phase or is_joint_phase: # Step learner optimizer if learner was trained
                        optimizer_learner.step()
                    self._log_vram(f"Epoch {epoch} Standard: After Optimizer Step")


                # Step the schedulers if using OneCycleLR, based on phase
                if scheduler_cl and (is_embedding_phase or is_joint_phase): # Step model scheduler if model trained
                    scheduler_cl.step()
                if scheduler_learner and (is_graph_learner_phase or is_joint_phase): # Step learner scheduler if learner trained
                    scheduler_learner.step()

                # Print current learning rates periodically using tqdm.write
                # Adapt LR reporting based on phase
                lr_model_str = f"{optimizer_cl.param_groups[0]['lr']:.6f}" if scheduler_cl else "N/A"
                lr_learner_str = f"{optimizer_learner.param_groups[0]['lr']:.6f}" if scheduler_learner else "N/A"
                # Only show active LRs
                if is_embedding_phase: lr_learner_str = "N/A (Phase 1)"
                if is_graph_learner_phase: lr_model_str = "N/A (Phase 2)"

                if args.verbose and epoch % 10 == 0:
                    tqdm.write(f"Epoch {epoch} [Phase {1 if is_embedding_phase else (2 if is_graph_learner_phase else 3)}] - LR model: {lr_model_str}, LR learner: {lr_learner_str}") # Changed to tqdm.write


                # Structure Bootstrapping - Should only happen when learner is active (Phase 2 & 3) and Adj exists
                if Adj is not None and (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    # This condition ensures Adj was computed (i.e., not Phase 1)
                    # Original bootstrapping logic follows:
                    if args.sparse:
                        # Memory-efficient sparse bootstrapping using DGL graph manipulation
                        num_nodes = anchor_adj.num_nodes()
                        dev = anchor_adj.device
                        self._log_vram(f"Epoch {epoch}: Before Bootstrapping")

                        # 1. Get edges and weights from anchor_adj (DGL graph)
                        u_a, v_a = anchor_adj.edges()
                        w_a = anchor_adj.edata['w'] if 'w' in anchor_adj.edata else torch.ones(anchor_adj.num_edges(), device=dev)

                        # 2. Get edges and weights from learned Adj (DGL graph)
                        # Ensure Adj is on the same device and detach its weights
                        Adj = Adj.to(dev)
                        u_l, v_l = Adj.edges()
                        # Detach learned weights before combining to prevent gradient flow through bootstrap update
                        w_l = Adj.edata['w'].detach() if 'w' in Adj.edata else torch.ones(Adj.num_edges(), device=dev)

                        # 3. Combine edges and calculate new weights
                        # Use a dictionary for efficient lookup: map (u, v) tuple to index
                        anchor_edge_dict = {}
                        for i in range(anchor_adj.num_edges()):
                            anchor_edge_dict[(u_a[i].item(), v_a[i].item())] = i

                        learned_edge_dict = {}
                        for i in range(Adj.num_edges()):
                            learned_edge_dict[(u_l[i].item(), v_l[i].item())] = i

                        new_u, new_v, new_w = [], [], []

                        # Combine keys (edges)
                        all_edges = set(anchor_edge_dict.keys()) | set(learned_edge_dict.keys())

                        tau = args.tau
                        one_minus_tau = 1.0 - tau

                        for u, v in all_edges:
                            w_a_val = 0.0
                            if (u, v) in anchor_edge_dict:
                                w_a_val = w_a[anchor_edge_dict[(u, v)]]

                            w_l_val = 0.0
                            if (u, v) in learned_edge_dict:
                                w_l_val = w_l[learned_edge_dict[(u, v)]]

                            final_w = tau * w_a_val + one_minus_tau * w_l_val

                            # Add edge if weight is non-zero (or above a small threshold if desired)
                            if final_w > EOS: # Using EOS threshold from utils
                                new_u.append(u)
                                new_v.append(v)
                                new_w.append(final_w)

                        # 4. Create the new anchor_adj DGL graph
                        if len(new_u) > 0:
                            new_anchor_adj = dgl.graph((torch.tensor(new_u, device=dev), torch.tensor(new_v, device=dev)),
                                                       num_nodes=num_nodes)
                            # Assign weights as tensors
                            new_anchor_adj.edata['w'] = torch.tensor(new_w, device=dev, dtype=torch.float32) # Ensure float type
                            anchor_adj = new_anchor_adj
                        else:
                            # Handle case where the combined graph might be empty
                            print(f"Warning: Bootstrapped graph at epoch {epoch} became empty.")
                            anchor_adj = dgl.graph(([], []), num_nodes=num_nodes, device=dev) # Keep same num_nodes
                        self._log_vram(f"Epoch {epoch}: After Bootstrapping")

                    else:
                        # Dense bootstrapping remains the same
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)


                # --- Update tqdm postfix --- # UNCOMMENT THIS BLOCK
                # Use averaged losses if using grad accum, otherwise use direct loss values
                report_loss = loss if grad_accumulation_steps > 1 else loss.item() # Use avg loss from accum or current loss
                phase_str = f"P{1 if is_embedding_phase else (2 if is_graph_learner_phase else 3)}"
                postfix_dict = {'Phase': phase_str, 'Loss': f"{report_loss:.4f}"}
                
                if use_classification_head:
                    # Use accuracy from accum or current
                    report_cls_acc = first_step_cls_accuracy.item() if grad_accumulation_steps > 1 else current_cls_accuracy.item()
                    # Only report accuracy if it was calculated (Phase 1 & 3)
                    if is_embedding_phase or is_joint_phase:
                        postfix_dict['ClsAcc'] = f"{report_cls_acc:.4f}"
                        cls_accuracies.append(report_cls_acc) # Track accuracy when it's calculated
                    else:
                        postfix_dict['ClsAcc'] = "N/A" # Phase 2
                
                
                # Add individual loss components to postfix based on phase and reporting source
                if grad_accumulation_steps > 1:
                     report_contrastive = avg_contrastive_loss
                     report_arcface = avg_arcface_loss
                     report_cls_loss = avg_cls_loss
                else: # Standard path, use print_loss_components dict
                     report_contrastive = print_loss_components.get('contrastive', 0.0)
                     report_arcface = print_loss_components.get('arcface_sampled', 0.0)
                     report_cls_loss = print_loss_components.get('classification', 0.0)
                
                # Only display losses relevant to the current phase
                if is_graph_learner_phase or is_joint_phase: postfix_dict['Contra'] = f"{report_contrastive:.4f}"
                if args.use_arcface and (is_embedding_phase or is_joint_phase): postfix_dict['ArcF'] = f"{report_arcface:.4f}"
                if use_classification_head and (is_embedding_phase or is_joint_phase): postfix_dict['ClsLoss'] = f"{report_cls_loss:.4f}"
                
                epoch_iterator.set_postfix(postfix_dict)
                # --- End tqdm postfix update ---

                # Log metrics to wandb at the end of each epoch
                log_dict = {'epoch': epoch}
                # Flatten the postfix_dict for wandb logging
                for key, value in postfix_dict.items():
                    try:
                        # Attempt to convert value to float for logging
                        log_dict[f'train/{key}'] = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, log as string or skip
                        log_dict[f'train/{key}_str'] = str(value) 
                        # Or handle specific non-numeric keys like 'Phase' differently if needed
                
                # Add learning rates
                if scheduler_cl: log_dict['train/lr_model'] = optimizer_cl.param_groups[0]['lr']
                if scheduler_learner: log_dict['train/lr_learner'] = optimizer_learner.param_groups[0]['lr']
                
                wandb.log(log_dict)


                # Periodic Checkpointing (remains the same, saves state at end of epoch)
                if epoch > 0 and epoch % args.checkpoint_freq == 0:
                    current_checkpoint_path = self.save_checkpoint(
                        epoch, model, graph_learner, optimizer_cl, optimizer_learner, anchor_adj, args
                    )
                    last_checkpoint_path = current_checkpoint_path # Update last known good path

                    # Also report average classification accuracy over period if using classification head
                    if use_classification_head and len(cls_accuracies) > 0:
                        avg_cls_acc = sum(cls_accuracies) / len(cls_accuracies)
                        if args.verbose:
                            # Use standard print for this message
                            print(f"Checkpoint @ Epoch {epoch}: Avg ClsAcc over last {len(cls_accuracies)} tracked epochs: {avg_cls_acc:.4f}")
                        cls_accuracies = []  # Reset for next period

                # --- Validation & Evaluation Step ---
                if epoch % args.eval_freq == 0:
                    model.eval()
                    graph_learner.eval()
                    # Initialize metrics for this eval step
                    val_cls_accu = torch.tensor(0.0, device=self.device)
                    val_cls_loss = torch.tensor(0.0, device=self.device)
                    Adj_eval = None # The graph used for evaluation/validation

                    with torch.no_grad():
                        # --- Get necessary outputs for validation/evaluation ---
                        features_val = features # Use original features for simplicity

                        # Get Anchor outputs (needed for P1 validation/P2&P3 downstream eval)
                        # Ensure anchor_adj is on the correct device
                        if isinstance(anchor_adj, dgl.DGLGraph): anchor_adj = anchor_adj.to(self.device)
                        elif isinstance(anchor_adj, torch.Tensor): anchor_adj = anchor_adj.to(self.device)
                        
                        z1_val, emb1_val, _, cls_output1_val = model(features_val, anchor_adj, 'anchor', include_features=True)

                        # Get Learner outputs if needed (P3 validation / P2&P3 downstream eval)
                        emb2_val, cls_output2_val = None, None
                        if not is_embedding_phase: # If in P2 or P3, use the learned graph 'Adj' from the training step
                            Adj_eval = Adj # Use the Adj computed during training step for this epoch
                            if Adj_eval is not None:
                                # Ensure Adj_eval is on correct device and detached
                                if isinstance(Adj_eval, dgl.DGLGraph):
                                    Adj_eval = Adj_eval.to(self.device)
                                    # Detach weights specifically if they exist
                                    if 'w' in Adj_eval.edata:
                                        Adj_eval_detached = dgl.graph(Adj_eval.edges(), num_nodes=Adj_eval.num_nodes(), device=self.device)
                                        Adj_eval_detached.edata['w'] = Adj_eval.edata['w'].detach()
                                        Adj_eval = Adj_eval_detached
                                    else: # If no weights, just ensure graph is on device
                                        Adj_eval = dgl.graph(Adj_eval.edges(), num_nodes=Adj_eval.num_nodes(), device=self.device)

                                elif isinstance(Adj_eval, torch.Tensor):
                                    Adj_eval = Adj_eval.to(self.device).detach()
                                # Get learner outputs using the detached evaluation graph
                                _, emb2_val, _, cls_output2_val = model(features_val, Adj_eval, 'learner', include_features=True)
                        else:
                                if args.verbose: tqdm.write(f"Warning: Adj is None during validation/eval in Phase {2 if is_graph_learner_phase else 3}.")
                                Adj_eval = None # Ensure it's None if something went wrong


                        # --- Perform Validation Calculations (Only in P1 & P3) ---
                        if is_embedding_phase or is_joint_phase:
                            validation_performed = False # Flag to print header once
                            if val_mask is not None and val_mask.any():
                                # Ensure val_mask is on device
                                current_val_mask = val_mask.to(self.device)

                                # CLASSIFICATION VALIDATION
                                if use_classification_head:
                                    cls_output_val = cls_output1_val if is_embedding_phase else cls_output2_val
                                    if cls_output_val is not None:
                                        masked_val_logits = cls_output_val[current_val_mask]
                                        # Ensure labels are on device before indexing
                                        labels_dev = labels.to(self.device) if labels is not None else None
                                        if labels_dev is not None:
                                            masked_val_labels = labels_dev[current_val_mask]

                                            # Further mask for valid binary labels within the val set
                                            val_cls_eval_mask = (masked_val_labels != -1)
                                            if val_cls_eval_mask.any():
                                                if args.verbose and not validation_performed:
                                                    tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---")
                                                    validation_performed = True
                                                
                                                val_cls_loss, val_cls_accu = self.loss_binary_cls(
                                                    masked_val_logits[val_cls_eval_mask],
                                                    masked_val_labels[val_cls_eval_mask]
                                                )
                                                if args.verbose:
                                                    tqdm.write(f"  Classification Val Loss: {val_cls_loss.item():.4f}, Acc: {val_cls_accu.item():.4f}")
                                                # Log validation classification metrics
                                                wandb.log({'epoch': epoch, 'val/cls_loss': val_cls_loss.item(), 'val/cls_accuracy': val_cls_accu.item()})
                                            elif args.verbose:
                                                 # Print header only if not already printed by potential subsequent ArcFace validation
                                                 if not (args.use_arcface and current_val_mask.sum().item() > 1):
                                                      tqdm.write(f"--- Epoch {epoch} Validation ---")
                                                 tqdm.write(f"  Classification Validation: No valid binary labels found in validation set.")

                                        else:
                                            if args.verbose: tqdm.write(f"--- Epoch {epoch} Validation --- \n  Classification Validation: Labels tensor is None.")
                                    else:
                                        if args.verbose: tqdm.write(f"--- Epoch {epoch} Validation --- \n  Classification Validation: Output is None.")

                                # --- ARCFACE VALIDATION (Pairwise Distance Quantiles) ---
                                if args.use_arcface:
                                    embedding_val = emb1_val if is_embedding_phase else emb2_val
                                    if embedding_val is not None:
                                        # Ensure val_mask and embedding_val are on the same device
                                        current_val_mask = current_val_mask.to(embedding_val.device)
                                        masked_val_embeddings = embedding_val[current_val_mask]
                                        num_val_samples = masked_val_embeddings.shape[0]

                                        if num_val_samples >= 2:
                                            try:
                                                # Normalize embeddings for cosine similarity calculation
                                                norm_embeddings = F.normalize(masked_val_embeddings, p=2, dim=1)
                                                
                                                # Decide on sampling strategy
                                                num_pairs_to_sample = 1000
                                                compute_all_pairs = num_val_samples < 50 
                                                
                                                if compute_all_pairs:
                                                    # Calculate all pairwise cosine similarities, then distances
                                                    # (Efficiently using matrix multiplication)
                                                    similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())
                                                    # Get upper triangle indices (excluding diagonal) to avoid duplicates and self-comparison
                                                    indices = torch.triu_indices(num_val_samples, num_val_samples, offset=1, device=similarity_matrix.device)
                                                    pairwise_similarities = similarity_matrix[indices[0], indices[1]]
                                                    # Handle potential numerical precision issues slightly outside [-1, 1]
                                                    pairwise_similarities = torch.clamp(pairwise_similarities, -1.0, 1.0)
                                                    pairwise_distances = 1.0 - pairwise_similarities 
                                                    num_pairs_calculated = len(pairwise_distances)
                                                else:
                                                    # Sample pairs of indices
                                                    idx1 = torch.randint(0, num_val_samples, (num_pairs_to_sample,), device=norm_embeddings.device)
                                                    idx2 = torch.randint(0, num_val_samples, (num_pairs_to_sample,), device=norm_embeddings.device)
                                                    # Ensure idx1 != idx2 for sampled pairs
                                                    valid_pair_mask = (idx1 != idx2)
                                                    idx1 = idx1[valid_pair_mask]
                                                    idx2 = idx2[valid_pair_mask]
                                                    num_pairs_calculated = len(idx1)

                                                    if num_pairs_calculated > 0:
                                                        # Calculate cosine similarity only for sampled pairs
                                                        emb_pairs1 = norm_embeddings[idx1]
                                                        emb_pairs2 = norm_embeddings[idx2]
                                                        # Element-wise dot product for cosine similarity
                                                        pairwise_similarities = torch.sum(emb_pairs1 * emb_pairs2, dim=1)
                                                        # Handle potential numerical precision issues slightly outside [-1, 1]
                                                        pairwise_similarities = torch.clamp(pairwise_similarities, -1.0, 1.0)
                                                        pairwise_distances = 1.0 - pairwise_similarities
                                                    else: # Handle unlikely case of no valid pairs sampled
                                                        pairwise_distances = torch.tensor([], device=norm_embeddings.device)
                                                
                                                # Calculate quantiles if we have distances
                                                if len(pairwise_distances) > 0:
                                                    quantiles = torch.tensor([0.1, 0.5], device=pairwise_distances.device)
                                                    distance_quantiles = torch.quantile(pairwise_distances, quantiles)
                                                    q10 = distance_quantiles[0].item()
                                                    q50 = distance_quantiles[1].item() # Median
                                                    
                                                    if args.verbose:
                                                        if not validation_performed: tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---"); validation_performed = True
                                                        strategy_msg = "all pairs" if compute_all_pairs else f"{num_pairs_calculated} sampled pairs"
                                                        tqdm.write(f"  ArcFace Val Cosine Dist ({strategy_msg}): 10%={q10:.4f}, 50%={q50:.4f}")
                                                        # Log ArcFace validation distance quantiles
                                                        wandb.log({'epoch': epoch, 'val/arcface_dist_q10': q10, 'val/arcface_dist_q50': q50})
                                                else:
                                                     if args.verbose:
                                                          if not validation_performed: tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---"); validation_performed = True
                                                          tqdm.write(f"  ArcFace Validation: No valid pairs found/sampled to calculate distances.")

                                            except Exception as e:
                                                if args.verbose:
                                                    if not validation_performed: tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---"); validation_performed = True
                                                    tqdm.write(f"  ArcFace Validation: Error calculating distance quantiles: {e}")
                                        else:
                                            if args.verbose:
                                                if not validation_performed: tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---"); validation_performed = True
                                                tqdm.write(f"  ArcFace Validation: Skipped distance quantiles (< 2 samples in val_mask: {num_val_samples}).")
                                    else:
                                        if args.verbose:
                                             if not validation_performed: tqdm.write(f"--- Epoch {epoch} Validation (Phase {1 if is_embedding_phase else 3}) ---"); validation_performed = True
                                             tqdm.write("  ArcFace Validation: Embedding is None.")

                            else:
                                 if args.verbose and (is_embedding_phase or is_joint_phase): # Only print if validation was expected
                                     tqdm.write(f"--- Epoch {epoch} Validation --- \n  Skipped validation: val_mask is None or empty.")


                    # Reset model/learner back to train mode for the next epoch
                    if is_embedding_phase: model.train(); graph_learner.eval() # Keep learner frozen in P1
                    elif is_graph_learner_phase: model.eval(); graph_learner.train() # Keep model frozen in P2
                    else: model.train(); graph_learner.train() # Both train in P3


                # --- End Validation & Evaluation Step ---
            
            # After all epochs, report best classification accuracy if tracked
            if use_classification_head:
                 final_best_cls_acc = best_cls_accuracy.item() if isinstance(best_cls_accuracy, torch.Tensor) else best_cls_accuracy
                 print(f"Best classification accuracy during training phases 1 & 3: {final_best_cls_acc:.4f}")
            self._log_vram(f"Trial {trial+1} End") # Log at end of trial

            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val)
                test_accuracies.append(best_val_test.item())
                if args.verbose:
                    print("\nTrial: ", trial + 1)
                    print(f"Best Eval val ACC: {best_val:.4f} (Epoch {best_epoch})") # Report best val accuracy from evaluation
                    print(f"Corresponding test ACC: {best_val_test.item():.4f}")
                    # Report best classification accuracy if used (already printed above)
                    # if use_classification_head:
                        # print("Best classification accuracy during training: ", best_cls_accuracy.item() if isinstance(best_cls_accuracy, torch.Tensor) else best_cls_accuracy) # Ensure it's a float

            # After training completes (remains the same)
            if args.save_model:
                # Save model, graph learner, features and final anchor adjacency matrix
                self.save_model(model, graph_learner, features, anchor_adj, args.sparse, args,
                               output_dir=args.output_dir)
                if args.verbose:
                    print(f"Model saved to {args.output_dir}")
                self._log_vram(f"Trial {trial+1}: Model Saved") # Log after saving

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies)
        self._log_vram("Train End") # Log at the very end
        
        # Finish the wandb run
        wandb.finish()

    def print_results(self, validation_accu, test_accu):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        print(s_val)
        print(s_test)

    def save_checkpoint(self, epoch, model, graph_learner, optimizer_cl, optimizer_learner, anchor_adj, args):
        """Saves the training state to a checkpoint file."""
        checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'graph_learner_state_dict': graph_learner.state_dict(),
            'optimizer_cl_state_dict': optimizer_cl.state_dict(),
            'optimizer_learner_state_dict': optimizer_learner.state_dict(),
            'anchor_adj': anchor_adj, # Save anchor_adj for bootstrapping
            'args': args # Save args used for this run
        }
        torch.save(state, checkpoint_path)
        if args.verbose:
            print(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path # Return path for tracking

    def load_checkpoint(self, model, graph_learner, optimizer_cl, optimizer_learner, args):
        """Loads the latest checkpoint from the checkpoint directory."""
        latest_checkpoint_path = None
        latest_epoch = -1

        if not os.path.isdir(args.checkpoint_dir):
            if args.verbose:
                print(f"Checkpoint directory {args.checkpoint_dir} not found. Starting from scratch.")
            return model, graph_learner, optimizer_cl, optimizer_learner, -1, None # Return initial state

        for filename in os.listdir(args.checkpoint_dir):
            if filename.startswith('checkpoint_epoch_') and filename.endswith('.pt'):
                try:
                    epoch_num = int(filename.split('_')[-1].split('.')[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint_path = os.path.join(args.checkpoint_dir, filename)
                except ValueError:
                    continue # Skip files that don't match the naming convention

        if latest_checkpoint_path:
            if args.verbose:
                print(f"Loading checkpoint from {latest_checkpoint_path}")
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=self.device)

                # Load model and graph learner states
                model.load_state_dict(checkpoint['model_state_dict'])
                graph_learner.load_state_dict(checkpoint['graph_learner_state_dict'])

                # Load optimizer states
                optimizer_cl.load_state_dict(checkpoint['optimizer_cl_state_dict'])
                optimizer_learner.load_state_dict(checkpoint['optimizer_learner_state_dict'])

                # Manually move optimizer state tensors to the correct device
                for state in optimizer_cl.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                
                for state in optimizer_learner.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

                start_epoch = checkpoint['epoch'] + 1
                anchor_adj = checkpoint['anchor_adj'] # Load anchor_adj

                # Move loaded anchor_adj to the correct device if it's a tensor/DGL graph
                if isinstance(anchor_adj, torch.Tensor):
                    anchor_adj = anchor_adj.to(self.device)
                elif isinstance(anchor_adj, dgl.DGLGraph):
                    anchor_adj = anchor_adj.to(self.device)

                print(f"Resuming training from epoch {start_epoch}")
                return model, graph_learner, optimizer_cl, optimizer_learner, start_epoch, anchor_adj
            except Exception as e:
                print(f"Error loading checkpoint {latest_checkpoint_path}: {e}")
                print("Starting training from scratch.")
                # Fall through to return initial state
        else:
            if args.verbose:
                print("No valid checkpoint found. Starting training from scratch.")

        return model, graph_learner, optimizer_cl, optimizer_learner, 0, None # Return initial state


def create_parser():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='person_data.csv',
                        help='Path to the dataset file')
    # Add annotated dataset arguments
    parser.add_argument('-annotated_dataset', type=str, default=None,
                        help='Path to annotated dataset with binary target column')
    parser.add_argument('-annotation_column', type=str, default='target',
                        help='Name of the binary target column in annotated dataset (values 0 or 1)')
    parser.add_argument('-annotation_loss_weight', type=float, default=1.0,
                        help='Weight for the binary classification loss')
    # Add argument for classification head layers
    parser.add_argument('-classification_head_layers', type=int, default=2,
                        help='Number of layers in the MLP classification head (default: 2)')
    parser.add_argument('-classification_dropout', type=float, default=0.3,
                        help='Dropout rate for the classification head MLP (default: 0.3)')
    # Add argument for dropping columns during KNN graph creation
    parser.add_argument('-drop_columns_file', type=str, default=None,
                        help='Path to CSV file containing a single column named "col" with names of features to exclude from initial KNN graph construction.')
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_refinement",
                        choices=['structure_inference', 'structure_refinement'])
    parser.add_argument('-eval_freq', type=int, default=5)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)
    parser.add_argument('-verbose', type=int, default=1, 
                       help='Control verbosity: 1 to show all prints, 0 to show only final results')
    # One Cycle Learning Rate Scheduler parameters
    parser.add_argument('-use_one_cycle', type=int, default=0,
                        help='Whether to use OneCycleLR scheduler (0=disabled, 1=enabled)')
    parser.add_argument('-one_cycle_pct_start', type=float, default=0.3,
                        help='Percentage of cycle spent increasing learning rate (default: 0.3)')
    parser.add_argument('-one_cycle_div_factor', type=float, default=25.0,
                        help='Initial learning rate is max_lr/div_factor (default: 25.0)')
    parser.add_argument('-one_cycle_final_div_factor', type=float, default=10000.0,
                        help='Final learning rate is max_lr/(div_factor*final_div_factor) (default: 10000.0)')

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=1000)
    parser.add_argument('-lr', type=float, default=0.01)
    # Add new argument for graph learner LR
    parser.add_argument('-lr_learner', type=float, default=0.01,
                        help='Learning rate specifically for the graph learner optimizer')
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-use_layer_norm', type=int, default=0,
                       help='Use Layer Normalization (0=disabled, 1=enabled)')
    parser.add_argument('-use_residual', type=int, default=0,
                       help='Use Residual Connections (0=disabled, 1=enabled)')
    # Add the new argument here
    parser.add_argument('--embedding_only_epochs', type=int, default=0,
                        help='Number of initial epochs to train only embeddings using anchor graph (ArcFace/Classification loss), disabling learner and contrastive loss.')
    parser.add_argument('--graph_learner_only_epochs', type=int, default=0,
                        help='Number of epochs to train only the graph learner, freezing the main GCL model.')

    # GCL Module -Augmentation
    parser.add_argument('-maskfeat_rate_learner', type=float, default=0.2)
    parser.add_argument('-maskfeat_rate_anchor', type=float, default=0.2)
    parser.add_argument('-dropedge_rate', type=float, default=0.5)

    # GSL Module
    parser.add_argument('-type_learner', type=str, default='fgp', choices=["fgp", "att", "mlp", "gnn"])
    parser.add_argument('-k', type=int, default=30)
    parser.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    parser.add_argument('-gamma', type=float, default=0.9)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0005)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    # Structure Bootstrapping
    parser.add_argument('-tau', type=float, default=1)
    parser.add_argument('-c', type=int, default=0)

    # New arguments for saving model
    parser.add_argument('-save_model', type=int, default=1)
    parser.add_argument('-output_dir', type=str, default='saved_models')
    
    # ArcFace arguments
    parser.add_argument('-use_arcface', type=int, default=0,
                       help='Whether to use ArcFace loss (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_scale', type=float, default=30.0,
                       help='Scale factor for ArcFace (s parameter)')
    parser.add_argument('-arcface_margin', type=float, default=0.5,
                       help='Angular margin for ArcFace (m parameter)')
    parser.add_argument('-arcface_weight', type=float, default=1.0,
                       help='Weight for ArcFace loss when combining with contrastive loss')
    
    # Memory optimization arguments (Remove batched, add sampled)
    # parser.add_argument('-use_batched_arcface', ...)
    # parser.add_argument('-arcface_batch_size', ...)
    parser.add_argument('-use_sampled_arcface', type=int, default=0,
                       help='Whether to use ultra memory-efficient sampled ArcFace (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_num_samples', type=int, default=5000,
                       help='Number of classes to sample in sampled ArcFace (default: 5000)')
    
    parser.add_argument('-grad_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before updating weights (1=disabled)')

    # Checkpointing Arguments
    parser.add_argument('-checkpoint_freq', type=int, default=200,
                       help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('-checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    # Gradient Clipping
    parser.add_argument('-clip_norm', type=float, default=1.0,
                       help='Max norm for gradient clipping (0 to disable)')

    # New argument for relationship dataset
    parser.add_argument('-relationship_dataset', type=str, default=None,
                        help='Path to the relationship dataset CSV file (e.g., relationships.csv)')

    # New argument for relationship edge weight
    parser.add_argument('--relationship_weight', type=float, default=1.0,
                        help='Weight assigned to edges created from the relationship dataset (default: 1.0)')

    # New argument to load pre-trained model weights
    parser.add_argument('-load_pretrained_path', type=str, default=None,
                        help='Path to directory containing pre-trained model.pt and graph_learner.pt to load weights from.')

    # New argument for KNN threshold type
    parser.add_argument('--knn_threshold_type', type=str, default='none', choices=['none', 'median_k', 'std_dev_k'],
                        help='Type of thresholding for KNN graph learners (none, median_k, std_dev_k)')
    parser.add_argument('--knn_std_dev_factor', type=float, default=1.0,
                        help='Factor (alpha) for std_dev_k threshold (mean - alpha*std_dev)')

    # Model Hyperparameters
    parser.add_argument('--model_type', type=str, default='sublime', help='Model type (sublime)')

    # Memory optimization parameters
    parser.add_argument('-graph_learner_chunk_size', type=int, default=100,
                     help='Chunk size for memory-efficient processing in graph learner (smaller = less memory but slower)')
    parser.add_argument('-offload_to_cpu', type=int, default=1,
                     help='Whether to offload tensors to CPU during graph learning (1=enabled, 0=disabled)')
    parser.add_argument('-cleanup_every_n_chunks', type=int, default=5,
                     help='Call torch.cuda.empty_cache() every N chunks to free up memory (0 to disable)')

    return parser

if __name__ == '__main__':
    parent_parser = create_parser()
    args = parent_parser.parse_args()

    experiment = Experiment()
    experiment.train(args, load_data)