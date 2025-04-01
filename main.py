import argparse
import copy
from datetime import datetime
import math
import contextlib

from tqdm import tqdm
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

# Import memory debugging utilities
from memory_debug import profile_arcface_memory, analyze_forward_pass, measure_arcface_memory_usage
# Import batched ArcFace functionality
from batched_arcface import BatchedArcFaceLayer, batched_arcface_loss
# Import memory optimization utilities
from memory_optimizations import train_with_memory_optimization, GradientAccumulation, enable_mixed_precision
# Import sampled ArcFace functionality
from sampled_arcface import SampledArcFaceLayer, arcface_loss_with_sampling

EOS = 1e-10

class Experiment:
    def __init__(self, device=None):
        super(Experiment, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)
        dgl.random.seed(seed)

    def loss_cls(self, model, mask, features, labels):
        logits = model(features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask], reduction='mean')
        accu = accuracy(logp[mask], labels[mask])
        return loss, accu

    def loss_gcl(self, model, graph_learner, features, anchor_adj, args):
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        z1, _ = model(features_v1, anchor_adj, 'anchor')

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

        z2, _ = model(features_v2, learned_adj, 'learner')

        # compute loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, args.contrast_batch_size)
            loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            loss = model.calc_loss(z1, z2)

        return loss, learned_adj
        
    def loss_arcface(self, model, graph_learner, features, anchor_adj, labels, args):
        """
        Calculate combined loss using both ArcFace loss and contrastive learning
        
        Args:
            model: The GCL model
            graph_learner: The graph learner
            features: Input features
            anchor_adj: Anchor adjacency matrix
            labels: Class labels (row indices in supervised mode)
            args: Arguments
            
        Returns:
            Tuple of (loss, learned_adjacency_matrix)
        """
        # Check if we should use the sampled version (highest memory savings)
        if hasattr(args, 'use_sampled_arcface') and args.use_sampled_arcface:
            # Use the most memory-efficient implementation with class sampling
            return self.loss_arcface_sampled(model, graph_learner, features, anchor_adj, labels, args)
        # Check if we should use the batched version
        elif hasattr(args, 'use_batched_arcface') and args.use_batched_arcface:
            # Use the memory-efficient batched implementation
            return self.loss_arcface_batched(model, graph_learner, features, anchor_adj, labels, args)
            
        # First calculate the original contrastive loss
        contrastive_loss, learned_adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)
        
        # Now calculate the ArcFace loss
        # Forward pass with labels for ArcFace
        if not hasattr(args, 'arcface_weight'):
            arcface_weight = 1.0  # Default weight
        else:
            arcface_weight = args.arcface_weight
            
        # For ArcFace, we need a separate forward pass to get the ArcFace outputs
        _, _, arcface_output = model(features, learned_adj, 'learner', labels)
        
        # Calculate cross-entropy loss
        criterion = nn.CrossEntropyLoss()
        arcface_loss = criterion(arcface_output, labels)
        
        # Combine both losses
        combined_loss = contrastive_loss + arcface_weight * arcface_loss
        
        if args.verbose and (not hasattr(args, '_loss_printed') or not args._loss_printed):
            print(f"Combined loss: contrastive_loss={contrastive_loss.item():.4f}, arcface_loss={arcface_loss.item():.4f}, weight={arcface_weight}")
            args._loss_printed = True
            
        return combined_loss, learned_adj
    
    def loss_arcface_sampled(self, model, graph_learner, features, anchor_adj, labels, args):
        """
        Most memory-efficient implementation of ArcFace loss using class sampling
        
        This implementation only computes ArcFace loss for a sample of classes
        (e.g., 5,000 classes instead of all 35,000), drastically reducing memory usage.
        """
        # First calculate the original contrastive loss
        contrastive_loss, learned_adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)
        
        # Get weight for ArcFace loss
        if not hasattr(args, 'arcface_weight'):
            arcface_weight = 1.0  # Default weight
        else:
            arcface_weight = args.arcface_weight
            
        # Get number of classes to sample
        if hasattr(args, 'arcface_num_samples'):
            num_samples = args.arcface_num_samples
        else:
            num_samples = min(5000, len(labels))  # Default to 5000 or total classes if less
            
        # Forward pass with include_features=True to get hidden representations
        z_from_model, hidden_representations, _ = model(features, learned_adj, include_features=True)
        
        # Verify that we're using the correct sampled ArcFace approach
        if not hasattr(model, 'sampled_arcface') or not model.sampled_arcface:
            raise ValueError("Model is not using sampled ArcFace. Set use_sampled_arcface=True in args")
            
        # Verify that model.arcface is an instance of SampledArcFaceLayer
        from sampled_arcface import SampledArcFaceLayer
        if not isinstance(model.arcface, SampledArcFaceLayer):
            raise ValueError("Model.arcface is not a SampledArcFaceLayer instance. Check model initialization.")
            
        if args.verbose and not hasattr(args, '_sampled_arcface_printed'):
            print(f"Using Fixed Sampled ArcFace for memory efficiency")
            args._sampled_arcface_printed = True
        
        # Forward pass through the sampling layer
        # Note: sampled_arcface is a boolean flag, the actual layer is still stored in model.arcface
        arcface_output, sampled_labels = model.arcface(hidden_representations, labels)
        
        # Calculate ArcFace loss using only the sampled classes
        arcface_loss = arcface_loss_with_sampling(arcface_output, sampled_labels)
        
        # Ensure both losses are torch tensors before combining
        if not isinstance(contrastive_loss, torch.Tensor):
            contrastive_loss = torch.tensor(contrastive_loss, device=arcface_output.device, requires_grad=True)
        if not isinstance(arcface_loss, torch.Tensor):
            arcface_loss = torch.tensor(arcface_loss, device=arcface_output.device, requires_grad=True)
        
        # Combine losses
        total_loss = contrastive_loss + arcface_weight * arcface_loss
        
        if args.verbose:
            print(f"Combined loss: contrastive_loss={contrastive_loss:.4f}, arcface_loss={arcface_loss:.4f}, weight={arcface_weight}")
            
        # Make sure we return a proper tensor
        return total_loss, learned_adj
        
    def loss_arcface_batched(self, model, graph_learner, features, anchor_adj, labels, args):
        """
        Memory-efficient version of ArcFace loss that processes data in batches
        
        Args:
            model: The GCL model
            graph_learner: The graph learner
            features: Input features
            anchor_adj: Anchor adjacency matrix
            labels: Class labels (row indices in supervised mode)
            args: Arguments
            
        Returns:
            Tuple of (loss, learned_adjacency_matrix)
        """
        # Import the batched implementation from batched_arcface.py
        return batched_arcface_loss(self, model, graph_learner, features, anchor_adj, labels, args)

    def evaluate_adj_by_cls(self, Adj, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args):
        model = GCN(in_channels=nfeats, hidden_channels=args.hidden_dim_cls, out_channels=nclasses, num_layers=args.nlayers_cls,
                    dropout=args.dropout_cls, dropout_adj=args.dropedge_cls, Adj=Adj, sparse=args.sparse)
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
        # Save a config file with all important parameters
        with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
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
            if sparse:
                hidden_dim = encoder.gnn_encoder_layers[0].linear.out_features
                emb_dim = encoder.gnn_encoder_layers[-1].linear.out_features
            else:
                hidden_dim = encoder.gnn_encoder_layers[0].linear.out_features
                emb_dim = encoder.gnn_encoder_layers[-1].linear.out_features
                
            f.write(f'hidden_dim: {hidden_dim}\n')
            f.write(f'emb_dim: {emb_dim}\n')
            
            # Get proj_dim from the first layer of the projection head
            proj_dim = encoder.proj_head[0].out_features
            f.write(f'proj_dim: {proj_dim}\n')
            
            f.write(f'dropout: {encoder.dropout}\n')
            f.write(f'dropout_adj: {encoder.dropout_adj_p}\n')
            
            # Graph learner parameters
            # Identify the learner type
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
            
            # These parameters should exist in all learner types
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
                
                # Check for sampled ArcFace
                if hasattr(model, 'sampled_arcface') and model.sampled_arcface:
                    f.write(f'use_sampled_arcface: True\n')
                    if hasattr(model.arcface, 'num_samples'):
                        f.write(f'arcface_num_samples: {model.arcface.num_samples}\n')
        
    def load_model(self, input_dir='saved_models'):
        """
        Load a saved model, graph learner, features and adjacency matrix
        
        Args:
            input_dir: Directory where models are saved
            
        Returns:
            tuple: (model, graph_learner, features, adj, sparse)
        """
        # Load config
        config = {}
        with open(os.path.join(input_dir, 'config.txt'), 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                if key == 'sparse':
                    config[key] = value.lower() == '1'
                elif key in ['feature_dim', 'num_nodes', 'nlayers', 'hidden_dim', 'emb_dim', 'proj_dim', 'k']:
                    config[key] = int(value)
                elif key in ['dropout', 'dropout_adj']:
                    config[key] = float(value)
                else:
                    config[key] = value
        
        sparse = config['sparse']
        
        # Load features
        features = torch.load(os.path.join(input_dir, 'features.pt')).to(self.device)
        
        # If model_params is not provided, use parameters from config
        model_params = {
            'nlayers': config.get('nlayers', 2),
            'hidden_dim': config.get('hidden_dim', 128),
            'emb_dim': config.get('emb_dim', 32),
            'proj_dim': config.get('proj_dim', 32),
            'dropout': config.get('dropout', 0.3),
            'dropout_adj': config.get('dropout_adj', 0.3),
            'k': config.get('k', 10),
            'sim_function': config.get('sim_function', 'cosine'),
            'activation_learner': config.get('activation_learner', 'relu')
        }
        
        # Check for arcface parameters in config
        use_arcface = config.get('use_arcface', 'False').lower() in ['true', '1', 't', 'y', 'yes']
        arcface_params = {}
        if use_arcface:
            arcface_params['use_arcface'] = True
            arcface_params['num_classes'] = int(config.get('num_classes', 0))
            arcface_params['arcface_scale'] = float(config.get('arcface_scale', 30.0))
            arcface_params['arcface_margin'] = float(config.get('arcface_margin', 0.5))
            
            # Check for sampled arcface parameters
            use_sampled_arcface = config.get('use_sampled_arcface', 'False').lower() in ['true', '1', 't', 'y', 'yes']
            if use_sampled_arcface:
                arcface_params['use_sampled_arcface'] = True
                arcface_params['arcface_num_samples'] = int(config.get('arcface_num_samples', 5000))
        
        # Initialize models with arcface parameters if needed
        if use_arcface:
            model = GCL(nlayers=model_params['nlayers'],
                       in_dim=features.shape[1],
                       hidden_dim=model_params['hidden_dim'],
                       emb_dim=model_params['emb_dim'],
                       proj_dim=model_params['proj_dim'],
                       dropout=model_params['dropout'],
                       dropout_adj=model_params['dropout_adj'],
                       sparse=sparse,
                       use_arcface=arcface_params['use_arcface'],
                       num_classes=arcface_params['num_classes'],
                       arcface_scale=arcface_params['arcface_scale'],
                       arcface_margin=arcface_params['arcface_margin'],
                       use_sampled_arcface=arcface_params.get('use_sampled_arcface', False),
                       arcface_num_samples=arcface_params.get('arcface_num_samples', None))
        else:
            model = GCL(nlayers=model_params['nlayers'],
                       in_dim=features.shape[1],
                       hidden_dim=model_params['hidden_dim'],
                       emb_dim=model_params['emb_dim'],
                       proj_dim=model_params['proj_dim'],
                       dropout=model_params['dropout'],
                       dropout_adj=model_params['dropout_adj'],
                       sparse=sparse)
        
        # Determine type of graph learner from config or saved file
        learner_type = config.get('type_learner', None)
        
        if learner_type is None and os.path.exists(os.path.join(input_dir, 'graph_learner.pt')):
            # Try to determine from saved state dict
            learner_state = torch.load(os.path.join(input_dir, 'graph_learner.pt'))
            
            # Check which type of learner it is
            if 'Adj' in learner_state:
                learner_type = 'fgp'
            elif any('layers.0.weight' in key for key in learner_state.keys()):
                # Determine if it's MLP, ATT, or GNN
                if any('w' in key for key in learner_state.keys()):
                    learner_type = 'att'
                else:
                    # Check if it's a GNN (need to load adjacency first)
                    # For now, assume MLP as default
                    learner_type = 'mlp'
        
        # Create the appropriate graph learner
        if learner_type == 'fgp':
            graph_learner = FGP_learner(features.cpu(), model_params['k'], 
                                      model_params['sim_function'], 6, sparse)
        elif learner_type == 'mlp':
            graph_learner = MLP_learner(2, features.shape[1], model_params['k'], 
                                     model_params['sim_function'], 6, sparse,
                                     model_params['activation_learner'])
        elif learner_type == 'att':
            graph_learner = ATT_learner(2, features.shape[1], model_params['k'], 
                                     model_params['sim_function'], 6, sparse,
                                     model_params['activation_learner'])
        elif learner_type == 'gnn':
            # Need to load adjacency first
            adj_data = self._load_adjacency(input_dir, sparse)
            graph_learner = GNN_learner(2, features.shape[1], model_params['k'], 
                                     model_params['sim_function'], 6, sparse,
                                     model_params['activation_learner'], adj_data)
        else:
            # Default to FGP learner if no specific type is found
            graph_learner = FGP_learner(features.cpu(), model_params['k'], 
                                      model_params['sim_function'], 6, sparse)
        
        # Load model weights
        model.load_state_dict(torch.load(os.path.join(input_dir, 'model.pt')))
        graph_learner.load_state_dict(torch.load(os.path.join(input_dir, 'graph_learner.pt')))
        
        # Load adjacency matrix
        adj = self._load_adjacency(input_dir, sparse)
        
        # Move models to device
        model = model.to(self.device)
        graph_learner = graph_learner.to(self.device)
        
        return model, graph_learner, features, adj, sparse
        
    def _load_adjacency(self, input_dir, sparse):
        """Helper method to load adjacency matrix"""
        adj_data = torch.load(os.path.join(input_dir, 'adjacency.pt'))
        
        if sparse:
            if isinstance(adj_data, dict):
                # This is a saved DGL graph
                edges = adj_data['edges']
                weights = adj_data['weights']
                num_nodes = adj_data['num_nodes']
                
                # Recreate DGL graph
                adj = dgl.graph(edges, num_nodes=num_nodes, device=self.device)
                adj.edata['w'] = weights
            else:
                # This is a torch sparse tensor
                adj = adj_data.to(self.device)
        else:
            adj = adj_data.to(self.device)
            
        return adj

    def process_new_point(self, new_point, model, graph_learner, features, adj, sparse, replace_idx=None):
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
            
        Returns:
            torch.Tensor: The node embedding for the new point
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
            new_adj = graph_learner(modified_features)
            
            # Process adjacency matrix based on sparse flag
            if not sparse:
                new_adj = symmetrize(new_adj)
                new_adj = normalize(new_adj, 'sym', sparse)
            
            # Get embeddings for all nodes including the new one
            # For ArcFace models, we don't need to pass labels during inference
            if hasattr(model, 'use_arcface') and model.use_arcface:
                # For ArcFace models, we still get embeddings without passing labels
                _, embeddings = model(modified_features, new_adj)
            else:
                # Standard model
                _, embeddings = model(modified_features, new_adj)
            
            # Return the embedding of the replaced point
            new_point_embedding = embeddings[replace_idx].detach()
            
            return new_point_embedding
            
    def train(self, args, load_data_fn=None):
        """
        Train the model
        
        Args:
            args: Arguments for training
            load_data_fn: Function to load the dataset. Should return:
                (features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj)
        """
        if load_data_fn is None:
            raise ValueError("Must provide a data loading function")
            
        if args.gsl_mode == 'structure_refinement':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj_original = load_data_fn(args)
        elif args.gsl_mode == 'structure_inference':
            features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, _ = load_data_fn(args)
        
        # For ArcFace, create labels as row indices if not provided
        if args.use_arcface:
            if labels is None:
                if args.verbose:
                    print("Using row indices as labels for ArcFace")
                labels = torch.arange(features.shape[0])
                
                # When using sampled ArcFace, limit the number of classes
                if hasattr(args, 'use_sampled_arcface') and args.use_sampled_arcface and hasattr(args, 'arcface_num_samples'):
                    nclasses = args.arcface_num_samples
                    if args.verbose:
                        print(f"Using sampled ArcFace with {nclasses} classes instead of {features.shape[0]}")
                else:
                    nclasses = features.shape[0]
               
            # Run memory profiling if requested and not using sampled ArcFace
            # (memory debugging doesn't work well with sampled ArcFace due to subgraph issues)
            if args.debug_memory:
                if hasattr(args, 'use_sampled_arcface') and args.use_sampled_arcface:
                    if args.verbose:
                        print("\nSkipping memory debugging when using sampled ArcFace\n")
                else:
                    if args.verbose:
                        print("\n=== Memory Profiling for ArcFace ===")
                        # Estimate memory usage
                        batch_size = features.shape[0]  # Full batch size
                        emb_dim = args.rep_dim  # Embedding dimension
                        profile_arcface_memory(batch_size, emb_dim, nclasses)
                        print("===================================\n")

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):
            self.setup_seed(trial)

            if args.gsl_mode == 'structure_inference':
                if args.sparse:
                    anchor_adj_raw = torch_sparse_eye(features.shape[0])
                else:
                    anchor_adj_raw = torch.eye(features.shape[0]).to(self.device)
            elif args.gsl_mode == 'structure_refinement':
                if args.sparse:
                    anchor_adj_raw = adj_original
                else:
                    anchor_adj_raw = torch.from_numpy(adj_original).to(self.device)

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            if args.type_learner == 'fgp':
                graph_learner = FGP_learner(features.cpu(), args.k, args.sim_function, 6, args.sparse)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)

            if args.use_arcface:
                model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                          emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                          dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse,
                          use_arcface=True, num_classes=nclasses,
                          arcface_scale=args.arcface_scale, arcface_margin=args.arcface_margin,
                          use_sampled_arcface=args.use_sampled_arcface if hasattr(args, 'use_sampled_arcface') else False,
                          arcface_num_samples=args.arcface_num_samples if hasattr(args, 'arcface_num_samples') else None)
                
                # If using both sampled and batched ArcFace, prioritize sampled (don't try to use both)
                if hasattr(args, 'use_sampled_arcface') and args.use_sampled_arcface and args.use_batched_arcface:
                    if args.verbose:
                        print("Warning: Both sampled and batched ArcFace specified. Using sampled ArcFace only.")
                # If using batched ArcFace, replace the standard ArcFace layer with batched version
                elif args.use_batched_arcface:
                    try:
                        # Get the properties from the original ArcFace layer
                        original_arcface = model.arcface
                        in_features = original_arcface.in_features
                        out_features = original_arcface.out_features
                        scale = original_arcface.scale
                        margin = original_arcface.margin
                        easy_margin = original_arcface.easy_margin
                    except AttributeError as e:
                        print(f"Error accessing attributes from ArcFace layer: {e}")
                        print("Skipping batched ArcFace initialization")
                        continue
                    
                    # Create new batched ArcFace layer
                    batched_layer = BatchedArcFaceLayer(
                        in_features=in_features,
                        out_features=out_features,
                        scale=scale,
                        margin=margin,
                        easy_margin=easy_margin,
                        batch_size=args.arcface_batch_size
                    )
                    
                    # Copy weights from original layer (important for fine-tuning)
                    batched_layer.weight.data.copy_(original_arcface.weight.data)
                    
                    # Replace the layer
                    model.arcface = batched_layer
                    
                    if args.verbose:
                        print(f"Using Batched ArcFace implementation with batch size {args.arcface_batch_size}")
                
                if args.verbose:
                    print(f"Using ArcFace loss with {nclasses} classes, scale={args.arcface_scale}, margin={args.arcface_margin}")
                    
                # Run memory debugging for model if requested
                if args.debug_memory:
                    # Skip memory debugging for sampled ArcFace due to subgraph issues
                    if hasattr(args, 'use_sampled_arcface') and args.use_sampled_arcface:
                        if args.verbose:
                            print("\nSkipping ArcFace forward pass memory analysis when using sampled ArcFace")
                            print("(This is due to subgraph size mismatches that occur during testing)")
                    else:
                        # Move to GPU if needed
                        model = model.to(self.device)
                        features_gpu = features.to(self.device)
                        labels_gpu = labels.to(self.device)
                        
                        if args.verbose:
                            print("\n=== ArcFace Forward Pass Memory Analysis ===")
                            result = analyze_forward_pass(
                                model,
                                features_gpu,
                                anchor_adj,
                                labels_gpu,
                                max_samples=args.max_debug_samples
                            )
                            print("==========================================\n")
            else:
                model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                          emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                          dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            
            # Set up OneCycleLR scheduler if enabled
            if hasattr(args, 'use_one_cycle') and args.use_one_cycle:
                print("Using One Cycle Policy learning rate scheduler")
                scheduler_cl = lr_scheduler.OneCycleLR(
                    optimizer_cl,
                    max_lr=args.lr,
                    total_steps=args.epochs,
                    pct_start=args.one_cycle_pct_start if hasattr(args, 'one_cycle_pct_start') else 0.3,
                    div_factor=args.one_cycle_div_factor if hasattr(args, 'one_cycle_div_factor') else 25.0,
                    final_div_factor=args.one_cycle_final_div_factor if hasattr(args, 'one_cycle_final_div_factor') else 10000.0
                )
                
                scheduler_learner = lr_scheduler.OneCycleLR(
                    optimizer_learner,
                    max_lr=args.lr,
                    total_steps=args.epochs,
                    pct_start=args.one_cycle_pct_start if hasattr(args, 'one_cycle_pct_start') else 0.3,
                    div_factor=args.one_cycle_div_factor if hasattr(args, 'one_cycle_div_factor') else 25.0,
                    final_div_factor=args.one_cycle_final_div_factor if hasattr(args, 'one_cycle_final_div_factor') else 10000.0
                )
            else:
                scheduler_cl = None
                scheduler_learner = None

            model = model.to(self.device)
            graph_learner = graph_learner.to(self.device)
            features = features.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)
            if train_mask is not None:
                train_mask = train_mask.to(self.device)
            if val_mask is not None:
                val_mask = val_mask.to(self.device)
            if test_mask is not None:
                test_mask = test_mask.to(self.device)

            if args.downstream_task == 'classification':
                best_val = 0
                best_val_test = 0
                best_epoch = 0

            # Mixed precision training is disabled by default to avoid type mismatch issues
            # with the graph learner's KNN function
            scaler = None
            if hasattr(args, 'use_mixed_precision') and args.use_mixed_precision:
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                if args.verbose:
                    print("Using mixed precision training (WARNING: may cause type mismatch in KNN)")
            
            # Initialize gradient accumulation steps
            grad_accumulation_steps = args.grad_accumulation_steps if hasattr(args, 'grad_accumulation_steps') else 1
            if grad_accumulation_steps > 1 and args.verbose:
                print(f"Using gradient accumulation with {grad_accumulation_steps} steps")
                
            for epoch in tqdm(range(1, args.epochs + 1), desc="Training", total=args.epochs):
                model.train()
                graph_learner.train()
                
                # Check if we should use memory-efficient training
                if hasattr(args, 'memory_efficient_training') and args.memory_efficient_training:
                    # Use our memory-efficient training function
                    loss, Adj = train_with_memory_optimization(
                        self, model, graph_learner, features, anchor_adj,
                        labels, args, optimizer_cl, optimizer_learner
                    )
                elif grad_accumulation_steps > 1:
                    # Use gradient accumulation
                    accumulated_loss = 0
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()
                    
                    for i in range(grad_accumulation_steps):
                        # Forward pass
                        if args.use_arcface:
                            with torch.cuda.amp.autocast() if scaler else contextlib.nullcontext():
                                mini_loss, Adj = self.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
                                mini_loss = mini_loss / grad_accumulation_steps  # Scale loss
                        else:
                            with torch.cuda.amp.autocast() if scaler else contextlib.nullcontext():
                                mini_loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)
                                mini_loss = mini_loss / grad_accumulation_steps  # Scale loss
                        
                        # Backward pass with scaling if using mixed precision
                        if scaler:
                            scaler.scale(mini_loss).backward()
                        else:
                            mini_loss.backward()
                            
                        accumulated_loss += mini_loss.item()
                    
                    # Update weights
                    if scaler:
                        scaler.step(optimizer_cl)
                        scaler.step(optimizer_learner)
                        scaler.update()
                    else:
                        optimizer_cl.step()
                        optimizer_learner.step()
                    
                    loss = accumulated_loss
                elif scaler:  # Mixed precision without gradient accumulation
                    # Forward pass with mixed precision
                    with torch.cuda.amp.autocast():
                        if args.use_arcface:
                            loss, Adj = self.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
                        else:
                            loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)
                    
                    # Backward and optimize with scaling
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_cl)
                    scaler.step(optimizer_learner)
                    scaler.update()
                else:
                    # Original training code path
                    if args.use_arcface:
                        loss, Adj = self.loss_arcface(model, graph_learner, features, anchor_adj, labels, args)
                    else:
                        loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()
                    loss.backward()
                    optimizer_cl.step()
                    optimizer_learner.step()
                
                # Step the schedulers if using OneCycleLR
                if hasattr(args, 'use_one_cycle') and args.use_one_cycle:
                    scheduler_cl.step()
                    scheduler_learner.step()
                    
                    # Print current learning rates periodically
                    if args.verbose and epoch % 10 == 0:
                        print(f"Epoch {epoch} - LR model: {optimizer_cl.param_groups[0]['lr']:.6f}, "
                              f"LR learner: {optimizer_learner.param_groups[0]['lr']:.6f}")

                # Structure Bootstrapping
                if (1 - args.tau) and (args.c == 0 or epoch % args.c == 0):
                    if args.sparse:
                        learned_adj_torch_sparse = dgl_graph_to_torch_sparse(Adj)
                        anchor_adj_torch_sparse = anchor_adj_torch_sparse * args.tau \
                                                  + learned_adj_torch_sparse * (1 - args.tau)
                        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj_torch_sparse)
                    else:
                        anchor_adj = anchor_adj * args.tau + Adj.detach() * (1 - args.tau)

                if args.verbose:
                    try:
                        print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()), args.downstream_task)
                    except:
                        print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss), args.downstream_task)


                if epoch % args.eval_freq == 0:
                    if args.downstream_task == 'classification':
                        model.eval()
                        graph_learner.eval()
                        f_adj = Adj

                        if args.sparse:
                            f_adj.edata['w'] = f_adj.edata['w'].detach()
                        else:
                            f_adj = f_adj.detach()

                        val_accu, test_accu, _ = self.evaluate_adj_by_cls(f_adj, features, nfeats, labels,
                                                                               nclasses, train_mask, val_mask, test_mask, args)

                        if val_accu > best_val:
                            best_val = val_accu
                            best_val_test = test_accu
                            best_epoch = epoch

                    elif args.downstream_task == 'clustering':
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        
                        embedding = embedding.cpu().detach().numpy()

                        # For unlabeled data, we'll use unsupervised metrics
                        if labels is None:
                            # Compute silhouette score to measure cluster quality
                            if args.verbose:
                                print(f"Embedding shape: {embedding.shape}")
                            kmeans = KMeans(n_clusters=nclasses, random_state=0).fit(embedding)
                            
                            predict_labels = kmeans.predict(embedding)
                            
                            # Silhouette score: higher is better (-1 to 1)
                            sil_score = silhouette_score(embedding, predict_labels)
                            
                            # Davies-Bouldin score: lower is better
                            db_score = davies_bouldin_score(embedding, predict_labels)
                            
                            # Inertia (within-cluster sum of squares): lower is better
                            inertia = kmeans.inertia_
                            
                            # Get cluster sizes
                            unique, counts = np.unique(predict_labels, return_counts=True)
                            cluster_sizes = dict(zip(unique, counts))
                            
                            if args.verbose:
                                print("\nClustering Evaluation:")
                                print(f"Silhouette Score: {sil_score:.4f} (higher is better, range: -1 to 1)")
                                print(f"Davies-Bouldin Score: {db_score:.4f} (lower is better)")
                                print(f"Inertia: {inertia:.4f} (lower is better)")
                            
                            # Calculate cluster statistics
                            cluster_stats = {}
                            all_distances = []
                            for cluster_id in unique:
                                cluster_mask = predict_labels == cluster_id
                                cluster_points = embedding[cluster_mask]
                                
                                cluster_center = kmeans.cluster_centers_[cluster_id]
                                
                                # Calculate mean distance to center
                                distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
                                all_distances.extend(distances)
                                
                                mean_dist = distances.mean()
                                std_dist = distances.std()
                                
                                cluster_stats[cluster_id] = {
                                    'size': counts[cluster_id],
                                    'mean_distance_to_center': mean_dist,
                                    'std_distance_to_center': std_dist
                                }
                            
                            # Calculate overall statistics
                            avg_cluster_size = np.mean(counts)
                            std_cluster_size = np.std(counts)
                            min_cluster_size = np.min(counts)
                            max_cluster_size = np.max(counts)
                            
                            avg_distance = np.mean(all_distances)
                            std_distance = np.std(all_distances)
                            
                            # Print condensed summary
                            if args.verbose:
                                print("\nCluster Statistics Summary:")
                                print(f"Number of clusters: {len(unique)}")
                                print(f"Cluster sizes: min={min_cluster_size}, max={max_cluster_size}, avg={avg_cluster_size:.2f}, std={std_cluster_size:.2f}")
                                print(f"Distance to center: avg={avg_distance:.4f}, std={std_distance:.4f}")
                                
                                # Replace detailed distribution with condensed statistics
                                sorted_clusters = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)
                                
                                # Only show top 3 and bottom 3 clusters if there are more than 6 clusters
                                if len(unique) > 6:
                                    print("\nLargest clusters:")
                                    for cluster_id, count in sorted_clusters[:3]:
                                        print(f"  Cluster {cluster_id}: {count} points ({count/len(predict_labels)*100:.1f}%)")
                                    
                                    print("\nSmallest clusters:")
                                    for cluster_id, count in sorted_clusters[-3:]:
                                        print(f"  Cluster {cluster_id}: {count} points ({count/len(predict_labels)*100:.1f}%)")
                                    
                                    # Add size distribution histogram using text-based visualization
                                    print("\nCluster size distribution (text histogram):")
                                    bin_count = min(10, len(unique))  # Use at most 10 bins
                                    hist, bin_edges = np.histogram(counts, bins=bin_count)
                                    max_bar_length = 40  # Maximum length of histogram bars
                                    
                                    for i in range(len(hist)):
                                        bin_start = int(bin_edges[i])
                                        bin_end = int(bin_edges[i+1])
                                        bar_length = int((hist[i] / max(hist)) * max_bar_length)
                                        bar = '█' * bar_length
                                        print(f"  {bin_start:4d}-{bin_end:<4d} | {bar} ({hist[i]})")
                                else:
                                    # If few clusters, show all of them
                                    print("\nCluster size distribution:")
                                    for cluster_id, count in sorted_clusters:
                                        print(f"  Cluster {cluster_id}: {count} points ({count/len(predict_labels)*100:.1f}%)")
                        
                        else:
                            # Original code for labeled data
                            acc_mr, nmi_mr, f1_mr, ari_mr = [], [], [], []
                            for clu_trial in range(n_clu_trials):
                                kmeans = KMeans(n_clusters=nclasses, random_state=clu_trial).fit(embedding)
                                predict_labels = kmeans.predict(embedding)
                                cm_all = clustering_metrics(labels.cpu().numpy(), predict_labels)
                                acc_, nmi_, f1_, ari_ = cm_all.evaluationClusterModelFromLabel(print_results=False)
                                acc_mr.append(acc_)
                                nmi_mr.append(nmi_)
                                f1_mr.append(f1_)
                                ari_mr.append(ari_)
                            
                            acc, nmi, f1, ari = np.mean(acc_mr), np.mean(nmi_mr), np.mean(f1_mr), np.mean(ari_mr)
                            if args.verbose:
                                print("Final ACC: ", acc)
                                print("Final NMI: ", nmi)
                                print("Final F-score: ", f1)
                                print("Final ARI: ", ari)

            if args.downstream_task == 'classification':
                validation_accuracies.append(best_val.item())
                test_accuracies.append(best_val_test.item())
                if args.verbose:
                    print("Trial: ", trial + 1)
                    print("Best val ACC: ", best_val.item())
                    print("Best test ACC: ", best_val_test.item())
            # elif args.downstream_task == 'clustering':
            #     if labels is not None:
            #         if args.verbose:
            #             print("Final ACC: ", acc)
            #             print("Final NMI: ", nmi)
            #             print("Final F-score: ", f1)
            #             print("Final ARI: ", ari)

            # After training completes
            if args.save_model:
                # Save model, graph learner, features and final adjacency matrix
                self.save_model(model, graph_learner, features, anchor_adj, args.sparse, args,
                               output_dir=args.output_dir)
                if args.verbose:
                    print(f"Model saved to {args.output_dir}")

        if args.downstream_task == 'classification' and trial != 0:
            self.print_results(validation_accuracies, test_accuracies)

    def print_results(self, validation_accu, test_accu):
        s_val = "Val accuracy: {:.4f} +/- {:.4f}".format(np.mean(validation_accu), np.std(validation_accu))
        s_test = "Test accuracy: {:.4f} +/- {:.4f}".format(np.mean(test_accu),np.std(test_accu))
        print(s_val)
        print(s_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='person_data.csv',
                        help='Path to the dataset file')
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=0)
    parser.add_argument('-gsl_mode', type=str, default="structure_inference",
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
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=512)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)

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
    parser.add_argument('-save_model', type=int, default=0)
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
    
    # Memory optimization arguments
    parser.add_argument('-use_batched_arcface', type=int, default=0,
                       help='Whether to use memory-efficient batched ArcFace implementation (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_batch_size', type=int, default=1000,
                       help='Number of classes to process in each ArcFace batch (default: 1000)')
    parser.add_argument('-debug_memory', type=int, default=0,
                       help='Run memory debugging before training (0=disabled, 1=enabled)')
    parser.add_argument('-max_debug_samples', type=int, default=10000,
                       help='Maximum samples to use for memory debugging (default: 10000)')
    
    parser.add_argument('-use_mixed_precision', type=int, default=0,
                       help='Use mixed precision training (FP16) to reduce memory usage (0=disabled, 1=enabled)')
    parser.add_argument('-grad_accumulation_steps', type=int, default=1,
                       help='Number of steps to accumulate gradients before updating weights (1=disabled)')
    parser.add_argument('-memory_efficient_training', type=int, default=0,
                       help='Use memory-efficient training with gradient accumulation and chunked processing (0=disabled, 1=enabled)')


    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args, load_data)
