import argparse
import copy
from datetime import datetime
import math

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

# Add sampled ArcFace imports
from sampled_arcface import SampledArcFaceLayer, arcface_loss_with_sampling
# Import memory optimization utilities (only gradient accumulation remains relevant)
from memory_optimizations import GradientAccumulation # Keep if grad_accumulation is used

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

    def loss_gcl_with_classification(self, model, graph_learner, features, anchor_adj, binary_labels, args):
        """
        Combined loss function that includes both contrastive learning and binary classification
        
        Args:
            model: The GCL model with classification head
            graph_learner: Graph structure learning module
            features: Node features
            anchor_adj: Anchor adjacency matrix
            binary_labels: Binary labels for classification (0 or 1)
            args: Training arguments
            
        Returns:
            total_loss: Combined loss from contrastive learning and classification
            learned_adj: Learned adjacency matrix
            cls_accuracy: Accuracy of binary classification
        """
        # First get contrastive loss and learned adjacency
        # view 1: anchor graph
        if args.maskfeat_rate_anchor:
            mask_v1, _ = get_feat_mask(features, args.maskfeat_rate_anchor)
            features_v1 = features * (1 - mask_v1)
        else:
            features_v1 = copy.deepcopy(features)

        # Get projection (z1) and embedding (emb1) from anchor view
        z1, emb1, _, _ = model(features_v1, anchor_adj, 'anchor', include_features=True)

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

        # Get projection (z2), embedding (emb2), and classification output
        z2, emb2, _, cls_output = model(features_v2, learned_adj, 'learner', include_features=True)

        # Compute contrastive loss
        if args.contrast_batch_size:
            node_idxs = list(range(features.shape[0]))
            batches = split_batch(node_idxs, args.contrast_batch_size)
            contrastive_loss = 0
            for batch in batches:
                weight = len(batch) / features.shape[0]
                contrastive_loss += model.calc_loss(z1[batch], z2[batch]) * weight
        else:
            contrastive_loss = model.calc_loss(z1, z2)

        # Compute classification loss
        cls_loss, cls_accuracy = self.loss_binary_cls(cls_output, binary_labels)
        
        # Combine losses using weight parameter
        total_loss = contrastive_loss + args.annotation_loss_weight * cls_loss
        
        print(f"Combined loss: contrastive={contrastive_loss.item():.4f}, classification={cls_loss.item():.4f}, "
                f"weight={args.annotation_loss_weight}, accuracy={cls_accuracy.item():.4f}")
            
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
                if key in ['sparse', 'use_layer_norm', 'use_residual', 'use_arcface', 'use_sampled_arcface', 'use_classification_head']:
                    config[key] = value.lower() in ['true', '1', 't', 'y', 'yes']
                elif key in ['feature_dim', 'num_nodes', 'nlayers', 'hidden_dim', 'emb_dim', 'proj_dim', 'k', 'num_classes', 'arcface_num_samples', 'classification_head_layers']:
                    config[key] = int(value)
                elif key in ['dropout', 'dropout_adj', 'arcface_scale', 'arcface_margin', 'classification_dropout']:
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
        use_arcface = config.get('use_arcface', False)
        arcface_params = {}
        if use_arcface:
            arcface_params['use_arcface'] = True
            arcface_params['num_classes'] = config.get('num_classes', 0)
            arcface_params['arcface_scale'] = config.get('arcface_scale', 30.0)
            arcface_params['arcface_margin'] = config.get('arcface_margin', 0.5)
            
            # Add sampled arcface config loading
            use_sampled_arcface = config.get('use_sampled_arcface', False)
            if use_sampled_arcface:
                arcface_params['use_sampled_arcface'] = True
                arcface_params['arcface_num_samples'] = config.get('arcface_num_samples', 5000)

        # Check for classification head parameters in config
        use_classification_head = config.get('use_classification_head', False)
        classification_params = {}
        if use_classification_head:
            classification_params['use_classification_head'] = True
            classification_params['classification_dropout'] = config.get('classification_dropout', 0.3)
            classification_params['classification_head_layers'] = config.get('classification_head_layers', 2)

        # Add flags to model loading defaults if needed (assuming they weren't saved)
        use_layer_norm_load = config.get('use_layer_norm', False)
        use_residual_load = config.get('use_residual', False)
        
        # Initialize models with arcface and classification parameters if needed
        init_params = {
            'nlayers': model_params['nlayers'],
            'in_dim': features.shape[1],
            'hidden_dim': model_params['hidden_dim'],
            'emb_dim': model_params['emb_dim'],
            'proj_dim': model_params['proj_dim'],
            'dropout': model_params['dropout'],
            'dropout_adj': model_params['dropout_adj'],
            'sparse': sparse,
            'use_layer_norm': use_layer_norm_load,
            'use_residual': use_residual_load
        }
        
        # Add ArcFace parameters if needed
        if use_arcface:
            init_params.update(arcface_params)
            
        # Add classification head parameters if needed
        if use_classification_head:
            init_params.update(classification_params)
            
        # Initialize model with all collected parameters
        model = GCL(**init_params)
        
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
        
        # Load adjacency matrix *before* initializing graph learner if needed (e.g., FGP)
        adj = self._load_adjacency(input_dir, sparse)
        initial_graph_data_loaded = adj # Assume loaded adj is the initial graph

        # Create the appropriate graph learner
        if learner_type == 'fgp':
            # Pass the loaded initial graph data
            fgp_elu_alpha = config.get('fgp_elu_alpha', 6) # Load alpha, default to 6
            graph_learner = FGP_learner(k=model_params['k'], knn_metric=model_params['sim_function'], i=fgp_elu_alpha,
                                      sparse=sparse, initial_graph_data=initial_graph_data_loaded)
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
            # Adjacency (initial_graph_data_loaded) is already loaded
            graph_learner = GNN_learner(2, features.shape[1], model_params['k'], 
                                     model_params['sim_function'], 6, sparse,
                                     model_params['activation_learner'], initial_graph_data_loaded)
        else:
            # Default to FGP learner if no specific type is found
            fgp_elu_alpha = config.get('fgp_elu_alpha', 6)
            graph_learner = FGP_learner(k=model_params['k'], knn_metric=model_params['sim_function'], i=fgp_elu_alpha,
                                      sparse=sparse, initial_graph_data=initial_graph_data_loaded)
        
        # Load model weights
        model.load_state_dict(torch.load(os.path.join(input_dir, 'model.pt')))
        graph_learner.load_state_dict(torch.load(os.path.join(input_dir, 'graph_learner.pt')))
        
        # Move models to device
        model = model.to(self.device)
        graph_learner = graph_learner.to(self.device)
        
        # Return the loaded components, including the initial graph data used by the learner
        return model, graph_learner, features, initial_graph_data_loaded, sparse

    def _load_adjacency(self, input_dir, sparse):
        """Helper method to load adjacency matrix"""
        adj_path = os.path.join(input_dir, 'adjacency.pt')
        
        if sparse:
            # Load the saved data (could be dict for DGL or sparse tensor)
            saved_data = torch.load(adj_path)
            if isinstance(saved_data, dict):
                # This is a saved DGL graph
                edges = saved_data['edges']
                weights = saved_data['weights']
                num_nodes = saved_data['num_nodes']
                
                # Recreate DGL graph
                adj = dgl.graph(edges, num_nodes=num_nodes, device=self.device)
                adj.edata['w'] = weights.to(self.device)
            else:
                # This is a torch sparse tensor
                adj = saved_data.to(self.device)
        else:
            adj = torch.load(adj_path).to(self.device)
            
        return adj

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
        if load_data_fn is None:
            raise ValueError("Must provide a data loading function")
            
        # Load data - Expected return: features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, initial_graph_data
        # Note: nclasses might be n_clusters from args in clustering mode
        features, nfeats, labels, nclasses_or_clusters, train_mask, val_mask, test_mask, initial_graph_data = load_data_fn(args)
        
        # Assign nclasses based on downstream task
        if args.downstream_task == 'clustering':
            nclasses = nclasses_or_clusters # Use n_clusters from args
            if labels is not None and args.verbose:
                print("Warning: Labels provided but downstream task is clustering. Labels will be used for evaluation only.")
        elif labels is not None:
            nclasses = nclasses_or_clusters # Use nclasses derived from data
        else:
            # Handle case where labels are None but task is classification (e.g., if using ArcFace with row indices)
            # Or if ArcFace is used in clustering without true labels
            nclasses = nclasses_or_clusters # Should be num_nodes if using row indices for ArcFace
            if args.use_arcface and labels is None:
                nclasses = features.shape[0] # ArcFace uses row indices
            elif args.verbose:
                print("Warning: No labels provided for classification task or nclasses couldn't be determined.")

        # Load annotated dataset if provided
        annotated_features = None
        binary_labels = None
        if args.annotated_dataset is not None:
            # Call the same load function but with annotated dataset
            saved_dataset = args.dataset
            args.dataset = args.annotated_dataset
            
            # Load annotated data (using the same loading function)
            annotated_features, annotated_nfeats, _, _, _, _, _, _ = load_data_fn(args)
            
            # Restore original dataset path
            args.dataset = saved_dataset
            
            # Check if dimensions match the main dataset
            if annotated_nfeats != nfeats:
                raise ValueError(f"Annotated dataset features dimension ({annotated_nfeats}) "
                                f"doesn't match main dataset ({nfeats})")
            
            # Extract binary labels from the annotated dataset
            # This assumes the binary labels are stored in the last column
            # We'll need to modify the data_loader.py to support this properly
            # binary_labels = torch.tensor(annotated_features[:, -1].astype(int)) # Comment out old line
            binary_labels = annotated_features[:, -1].int() # Use PyTorch's .int() method

            # Remove the label column from features
            annotated_features = annotated_features[:, :-1]
            
            if args.verbose:
                print(f"Loaded annotated dataset with {len(binary_labels)} samples")
                print(f"Binary label distribution: 0s={int((binary_labels == 0).sum())}, "
                     f"1s={int((binary_labels == 1).sum())}")

        # For ArcFace, create labels as row indices if not provided
        if args.use_arcface:
            if labels is None:
                if args.verbose:
                    print("Using row indices as labels for ArcFace")
                labels = torch.arange(features.shape[0])
                # Simplified nclasses assignment - only depends on feature shape now
                nclasses = features.shape[0]

        if args.downstream_task == 'classification':
            test_accuracies = []
            validation_accuracies = []
        elif args.downstream_task == 'clustering':
            n_clu_trials = copy.deepcopy(args.ntrials)
            args.ntrials = 1

        for trial in range(args.ntrials):
            self.setup_seed(trial)

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

            anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)

            if args.sparse:
                anchor_adj_torch_sparse = copy.deepcopy(anchor_adj)
                anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

            if args.type_learner == 'fgp':
                # Pass pre-computed initial graph data to FGP learner
                graph_learner = FGP_learner(k=args.k, knn_metric=args.sim_function, i=6, # Note: i=6 seems hardcoded, consider making it an arg
                                          sparse=args.sparse, initial_graph_data=initial_graph_data)
            elif args.type_learner == 'mlp':
                graph_learner = MLP_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner)
            elif args.type_learner == 'att':
                graph_learner = ATT_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                          args.activation_learner)
            elif args.type_learner == 'gnn':
                graph_learner = GNN_learner(2, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                     args.activation_learner, anchor_adj)

            # Determine if we should use classification head based on whether annotated_dataset was provided
            use_classification_head = annotated_features is not None and binary_labels is not None

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
                    'num_classes': nclasses,
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

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)
            
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
                
            # Move annotated data to device if provided
            if annotated_features is not None:
                annotated_features = torch.tensor(annotated_features, dtype=torch.float32).to(self.device)
            if binary_labels is not None:
                binary_labels = binary_labels.to(self.device)

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
                
            for epoch in tqdm(range(start_epoch, args.epochs), desc="Training", initial=start_epoch, total=args.epochs):
                model.train()
                graph_learner.train()
                
                # Simplified training loop with annotated dataset support
                if grad_accumulation_steps > 1:
                    # Gradient accumulation path
                    accumulated_loss = 0
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()

                    for i in range(grad_accumulation_steps):
                        # Forward pass
                        if use_classification_head:
                            # Use combined loss with classification
                            mini_loss, Adj, cls_accuracy = self.loss_gcl_with_classification(
                                model, graph_learner, annotated_features, anchor_adj, binary_labels, args
                            )
                            # Track classification accuracy
                            if i == 0:  # only track for first accumulation step
                                current_cls_accuracy = cls_accuracy
                        elif args.use_arcface:
                            # Use ArcFace loss
                            mini_loss, Adj = self.loss_arcface_sampled(model, graph_learner, features, anchor_adj, labels, args)
                        else:
                            # Use standard contrastive loss
                            mini_loss, Adj, _ = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                        mini_loss = mini_loss / grad_accumulation_steps # Scale loss

                        # Backward pass
                        mini_loss.backward()

                        accumulated_loss += mini_loss.item() * grad_accumulation_steps # Unscale for reporting

                    # Gradient Clipping
                    if args.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)

                    # Update weights
                    optimizer_cl.step()
                    optimizer_learner.step()

                    loss = accumulated_loss / grad_accumulation_steps # Average loss for the step

                else:
                    # Standard training path (no gradient accumulation)
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()

                    # Choose appropriate loss function based on available data
                    if use_classification_head:
                        # Use combined loss with classification
                        loss, Adj, current_cls_accuracy = self.loss_gcl_with_classification(
                            model, graph_learner, annotated_features, anchor_adj, binary_labels, args
                        )
                        # Track best classification accuracy
                        if current_cls_accuracy > best_cls_accuracy:
                            best_cls_accuracy = current_cls_accuracy
                    elif args.use_arcface:
                        # Use ArcFace loss
                        loss, Adj = self.loss_arcface_sampled(model, graph_learner, features, anchor_adj, labels, args)
                    else:
                        # Use standard contrastive loss
                        loss, Adj, _ = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                    # NaN Loss Check
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
                                continue # Skip the rest of the loop for this epoch
                            except Exception as e:
                                print(f"Error reloading checkpoint {last_checkpoint_path} after NaN loss: {e}")
                                print("Stopping training trial due to unrecoverable NaN loss.")
                                break # Exit the epoch loop for this trial
                        else:
                            print("No previous checkpoint found to revert to after NaN loss. Stopping training trial.")
                            break # Exit the epoch loop for this trial

                    # Backward and optimize
                    loss.backward()

                    # Gradient Clipping
                    if args.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)

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
                        loss_msg = "Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item())
                        if use_classification_head:
                            # Add classification accuracy to the message
                            loss_msg += " | Cls Acc {:.4f}".format(current_cls_accuracy.item())
                            cls_accuracies.append(current_cls_accuracy.item())
                        loss_msg += " | " + args.downstream_task
                        print(loss_msg)
                    except:
                        print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss), args.downstream_task)

                # Periodic Checkpointing
                if epoch > 0 and epoch % args.checkpoint_freq == 0:
                    current_checkpoint_path = self.save_checkpoint(
                        epoch, model, graph_learner, optimizer_cl, optimizer_learner, anchor_adj, args
                    )
                    last_checkpoint_path = current_checkpoint_path # Update last known good path
                    
                    # Also report average classification accuracy over period if using classification head
                    if use_classification_head and len(cls_accuracies) > 0:
                        avg_cls_acc = sum(cls_accuracies) / len(cls_accuracies)
                        if args.verbose:
                            print(f"Average classification accuracy over last {len(cls_accuracies)} epochs: {avg_cls_acc:.4f}")
                        cls_accuracies = []  # Reset for next period

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
                            best_val = val_accu.item() if isinstance(val_accu, torch.Tensor) else val_accu # Ensure float
                            best_val_test = test_accu
                            best_epoch = epoch

                    elif args.downstream_task == 'clustering' and labels is not None:
                        model.eval()
                        graph_learner.eval()
                        _, embedding = model(features, Adj)
                        
                        embedding = embedding.cpu().detach().numpy()
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
                validation_accuracies.append(best_val)
                test_accuracies.append(best_val_test.item())
                if args.verbose:
                    print("Trial: ", trial + 1)
                    print("Best val ACC: ", best_val)
                    print("Best test ACC: ", best_val_test.item())
                    
                    # Report best classification accuracy if used
                    if use_classification_head:
                        print("Best classification accuracy: ", best_cls_accuracy.item())

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

    return parser

if __name__ == '__main__':
    parent_parser = create_parser()
    args = parent_parser.parse_args()

    experiment = Experiment()
    experiment.train(args, load_data)