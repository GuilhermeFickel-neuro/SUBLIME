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

# Import batched ArcFace functionality
from batched_arcface import BatchedArcFaceLayer, batched_arcface_loss
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
        # Use the dedicated function from batched_arcface.py
        return batched_arcface_loss(self, model, graph_learner, features, anchor_adj, labels, args)

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
            self.save_model_config(f, args, model, graph_learner, features, sparse)

    def save_model_config(self, f, args, model, graph_learner, features, sparse):
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
            
        # Add flags to model loading defaults if needed (assuming they weren't saved)
        use_layer_norm_load = config.get('use_layer_norm', 'False').lower() in ['true', '1']
        use_residual_load = config.get('use_residual', 'False').lower() in ['true', '1']
        
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
                       use_layer_norm=use_layer_norm_load,
                       use_residual=use_residual_load,
                       use_arcface=arcface_params['use_arcface'],
                       num_classes=arcface_params['num_classes'],
                       arcface_scale=arcface_params['arcface_scale'],
                       arcface_margin=arcface_params['arcface_margin'],
                       )
        else:
            model = GCL(nlayers=model_params['nlayers'],
                       in_dim=features.shape[1],
                       hidden_dim=model_params['hidden_dim'],
                       emb_dim=model_params['emb_dim'],
                       proj_dim=model_params['proj_dim'],
                       dropout=model_params['dropout'],
                       dropout_adj=model_params['dropout_adj'],
                       sparse=sparse,
                       use_layer_norm=use_layer_norm_load,
                       use_residual=use_residual_load)
        
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
            new_adj = graph_learner(modified_features, faiss_index=faiss_index)
            
            # Process adjacency matrix based on sparse flag
            if not sparse:
                new_adj = symmetrize(new_adj)
                new_adj = normalize(new_adj, 'sym', sparse)
            
            # Get embeddings for all nodes including the new one.
            # The if/else based on use_arcface was redundant here because labels are not passed.
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
                model = GCL(nlayers=args.nlayers, in_dim=nfeats,
                           hidden_dim=args.hidden_dim, emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                           dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse,
                           use_layer_norm=bool(args.use_layer_norm),
                           use_residual=bool(args.use_residual),
                           use_arcface=True, num_classes=nclasses,
                           arcface_scale=args.arcface_scale, arcface_margin=args.arcface_margin,
                           )
                
                # Logic for using BatchedArcFace remains
                if args.use_batched_arcface:
                    # Check if the model already has an ArcFace layer to replace
                    if hasattr(model, 'arcface') and model.arcface is not None:
                        try:
                            original_arcface = model.arcface
                            in_features = original_arcface.in_features
                            out_features = original_arcface.out_features
                            scale = original_arcface.s # Use .s for scale
                            margin = original_arcface.m # Use .m for margin
                            easy_margin = original_arcface.easy_margin
                        except AttributeError as e:
                            print(f"Error accessing attributes from original ArcFace layer: {e}")
                            print("Skipping batched ArcFace replacement.")
                        else:
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
                            if original_arcface.weight is not None:
                                batched_layer.weight.data.copy_(original_arcface.weight.data)
                            # Replace the layer
                            model.arcface = batched_layer
                            if args.verbose:
                                print(f"Using Batched ArcFace implementation with batch size {args.arcface_batch_size}")
                    else:
                         print("Warning: use_batched_arcface is True, but the model doesn't have an initial 'arcface' layer to replace.")

                if args.verbose:
                    print(f"Using ArcFace loss with {nclasses} classes, scale={args.arcface_scale}, margin={args.arcface_margin}")
            else:
                model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                          emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                          dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

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

            if args.downstream_task == 'classification':
                best_val = 0.0 # Initialize as float
                best_val_test = 0
                best_epoch = 0

            # Initialize gradient accumulation steps
            grad_accumulation_steps = args.grad_accumulation_steps if hasattr(args, 'grad_accumulation_steps') else 1
            if grad_accumulation_steps > 1 and args.verbose:
                print(f"Using gradient accumulation with {grad_accumulation_steps} steps")
                
            for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Training", initial=start_epoch, total=args.epochs):
                model.train()
                graph_learner.train()
                
                # Simplified training loop: Removed mixed precision and logic for different ArcFace types
                if grad_accumulation_steps > 1:
                    # Gradient accumulation path (without mixed precision)
                    accumulated_loss = 0
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()

                    for i in range(grad_accumulation_steps):
                        # Forward pass
                        if args.use_arcface:
                            # Always use batched version if ArcFace is enabled
                            mini_loss, Adj = self.loss_arcface_batched(model, graph_learner, features, anchor_adj, labels, args)
                        else:
                            mini_loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                        mini_loss = mini_loss / grad_accumulation_steps # Scale loss

                        # Backward pass (no scaler)
                        mini_loss.backward()

                        accumulated_loss += mini_loss.item() * grad_accumulation_steps # Unscale for reporting

                    # --- Gradient Clipping ---
                    if args.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)
                    # --- End Gradient Clipping ---

                    # Update weights
                    optimizer_cl.step()
                    optimizer_learner.step()

                    loss = accumulated_loss / grad_accumulation_steps # Average loss for the step

                else:
                    # Standard training path (no gradient accumulation, no mixed precision)
                    optimizer_cl.zero_grad()
                    optimizer_learner.zero_grad()

                    if args.use_arcface:
                        # Always use batched version if ArcFace is enabled
                        loss, Adj = self.loss_arcface_batched(model, graph_learner, features, anchor_adj, labels, args)
                    else:
                        loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                    # --- NaN Loss Check ---
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
                    # --- End NaN Loss Check ---

                    # Backward and optimize
                    loss.backward()

                    # --- Gradient Clipping ---
                    if args.clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                        torch.nn.utils.clip_grad_norm_(graph_learner.parameters(), args.clip_norm)
                    # --- End Gradient Clipping ---

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

                # --- Periodic Checkpointing ---
                if epoch > 0 and epoch % args.checkpoint_freq == 0:
                    current_checkpoint_path = self.save_checkpoint(
                        epoch, model, graph_learner, optimizer_cl, optimizer_learner, anchor_adj, args
                    )
                    last_checkpoint_path = current_checkpoint_path # Update last known good path
                # --- End Periodic Checkpointing ---

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
    
    # Memory optimization arguments (kept batched arcface and grad accum)
    parser.add_argument('-use_batched_arcface', type=int, default=0,
                       help='Whether to use memory-efficient batched ArcFace implementation (0=disabled, 1=enabled)')
    parser.add_argument('-arcface_batch_size', type=int, default=1000,
                       help='Number of classes to process in each ArcFace batch (default: 1000)')
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