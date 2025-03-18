import argparse
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from model import GCN, GCL
from graph_learners import *
from utils import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dgl
from data_loader import load_data

import random
import os

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

    def save_model(self, model, graph_learner, features, adj, sparse, output_dir='saved_models'):
        """
        Save model, graph learner, features and adjacency matrix
        
        Args:
            model: The GCL model
            graph_learner: The graph learner model
            features: Node features
            adj: Adjacency matrix
            sparse: Whether the adjacency matrix is sparse
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
                    config[key] = value.lower() == 'true'
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
        
        # Initialize models
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

            model = GCL(nlayers=args.nlayers, in_dim=nfeats, hidden_dim=args.hidden_dim,
                         emb_dim=args.rep_dim, proj_dim=args.proj_dim,
                         dropout=args.dropout, dropout_adj=args.dropedge_rate, sparse=args.sparse)

            optimizer_cl = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.w_decay)

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

            for epoch in range(1, args.epochs + 1):
                model.train()
                graph_learner.train()

                loss, Adj = self.loss_gcl(model, graph_learner, features, anchor_adj, args)

                optimizer_cl.zero_grad()
                optimizer_learner.zero_grad()
                loss.backward()
                optimizer_cl.step()
                optimizer_learner.step()

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
                    print("Epoch {:05d} | CL Loss {:.4f}".format(epoch, loss.item()), args.downstream_task)

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
            elif args.downstream_task == 'clustering':
                if labels is not None:
                    if args.verbose:
                        print("Final ACC: ", acc)
                        print("Final NMI: ", nmi)
                        print("Final F-score: ", f1)
                        print("Final ARI: ", ari)

            # After training completes
            if args.save_model:
                # Save model, graph learner, features and final adjacency matrix
                self.save_model(model, graph_learner, features, anchor_adj, args.sparse, 
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

    args = parser.parse_args()

    experiment = Experiment()
    experiment.train(args, load_data)
