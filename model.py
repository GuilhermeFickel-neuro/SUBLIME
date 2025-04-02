import copy
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F

from graph_learners import *
from layers import GCNConv_dense, GCNConv_dgl, SparseDropout
from torch.nn import Sequential, Linear, ReLU

# GCN for evaluation.
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, Adj, sparse, use_layer_norm=False, use_residual=False):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        self.dropout = dropout # Store dropout rate

        if use_layer_norm:
            self.norm_layers = nn.ModuleList()
        if use_residual and in_channels != hidden_channels:
            # Projection layer for the first residual connection if dimensions don't match
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            self.input_proj = None

        LayerClass = GCNConv_dgl if sparse else GCNConv_dense

        # Input layer
        self.layers.append(LayerClass(in_channels, hidden_channels))
        if use_layer_norm:
            self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(LayerClass(hidden_channels, hidden_channels))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_channels))

        # Output layer
        self.layers.append(LayerClass(hidden_channels, out_channels))

        self.dropout_adj_p = dropout_adj
        self.Adj = Adj
        self.Adj.requires_grad = False
        self.sparse = sparse

        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

    def forward(self, x):
        if self.sparse:
            Adj = copy.deepcopy(self.Adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(self.Adj)

        h_prev = x # Store input for first potential residual connection

        for i, conv in enumerate(self.layers[:-1]):
            h = conv(x, Adj)

            # Apply residual connection before normalization/activation
            if self.use_residual:
                if i == 0 and self.input_proj is not None:
                    # Project input for the first layer's residual connection
                    residual = self.input_proj(h_prev)
                elif i > 0 or self.input_proj is None:
                     # For hidden layers or if input_proj wasn't needed
                    residual = h_prev
                else: # Should not happen based on logic, but for safety
                    residual = 0

                # Add residual before norm/activation
                h = h + residual
                h_prev = h # Update h_prev for the next layer's residual

            # Apply LayerNorm if enabled
            if self.use_layer_norm:
                h = self.norm_layers[i](h)

            # Activation and Dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = h # Update x for the next iteration

        # Output layer (no norm, residual, activation, or dropout typically)
        x = self.layers[-1](x, Adj)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse, use_layer_norm=False, use_residual=False):
        super(GraphEncoder, self).__init__()
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj
        self.sparse = sparse
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual

        self.gnn_encoder_layers = nn.ModuleList()
        if use_layer_norm:
            self.norm_layers = nn.ModuleList()
        if use_residual and in_dim != hidden_dim:
            # Projection layer for the first residual connection
            self.input_proj = nn.Linear(in_dim, hidden_dim)
        else:
             self.input_proj = None


        LayerClass = GCNConv_dgl if sparse else GCNConv_dense

        # Input layer
        self.gnn_encoder_layers.append(LayerClass(in_dim, hidden_dim))
        if use_layer_norm:
            self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Hidden layers
        for _ in range(nlayers - 2):
            self.gnn_encoder_layers.append(LayerClass(hidden_dim, hidden_dim))
            if use_layer_norm:
                self.norm_layers.append(nn.LayerNorm(hidden_dim))

        # Embedding layer (output layer of the encoder part)
        self.gnn_encoder_layers.append(LayerClass(hidden_dim, emb_dim))

        # Adjacency dropout
        if self.sparse:
            self.dropout_adj = SparseDropout(dprob=dropout_adj)
        else:
            self.dropout_adj = nn.Dropout(p=dropout_adj)

        # Projection head (remains the same)
        self.proj_head = Sequential(Linear(emb_dim, proj_dim), ReLU(inplace=True),
                                           Linear(proj_dim, proj_dim))

    def forward(self, x, Adj_, branch=None):
        if self.sparse:
            if branch == 'anchor':
                Adj = copy.deepcopy(Adj_)
            else:
                Adj = Adj_
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=self.dropout_adj_p, training=self.training)
        else:
            Adj = self.dropout_adj(Adj_)

        h_prev = x # Store input for first potential residual connection

        # Process through GNN layers (excluding the final embedding layer)
        for i, conv in enumerate(self.gnn_encoder_layers[:-1]):
            h = conv(x, Adj)

            # Apply residual connection
            if self.use_residual:
                if i == 0 and self.input_proj is not None:
                    residual = self.input_proj(h_prev)
                elif i > 0 or self.input_proj is None:
                    residual = h_prev
                else: # Should not happen
                    residual = 0

                h = h + residual # Add residual before norm/activation

            # Apply LayerNorm if enabled
            if self.use_layer_norm:
                h = self.norm_layers[i](h)

            # Activation and Dropout
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            # Update x and h_prev for the next iteration
            x = h
            if self.use_residual: # Only need h_prev if using residuals
                h_prev = h


        # Final embedding layer (no norm, residual, activation, or dropout)
        embedding = self.gnn_encoder_layers[-1](x, Adj)

        # Projection head
        z = self.proj_head(embedding)
        return z, embedding

class GCL(nn.Module):
    def __init__(self, nlayers, in_dim, hidden_dim, emb_dim, proj_dim, dropout, dropout_adj, sparse,
                 use_layer_norm=False, use_residual=False, # <-- Pass flags here
                 use_arcface=False, num_classes=None, arcface_scale=30.0, arcface_margin=0.5,
                 use_sampled_arcface=False, arcface_num_samples=None):
        super(GCL, self).__init__()

        # Pass the flags to the GraphEncoder
        self.encoder = GraphEncoder(nlayers, in_dim, hidden_dim, emb_dim, proj_dim,
                                    dropout, dropout_adj, sparse,
                                    use_layer_norm=use_layer_norm, use_residual=use_residual)
        self.use_arcface = use_arcface
        self.use_sampled_arcface = use_sampled_arcface
        
        # Add ArcFace layer if specified
        if use_arcface and num_classes is not None:
            if use_sampled_arcface and arcface_num_samples is not None:
                # Use the memory-efficient sampled ArcFace implementation
                from sampled_arcface import SampledArcFaceLayer
                self.sampled_arcface = True
                self.arcface = SampledArcFaceLayer(
                    in_features=emb_dim,
                    max_classes=num_classes,
                    num_samples=arcface_num_samples,
                    scale=arcface_scale,
                    margin=arcface_margin
                )
                print(f"Using Fixed Sampled ArcFace with {arcface_num_samples}/{num_classes} classes")
            else:
                # Use the standard ArcFace implementation
                from layers import ArcFaceLayer
                self.sampled_arcface = False
                self.arcface = ArcFaceLayer(emb_dim, num_classes, scale=arcface_scale, margin=arcface_margin)

    def forward(self, x, Adj_, branch=None, labels=None, include_features=False):
        z, embedding = self.encoder(x, Adj_, branch)
        
        # Return hidden features if requested (for use in loss functions)
        if include_features:
            return z, embedding, None
        
        # If using ArcFace and we have labels, return ArcFace outputs too
        if self.use_arcface and hasattr(self, 'arcface') and labels is not None:
            if hasattr(self, 'sampled_arcface') and self.sampled_arcface:
                arcface_output, _ = self.arcface(embedding, labels)
            else:
                arcface_output = self.arcface(embedding, labels)
            return z, embedding, arcface_output
        
        return z, embedding

    @staticmethod
    def calc_loss(x, x_aug, temperature=0.2, sym=True):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        if sym:
            loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

            loss_0 = - torch.log(loss_0).mean()
            loss_1 = - torch.log(loss_1).mean()
            loss = (loss_0 + loss_1) / 2.0
            return loss
        else:
            loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss_1 = - torch.log(loss_1).mean()
            return loss_1