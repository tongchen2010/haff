import os
import time
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from functools import partial
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric import seed_everything
import math
#import torch_scatter
# ------------------------------
# Models and Masking Functions
# ------------------------------

class GraphTransformerPYG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=hidden_channels, dropout=dropout)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=hidden_channels, dropout=dropout)

        self.pool = global_mean_pool
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))

        x = self.pool(x, batch)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


def mask_node(x, mask_rate=0.2, noise=0.05):
    """
    Randomly mask rows (node features) in the feature matrix.

    Args:
        x (torch.Tensor or np.ndarray): Node feature matrix of shape (num_nodes, num_features).
        mask_rate (float): Proportion of nodes to mask.
        noise (float): Proportion of masked nodes to replace with noise.

    Returns:
        masked_x (torch.Tensor): Modified feature matrix with random masking.
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes, num_features = x.size()
    # Randomly permute the node indices
    perm = torch.randperm(num_nodes, device=x.device)
    num_mask_nodes = int(mask_rate * num_nodes)
    if num_mask_nodes == 0:
        return x
    # Select nodes to mask
    mask_nodes = perm[:num_mask_nodes]
    
    # Determine noisy nodes and token nodes
    num_noise_nodes = int(noise * num_mask_nodes)
    perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    token_nodes = mask_nodes[perm_mask[:num_mask_nodes - num_noise_nodes]]
    noise_nodes = mask_nodes[perm_mask[num_mask_nodes - num_noise_nodes:]]
    
    # Generate random noise
    noise_value = torch.randn(noise_nodes.size(0), num_features, device=x.device)
    
    # Clone the feature matrix to apply masking
    masked_x = x.clone()
    # Replace token nodes with zeros and noise nodes with random noise
    masked_x[token_nodes] = 0.0
    masked_x[noise_nodes] = noise_value

    return masked_x


def mask_edge(adj_matrix, mask_rate=0.2, noise=0.05):
    """
    Randomly drop some edges and add random ones to the adjacency matrix.
    
    Args:
        adj_matrix (torch.Tensor or np.ndarray): Adjacency matrix of shape (num_nodes, num_nodes).
        mask_rate (float): Proportion of edges to drop.
        noise (float): Proportion of noise edges to add (relative to original number of edges).
    
    Returns:
        adj_matrix (torch.Tensor): Modified adjacency matrix.
    """
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Remove self-loops (if any)
    adj_matrix = adj_matrix.clone()
    adj_matrix.fill_diagonal_(0)
    
    # Mask edges (drop edges)
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False)
    num_edges = edge_indices.size(0)
    num_edges_to_drop = int(num_edges * mask_rate)
    if num_edges > 0 and num_edges_to_drop > 0:
        drop_indices = torch.randperm(num_edges)[:num_edges_to_drop]
        adj_matrix[edge_indices[drop_indices, 0], edge_indices[drop_indices, 1]] = 0
    
    # Add noise (random edges)
    num_nodes = adj_matrix.size(0)
    num_edges_to_add = int(num_edges * noise)
    for _ in range(num_edges_to_add):
        i, j = torch.randint(0, num_nodes, (2,))
        if i != j:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # Assuming an undirected graph
    
    # Add self-loops
    adj_matrix.fill_diagonal_(1)
    
    return adj_matrix


class GraphTransformer(nn.Module):
    def __init__(self, in_hidden, out_hidden, num_heads, dropout, num_layers=2):
        super(GraphTransformer, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Linear layer to project input features
        self.input_proj = nn.Linear(in_hidden, out_hidden)
        self.pos_proj = nn.Linear(in_hidden, out_hidden)

        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=out_hidden,
                nhead=num_heads,
                dim_feedforward=out_hidden * 4,
                dropout=dropout,
                activation='gelu',
            )
            for _ in range(num_layers)
        ])

        # Final output layer
        self.output_proj = nn.Linear(out_hidden, out_hidden)

    def forward(self, adj_matrix, feat):
        # Apply input projection
        h = self.input_proj(feat)

        # Compute normalized adjacency matrix (using symmetric normalization)
        deg = adj_matrix.sum(dim=1).pow(-0.5)
        deg[deg == float('inf')] = 0
        norm_adj = deg.unsqueeze(1) * adj_matrix * deg.unsqueeze(0)

        # Positional encoding via neighborhood aggregation
        pos_enc = torch.matmul(norm_adj, feat)
        pos_enc = self.pos_proj(pos_enc)

        h = h + pos_enc

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            h = transformer_layer(h)

        # Apply output projection
        h = self.output_proj(h)

        return h
    



def roi_pooling(x, roi_labels, batch, num_rois, pooling_type='mean'):
    """
    Perform ROI pooling ensuring that every sample has `num_rois` slots, even if some ROIs are missing.

    Args:
        x (torch.Tensor): Node embeddings of shape (n, d), where n is the number of nodes, and d is the feature dimension.
        roi_labels (torch.Tensor): ROI labels of shape (n,), indicating which ROI each node belongs to.
        batch (torch.Tensor): Batch indices of shape (n,), indicating which sample each node belongs to.
        num_rois (int): The total number of ROIs to ensure each sample has `num_rois * d` shape.
        pooling_type (str): Type of pooling ('mean', 'max', or 'sum').

    Returns:
        torch.Tensor: Aggregated ROI features of shape (batch_size, num_rois, d).
    """
    batch_size = batch.max().item() + 1  # Number of unique samples
    d = x.shape[1]  # Feature dimension

    # Initialize pooled features with zeros (batch_size, num_rois, d)
    pooled_features = torch.zeros((batch_size, num_rois, d), dtype=x.dtype, device=x.device)

    for b in range(batch_size):  # Iterate through each sample in the batch
        for r in range(num_rois):  # Iterate through each ROI
            mask = (batch == b) & (roi_labels == r)  # Find nodes in the current (batch, ROI)
            
            if mask.any():  # If there are nodes in this ROI
                if pooling_type == 'mean':
                    pooled_features[b, r] = x[mask].mean(dim=0)
                elif pooling_type == 'max':
                    pooled_features[b, r] = x[mask].max(dim=0)[0]
                elif pooling_type == 'sum':
                    pooled_features[b, r] = x[mask].sum(dim=0)
                else:
                    raise ValueError("Unsupported pooling_type. Choose from 'mean', 'max', or 'sum'.")
    #print(f"pooled_features.shape: {pooled_features.shape}")
    return pooled_features.reshape(batch_size, -1)
    #return pooled_features



def roi_pooling_for_attn(x, roi_labels, batch, num_rois, pooling_type='mean'):
    """
    Perform ROI pooling ensuring that every sample has `num_rois` slots, even if some ROIs are missing.

    Args:
        x (torch.Tensor): Node embeddings of shape (n, d), where n is the number of nodes, and d is the feature dimension.
        roi_labels (torch.Tensor): ROI labels of shape (n,), indicating which ROI each node belongs to.
        batch (torch.Tensor): Batch indices of shape (n,), indicating which sample each node belongs to.
        num_rois (int): The total number of ROIs to ensure each sample has `num_rois * d` shape.
        pooling_type (str): Type of pooling ('mean', 'max', or 'sum').

    Returns:
        torch.Tensor: Aggregated ROI features of shape (batch_size, num_rois, d).
    """
    batch_size = batch.max().item() + 1  # Number of unique samples
    d = x.shape[1]  # Feature dimension

    # Initialize pooled features with zeros (batch_size, num_rois, d)
    pooled_features = torch.zeros((batch_size, num_rois, d), dtype=x.dtype, device=x.device)

    for b in range(batch_size):  # Iterate through each sample in the batch
        for r in range(num_rois):  # Iterate through each ROI
            mask = (batch == b) & (roi_labels == r)  # Find nodes in the current (batch, ROI)
            
            if mask.any():  # If there are nodes in this ROI
                if pooling_type == 'mean':
                    pooled_features[b, r] = x[mask].mean(dim=0)
                elif pooling_type == 'max':
                    pooled_features[b, r] = x[mask].max(dim=0)[0]
                elif pooling_type == 'sum':
                    pooled_features[b, r] = x[mask].sum(dim=0)
                else:
                    raise ValueError("Unsupported pooling_type. Choose from 'mean', 'max', or 'sum'.")
    #print(f"pooled_features.shape: {pooled_features.shape}")
    return pooled_features
    #return pooled_features


# def roi_pooling(x, roi_labels, batch, num_rois, pooling_type='mean'):
#     """
#     Vectorized ROI pooling using scatter operations.
#     """
#     batch_size = batch.max().item() + 1  # Number of unique samples
#     d = x.shape[1]
    
#     # Compute a unique index per (batch, ROI)
#     indices = (batch * num_rois + roi_labels).long()  # Ensure indices are int64

#     if pooling_type == 'mean':
#         # Use scatter_mean which returns (pooled, count)
#         pooled = torch_scatter.scatter_mean(x, indices, dim=0, dim_size=batch_size * num_rois)
#     elif pooling_type == 'sum':
#         pooled = torch_scatter.scatter_add(x, indices, dim=0, dim_size=batch_size * num_rois)
#     elif pooling_type == 'max':
#         # scatter_max returns a tuple (values, argmax)
#         pooled, _ = torch_scatter.scatter_max(x, indices, dim=0, dim_size=batch_size * num_rois)
#         # Replace initial -inf values with zeros if necessary
#         pooled[pooled == float('-inf')] = 0
#     else:
#         raise ValueError("Unsupported pooling_type. Choose from 'mean', 'max', or 'sum'.")

#     # Reshape the pooled result to (batch_size, num_rois, d)
#     pooled = pooled.reshape(batch_size, num_rois, d)
#     # Flatten the ROI dimension if needed: (batch_size, num_rois * d)
#     return pooled.reshape(batch_size, -1)

# Define the GCN Model
class ModelGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, in_channels_roi, hidden_channels_roi, num_layers, out_channels, dropout=0.5, pooling='roi', tree_embed_dim=32):
        super(ModelGCN, self).__init__()

        self.in_channels = in_channels
        self.in_channels_roi = in_channels_roi
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.embedding_dim = in_channels*148

        # self.roi_pooling = roi_pooling

        self.convs_roi = torch.nn.ModuleList()
        self.convs_roi.append(GCNConv(in_channels_roi, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs_roi.append(GCNConv(hidden_channels, hidden_channels))
        self.convs_roi.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.pooling = pooling
        self.mlp_roi_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels*148, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )
        self.mlp_global_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )

    def forward(self, data, data2):
        #x, edge_index, batch, nodeLabelIndex, subject_id, roi_feat, roi_edge_index, batch2= data.x1, data.edge_index, data.batch1, data.nodeLabelIndex, data.subject_id, data.x2, data.roi_edge_index, data.batch2
        #x, edge_index, batch, nodeLabelIndex, subject_id= data.x, data.edge_index, data.batch, data.nodeLabelIndex, data.subject_id
        #roi_feat, roi_edge_index, batch2, subject_id2= data2.x, data2.edge_index, data2.batch, data2.subject_id
        device = data.x.device
        x, edge_index, edge_attr, batch, nodeLabelIndex, subject_id = (
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
            data.batch.to(device),
            data.nodeLabelIndex.to(device),
            data.subject_id,
        )

        roi_feat, roi_edge_index, roi_edge_attr, batch2, subject_id2 = (
            data2.x.to(device),
            data2.edge_index.to(device),
            data2.edge_attr.to(device),
            data2.batch.to(device),
            data2.subject_id,
        )

        y = data.y
        subject_id = data.subject_id
        nodeLabelIndex = x[:,-1]
        x = x[:,:-1]

        roiLabelIndex = roi_feat[:,-1]
        x2 = roi_feat[:,:-1]

        #print(f"x2.shape: {x2}")
        for conv_roi in self.convs_roi[:-1]:
            x2 = conv_roi(x2, roi_edge_index, roi_edge_attr)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = roi_pooling(x2, roiLabelIndex, batch2, 148)
        embedding_roi = x2

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == 'global':
            x = global_mean_pool(x, batch)
            embedding = x
            x = self.mlp_global_pooling_classifier(x)
        elif self.pooling == 'roi':
            x = roi_pooling(x, nodeLabelIndex, batch, 148)
            embedding = x
            x = x + x2
            x = self.mlp_roi_pooling_classifier(x)
        embedding_sum = embedding + embedding_roi
        F.log_softmax(x, dim=1)
        output_dict = {"out": x, "embedding": embedding, "embedding_roi": embedding_roi, "embedding_sum": embedding_sum, "subject_id": subject_id}

        return output_dict

def is_one_hot(dist, max_threshold=0.9, other_threshold=0.01):
    # dist is a 1D tensor of attention weights for one query
    max_val = dist.max()
    # Zero out the maximum element and sum the rest
    others_sum = (dist - max_val).abs().sum()
    return (max_val > max_threshold) and (others_sum < other_threshold)


class SelfAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(SelfAttnBlock, self).__init__()
        # nn.MultiheadAttention expects input shape (seq_len, batch, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x is expected to have shape (batch, seq_len, embed_dim)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        # print(f"attn_out: {attn_out}")
        #print(f"attn_weights.shape: {attn_weights.shape}")
  
        # print(f"x: {x}")
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class ModelGCNAttn(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, in_channels_roi, hidden_channels_roi, num_layers, out_channels, dropout=0.5, pooling='roi', tree_embed_dim=32):
        super(ModelGCNAttn, self).__init__()

        self.in_channels = in_channels
        self.in_channels_roi = in_channels_roi
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.embedding_dim = in_channels*148

        self.roi_pooling = roi_pooling_for_attn

        self.convs_roi = torch.nn.ModuleList()
        self.convs_roi.append(GCNConv(in_channels_roi, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs_roi.append(GCNConv(hidden_channels, hidden_channels))
        self.convs_roi.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.pooling = pooling
        self.mlp_roi_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels*148, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )
        self.mlp_global_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )
        self.attn_embedding = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)
        self.attn_embedding_roi = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)
        self.attn_embedding_sum = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)

    def forward(self, data, data2):
        #x, edge_index, batch, nodeLabelIndex, subject_id, roi_feat, roi_edge_index, batch2= data.x1, data.edge_index, data.batch1, data.nodeLabelIndex, data.subject_id, data.x2, data.roi_edge_index, data.batch2
        #x, edge_index, batch, nodeLabelIndex, subject_id= data.x, data.edge_index, data.batch, data.nodeLabelIndex, data.subject_id
        #roi_feat, roi_edge_index, batch2, subject_id2= data2.x, data2.edge_index, data2.batch, data2.subject_id
        device = data.x.device
        x, edge_index, edge_attr, batch, nodeLabelIndex, subject_id = (
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
            data.batch.to(device),
            data.nodeLabelIndex.to(device),
            data.subject_id,
        )
        roi_feat, roi_edge_index, roi_edge_attr, batch2, subject_id2 = (
            data2.x.to(device),
            data2.edge_index.to(device),
            data2.edge_attr.to(device),
            data2.batch.to(device),
            data2.subject_id,
        )

        y = data.y
        subject_id = data.subject_id
        nodeLabelIndex = x[:,-1]
        x = x[:,:-1]
        batch_size = batch.max().item() + 1  # Number of unique samples
        roiLabelIndex = roi_feat[:,-1]
        x2 = roi_feat[:,:-1]

        #print(f"x2.shape: {x2}")
        for conv_roi in self.convs_roi[:-1]:
            x2 = conv_roi(x2, roi_edge_index, roi_edge_attr)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.roi_pooling(x2, roiLabelIndex, batch2, 148)
        embedding_roi = x2

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.pooling == 'global':
            x = global_mean_pool(x, batch)
            embedding = x
            x = self.mlp_global_pooling_classifier(x)
        elif self.pooling == 'roi':
            x = self.roi_pooling(x, nodeLabelIndex, batch, 148)
            embedding = x
            x = x 
            # x = self.mlp_roi_pooling_classifier(x)
        embedding_sum = x + x2

        #x = self.attn_embedding(x)
        #x2 = self.attn_embedding_roi(x2)


        transformer_in_roi = x2
        transformer_in_combined = x + transformer_in_roi
        # transformer_in_roi.retain_grad()
        # transformer_in_combined.retain_grad()
        # transformer_in_combined.requires_grad_() 
        transformer_out, attn_weights = self.attn_embedding_sum(transformer_in_combined)
        transformer_out.requires_grad_() 
        transformer_out.retain_grad()
        x = transformer_out.reshape(batch_size, -1)
        x = self.mlp_roi_pooling_classifier(x)
        # print(f"x.shape: {x.shape}")
        # print(f"x2.shape: {x2.shape}")
        # print(f"embedding.shape: {embedding.shape}")
        # print(f"embedding_roi.shape: {embedding_roi.shape}")
        # attn_emb_main = self.attn_embedding(embedding)
        # attn_emb_roi = self.attn_embedding_roi(embedding_roi)
        # attn_emb_sum = self.attn_embedding_sum(embedding_sum)

        F.log_softmax(x, dim=1)
        output_dict = {"out": x, 
                       "embedding": embedding, 
                       "embedding_roi": embedding_roi, 
                       "embedding_sum": embedding_sum, 
                       "subject_id": subject_id,
                       "transformer_in_roi": transformer_in_roi,
                       "transformer_in_combined": transformer_in_combined,
                       "transformer_out": transformer_out,
                       "attn_weights": attn_weights,}

        return output_dict



class ModelGCNAttn3h(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, in_channels_roi, hidden_channels_roi, num_layers, out_channels, dropout, pooling='roi', tree_embed_dim=32):
        super(ModelGCNAttn3h, self).__init__()

        self.in_channels = in_channels
        self.in_channels_roi = in_channels_roi
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.embedding_dim = in_channels*148

        self.roi_pooling = roi_pooling_for_attn

        self.convs_roi = torch.nn.ModuleList()
        self.convs_roi.append(GCNConv(in_channels_roi, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs_roi.append(GCNConv(hidden_channels, hidden_channels))
        self.convs_roi.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

        self.pooling = pooling
        self.mlp_roi_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels*148, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )
        self.mlp_global_pooling_classifier = nn.Sequential(
            nn.Linear(hidden_channels, 1000),
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(),
            nn.Linear(1000, out_channels),
        )
        self.attn_embedding = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)
        self.attn_embedding_roi = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)
        self.attn_embedding_sum = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)
        self.mha = SelfAttnBlock(embed_dim=hidden_channels, num_heads=4)

    def forward(self, data, data2):
        #x, edge_index, batch, nodeLabelIndex, subject_id, roi_feat, roi_edge_index, batch2= data.x1, data.edge_index, data.batch1, data.nodeLabelIndex, data.subject_id, data.x2, data.roi_edge_index, data.batch2
        #x, edge_index, batch, nodeLabelIndex, subject_id= data.x, data.edge_index, data.batch, data.nodeLabelIndex, data.subject_id
        #roi_feat, roi_edge_index, batch2, subject_id2= data2.x, data2.edge_index, data2.batch, data2.subject_id
        device = data.x.device
        x, edge_index, edge_attr, batch, nodeLabelIndex, subject_id = (
            data.x.to(device),
            data.edge_index.to(device),
            data.edge_attr.to(device),
            data.batch.to(device),
            data.nodeLabelIndex.to(device),
            data.subject_id,
        )

        roi_feat, roi_edge_index, roi_edge_attr, batch2, subject_id2 = (
            data2.x.to(device),
            data2.edge_index.to(device),
            data2.edge_attr.to(device),
            data2.batch.to(device),
            data2.subject_id,
        )

        y = data.y
        subject_id = data.subject_id

        nodeLabelIndex = x[:,-3]
        node_vertex_id = x[:,-2]
        node_lh_rh = x[:,-1]
        x = x[:,:-3]
        
        batch_size = batch.max().item() + 1  # Number of unique samples
        roiLabelIndex = roi_feat[:,-1]
        x2 = roi_feat[:,:-1]

        #print(f"x2.shape: {x2}")
        for conv_roi in self.convs_roi[:-1]:
            x2 = conv_roi(x2, roi_edge_index, roi_edge_attr)
            x2 = F.relu(x2)
            x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.roi_pooling(x2, roiLabelIndex, batch2, 148)
        embedding_roi = x2

        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for b in range(batch_size):
            pass

        embedding_3h = x

        all_sub_embeddings_lh_rh = []
        all_atten_weights_3h = []
        all_sub_embeddings_vertex_id = []
        updated_embeddings = torch.zeros_like(embedding_3h)
        updated_vertex_ids = torch.zeros_like(node_vertex_id)
        updated_lh_rh = torch.zeros_like(node_lh_rh)
        # Iterate over each graph.
        unique_graphs = sorted(torch.unique(batch))
        #print(f"unique_graphs: {unique_graphs}")
        for graph in unique_graphs:
            # Get the indices of nodes that belong to this graph.
            graph_indices = (batch == graph).nonzero(as_tuple=False).squeeze(1)
            sub_embeddings = embedding_3h[graph_indices]  # (num_nodes_in_graph, d)
            sub_embeddings_vertex_id = node_vertex_id[graph_indices]
            sub_embeddings_lh_rh = node_lh_rh[graph_indices]
            # If there's only one node, no attention is applied.
            if sub_embeddings.size(0) == 1:
                updated_embeddings[graph_indices] = sub_embeddings
                updated_vertex_ids[graph_indices] = sub_embeddings_vertex_id
                updated_lh_rh[graph_indices] = sub_embeddings_lh_rh
            else:
                # nn.MultiheadAttention expects input of shape (seq_len, batch_size, embed_dim)
                # Here, treat each graph's node embeddings as a sequence with batch_size=1.
                sub_seq = sub_embeddings.unsqueeze(1)  # (num_nodes_in_graph, 1, d)
                sub_seq = sub_seq.transpose(0, 1)         # (1, num_nodes_in_graph, d)
                
                # Apply multi-head self-attention.
                attn_output_3h, atten_weights_3h = self.mha(sub_seq)
                # Remove the extra batch dimension to get back to (num_nodes_in_graph, d)
                attn_output_3h = attn_output_3h.transpose(0, 1).squeeze(1)
                updated_embeddings[graph_indices] = attn_output_3h
                updated_vertex_ids[graph_indices] = sub_embeddings_vertex_id
                updated_lh_rh[graph_indices] = sub_embeddings_lh_rh
            all_atten_weights_3h.append(atten_weights_3h)
            #all_sub_embeddings_vertex_id.append(sub_embeddings_vertex_id.cpu().astype(np.int32))
            all_sub_embeddings_vertex_id.append(sub_embeddings_vertex_id.cpu().numpy().astype(np.int32))
            all_sub_embeddings_lh_rh.append(sub_embeddings_lh_rh.cpu().numpy())
        # print(f"updated_embeddings.shape: {updated_embeddings.shape}")
        # print(f"embedding_3h.shape: {embedding_3h.shape}")
        # print(f"attn_output_3h.shape: {attn_output_3h.shape}")
        # print(f"atten_weights_3h.shape: {atten_weights_3h.shape}")

        x = self.roi_pooling(embedding_3h, nodeLabelIndex, batch, 148)
        embedding = x
        x = x 

        # x = self.mlp_roi_pooling_classifier(x)
        embedding_sum = x + x2

        #x = self.attn_embedding(x)
        #x2 = self.attn_embedding_roi(x2)


        transformer_in_roi = x2
        transformer_in_combined = x + transformer_in_roi
        # transformer_in_roi.retain_grad()
        # transformer_in_combined.retain_grad()
        # transformer_in_combined.requires_grad_() 
        transformer_out, attn_weights = self.attn_embedding_sum(transformer_in_combined)
        transformer_out.requires_grad_() 
        transformer_out.retain_grad()
        x = transformer_out.reshape(batch_size, -1)
        x = self.mlp_roi_pooling_classifier(x)
        # print(f"x.shape: {x.shape}")
        # print(f"x2.shape: {x2.shape}")
        # print(f"embedding.shape: {embedding.shape}")
        # print(f"embedding_roi.shape: {embedding_roi.shape}")
        # attn_emb_main = self.attn_embedding(embedding)
        # attn_emb_roi = self.attn_embedding_roi(embedding_roi)
        # attn_emb_sum = self.attn_embedding_sum(embedding_sum)

        F.log_softmax(x, dim=1)
        output_dict = {"out": x, 
                       "embedding": embedding, 
                       "embedding_roi": embedding_roi, 
                       "embedding_sum": embedding_sum, 
                       "subject_id": subject_id,
                       "transformer_in_roi": transformer_in_roi,
                       "transformer_in_combined": transformer_in_combined,
                       "transformer_out": transformer_out,
                       "attn_weights": attn_weights,
                       "embedding_3h": embedding_3h,
                       "all_atten_weights_3h": all_atten_weights_3h,
                       "all_sub_embeddings_vertex_id": all_sub_embeddings_vertex_id,
                       "all_sub_embeddings_lh_rh": all_sub_embeddings_lh_rh,}

        return output_dict
