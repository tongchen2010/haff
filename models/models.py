import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

class GraphTransformerPYG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1, beta=None):
        """
        A graph classification model using TransformerConv layers.
        
        Args:
            in_channels (int): Dimension of input node features.
            hidden_channels (int): Dimension of hidden representations.
            out_channels (int): Number of output classes.
            heads (int, optional): Number of attention heads. Default is 4.
            dropout (float, optional): Dropout probability applied to attention coefficients.
            beta (float or None, optional): If provided, enables skip fusion with weight beta.
                (See PyG docs for TransformerConv for details.) Default is None.
        """
        super().__init__()
        # First TransformerConv layer: it projects the input features into a hidden space.
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                       dropout=dropout, beta=beta)
        # Second TransformerConv layer: note that if concatenation is enabled (default),
        # the output dimension becomes heads * hidden_channels.
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, 
                                       dropout=dropout, beta=beta)
        # Global pooling (mean) to aggregate node features into a graph-level representation.
        self.pool = global_mean_pool
        # Final classifier: maps the pooled representation to output classes.
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Apply the first TransformerConv layer and a non-linear activation.
        x = F.elu(self.conv1(x, edge_index))
        # Apply the second TransformerConv layer.
        x = F.elu(self.conv2(x, edge_index))
        # Aggregate node features to obtain a graph-level embedding.
        x = self.pool(x, batch)
        # Classify the graph.
        x = self.lin(x)
        return F.log_softmax(x, dim=-1)


def mask_node(x, mask_rate=0.2, noise=0.05):
    """
    Randomly mask rows (node features) in the feature matrix.

    Args:
        x (torch.Tensor): Node feature matrix of shape (num_nodes, num_features).
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

    # Select nodes to mask
    mask_nodes = perm[: num_mask_nodes]
    
    # Determin noisy nodes and token nodes
    num_noise_nodes = int(noise * num_mask_nodes)
    perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    token_nodes = mask_nodes[perm_mask[:num_mask_nodes - num_noise_nodes]]
    noise_nodes = mask_nodes[perm_mask[num_mask_nodes - num_noise_nodes:]]

    # Generate random noise
    noise_value = torch.randn(num_noise_nodes, num_features, device=x.device)

    # Clone the feature matrix to apply masking
    masked_x = x.clone()

    # Replace masked nodes with noise
    masked_x[token_nodes] = 0.0
    masked_x[noise_nodes] = noise_value

    return masked_x

def mask_edge(adj_matrix, mask_rate=0.2, noise=0.05):
    
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Remove self-loops
    adj_matrix = adj_matrix.clone()
    adj_matrix.fill_diagonal_(0)
    
    # Mask edges (drop edges)
    edge_indices = torch.nonzero(adj_matrix, as_tuple=False)
    num_edges = edge_indices.size(0)
    num_edges_to_drop = int(num_edges * mask_rate)
    drop_indices = torch.randperm(num_edges)[:num_edges_to_drop]
    adj_matrix[edge_indices[drop_indices, 0], edge_indices[drop_indices, 1]] = 0
    
    # Add noise (random edges)
    num_nodes = adj_matrix.size(0)
    num_edges_to_add = int(num_edges * noise)
    for _ in range(num_edges_to_add):
        i, j = torch.randint(0, num_nodes, (2,))
        if i != j:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1  # For undirected graph
    
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

        # Positional encoding for graph structure
        # self.positional_encoding = nn.Parameter(torch.randn(1, out_hidden))

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=out_hidden,
                nhead=num_heads,
                dim_feedforward=out_hidden*4,
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

        deg = adj_matrix.sum(dim=1).pow(-0.5)
        deg[deg == float('inf')] = 0
        norm_adj = deg.unsqueeze(1) * adj_matrix * deg.unsqueeze(0) # normalize

        pos_enc = torch.matmul(norm_adj, feat)
        pos_enc = self.pos_proj(pos_enc)

        h = h + pos_enc

        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            h = transformer_layer(h)

        # Apply output projection
        h = self.output_proj(h)

        return h
    

class TreeModel(nn.Module):
    def __init__(self, in_hidden, out_hidden, p1, p2, p3, beta, beta1, alpha, adj_dim, feat_dim):
        super(MG, self).__init__()
        # self.enc = Encoder(in_hidden, out_hidden, p1)
        self.gtrans = GraphTransformer(in_hidden, out_hidden, 8, p1, 2)
        self.dec1 = Decoder1(out_hidden, in_hidden, p2, adj_dim)
        self.dec2 = Decoder2(out_hidden, in_hidden, p3, feat_dim)
        # self.dec1 = TransformerDecoder(out_hidden, in_hidden, 8, p2, 2)
        # self.dec2 = TransformerDecoder2(out_hidden, in_hidden, 8, p3, 2)
        # self.rate_node = rate_node
        # self.rate_edge = rate_edge
        self.alpha = alpha
        # self.noise_node = noise_node
        # self.noise_edge = noise_edge
        # self.enc_mask_token = nn.Parameter(torch.zeros(1, in_hidden))
        # self.enc_mask_token1 = nn.Parameter(torch.zeros(1, in_hidden))

        self.criterion = self.setup_loss_fn("sce", beta)
        self.criterion1 = self.setup_loss_fn("mse", beta1)

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, masked_adj, masked_feat, label_adj, label_feat):

        h1 = self.gtrans(masked_adj, masked_feat)
        # print(f"h1.shape: {h1.shape}")

        # print(f"h1.shape: {h1.shape}, feat.shape: {feat.shape}")

        h2 = self.dec1(h1)
        h3 = self.dec2(h1)

        loss1 = self.criterion1(h2, label_adj)
        loss2 = self.criterion1(h3, label_feat)

        return self.alpha * loss1 + loss2 * (1 - self.alpha), h3

    def get_embed(self, masked_adj, masked_feat):
        # h1 = self.enc(graph, feat)
        h1 = self.gtrans(masked_adj, masked_feat)

        return h1


