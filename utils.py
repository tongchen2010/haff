import os
import time
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from models.models_draft import GraphTransformerPYG, mask_node, mask_edge, ModelGCN
from models.center_models import CenterModel, CenterModel_zl

#from models.model_zy import MG, mask_node, mask_edge

import datetime
import pandas as pd 
from functools import partial
from torch.utils.data import random_split, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.utils import dense_to_sparse
from torch_geometric import seed_everything
import time
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset
from utils import *

# def collate_fn(batch):
#     """
#     Collate function for PyG Data objects.
#     """
#     batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

#     # Convert tensors to a batch format
#     batch_dict['feat'] = torch.stack(batch_dict['feat'])
#     batch_dict['adj'] = torch.stack(batch_dict['adj'])
#     batch_dict['label'] = torch.stack(batch_dict['label'])
#     batch_dict['nodeLabelIndex'] = torch.stack(batch_dict['nodeLabelIndex'])
#     batch_dict['subject_id'] = torch.stack(batch_dict['subject_id'])
#     batch_dict['roi_adj'] = torch.stack(batch_dict['roi_adj'])
#     batch_dict['roi_feat'] = torch.stack(batch_dict['roi_feat'])
#     return batch_dict


class CustomGraphDataset(Dataset):
    def __init__(self, adj_dir, feat_dir, roi_adj_dir, roi_feat_dir, label_dir, excel_dir, node_label_index_dict_dir, device='cuda'):
        self.adj_dir = adj_dir
        self.feat_dir = feat_dir
        self.roi_adj_dir = roi_adj_dir
        self.roi_feat_dir = roi_feat_dir
        self.label_dir = label_dir
        self.excel_dir = excel_dir
        self.node_label_index_dict_dir = node_label_index_dict_dir
        self.device = device
        self.all_adj = []
        
        self.graphs = []
        self.load_data()

    def load_data(self):
        label_df = pd.read_csv(self.label_dir)
        subject_ids = sorted(os.listdir(self.adj_dir))
        subject_ids = [subject_file.split('.')[0] for subject_file in subject_ids]
        min_value = 999
        max_value = 0
        for subject_id in subject_ids:
            adj_path = os.path.join(self.adj_dir, f"{subject_id}.txt")
            adj = np.loadtxt(adj_path)
            if adj.min() < min_value:
                min_value = adj.min()
            if adj.max() > max_value:
                max_value = adj.max()



        for subject_id in subject_ids:
            print(f"subject_id {subject_id}")
            adj_path = os.path.join(self.adj_dir, f"{subject_id}.txt")
            feat_path = os.path.join(self.feat_dir, f"{subject_id}.txt")
            roi_adj_path = os.path.join(self.roi_adj_dir, f"{subject_id}.txt")
            roi_feat_path = os.path.join(self.roi_feat_dir, f"{subject_id}.txt")

            excel_path = os.path.join(self.excel_dir, f"{subject_id}_combined.xlsx")

            label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}  # Grouping 3,4,5 into class 3

            adj = np.loadtxt(adj_path)
            adj = (adj - min_value) / (max_value - min_value)
            print(f"feat_path {feat_path}")
            print(f"excel_path {excel_path}")
            if "3h" in feat_path:
                feat = np.loadtxt(feat_path, delimiter=',')
            else:
                feat = np.loadtxt(feat_path)
            roi_adj = np.loadtxt(roi_adj_path)
            roi_feat = np.loadtxt(roi_feat_path)

            label = label_df[label_df["subject_id"] == subject_id]["diagnosis"].values[0]
            label = label_mapping[label]
            
            with open(self.node_label_index_dict_dir, 'r') as f:
                index_nodeLabel_dict = json.load(f)  # This loads the JSON data into a dictionary


            nodeLabel_index_dict = {v: k for k, v in index_nodeLabel_dict.items()}
            node_labels = pd.read_excel(excel_path)["center 3hinge label"].values
            node_labels = [x.replace(".label", "") for x in node_labels]
            node_labels = [x.replace(".", "_") for x in node_labels]
            node_labels = [x.replace("&", "_and_") for x in node_labels]
            #print(f"node_labels {len(node_labels)}")

            #node_label_index = [nodeLabel_index_dict[label] for label in node_labels]
            node_label_index = []
            for node_label in node_labels:
                if 'nknown' in node_label:
                    node_label_index.append('-1')
                else:
                    node_label_index.append(nodeLabel_index_dict[node_label])
            node_label_index = [int(x) for x in node_label_index]
            # print(len(node_label_index))
            # print(node_label_index)
            edge_index, edge_attr = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))
            roi_edge_index, roi_edge_attr = dense_to_sparse(torch.tensor(roi_adj, dtype=torch.float32))


            roi_label_index = [i for i in range(len(roi_feat))]
            roi_feat_with_roiLabelIndex = np.column_stack((roi_feat, roi_label_index))
            print(f"feat {feat.shape}")
            print(f"node_label_index {len(node_label_index)}")
            feat_with_nodeLabelIndex = np.column_stack((feat, node_label_index))

            num_nodes_x1 = feat_with_nodeLabelIndex.shape[0]
            num_nodes_x2 = roi_feat_with_roiLabelIndex.shape[0]

            data = Data(
                x1=torch.tensor(feat_with_nodeLabelIndex, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                nodeLabelIndex=torch.tensor(node_label_index, dtype=torch.long),
                subject_id=subject_id,
                roi_edge_index=roi_edge_index,
                x2=torch.tensor(roi_feat_with_roiLabelIndex, dtype=torch.float32),
                roi_edge_attr=roi_edge_attr,
                num_nodes_x1=num_nodes_x1,  # Explicitly set num_nodes
                num_nodes_x2=num_nodes_x2
            )
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


# class CustomGraphDataLoaderROI:
#     def __init__(self, dataset, batch_size=4, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle

#     def __iter__(self):
#         indices = list(range(len(self.dataset)))
#         if self.shuffle:
#             np.random.shuffle(indices)

#         for i in range(0, len(indices), self.batch_size):
#             batch_indices = indices[i:i + self.batch_size]
#             batch = [self.dataset[idx] for idx in batch_indices]
            
#             batched_graph = Batch.from_data_list(batch)


#             batched_graph.batch1 = torch.cat([
#                 torch.full((graph.x1.shape[0],), i, dtype=torch.long) for i, graph in enumerate(batch)
#             ])
            
#             batched_graph.batch2 = torch.cat([
#                 torch.full((graph.x2.shape[0],), i, dtype=torch.long) for i, graph in enumerate(batch)
#             ])
#             abc = 0
#             yield batched_graph, abc
#     def __len__(self):
#         return len(self.dataset) // self.batch_size

class CustomGraphDataLoaderROI:
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            # **Separate Main Graph and ROI Graph**
            batch_main = []
            batch_roi = []

            for data in batch:
                main_graph = Data(
                    x=data.x1,
                    edge_index=data.edge_index,
                    edge_attr=data.edge_attr,
                    y=data.y,
                    nodeLabelIndex=data.nodeLabelIndex,
                    subject_id=data.subject_id,
                )

                roi_graph = Data(
                    x=data.x2,
                    edge_index=data.roi_edge_index,
                    edge_attr=data.roi_edge_attr,
                    subject_id=data.subject_id,
                )

                batch_main.append(main_graph)
                batch_roi.append(roi_graph)

            # **Batch Both Graphs Separately**
            batched_main_graph = Batch.from_data_list(batch_main)
            batched_roi_graph = Batch.from_data_list(batch_roi)

            yield batched_main_graph, batched_roi_graph

    def __len__(self):
        return len(self.dataset) // self.batch_size


class CustomGraphDataLoader:
    def __init__(self, dataset, batch_size=4, shuffle=True):
        self.dataset = dataset
        print(f"Length of dataset: {len(dataset)}")

        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield Batch.from_data_list(batch)  # PyG batch support
    def __len__(self):
        return len(self.dataset) // self.batch_size


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = torch.optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = torch.optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = torch.optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = torch.optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        optimizer = torch.optim.SGD(parameters, **opt_args)
    else:
        raise ValueError("Invalid optimizer type specified.")
    return optimizer


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2):
    """
    Split a dataset into train, validation, and test subsets.
    """
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    val_len = int(val_ratio * total_len)
    test_len = total_len - train_len - val_len
    return random_split(dataset, [train_len, val_len, test_len])


def ordinal_embedding_loss(embeddings, labels, alpha=1.0):
    """
    Enforce that the pairwise distance between embeddings approximates
    alpha * |difference in ordinal labels|.

    Parameters:
      - embeddings: Tensor of shape [batch_size, d].
      - labels: Tensor of shape [batch_size] containing integer ordinal labels.
      - alpha: A scaling factor relating label differences to embedding distances.
      
    Returns:
      - A scalar tensor representing the ordinal embedding loss.
    """
    # Convert labels to float for computation.
    labels = labels.float()
    batch_size = embeddings.size(0)
    
    # Compute pairwise absolute differences in labels.
    label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))  # Shape: [batch_size, batch_size]
    
    # Compute pairwise Euclidean distances between embeddings.
    emb_dist = torch.cdist(embeddings, embeddings, p=2)  # Shape: [batch_size, batch_size]
    
    # Compute the loss as the mean absolute error between the embedding distances and alpha-scaled label differences.
    loss = torch.mean(torch.abs(emb_dist - alpha * label_diff))
    return loss

def calculate_combine_loss(output_dict, data_y, center_embedding, alpha=1.0):
    """
    Calculate the combined loss for the model.
    """

    cls_loss = nn.CrossEntropyLoss()(output_dict['out'], data_y)
    center_embedding_clone = center_embedding.clone().detach()
    embedding_tensor = output_dict['tree_embed']  # Shape: [batch_size, d]
    ord_emb_loss = ordinal_embedding_loss(embedding_tensor, data_y, alpha=alpha)
    
    loss1 = tree_dce_loss(embedding_tensor, data_y, center_embedding_clone, 1.0)
    loss2 = tree_pl_loss(embedding_tensor, data_y, center_embedding_clone)
    # Combine the losses.
    # loss = 0.9 * cls_loss + 0.0001 * ord_emb_loss + 0.1*loss1 + 0.1*loss2
    # loss = 0.9 * cls_loss + 0.1*loss1 + 0.1*loss2
    loss = loss1 + 0.1*loss2
    return loss

def tree_acc(features, labels, centers):
    dist = tree_distance(features, centers)
    prediction = torch.min(dist, dim=1)[1]
    correct = torch.eq(prediction.int(), labels)
    acc = torch.sum(correct.float())

    return prediction, labels, acc

def tree_softmax_loss(logits, labels):
    # Ensure logits are of type FloatTensor
    logits = logits.float()

    # Ensure labels are of type LongTensor
    labels = labels.long()

    # Cross-entropy loss
    criterion = torch.nn.CrossEntropyLoss()
    mean_cross_entropy = criterion(logits, labels)

    return mean_cross_entropy

def tree_pl_loss(features, labels, centers):
    # pdb.set_trace()
    # print(f'features.shape {features.shape}')
    # print(f'labels.shape {labels.shape}')
    # print(f'centers.shape {centers.shape}')
    # print(f'labels.shape {labels.shape}')

    batch_num = float(features.shape[0])
    labels = labels.type(torch.LongTensor).to(labels.device)
    #batch_centers = centers[labels]
    batch_centers = torch.index_select(centers, 0, labels)
    dis = features - batch_centers
    # print(f'dis.shape {dis.shape}')
    # abcd = dis.pow(2)
    # print(f'abcd.shape {abcd.shape}')
    return torch.div(torch.sum(dis.pow(2)) * 0.5, batch_num)

def tree_distance(features, centers):
    # print(f'features {features.shape}')
    f_2 = torch.sum(torch.pow(features, 2), dim=1, keepdim=True)
    c_2 = torch.sum(torch.pow(centers, 2), dim=1, keepdim=True)
    # print(f'features {features.shape}')
    # print(f'centers {centers.shape}')
    dist = f_2 - 2 * torch.matmul(features, torch.transpose(centers, 1, 0)) + torch.transpose(c_2, 1, 0)
    return dist

def tree_dce_loss(features, labels, centers, T):
    #pdb.set_trace()
    dist = tree_distance(features, centers)
    logits = -dist / T
    #logits = -torch.log(dist / T)
    mean_loss = tree_softmax_loss(logits, labels)
    # print(features.shape)
    # print(labels.shape)
    # print(centers.shape)
    # print(dist.shape)
    # print(dist[0])
    # print(logits[0])

    return mean_loss


def save_model(model, path):
    """
    Save the model state dictionary.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path, device):
    """
    Load model state dictionary from a file.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {path}")
    return model

# Setup Logging
def setup_logging(log_dir, filename="experiment_logs.txt"):
    log_path = os.path.join(log_dir, filename)
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logging.info("Logging initialized.")

def log_experiment_details(args, train_acc, val_acc, test_acc):
    logging.info(f"Experiment Run:")
    logging.info(f"Learning Rate: {args.lr}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Epochs: {args.epoch}")
    logging.info(f"Train Accuracy: {train_acc:.4f}")
    logging.info(f"Validation Accuracy: {val_acc:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.4f}")
    logging.info("-" * 50)