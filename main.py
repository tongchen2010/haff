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
from models.models_draft import GraphTransformerPYG, mask_node, mask_edge, ModelGCN, ModelGCNAttn, ModelGCNAttn3h
from models.center_models import CenterModel, CenterModel_zl
import matplotlib.pyplot as plt

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
import datetime

class CustomGraphDataset3hID(Dataset):
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
            adj_path = os.path.join(self.adj_dir, f"{subject_id}.txt")
            feat_path = os.path.join(self.feat_dir, f"{subject_id}.txt")
            roi_adj_path = os.path.join(self.roi_adj_dir, f"{subject_id}.txt")
            roi_feat_path = os.path.join(self.roi_feat_dir, f"{subject_id}.txt")

            excel_path = os.path.join(self.excel_dir, f"{subject_id}_combined.xlsx")

            label_mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}  # Grouping 3,4,5 into class 3

            adj = np.loadtxt(adj_path)
            adj = (adj - min_value) / (max_value - min_value)
            if "data_cambridge" in feat_path:
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
            excel_df = pd.read_excel(excel_path)
            node_labels = excel_df["center 3hinge label"].values
            node_vertex_id = excel_df["center 3hinge ID"].values
            node_lh_rh = excel_df["hemi"].values
            node_lh_rh_binary = [0 if val == "lh" else 1 for val in node_lh_rh]

            #node_lh_rh
            # print(f"node_lh_rh: {node_lh_rh.shape}")
            # print(f"node_vertex_id: {node_vertex_id.shape}")
            node_labels = [x.replace(".label", "") for x in node_labels]
            node_labels = [x.replace(".", "_") for x in node_labels]
            node_labels = [x.replace("&", "_and_") for x in node_labels]
            
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

            feat_with_nodeLabelIndex = np.column_stack((feat, node_label_index,node_vertex_id, node_lh_rh_binary))
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

def evaluate_model(model, prototype_model, loader, fold_num, data_split, output_dir, device, alpha=1.0, timestamp=None, epoch=None):
    """
    Evaluate the model on a given dataset loader.
    Returns average loss and accuracy.
    """
    model.eval()
    prototype_model.eval()
    # center_predicts, center_embedding, center_labels, order_matrix = prototype_model()

    total_loss = 0
    correct = 0
    total = 0
    subject_id_list = []
    all_embedding= []

    predictions_list = []
    labels_list = []
    all_subject_importance = []
    all_all_sub_embeddings_vertex_id = []
    all_all_sub_embeddings_lh_rh = []
    with torch.no_grad():
        for batch, roi_batch in loader:
            # subject_id_list += batch.subject_id
            batch = batch.to(device)
            output_dict = model(batch, roi_batch)
            out = output_dict['out']
            all_atten_weights_3h = output_dict['all_atten_weights_3h']
            all_sub_embeddings_vertex_id = output_dict['all_sub_embeddings_vertex_id']
            all_sub_embeddings_lh_rh = output_dict['all_sub_embeddings_lh_rh']

            # print(f"all_sub_embeddings_vertex_id: {len(all_sub_embeddings_vertex_id)}")
            # print(f"attn_weights shape: {attn_weights.shape}")
            # print(f"all_atten_weights_3h: {len(all_atten_weights_3h)}")
            # print(f"all_sub_embeddings_vertex_id: {all_sub_embeddings_vertex_id[0].shape}")
            # print(f"all_atten_weights_3h: {all_atten_weights_3h[0].shape}")
            cls_attn = []
            cls_attn_avg = []
            for i in all_atten_weights_3h:
                temp_i = i[:, :, 0, :]
                temp_i = temp_i.mean(dim=1)
                temp_i = temp_i.squeeze(0)
                cls_attn_avg.append(temp_i)
            importance_rankings = []
            batch_size = len(batch)

            for i in range(batch_size):
                scores = cls_attn_avg[i]  # shape: (seq_len,)
                sorted_indices = torch.argsort(scores, descending=True)
                importance_rankings.append(sorted_indices.tolist())
            all_subject_importance += importance_rankings
            all_all_sub_embeddings_vertex_id += all_sub_embeddings_vertex_id
            all_all_sub_embeddings_lh_rh += all_sub_embeddings_lh_rh

            embedding_tensor = output_dict['embedding_sum']
            embedding_np = embedding_tensor.cpu().numpy()
            # print(f"embedding_np shape: {embedding_np.shape}")
            embedding_np = embedding_np.reshape(len(batch), -1)
            #print(f"embedding_np shape: {embedding_np.shape}")
            # embedding_sum = output_dict['embedding_sum']
            # embedding = output_dict['embedding']
            # embedding_roi = output_dict['embedding_roi']
            


            all_embedding.append(embedding_np)
            subject_id = output_dict['subject_id']
            subject_id_list += subject_id


            # loss = nn.CrossEntropyLoss(out, batch.y)
            loss = nn.CrossEntropyLoss()(output_dict['out'], batch.y)
            # loss = calculate_combine_loss(output_dict, batch.y, center_embedding, alpha=alpha)
            total_loss += loss.item()
            pred = out.argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs

            predictions_list.extend(pred.cpu().numpy().tolist())
            labels_list.extend(batch.y.cpu().numpy().tolist())
    # Stack all collected embeddings vertically
    if all_embedding:
        all_embedding = np.vstack(all_embedding)
    else:
        all_embedding = np.array([])

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    acc = correct / total if total > 0 else 0

    output_embedding_path = os.path.join(output_dir, f"{data_split}_results", f"embedding_fold_{str(fold_num+1)}.txt")
    output_subject_id_path = os.path.join(output_dir, f"{data_split}_results", f"subject_id_fold_{str(fold_num+1)}.txt")
    output_pred_path = os.path.join(output_dir, f"{data_split}_results", f"predictions_fold_{str(fold_num+1)}.csv")

    if epoch is not None:
        output_embedding_path = os.path.join(output_dir, f"{data_split}_results", f"epoch_{epoch}_embedding_fold_{str(fold_num+1)}.txt")
        output_subject_id_path = os.path.join(output_dir, f"{data_split}_results", f"epoch_{epoch}_subject_id_fold_{str(fold_num+1)}.txt")
        output_pred_path = os.path.join(output_dir, f"{data_split}_results", f"epoch_{epoch}_predictions_fold_{str(fold_num+1)}.csv")

    output_importance_dir = os.path.join(output_dir, "importance_3h")
    os.makedirs(output_importance_dir, exist_ok=True)
    output_importance_path = os.path.join(output_importance_dir, f"{data_split}_importance_rankings_fold_{str(fold_num+1)}.csv")
    df_importance_rankings = pd.DataFrame({
        'subject_id': subject_id_list,
        'importance_ranking_3h': all_subject_importance,
        'all_all_sub_embeddings_vertex_id': all_all_sub_embeddings_vertex_id,
        'all_all_sub_embeddings_lh_rh': all_all_sub_embeddings_lh_rh,
    })


    np.savetxt(output_embedding_path, all_embedding, fmt="%.6f")
    np.savetxt(output_subject_id_path, subject_id_list, fmt="%s")
    df = pd.DataFrame({
        'prediction': predictions_list,
        'ground_truth': labels_list
    })
    df.to_csv(output_pred_path, index=False)
    df_importance_rankings.to_csv(output_importance_path, index=False)

    return avg_loss, acc



def save_training_results(epochs, fold, output_dir, train_losses, train_accuracies, val_accuracies):
    # Plot training loss and accuracy
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training Loss (Fold {fold+1})")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Training Accuracy (Fold {fold+1})")
    plt.legend()


    # Plot accuracy
    plt.subplot(2, 2, 3)
    plt.plot(range(1, epochs + 1), val_accuracies, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"Validation Accuracy (Fold {fold+1})")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_plots_fold_{fold}.png"))
    plt.close()
# ------------------------------
# Main function
# ------------------------------

def main(args):
    seed_everything(args.seed_num)
    args.feat_type = "3h"
    args.adj_type = "sc"
    model_dir = f"models_feat_{args.feat_type}_featVer_{args.featVer}_adj_{args.adj_type}_fr_{args.featRing}_ar_{args.adjRing}_bs_{args.batch_size}_lr_{args.lr}_epoch_{args.epoch}_ed_{args.embed_dim}_pooling_{args.pooling}_do_{args.dropout_rate}_seed_{args.seed_num}"
    model_path = os.path.join(args.result_dir, model_dir)
    logging.info(f"Starting Experiment with lr={args.lr}, batch_size={args.batch_size}")
    
    pth_dir = "checkpoints"
    run_dir = "runs"
    output_test_result_dir = os.path.join(args.result_dir, model_dir, "test_results")
    output_train_result_dir = os.path.join(args.result_dir, model_dir, "train_results")
    output_val_result_dir = os.path.join(args.result_dir, model_dir, "val_results")
    output_figure_dir = os.path.join(args.result_dir, model_dir, "figures")
    log_dir = os.path.join(args.result_dir, model_dir)
    
    
    setup_logging(log_dir)
    os.makedirs(output_test_result_dir, exist_ok=True)

    model_checkpoint_dir = os.path.join(args.result_dir, model_dir, pth_dir)
    os.makedirs(model_checkpoint_dir, exist_ok=True)
    os.makedirs(output_train_result_dir, exist_ok=True)
    os.makedirs(output_val_result_dir, exist_ok=True)
    os.makedirs(output_figure_dir, exist_ok=True)


    # Directories for adjacency and feature matrices (adjust paths as needed)
    # adj_dir = '/media/yanzhuang/8T_ZY/Graph_transformer/Dataset/adj_matrix_3ring'
    # feat_dir = '/media/yanzhuang/8T_ZY/Graph_transformer/Dataset/feat_matrix'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    


    # Create the dataset; mask functions are optional (set to None to disable)
    dataset = CustomGraphDataset3hID(
        adj_dir=args.adj_dir,
        feat_dir=args.feat_dir,
        roi_adj_dir=args.roi_adj_dir, 
        roi_feat_dir=args.roi_feat_dir, 
        label_dir=args.label_dir,
        excel_dir=args.excel_dir,
        node_label_index_dict_dir=args.node_label_index_dict_dir,
        device=device
    )

    labels = np.array([data.y.item() for data in dataset])
    
    # Set up 5-fold cross validation with stratification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed_num)
    fold_accuracies = []

    csv_file_path = f'./experiment_results_2.csv'
    csv_header = ['test_acc', 
                  'fold', 
                  'feat_type', 
                  'feat_version',
                  'adj_type', 
                  'featRing', 
                  'adjRing',
                  'optimizer', 
                  'batch_size', 
                  'model_lr',
                  'epoch', 
                  'dropout_rate',
                  'embed_dim', 
                  'train_loss', 
                  'train_acc', 
                  'train_center_loss', 
                  'train_center_acc', 
                  'timestamp', 
                  'seed']
    # Create the CSV file if it doesn't exist
    if not os.path.isfile(csv_file_path):
        with open(csv_file_path, mode='w', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow(csv_header)


    for fold, (train_val_idx, test_idx) in enumerate(skf.split(np.arange(len(dataset)), labels)):
        print(f"\n--- Fold {fold+1} / 5 ---")
        logging.info(f"Starting fold {fold+1}")

        train_val_labels = labels[train_val_idx]
        # train_idx, val_idx = train_test_split(
        #     train_val_idx, 
        #     test_size=0.2, 
        #     stratify=train_val_labels, 
        #     random_state=args.seed_num
        # )
        train_idx = train_val_idx
        val_idx = test_idx


        print(f"train length: {len(train_idx)}")
        print(f"val length: {len(val_idx)}")
        print(f"test length: {len(test_idx)}")
        # Create subsets for training, validation, and test
        train_dataset = Subset(dataset, train_idx)
        val_dataset   = Subset(dataset, val_idx)
        test_dataset  = Subset(dataset, test_idx)
        
        # Use custom dataloader
        train_loader = CustomGraphDataLoaderROI(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = CustomGraphDataLoaderROI(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = CustomGraphDataLoaderROI(test_dataset, batch_size=args.batch_size, shuffle=False)

        sample_data = dataset[0]
        in_channels = sample_data.x1.size(1) - 3
        hidden_channels = args.embed_dim
        out_channels = args.group_num   
        in_channels_roi = sample_data.x2.size(1) -1
        hidden_channels_roi = args.embed_dim

        model = ModelGCNAttn3h(in_channels=in_channels, 
                                hidden_channels=hidden_channels, 
                                in_channels_roi=in_channels_roi, 
                                hidden_channels_roi=hidden_channels_roi, 
                                num_layers=2, 
                                out_channels=out_channels,
                                dropout=args.dropout_rate,
                                pooling=args.pooling).to(device)

        prototype_model = CenterModel_zl(device=device, 
                                        dimension=args.embed_dim*148, 
                                        layer_dims=[args.embed_dim*148,args.embed_dim*148], 
                                        num_classes=4, 
                                        num_prototype=1)

        # prototype_optimizer = create_prototype_optimizer(prototype_model, args)
        # prototype_scheduler = create_prototype_scheduler(prototype_optimizer, args)

        fold_model_path = os.path.join(model_checkpoint_dir, f"model_fold_{str(fold+1)}.pth")
        model = load_model(model, fold_model_path, device)
        model.eval()
        fold_prototype_model_path = os.path.join(model_checkpoint_dir, f"prototype_model_fold_{str(fold+1)}.pth")
 
        # # Evaluate on the test set.
        test_loss, test_acc = evaluate_model(model, prototype_model, test_loader, fold_num=fold, data_split='test', output_dir=model_path,device=device)
        print(f"Fold {fold+1} - Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        logging.info(f"Fold {fold+1} - Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        fold_accuracies.append(test_acc)

        train_loss, train_acc = evaluate_model(model, prototype_model, train_loader, fold_num=fold, data_split='train', output_dir=model_path,device=device)
        val_loss, val_acc = evaluate_model(model, prototype_model, val_loader, fold_num=fold, data_split='val', output_dir=model_path,device=device)

        log_experiment_details(args, train_acc, val_acc, test_acc)

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        train_center_loss = 0
        train_center_acc = 0
        with open(csv_file_path, mode='a', newline='') as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([
                test_acc,             # test accuracy for this fold
                fold + 1,             # current fold (1-indexed)
                args.feat_type,       # feature type
                args.featVer,         # feature version
                args.adj_type,        # adjacency type
                args.featRing,        # featRing value
                args.adjRing,         # adjRing value
                args.optimizer,       # optimizer type
                args.batch_size,      # batch size
                args.lr,              # learning rate
                args.epoch,           # number of epochs
                args.dropout_rate,    # dropout rate
                args.embed_dim,       # embedding dimension
                train_loss,           # training loss
                train_acc,            # training accuracy
                train_center_loss,    # training center loss (if any)
                train_center_acc,     # training center accuracy (if any)
                current_time,            # timestamp
                args.seed_num         # random seed used
            ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Transformer for 3-hinge")
    # (You can add additional command line arguments here)
    
    parser.add_argument('-feat_dir', '--feat_dir', type=str, help='')
    parser.add_argument('-adj_dir', '--adj_dir', type=str, help='')
    
    parser.add_argument('-root_data_dir', '--root_data_dir', type=str, help='')

    parser.add_argument('-roi_feat_dir', '--roi_feat_dir', type=str, help='')
    parser.add_argument('-roi_adj_dir', '--roi_adj_dir', type=str, help='')    
    
    parser.add_argument('-label_dir', '--label_dir', type=str, help='')
    parser.add_argument('-excel_dir', '--excel_dir', type=str, help='P')
    parser.add_argument('-node_label_index_dict_dir', help='')
    
    parser.add_argument('-result_dir', '--result_dir', type=str, help='')
    
    parser.add_argument('-featVer', '--featVer', type=str, help='')
    parser.add_argument('-feat_type', '--feat_type', type=str, help='')
    parser.add_argument('-adj_type', '--adj_type', type=str, help='')
    parser.add_argument('-featRing', '--featRing', help='')
    parser.add_argument('-adjRing', '--adjRing', help='')
    parser.add_argument('-pooling', '--pooling', help='')


    parser.add_argument('-batch_size', '--batch_size', type=int, default=16, help='')
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='')
    parser.add_argument('-epoch', '--epoch', type=int, default=1, help='')
    parser.add_argument('-dropout_rate', '--dropout_rate', type=float, default=0.2, help='')

    parser.add_argument('-exp_num', '--exp_num', type=int, default=1, help='')
    parser.add_argument('-seed_num', '--seed_num', type=int, default=12345, help='')

    parser.add_argument('-weight_decay', '--weight_decay', type=float, default=0.0001, help='')
    parser.add_argument('-optimizer', '--optimizer', type=str, default="adamw", help='')
    parser.add_argument('-model', '--model', type=str, default="graph_transformer", help='')
    parser.add_argument('-embed_dim', '--embed_dim', type=int, default=16, help='')
    parser.add_argument('-group_num', '--group_num', type=int, default=4, help='')



