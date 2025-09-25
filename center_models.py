import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np


class CenterModel_zl(nn.Module):
    def __init__(self, device, layer_dims, dimension, num_classes=4, num_prototype=1, dropout_rate=0.1, tree_embed_dim=32):
        super(CenterModel_zl, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.dimension = dimension

        # Initialize centers with requires_grad=True to make them trainable
        self.centers = self.center_initialization().to(self.device)
        self.label = self.label_generator()
        self.order_matrix = self.similarity_matrix_generator()

        # Build layers of the model
        dims = [dimension] + layer_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=dims[i + 1], affine=False)
            ))
        self.layers = nn.ModuleList(layers)
        self.class_layer = nn.Linear(dims[-1], num_classes)

    def center_initialization(self):
        center_list = []
        for class_num in range(self.num_classes):
            for prototype_num in range(self.num_prototype):
                center_list.append([0.2 * (class_num + 1)] * self.dimension)

        #np.random.seed(42)
        #center_list = np.random.uniform(-1, 1, (3, 20))
        #print(center_list)
        centers = torch.FloatTensor(center_list)
        centers = (centers - 0.5) / 1.0
        centers = Parameter(centers, requires_grad=True)  # Now centers are trainable
        return centers

    # def center_initialization(self):
    #     center_list = list()
    #     for class_num in range(self.num_classes):
    #         for prototype_num in range(self.num_prototype):
    #             center_list.append([0.2 * (class_num + 1)] * self.dimension)
    #     #print(center_list)
    #     centers = torch.FloatTensor(center_list)
    #     centers = (centers-0.5) / 1.0
    #     centers = Parameter(centers, requires_grad=False)
    #     #print(centers)
    #     return centers

    def label_generator(self):
        label = []
        for i in range(self.num_classes):
            label += [i] * self.num_prototype
        label = torch.LongTensor(label)
        return label

    # def similarity_matrix_generator(self):
    #     n = self.label.shape[0]
    #     distance_similarities = {
    #         0: 1.0,  # Same label
    #         1: 3.0,  # Labels differ by 1
    #         2: 2.0,  # Labels differ by 2
    #         3: 1.0  # Labels differ by 3
    #     }
    #
    #     label_similarity = torch.zeros(n, n)
    #     for i in range(n):
    #         for j in range(n):
    #             distance = math.fabs(self.label[i] - self.label[j])
    #             similarity = distance_similarities.get(distance, 0.0)  # Default similarity if not found
    #             label_similarity[i, j] = similarity
    #
    #     #print(label_similarity)
    #     row_sums = label_similarity.sum(axis=1)
    #     D = torch.diag(row_sums)
    #     # print(f'D {D}')
    #     # print(f'label_similarity {label_similarity}')
    #     #order_matrix = D + label_similarity-2
    #     order_matrix = D - label_similarity
    #     return order_matrix.float()

    def similarity_matrix_generator(self):
        n = self.label.shape[0]
        label_similarity = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if self.label[i] == self.label[j]:
                    label_similarity[i, j] = 1.0
                elif math.fabs(self.label[i] - self.label[j]) == 1:
                    label_similarity[i, j] = 1.0
                elif math.fabs(self.label[i] - self.label[j]) == 2:
                    label_similarity[i, j] = 0.6
                elif math.fabs(self.label[i] - self.label[j]) == 3:
                    label_similarity[i, j] = 0.0
                else:
                    label_similarity[i, j] = 0.0

                # label_similarity[i,j] = math.exp( - weight * (ytr[i] - ytr[j]) ** 2)
        #print(f'label_similarity {label_similarity}')
        label_similarity = [[1.0,0.8,0.4,0.01],
                            [0.4,1.0,0.4,0.01],
                            [0.4,0.8,1.0,0.01],
                            [0.01,0.01,0.01,1.0]]
        label_similarity = np.array(label_similarity, dtype=np.float32)
        label_similarity = torch.from_numpy(label_similarity)
        #print(f'label_similarity {label_similarity}')
        order_matrix = torch.diag(label_similarity.sum(axis=1)) - label_similarity
        order_matrix = order_matrix.float()


        return order_matrix

    def forward(self):
        x = self.centers
        # print(f'initial center {x}')
        x = x.to(self.device)
        for layer in self.layers:
            x = layer(x)
            # print(f'layer_x {x}')
        learnable_centers = x
        # print(f'learnable_centers {learnable_centers}')
        out_class = self.class_layer(x)


        # Ensure label and order_matrix are on the correct device
        self.label = self.label.to(self.device)
        self.order_matrix = self.order_matrix.to(self.device)
        #print(f'order_matrix {self.order_matrix}')
        #print(f'centers {self.centers}')
        return out_class, learnable_centers, self.label, self.order_matrix

class CenterModel_zl_2(nn.Module):
    def __init__(self, device, layer_dims, dimension, num_classes=4, num_prototype=1, dropout_rate=0.5):
        """
        Args:
            device: torch.device
            layer_dims: list of hidden layer dimensions (for the feature extractor)
            dimension: dimension of the input features
            num_classes: number of classes (here 4)
            num_prototype: number of prototypes per class (default=1)
            dropout_rate: dropout rate for regularization
        """
        super(CenterModel_zl_2, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.dimension = dimension
        self.dropout_rate = dropout_rate

        # Initialize centers (trainable prototypes)
        self.centers = self.initialize_centers().to(self.device)
        self.labels = self.generate_labels()
        self.order_matrix = self.generate_similarity_matrix()  # updated for 4 classes

        # Build a feature extractor network with dropout and batch normalization (affine=True)
        dims = [dimension] + layer_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=dims[i + 1], affine=True),
                nn.Dropout(self.dropout_rate)
            ))
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Linear(dims[-1], num_classes)
        self._initialize_weights()

    def initialize_centers(self):
        """
        Initialize centers for each class-prototype pair.
        Here we use a simple constant initialization.
        """
        center_list = []
        for class_num in range(self.num_classes):
            for _ in range(self.num_prototype):
                center_list.append([0.2 * (class_num + 1)] * self.dimension)
        centers = torch.tensor(center_list, dtype=torch.float32)
        centers = centers - 0.5  # optionally normalize or shift
        return Parameter(centers, requires_grad=True)

    def generate_labels(self):
        """
        Generate a tensor of labels corresponding to each center.
        For example, if num_prototype=1 then for 4 classes, labels=[0,1,2,3]
        """
        label_list = []
        for i in range(self.num_classes):
            label_list.extend([i] * self.num_prototype)
        return torch.tensor(label_list, dtype=torch.long, device=self.device)

    def generate_similarity_matrix(self):
        """
        Generate a similarity matrix for the 4 classes that reflects the order relationships:
            - CN (0) and MCI (1): related (set to 0.8)
            - MCI (1) and AD (2): related (set to 0.8)
            - CN (0) and LBD (3): related (set to 0.8)
        All other off-diagonals for different classes are 0.
        Diagonals are 1.
        Then compute an order matrix as the Laplacian: D - S.
        """
        # Initialize a 4x4 matrix with zeros
        S = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32, device=self.device)
        # Set diagonals to 1 (same class)
        for i in range(self.num_classes):
            S[i, i] = 1.0
        
        # Set the desired order relationships (symmetric)
        S[0, 1] = S[1, 0] = 0.8  # CN and MCI
        S[1, 2] = S[2, 1] = 0.8  # MCI and AD
        S[0, 3] = S[3, 0] = 0.8  # CN and LBD
        
        # Other pairs remain 0 (no order relationship)
        # If you prefer to have a very small nonzero value (e.g., 0.0) for no relation, leave as 0.
        
        # Now, if you have multiple prototypes per class, you might want to expand this matrix.
        # For now, assume one prototype per class.
        
        # Compute the degree matrix D and then the Laplacian-like order matrix L = D - S
        degree = torch.diag(S.sum(dim=1))
        order_matrix = degree - S
        return order_matrix

    def _initialize_weights(self):
        """
        Initialize weights for linear layers using Xavier uniform initialization.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0.0)
        for module in self.feature_extractor:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        """
        Forward pass: extracts features and produces classification logits.
        Returns:
            logits: classification outputs
            features: embeddings from the feature extractor
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def compute_center_loss(self, features, targets):
        """
        Compute center loss as the mean squared error between feature embeddings and their corresponding class centers.
        Here we assume one prototype per class. For multiple prototypes, consider choosing the closest one.
        """
        centers_batch = self.centers[targets]
        loss = F.mse_loss(features, centers_batch)
        return loss



class CenterModel(nn.Module):
    def __init__(self, device, layer_dims, dimension, num_classes=4, num_prototype=1, dropout_rate=0.5):
        """
        Args:
            device (torch.device): The device to run the model on.
            layer_dims (list): List of hidden layer dimensions for the feature extractor.
            dimension (int): Dimension of the input features.
            num_classes (int): Number of classes.
            num_prototype (int): Number of prototypes per class.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(CenterModel, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_prototype = num_prototype
        self.dimension = dimension
        self.dropout_rate = dropout_rate

        # Initialize centers (trainable prototypes)
        self.centers = self.initialize_centers().to(self.device)
        self.labels = self.generate_labels().to(self.device)
        self.order_matrix = self.generate_similarity_matrix().to(self.device)  # Order matrix for class relationships

        # Build feature extractor network
        dims = [dimension] + layer_dims
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=dims[i + 1], affine=True),
                nn.Dropout(self.dropout_rate)
            ))
        self.feature_extractor = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Linear(dims[-1], num_classes)

        # Initialize all weights recursively
        self._initialize_weights()

    def initialize_centers(self):
        """
        Initialize centers for each class-prototype pair using a simple strategy.
        Returns:
            centers (Parameter): A trainable tensor of shape (num_classes*num_prototype, dimension).
        """
        center_list = []
        for class_num in range(self.num_classes):
            for _ in range(self.num_prototype):
                # Initialize with a constant value scaled by class index; then shift by -0.5
                center_list.append([0.2 * (class_num + 1)] * self.dimension)
        centers = torch.tensor(center_list, dtype=torch.float32)
        centers = centers - 0.5  # Optionally shift centers
        return Parameter(centers, requires_grad=True)

    def generate_labels(self):
        """
        Generate labels corresponding to each center.
        Returns:
            labels (torch.Tensor): Tensor of shape (num_classes*num_prototype,) with class labels.
        """
        label_list = []
        for i in range(self.num_classes):
            label_list.extend([i] * self.num_prototype)
        return torch.tensor(label_list, dtype=torch.long)

    def generate_similarity_matrix(self):
        """
        Generate a similarity matrix (and corresponding Laplacian) reflecting order relationships.
        Specific relationships:
            - CN (0) and MCI (1): 0.8 similarity
            - MCI (1) and AD (2): 0.8 similarity
            - CN (0) and LBD (3): 0.8 similarity
        Diagonals are 1 and other off-diagonals are 0.
        Returns:
            order_matrix (torch.Tensor): The Laplacian matrix (degree - similarity).
        """
        S = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32)
        # Set self-similarity to 1
        for i in range(self.num_classes):
            S[i, i] = 1.0
        
        # Set specified relationships (symmetric)
        S[0, 1] = S[1, 0] = 0.8  # CN and MCI
        S[1, 2] = S[2, 1] = 0.8  # MCI and AD
        S[0, 3] = S[3, 0] = 0.8  # CN and LBD
        
        # Compute Laplacian: L = D - S
        degree = torch.diag(S.sum(dim=1))
        order_matrix = degree - S
        return order_matrix

    def _initialize_weights(self):
        """
        Recursively initialize weights for all linear layers using Xavier uniform initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
        Forward pass: extracts features and produces classification logits.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            logits (torch.Tensor): Classification outputs.
            features (torch.Tensor): Embeddings from the feature extractor.
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

    def compute_center_loss(self, features, targets):
        """
        Compute center loss between feature embeddings and their corresponding class centers.
        If multiple prototypes per class exist, the closest prototype is used for each sample.
        
        Args:
            features (torch.Tensor): Feature embeddings of shape (batch_size, dimension).
            targets (torch.Tensor): Ground truth labels of shape (batch_size,).
            
        Returns:
            loss (torch.Tensor): The computed center loss.
        """
        batch_size = features.size(0)
        loss = 0.0
        # Reshape centers for easier distance computation
        # centers shape: (num_classes, num_prototype, dimension)
        centers = self.centers.view(self.num_classes, self.num_prototype, self.dimension)
        
        for i in range(batch_size):
            target = targets[i].item()
            # Get the prototypes for the target class: (num_prototype, dimension)
            class_centers = centers[target]
            # Compute distances between feature and each prototype
            distances = torch.norm(features[i] - class_centers, dim=1)  # Euclidean distances
            # Use the smallest distance
            loss += torch.min(distances) ** 2
        
        loss = loss / batch_size
        return loss

# # Example usage:
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Example parameters
#     layer_dims = [128, 64]
#     input_dimension = 256
#     num_classes = 4
#     num_prototype = 2  # For instance, using 2 prototypes per class
    
#     model = CenterModel(device, layer_dims, input_dimension, num_classes, num_prototype, dropout_rate=0.5).to(device)
    
#     # Create a dummy input batch
#     x_dummy = torch.randn(8, input_dimension).to(device)
#     targets_dummy = torch.randint(0, num_classes, (8,)).to(device)
    
#     logits, features = model(x_dummy)
#     center_loss = model.compute_center_loss(features, targets_dummy)
    
#     print("Logits shape:", logits.shape)
#     print("Features shape:", features.shape)
#     print("Center loss:", center_loss.item())
