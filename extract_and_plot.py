import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import torch.nn.functional as F
import pdb
from collections import defaultdict


# Define a function to get features from the penultimate layer:
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        # Assuming ResNet18, penultimate layer is the layer before final FC
        self.features = nn.Sequential(*list(model.children())[:-1])  # all but the last FC layer
    def forward(self, x):
        return self.features(x).view(x.size(0), -1)


def extract_embeddings(model, device, data_loader):
    feature_extractor = FeatureExtractor(model).to(device)
    label_to_features = defaultdict(list)
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            f = feature_extractor(x)
            for feature, label in zip(f.cpu(), y):
                label_to_features[label.item()].append(feature)
    return label_to_features

def sample_features(label_to_features, n_examples_per_class):
    sampled_features = []
    sampled_labels = []
    for label, features in label_to_features.items():
        if len(features) >= n_examples_per_class:
            sampled_features.extend(features[:n_examples_per_class])
            sampled_labels.extend([label] * n_examples_per_class)
        else:
            sampled_features.extend(features)
            sampled_labels.extend([label] * len(features))
    return torch.stack(sampled_features), torch.tensor(sampled_labels)

if __name__ == '__main__':
    # 1. Load CIFAR-10 (ID) 
    transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                     (0.2470, 0.2435, 0.2616))])

    trainset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=True, download=False, transform=transform)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='/Users/tanmoy/research/data', train=False, download=False, transform=transform)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # Load CIFAR-100 (Near OOD)
    oodset = torchvision.datasets.CIFAR100(root='/Users/tanmoy/research/data', train=False, download=False, transform=transform)
    ood_loader = DataLoader(oodset, batch_size=100, shuffle=False, num_workers=4)

    device = 'cuda' if torch.cuda.is_available() else 'mps'

    # 2. A simple pretrained model on CIFAR-10 (e.g. ResNet18)
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)

    # Suppose we train normally (not shown full training loop for brevity)
    # train_model(model, train_loader, ...)
    n_examples_per_class = 100
    # After training, we assume model is well-trained on CIFAR-10.
    labels_to_features = extract_embeddings(model, device, test_loader)
    features,labels = sample_features(labels_to_features, n_examples_per_class)
    pdb.set_trace()
   
    mu = features.mean(dim=0)
    diff = features - mu
    Sigma = (diff.t() @ diff) / (features.size(0) - 1)
    Sigma_inv = torch.inverse(Sigma + 1e-5 * torch.eye(Sigma.size(0)))  # Regularize if needed

    # Compute Mahalanobis distances on ID
    mahalanobis = []
    for z in features:
        diff_z = (z - mu).unsqueeze(0)
        M = torch.sqrt((diff_z @ Sigma_inv @ diff_z.t()).squeeze())
        mahalanobis.append(M.item())

    mahalanobis = torch.tensor(mahalanobis)
    mean_dist = mahalanobis.mean().item()
    std_dist = mahalanobis.std().item()


    # 4. Define uncertainty region
    tau = mean_dist + std_dist   # Just outside ID region
    delta = 0.2 * std_dist       # A small band around tau

    def mahalanobis_distance(z, mu, Sigma_inv):
        diff_z = z - mu
        return torch.sqrt(diff_z @ Sigma_inv @ diff_z)

    # Sample an embedding from uncertainty region:
    def sample_uncertainty_embedding(mu, Sigma, Sigma_inv, tau, delta, max_tries=10000):
        d = mu.shape[0]
        # To sample from N(mu, Sigma), we can use Cholesky decomposition
        L = torch.linalg.cholesky(Sigma + 1e-5*torch.eye(d))
        for _ in range(max_tries):
            z_candidate = mu + torch.randn(d) @ L
            M = mahalanobis_distance(z_candidate, mu, Sigma_inv)
            if tau <= M <= tau + delta:
                return z_candidate
            return None

    z_neg = sample_uncertainty_embedding(mu, Sigma, Sigma_inv, tau, delta)
