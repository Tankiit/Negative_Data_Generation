import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cvxpy as cp
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict
from pytorch_ood.model import WideResNet

class DROEnergyDetector(nn.Module):
    def __init__(
        self, 
        feature_extractor, 
        feature_dim, 
        num_classes=10,
        batch_size=128,
        device=None,
        use_hierarchical_features=True,
        energy_temperature=1.0,
        energy_weight=0.5
    ):
        super().__init__()
        self.device = device if device is not None else get_device('auto')
        print(f"Using device: {self.device}")
        
        # Move feature extractor to device
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.use_hierarchical_features = use_hierarchical_features
        
        # Initialize parameters
        self.register_buffer('theta', torch.zeros(feature_dim, dtype=torch.float32, device=self.device))
        self.register_buffer('bias', torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer('feature_mean', torch.zeros(feature_dim, dtype=torch.float32, device=self.device))
        self.register_buffer('feature_std', torch.ones(feature_dim, dtype=torch.float32, device=self.device))
        self.register_buffer('precision_matrix', torch.eye(feature_dim, dtype=torch.float32, device=self.device))
        
        # Energy-based parameters
        self.energy_temperature = energy_temperature
        self.energy_weight = energy_weight
        
        # For WideResNet, feature_dim is the output of the last layer before classifier
        if isinstance(feature_extractor, WideResNet):
            self.feature_dim = feature_extractor.nChannels  # WideResNet's final feature dimension
            
        # Create classifier head for energy scoring and move to device
        self.classifier = nn.Linear(self.feature_dim, num_classes).to(self.device)
        
        # Energy calibration parameters
        self.register_buffer('energy_mean_id', torch.tensor(0.0, dtype=torch.float32, device=self.device))
        self.register_buffer('energy_std_id', torch.tensor(1.0, dtype=torch.float32, device=self.device))

    def extract_features(self, x):
        """Extract features from the feature extractor"""
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        if isinstance(self.feature_extractor, WideResNet):
            with torch.no_grad():
                # WideResNet feature extraction
                out = self.feature_extractor.conv1(x)
                out = self.feature_extractor.block1(out)
                out = self.feature_extractor.block2(out)
                out = self.feature_extractor.block3(out)
                out = self.feature_extractor.relu(self.feature_extractor.bn1(out))
                out = F.avg_pool2d(out, out.size()[2:])
                features = out.view(-1, self.feature_extractor.nChannels)
                return features
        else:
            with torch.no_grad():
                if isinstance(self.feature_extractor, torchvision.models.ResNet):
                    x = self.feature_extractor.conv1(x)
                    x = self.feature_extractor.bn1(x)
                    x = self.feature_extractor.relu(x)
                    x = self.feature_extractor.maxpool(x)
                    
                    x = self.feature_extractor.layer1(x)
                    x = self.feature_extractor.layer2(x)
                    x = self.feature_extractor.layer3(x)
                    x = self.feature_extractor.layer4(x)
                    
                    x = self.feature_extractor.avgpool(x)
                    features = torch.flatten(x, 1)
                    return features
                else:
                    raise NotImplementedError(f"Feature extraction not implemented for {type(self.feature_extractor)}")

    def extract_hierarchical_features(self, x):
        """Extract hierarchical features from different network layers"""
        if not self.use_hierarchical_features:
            features = self.extract_features(x)
            return features.to(dtype=torch.float32)
            
        activations = []
        def hook_fn(module, input, output):
            activations.append(output.to(dtype=torch.float32))
        
        # Add hooks to network layers
        hooks = []
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d) and any(f'layer{i}' in name for i in [1, 2, 3, 4]):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = self.feature_extractor(x.to(self.device))
        
        # Collect features
        features = []
        for feat in activations:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            features.append(pooled)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        if len(features) > 0:
            # Take representative features from different layers
            if len(features) > 3:
                indices = [0, len(features)//2, -1]
                selected_features = torch.cat([features[i] for i in indices], dim=1)
            else:
                selected_features = torch.cat(features, dim=1)
            
            # Project to expected dimension if needed
            if selected_features.shape[1] != self.feature_dim:
                if not hasattr(self, 'projection'):
                    self.projection = nn.Linear(
                        selected_features.shape[1], 
                        self.feature_dim
                    ).to(self.device)
                    nn.init.orthogonal_(self.projection.weight)
                selected_features = self.projection(selected_features)
            
            return selected_features
        else:
            return self.extract_features(x).to(dtype=torch.float32)
    
    def compute_energy(self, logits):
        """Compute energy scores from logits"""
        # Ensure logits are on the correct device
        logits = logits.to(self.device)
        energy = -self.energy_temperature * torch.logsumexp(logits / self.energy_temperature, dim=1)
        return energy
    
    def solve_dro_optimization(self, features, radius=10.0, reg_param=1.0):
        """
        Solve DRO optimization with direct cvxpy
        """
        # CPU-based computation is more stable for cvxpy
        features_np = features.detach().cpu().numpy().astype(np.float32)
        n_samples, n_features = features_np.shape
        
        # Define variables
        theta = cp.Variable(n_features)
        bias = cp.Variable()
        
        # Compute predictions
        predictions = features_np @ theta + bias
        
        # Define objective terms
        hinge_loss = cp.sum(cp.pos(1.0 - predictions))
        wasserstein_penalty = radius * cp.norm(theta, 2)
        regularization = reg_param * cp.sum_squares(theta)
        
        # Total objective
        objective = hinge_loss + wasserstein_penalty + regularization
        
        # Create problem
        problem = cp.Problem(cp.Minimize(objective))
        
        try:
            # Solve with cvxpy directly
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                theta_val = theta.value.astype(np.float32)
                bias_val = float(bias.value)
                return theta_val, bias_val, float(problem.value)
            else:
                return None, None, float('inf')
        except Exception as e:
            print(f"Optimization error: {e}")
            return None, None, float('inf')
    
    def train_energy_classifier(self, train_loader, val_loader=None, num_epochs=5):
        """Train the energy-based classifier"""
        self.train()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            train_loss = 0.0
            train_acc = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                
                # Extract features and compute logits
                features = self.extract_hierarchical_features(inputs)
                logits = self.classifier(features)
                
                # Compute cross-entropy loss
                ce_loss = criterion(logits, targets)
                
                # Compute energy scores
                energy_scores = self.compute_energy(logits)
                
                # Energy regularization loss
                energy_reg = torch.pow(energy_scores, 2).mean()
                
                # Total loss
                loss = ce_loss + self.energy_weight * energy_reg
                
                loss.backward()
                optimizer.step()
                
                # Update metrics
                train_loss += loss.item()
                _, predicted = logits.max(1)
                train_acc += predicted.eq(targets).sum().item()
                
                # Update progress bar
                avg_loss = train_loss / (batch_idx + 1)
                avg_acc = 100. * train_acc / ((batch_idx + 1) * inputs.size(0))
                progress_bar.set_postfix({
                    'loss': f"{avg_loss:.3f}",
                    'acc': f"{avg_acc:.2f}%"
                })
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_acc = 0.0
                val_energy_scores = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        
                        # Forward pass
                        features = self.extract_hierarchical_features(inputs)
                        logits = self.classifier(features)
                        energy_scores = self.compute_energy(logits)
                        
                        # Compute loss
                        ce_loss = criterion(logits, targets)
                        energy_reg = torch.pow(energy_scores, 2).mean()
                        loss = ce_loss + self.energy_weight * energy_reg
                        
                        # Update metrics
                        val_loss += loss.item()
                        _, predicted = logits.max(1)
                        val_acc += predicted.eq(targets).sum().item()
                        val_energy_scores.extend(energy_scores.cpu().numpy())
                
                # Calculate validation metrics
                val_loss /= len(val_loader)
                val_acc = 100. * val_acc / len(val_loader.dataset)
                
                # Update energy calibration parameters
                val_energy_scores = np.array(val_energy_scores)
                self.energy_mean_id = torch.tensor(np.mean(val_energy_scores), device=self.device)
                self.energy_std_id = torch.tensor(np.std(val_energy_scores), device=self.device)
                
                print(f"\nValidation - Loss: {val_loss:.3f}, Accuracy: {val_acc:.2f}%")
                self.train()

    def evaluate(self, id_loader, ood_loader):
        """Evaluate the model on ID and OOD data"""
        self.eval()
        id_scores = []
        ood_scores = []
        
        # Evaluate on ID data
        print("\nEvaluating on ID data...")
        with torch.no_grad():
            for inputs, _ in tqdm(id_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                features = self.extract_hierarchical_features(inputs)
                logits = self.classifier(features)
                energy_scores = self.compute_energy(logits)
                id_scores.extend(energy_scores.cpu().numpy())
        
        # Evaluate on OOD data
        print("Evaluating on OOD data...")
        with torch.no_grad():
            for inputs, _ in tqdm(ood_loader):
                inputs = inputs.to(self.device, non_blocking=True)
                features = self.extract_hierarchical_features(inputs)
                logits = self.classifier(features)
                energy_scores = self.compute_energy(logits)
                ood_scores.extend(energy_scores.cpu().numpy())
        
        # Convert to numpy arrays
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        # Compute metrics
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
        
        auroc = roc_auc_score(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        aupr = auc(recall, precision)
        
        fpr, tpr, _ = roc_curve(labels, scores)
        
        return {
            'auroc': auroc,
            'aupr': aupr,
            'fpr': fpr,
            'tpr': tpr,
            'id_scores': id_scores,
            'ood_scores': ood_scores
        }
    
    def fit(self, train_loader, val_loader=None, radius_values=None, reg_values=None, energy_epochs=5):
        """Train the detector with both DRO and energy-based components"""
        if radius_values is None:
            radius_values = [1.0, 5.0, 10.0, 20.0, 50.0]
        if reg_values is None:
            reg_values = [0.01, 0.1, 1.0, 10.0]
        
        # First, train the energy-based classifier
        self.train_energy_classifier(train_loader, val_loader, num_epochs=energy_epochs)
        
        # Compute feature statistics
        print("\nComputing feature statistics...")
        features_sum = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
        features_squared_sum = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
        all_features = []
        n_samples = 0
        
        with torch.no_grad():  
            for images, _ in tqdm(train_loader, desc="Computing statistics"):
                images = images.to(self.device)
                features = self.extract_hierarchical_features(images)
                
                features_sum += features.sum(dim=0)
                features_squared_sum += (features ** 2).sum(dim=0)
                n_samples += features.size(0)
                
                if len(all_features) < 10000:
                    all_features.append(features.cpu())
        
        # Compute statistics
        self.feature_mean = features_sum / n_samples
        self.feature_std = torch.sqrt(
            (features_squared_sum / n_samples) - (self.feature_mean ** 2) + 1e-8
        )
        
        # Compute precision matrix for Mahalanobis distance
        if all_features:
            features_mat = torch.cat(all_features, dim=0)
            if features_mat.size(0) > 10000:
                idx = torch.randperm(features_mat.size(0))[:10000]
                features_mat = features_mat[idx]
            
            norm_features = (features_mat - self.feature_mean.cpu()) / self.feature_std.cpu()
            cov = torch.mm(norm_features.t(), norm_features) / norm_features.size(0)
            cov += torch.eye(cov.size(0)) * 1e-5
            
            try:
                self.precision_matrix = torch.inverse(cov).to(self.device)
                print("Successfully computed precision matrix for Mahalanobis distance")
            except:
                print("Warning: Using identity matrix for Mahalanobis distance")
        
        # Optimize DRO parameters
        print("\nOptimizing DRO parameters...")
        best_loss = float('inf')
        best_theta = None
        best_bias = None
        
        for radius in radius_values:
            for reg in reg_values:
                print(f"\nTrying radius={radius:.3f}, reg_param={reg:.3f}")
                try:
                    for images, _ in tqdm(train_loader, desc="Training"):
                        images = images.to(self.device)
                        features = self.extract_hierarchical_features(images)
                        norm_features = (features - self.feature_mean) / self.feature_std
                        
                        if norm_features.size(0) < self.batch_size:
                            padded = torch.zeros(
                                (self.batch_size, self.feature_dim), 
                                device=self.device
                            )
                            padded[:norm_features.size(0)] = norm_features
                            norm_features = padded
                        elif norm_features.size(0) > self.batch_size:
                            norm_features = norm_features[:self.batch_size]
                        
                        theta, bias, loss = self.solve_dro_optimization(norm_features, radius, reg)
                        
                        if loss < best_loss:
                            best_loss = loss
                            best_theta = theta
                            best_bias = bias
                            print("New best model found!")
                
                except Exception as e:
                    print(f"Error with parameters: {e}")
                    continue
        
        if best_theta is not None:
            self.theta = torch.tensor(best_theta, dtype=torch.float32, device=self.device)
            self.bias = torch.tensor(best_bias, dtype=torch.float32, device=self.device)
            print(f"Optimization complete. Theta norm: {torch.norm(self.theta).item():.4f}, "
                  f"Bias: {self.bias.item():.4f}")
            return True
        
        print("Optimization failed!")
        return False
    
    def get_dro_score(self, features):
        """Compute DRO-based score from features"""
        norm_features = (features - self.feature_mean) / self.feature_std
        dro_score = -(norm_features @ self.theta + self.bias)
        return dro_score
    
    def get_mahalanobis_score(self, features):
        """Compute Mahalanobis-based score from features"""
        norm_features = (features - self.feature_mean) / self.feature_std
        
        try:
            # Full Mahalanobis calculation
            delta = norm_features.unsqueeze(1) - self.feature_mean.unsqueeze(0)
            mahal_dist = torch.bmm(
                torch.bmm(delta, self.precision_matrix.unsqueeze(0).expand(features.size(0), -1, -1)),
                delta.transpose(1, 2)
            ).squeeze()
            return mahal_dist
        except:
            # Fallback to simpler calculation
            return torch.sum(norm_features ** 2, dim=1)
    
    def get_energy_score(self, features):
        """Compute calibrated energy score"""
        raw_energy = self.compute_energy(self.classifier(features))
        # Convert to a score where higher values indicate OOD samples
        normalized_energy = (raw_energy - self.energy_mean_id) / self.energy_std_id
        return normalized_energy
    
    def get_combined_score(self, x, alpha=0.4, beta=0.4, gamma=0.2):
        """Compute combined OOD score"""
        with torch.no_grad():
            features = self.extract_hierarchical_features(x)
            
            # DRO score
            dro_score = self.get_dro_score(features)
            
            # Energy score
            energy_score = self.get_energy_score(features)
            
            # Mahalanobis score
            mahal_score = self.get_mahalanobis_score(features)
            
            # Normalize scores to similar ranges (optional)
            dro_score = dro_score / (dro_score.abs().max() + 1e-8)
            energy_score = energy_score / (energy_score.abs().max() + 1e-8)
            mahal_score = mahal_score / (mahal_score.max() + 1e-8)
            
            # Combine scores
            combined_score = alpha * dro_score + beta * energy_score + gamma * mahal_score
        
        return combined_score
    
    @torch.no_grad()
    def forward(self, x):
        """Forward pass returning OOD scores"""
        return self.get_combined_score(x)

def create_feature_extractor(model_name='resnet18', num_classes=10, pretrained_key=None):
    """
    Create and initialize the feature extractor
    Args:
        model_name: Base model architecture
        num_classes: Number of output classes
        pretrained_key: Key for pytorch_ood pre-trained model
    """
    if model_name == 'wideresnet' or pretrained_key:
        # Use WideResNet from pytorch_ood
        depth = 28
        widen_factor = 10 if not pretrained_key or not pretrained_key.endswith('pixmix') else 4
        model = WideResNet(num_classes=num_classes, depth=depth, widen_factor=widen_factor)
        
        # Load pre-trained weights if specified
        if pretrained_key:
            try:
                weights_path = f"pytorch_ood_weights/{pretrained_key}.pt"
                if os.path.exists(weights_path):
                    state_dict = torch.load(weights_path)
                    model.load_state_dict(state_dict)
                else:
                    print(f"Pre-trained weights not found: {weights_path}")
                    print("Using randomly initialized weights")
            except Exception as e:
                print(f"Error loading pre-trained weights: {e}")
                print("Using randomly initialized weights")
    else:
        # Use standard torchvision models
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=True)
            feature_dim = model.fc.in_features
            model.fc = nn.Linear(feature_dim, num_classes)
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=True)
            feature_dim = model.fc.in_features
            model.fc = nn.Linear(feature_dim, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return model

def get_transform(dataset_name, model_type=None):
    """
    Get appropriate transforms based on dataset and model type.
    Ensures consistent 32x32 image size across all datasets.
    
    Args:
        dataset_name: Name of the dataset (cifar10, cifar100, svhn, mnist, etc.)
        model_type: Type of model (wideresnet or None for torchvision models)
    """
    # Set normalization parameters based on model type
    if model_type == 'wideresnet' or model_type is not None:
        mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std = [x / 255.0 for x in [63.0, 62.1, 66.7]]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    dataset_name = dataset_name.lower()
    
    # Base transforms to ensure consistent size
    if dataset_name == 'dtd':
        # DTD images can have varying sizes and aspect ratios
        size_transforms = [
            transforms.Resize(36),  # Slightly larger for better cropping
            transforms.CenterCrop(32),  # Then crop to exact size
        ]
    elif dataset_name == 'tiny-imagenet':
        # Tiny-ImageNet images are originally 64x64
        size_transforms = [
            transforms.Resize(32),
            transforms.CenterCrop(32),
        ]
    elif dataset_name == 'lsun':
        # LSUN images can be large and varied
        size_transforms = [
            transforms.Resize(36),  # Slightly larger
            transforms.CenterCrop(32),  # Then crop to exact size
        ]
    else:
        # For CIFAR, SVHN, MNIST, FashionMNIST
        size_transforms = [
            transforms.Resize((32, 32)),  # Force exact size
        ]
    
    # Compose the final transform
    if dataset_name in ['mnist', 'fashionmnist']:
        transform = transforms.Compose([
            *size_transforms,
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to RGB
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            *size_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def load_dataset(dataset_name, data_dir, train=True, model_type=None):
    """Load specified dataset with appropriate transforms."""
    transform = get_transform(dataset_name, model_type)

    # Standard datasets
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 100
    elif dataset_name.lower() == 'svhn':
        split = 'train' if train else 'test'
        dataset = torchvision.datasets.SVHN(root=data_dir, split=split, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'mnist':
        dataset = torchvision.datasets.MNIST(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'fashionmnist':
        dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=train, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'tiny-imagenet':
        dataset = TinyImageNet(root=os.path.join(data_dir, 'tiny-imagenet-200'), train=train, transform=transform)
        num_classes = 200
    elif dataset_name.lower() == 'dtd':
        split = 'train' if train else 'test'
        dataset = torchvision.datasets.DTD(root=data_dir, split=split, download=True, transform=transform)
        num_classes = 47
    elif dataset_name.lower() == 'lsun':
        if not train:  # Only use test set for OOD
            dataset = torchvision.datasets.LSUN(root=data_dir, classes=['test'], transform=transform)
            num_classes = 10
        else:
            raise ValueError("LSUN training set not supported for OOD detection")
    elif dataset_name.lower() in ['gaussian', 'uniform']:
        dataset = create_noise_dataset(noise_type=dataset_name.lower())
        num_classes = 1
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return dataset, num_classes

def parse_args():
    parser = argparse.ArgumentParser(description='DRO Energy-based Out-of-Distribution Detection')
    
    # Dataset arguments
    parser.add_argument('--dataset-dir', type=str, default='/Users/tanmoy/research/data',
                        help='Directory where datasets are stored/downloaded (default: ./data)')
    parser.add_argument('--id-dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn'],
                        help='In-distribution dataset (default: cifar10)')
    parser.add_argument('--ood-dataset', type=str, default='svhn',
                        choices=['cifar10', 'cifar100', 'svhn', 'mnist', 'fashionmnist',
                                'tiny-imagenet', 'dtd', 'lsun', 'gaussian', 'uniform'],
                        help='Out-of-distribution dataset (default: svhn)')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='wideresnet',
                        choices=['wideresnet', 'resnet18', 'resnet50'],
                        help='Model architecture (default: wideresnet)')
    parser.add_argument('--pretrained-key', type=str, default=None,
                        choices=['oe-cifar100-tune', 'oe-cifar10-tune',
                                'er-cifar10-tune', 'er-cifar100-tune',
                                'cifar100-pt', 'cifar10-pt',
                                'cifar10-pixmix', 'cifar100-pixmix'],
                        help='Pre-trained model key from pytorch_ood (only for WideResNet)')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use (default: auto)')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=5,
                        help='Number of epochs to train (default: 5)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for energy-based model (default: 1.0)')
    parser.add_argument('--energy-weight', type=float, default=0.5,
                        help='Weight for energy regularization (default: 0.5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    args = parser.parse_args()
    
    # Print device information
    if args.device == 'auto':
        print("Device selection: automatic")
    else:
        print(f"Device selection: {args.device}")
    
    if args.device in ['cuda', 'auto'] and torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    if args.device in ['mps', 'auto'] and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    
    return args

def create_noise_dataset(noise_type='gaussian', size=(3, 32, 32), num_samples=10000, device=None):
    """Create a synthetic noise dataset"""
    if device is None:
        device = get_device('auto')
    
    if noise_type == 'gaussian':
        data = torch.randn((num_samples,) + size, device=device)
    elif noise_type == 'uniform':
        data = torch.rand((num_samples,) + size, device=device)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Create dummy labels
    labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    
    return TensorDataset(data, labels)

def get_device(device_str='auto'):
    """
    Get the appropriate device based on availability and user preference
    Args:
        device_str: 'auto', 'cuda', 'mps', or 'cpu'
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    elif device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_str == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif device_str == 'cpu':
        return torch.device('cpu')
    else:
        print(f"Warning: Requested device '{device_str}' not available. Using CPU.")
        return torch.device('cpu')

def evaluate_detector(detector, id_loader, ood_loader, device, detailed=False):
    """Evaluate OOD detection performance with detailed component analysis"""
    detector.eval()
    
    # Collect different score components for analysis
    results = {
        'id_scores': [],
        'ood_scores': [],
        'id_dro': [],
        'ood_dro': [],
        'id_energy': [],
        'ood_energy': [],
        'id_mahalanobis': [],
        'ood_mahalanobis': []
    }
    
    with torch.no_grad():
        # Process ID data
        for images, _ in tqdm(id_loader, desc="Evaluating ID"):
            images = images.to(device)
            features = detector.extract_hierarchical_features(images)
            
            # Combined score
            combined = detector(images)
            results['id_scores'].extend(combined.cpu().numpy())
            
            # Individual components (if detailed evaluation requested)
            if detailed:
                dro = detector.get_dro_score(features)
                energy = detector.get_energy_score(features)
                mahal = detector.get_mahalanobis_score(features)
                
                results['id_dro'].extend(dro.cpu().numpy())
                results['id_energy'].extend(energy.cpu().numpy())
                results['id_mahalanobis'].extend(mahal.cpu().numpy())
        
        # Process OOD data
        for images, _ in tqdm(ood_loader, desc="Evaluating OOD"):
            images = images.to(device)
            features = detector.extract_hierarchical_features(images)
            
            # Combined score
            combined = detector(images)
            results['ood_scores'].extend(combined.cpu().numpy())
            
            # Individual components (if detailed evaluation requested)
            if detailed:
                dro = detector.get_dro_score(features)
                energy = detector.get_energy_score(features)
                mahal = detector.get_mahalanobis_score(features)
                
                results['ood_dro'].extend(dro.cpu().numpy())
                results['ood_energy'].extend(energy.cpu().numpy())
                results['ood_mahalanobis'].extend(mahal.cpu().numpy())
    
    # Convert to numpy arrays
    for key in results:
        if results[key]:
            results[key] = np.array(results[key])
    
    # Compute metrics for combined score
    id_scores = results['id_scores']
    ood_scores = results['ood_scores']
    
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    
    # FPR at 95% TPR
    threshold = np.percentile(id_scores, 95)
    fpr = (ood_scores <= threshold).mean()
    
    # Print metrics for combined score
    print(f"\nCombined Score Results:")
    print(f"ID scores: {np.mean(id_scores):.4f} ± {np.std(id_scores):.4f}")
    print(f"OOD scores: {np.mean(ood_scores):.4f} ± {np.std(ood_scores):.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR at 95% TPR: {fpr:.4f}")
    
    # Compute metrics for individual components if detailed evaluation
    component_metrics = {}
    if detailed:
        for component in ['dro', 'energy', 'mahalanobis']:
            id_comp = results[f'id_{component}']
            ood_comp = results[f'ood_{component}']
            
            y_true_comp = np.concatenate([np.zeros(len(id_comp)), np.ones(len(ood_comp))])
            y_score_comp = np.concatenate([id_comp, ood_comp])
            
            comp_auroc = roc_auc_score(y_true_comp, y_score_comp)
            comp_precision, comp_recall, _ = precision_recall_curve(y_true_comp, y_score_comp)
            comp_aupr = auc(comp_recall, comp_precision)
            
            # FPR at 95% TPR for component
            comp_threshold = np.percentile(id_comp, 95)
            comp_fpr = (ood_comp <= comp_threshold).mean()
            
            component_metrics[component] = {
                'auroc': comp_auroc,
                'aupr': comp_aupr,
                'fpr': comp_fpr
            }
            
            print(f"\n{component.capitalize()} Component Results:")
            print(f"ID scores: {np.mean(id_comp):.4f} ± {np.std(id_comp):.4f}")
            print(f"OOD scores: {np.mean(ood_comp):.4f} ± {np.std(ood_comp):.4f}")
            print(f"AUROC: {comp_auroc:.4f}")
            print(f"AUPR: {comp_aupr:.4f}")
            print(f"FPR at 95% TPR: {comp_fpr:.4f}")
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr': fpr,
        'id_scores': id_scores,
        'ood_scores': ood_scores,
        'component_metrics': component_metrics
    }

def visualize_results(results, save_path, title=None):
    """Create comprehensive visualizations of detection results"""
    id_scores = results['id_scores']
    ood_scores = results['ood_scores']
    
    # Create figure with subplots
    if 'component_metrics' in results and results['component_metrics']:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = axes.reshape(1, 2)
    
    # Main title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Combined score distribution
    axes[0, 0].hist(id_scores, bins=50, alpha=0.5, label='ID', density=True)
    axes[0, 0].hist(ood_scores, bins=50, alpha=0.5, label='OOD', density=True)
    axes[0, 0].set_xlabel('OOD Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].set_title('Combined Score Distributions')
    
    # ROC curve
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = results['auroc']
    
    axes[0, 1].plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()
    
    # Plot component performance if available
    if 'component_metrics' in results and results['component_metrics']:
        # Component AUROC comparison
        components = list(results['component_metrics'].keys())
        auroc_values = [results['component_metrics'][c]['auroc'] for c in components]
        auroc_values.append(results['auroc'])  # Add combined score
        components.append('combined')
        
        # Bar chart of AUROC values
        axes[1, 0].bar(components, auroc_values)
        axes[1, 0].set_ylim([0.5, 1.0])
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('AUROC')
        axes[1, 0].set_title('AUROC Comparison')
        
        # FPR comparison
        fpr_values = [results['component_metrics'][c]['fpr'] for c in components[:-1]]
        fpr_values.append(results['fpr'])  # Add combined score
        
        axes[1, 1].bar(components, fpr_values)
        axes[1, 1].set_ylim([0, 1.0])
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('FPR at 95% TPR')
        axes[1, 1].set_title('FPR at 95% TPR Comparison')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Load datasets with appropriate transforms
    print(f"\nLoading ID dataset: {args.id_dataset}")
    train_dataset, num_classes = load_dataset(args.id_dataset, args.dataset_dir, train=True, 
                                            model_type=args.model if args.pretrained_key else None)
    test_dataset, _ = load_dataset(args.id_dataset, args.dataset_dir, train=False, 
                                 model_type=args.model if args.pretrained_key else None)
    
    print(f"Loading OOD dataset: {args.ood_dataset}")
    ood_dataset, _ = load_dataset(args.ood_dataset, args.dataset_dir, train=False, 
                                model_type=args.model if args.pretrained_key else None)
    
    # Create data loaders with appropriate device placement
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                            pin_memory=device.type != 'cpu')  # Enable pin_memory for GPU
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                           pin_memory=device.type != 'cpu')
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                          pin_memory=device.type != 'cpu')
    
    # Create feature extractor
    print(f"\nCreating feature extractor...")
    if args.pretrained_key:
        print(f"Using pre-trained model: {args.pretrained_key}")
    feature_extractor = create_feature_extractor(args.model, num_classes, args.pretrained_key)
    
    # Determine feature dimension
    if isinstance(feature_extractor, WideResNet):
        feature_dim = feature_extractor.nChannels
    elif hasattr(feature_extractor, 'fc'):
        feature_dim = feature_extractor.fc.in_features
    else:
        # Run a sample batch through the network as fallback
        sample_batch, _ = next(iter(train_loader))
        with torch.no_grad():
            sample_features = feature_extractor(sample_batch.to(device))
        feature_dim = sample_features.shape[1]
    
    print(f"Feature dimension: {feature_dim}")
    
    # Create detector with proper device handling
    detector = DROEnergyDetector(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=num_classes,
        batch_size=args.batch_size,
        device=device,
        energy_temperature=args.temperature,
        energy_weight=args.energy_weight
    )
    
    # Train energy classifier
    detector.train_energy_classifier(train_loader, val_loader=test_loader, num_epochs=args.num_epochs)
    
    # Evaluate detector
    print("\nEvaluating detector...")
    results = evaluate_detector(detector, test_loader, ood_loader, device, detailed=True)
    
    # Save and visualize results
    results_file = os.path.join(args.output_dir, f'results_{args.id_dataset}_vs_{args.ood_dataset}.pt')
    torch.save(results, results_file)
    
    title = f'OOD Detection: {args.id_dataset.upper()} (ID) vs {args.ood_dataset.upper()} (OOD)'
    visualize_results(results, os.path.join(args.output_dir, f'{args.id_dataset}_vs_{args.ood_dataset}.png'), title=title)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
