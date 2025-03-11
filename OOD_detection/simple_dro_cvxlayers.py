import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os

class FixedDRODetector(nn.Module):
    def __init__(
        self, 
        feature_extractor, 
        feature_dim, 
        batch_size=128,
        device=None,
        use_hierarchical_features=True
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.use_hierarchical_features = use_hierarchical_features
        
        # Initialize parameters with float32 dtype
        self.register_buffer('theta', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('feature_mean', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('feature_std', torch.ones(feature_dim, dtype=torch.float32))
        self.register_buffer('precision_matrix', torch.eye(feature_dim, dtype=torch.float32))
        
        # For energy-based scoring
        self.temperature = 1.0
    
    def extract_hierarchical_features(self, x):
        """Extract hierarchical features from different network layers"""
        if not self.use_hierarchical_features:
            features = self.feature_extractor(x.to(self.device))
            return features.to(dtype=torch.float32)
            
        activations = []
        def hook_fn(module, input, output):
            # Ensure output is float32
            activations.append(output.to(dtype=torch.float32))
        
        # Add hooks to ResNet layers
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d) and any(f'layer{i}' in name for i in [1, 2, 3, 4]):
                module.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = self.feature_extractor(x.to(self.device))
        
        # Collect features
        features = []
        for feat in activations:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
            features.append(pooled)
        
        # Clean up
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d) and any(f'layer{i}' in name for i in [1, 2, 3, 4]):
                module._forward_hooks.clear()
        
        if len(features) > 0:
            # Take representative features
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
            return self.feature_extractor(x.to(self.device)).to(dtype=torch.float32)
    
    def solve_dro_optimization(self, features, radius=10.0, reg_param=1.0):
        """
        Solve DRO optimization with direct cvxpy instead of cvxpylayers
        """
        # CPU-based computation is more stable for cvxpy
        features_np = features.detach().cpu().numpy().astype(np.float32)  # Convert to float32
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
                # Convert to float32
                theta_val = theta.value.astype(np.float32)
                bias_val = float(bias.value)  # Convert to Python float
                return theta_val, bias_val, float(problem.value)
            else:
                return None, None, float('inf')
        except Exception as e:
            print(f"Optimization error: {e}")
            return None, None, float('inf')
    
    def fit(self, train_loader, radius_values=None, reg_values=None):
        """Train the detector with multiple scoring methods"""
        if radius_values is None:
            radius_values = [1.0, 5.0, 10.0, 20.0, 50.0]
        if reg_values is None:
            reg_values = [0.01, 0.1, 1.0, 10.0]
        
        # Compute feature statistics and precision matrix
        print("Computing feature statistics...")
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
        
        # Compute precision matrix
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
        
        # Collect all training features for batch optimization
        print("\nCollecting training features for optimization...")
        all_training_features = []
        
        with torch.no_grad():  
            for images, _ in tqdm(train_loader, desc="Collecting features"):
                images = images.to(self.device)
                features = self.extract_hierarchical_features(images)
                norm_features = (features - self.feature_mean) / self.feature_std
                all_training_features.append(norm_features.cpu())
        
        # Optimize DRO parameters
        print("\nOptimizing DRO parameters...")
        best_loss = float('inf')
        best_theta = None
        best_bias = None
        
        for radius in radius_values:
            for reg in reg_values:
                print(f"\nTrying radius={radius:.3f}, reg_param={reg:.3f}")
                try:
                    epoch_losses = []
                    
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
            # Convert to float32 tensors
            self.theta = torch.tensor(best_theta, dtype=torch.float32, device=self.device)
            self.bias = torch.tensor(best_bias, dtype=torch.float32, device=self.device)
            print(f"Optimization complete. Theta norm: {torch.norm(self.theta).item():.4f}, "
                  f"Bias: {self.bias.item():.4f}")
            return True
        
        print("Optimization failed!")
        return False
    
    def get_combined_score(self, x, alpha=0.4, beta=0.4, gamma=0.2):
        """Compute combined OOD score"""
        with torch.no_grad():
            features = self.extract_hierarchical_features(x)
            norm_features = (features - self.feature_mean) / self.feature_std
            
            # Ensure all tensors are float32
            norm_features = norm_features.to(dtype=torch.float32)
            theta = self.theta.to(dtype=torch.float32)
            bias = self.bias.to(dtype=torch.float32)
            
            # DRO score
            dro_score = -(norm_features @ theta + bias)
            combined_score = alpha * dro_score
            
            # Mahalanobis distance
            try:
                delta = norm_features.unsqueeze(1) - self.feature_mean.unsqueeze(0)
                precision_matrix = self.precision_matrix.to(dtype=torch.float32)
                mahal_dist = torch.bmm(
                    torch.bmm(delta, precision_matrix.unsqueeze(0).expand(features.size(0), -1, -1)),
                    delta.transpose(1, 2)
                ).squeeze()
                mahal_norm = mahal_dist / (mahal_dist.max() + 1e-8)
                combined_score += gamma * mahal_norm
            except Exception as e:
                # Fallback to simpler calculation if the above fails
                try:
                    # Simplified Mahalanobis (squared distance from mean)
                    mahal_simple = torch.sum(norm_features ** 2, dim=1)
                    mahal_norm = mahal_simple / (mahal_simple.max() + 1e-8)
                    combined_score += gamma * mahal_norm
                except:
                    pass
            
            # Energy score
            if hasattr(self.feature_extractor, 'fc') and isinstance(self.feature_extractor.fc, nn.Linear):
                try:
                    logits = self.feature_extractor.fc(features)
                    energy = -torch.logsumexp(logits / self.temperature, dim=1)
                    energy_norm = energy / (energy.max() + 1e-8)
                    combined_score += beta * energy_norm
                except:
                    pass
        
        return combined_score.to(dtype=torch.float32)
    
    @torch.no_grad()
    def forward(self, x):
        """Forward pass returning OOD scores"""
        return self.get_combined_score(x)

def evaluate_detector(detector, id_loader, ood_loader, device):
    """Evaluate OOD detection performance"""
    detector.eval()
    
    # Collect scores
    id_scores = []
    ood_scores = []
    
    with torch.no_grad():
        for images, _ in tqdm(id_loader, desc="Evaluating ID"):
            images = images.to(device)
            id_scores.extend(detector(images).cpu().numpy())
            
        for images, _ in tqdm(ood_loader, desc="Evaluating OOD"):
            images = images.to(device)
            ood_scores.extend(detector(images).cpu().numpy())
    
    id_scores = np.array(id_scores)
    ood_scores = np.array(ood_scores)
    
    # Compute metrics
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    
    auroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    aupr = auc(recall, precision)
    
    # FPR at 95% TPR
    threshold = np.percentile(id_scores, 95)
    fpr = (ood_scores <= threshold).mean()
    
    print(f"\nResults:")
    print(f"ID scores: {np.mean(id_scores):.4f} ± {np.std(id_scores):.4f}")
    print(f"OOD scores: {np.mean(ood_scores):.4f} ± {np.std(ood_scores):.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR at 95% TPR: {fpr:.4f}")
    
    return {
        'auroc': auroc,
        'aupr': aupr,
        'fpr': fpr,
        'id_scores': id_scores,
        'ood_scores': ood_scores
    }

def train_and_evaluate(train_loader, ood_loader, feature_extractor, feature_dim=128, batch_size=1000):
    """Train and evaluate the DRO OOD detector"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    detector = FixedDRODetector(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        batch_size=batch_size
    ).to(device)
    
    # Train detector
    success = detector.fit(train_loader)
    if not success:
        print("Training failed!")
        return None, None
    
    # Evaluate
    print("\nEvaluating...")
    results = evaluate_detector(detector, train_loader, ood_loader, device)
    
    return detector, results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced DRO-based OOD Detection")
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='/Users/tanmoy/research/data', 
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'wideresnet'], 
                        help='Model architecture')
    parser.add_argument('--hierarchical', action='store_true', 
                        help='Use hierarchical features')
    
    # DRO parameters
    parser.add_argument('--alpha', type=float, default=0.4, 
                        help='Weight for DRO score')
    parser.add_argument('--beta', type=float, default=0.4, 
                        help='Weight for energy score')
    parser.add_argument('--gamma', type=float, default=0.2, 
                        help='Weight for Mahalanobis score')
    
    # Other parameters
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--save-dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--ood-dataset', type=str, default='both',
                        choices=['svhn', 'mnist', 'both'],
                        help='OOD dataset to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    return args

def create_feature_extractor(model_name='resnet18'):
    """Create and initialize the feature extractor"""
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        feature_dim = 512
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
        feature_dim = 2048
    else:  # wideresnet
        from pytorch_ood.model import WideResNet
        model = WideResNet(
            num_classes=10,
            depth=40,
            widen_factor=2,
            dropout_rate=0.3,
            in_channels=3
        )
        feature_dim = 128
    
    # Save original FC layer for energy scoring
    if model_name in ['resnet18', 'resnet50']:
        original_fc = model.fc
        model.fc = torch.nn.Identity()
    
    return model, feature_dim

def visualize_results(id_scores, ood_scores, save_path):
    """Visualize score distributions and ROC curve"""
    plt.figure(figsize=(12, 5))
    
    # Score distributions
    plt.subplot(1, 2, 1)
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID (CIFAR10)', density=True)
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD', density=True)
    plt.xlabel('OOD Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Score Distributions')
    
    # ROC curve
    plt.subplot(1, 2, 2)
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_score = np.concatenate([id_scores, ood_scores])
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    
    plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    mnist_transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    print("Loading datasets...")
    cifar10_train = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    svhn_test = torchvision.datasets.SVHN(
        root=args.data_dir, split='test', download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=mnist_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        cifar10_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    svhn_loader = DataLoader(
        svhn_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    mnist_loader = DataLoader(
        mnist_test, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Create feature extractor
    feature_extractor, feature_dim = create_feature_extractor(args.model)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.eval()
    
    # Create and train detector
    detector = FixedDRODetector(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        batch_size=args.batch_size,
        device=args.device,
        use_hierarchical_features=args.hierarchical
    ).to(args.device)
    
    # Train on CIFAR10 vs SVHN
    print("\nTraining on CIFAR10 vs SVHN...")
    success = detector.fit(train_loader)
    
    if success:
        # Evaluate on SVHN
        print("\nEvaluating on SVHN...")
        svhn_metrics = evaluate_detector(detector, test_loader, svhn_loader, args.device)
        visualize_results(
            svhn_metrics['id_scores'],
            svhn_metrics['ood_scores'],
            os.path.join(args.save_dir, 'dro_results_svhn.png')
        )
        
        # Evaluate on MNIST if requested
        if args.ood_dataset in ['mnist', 'both']:
            print("\nEvaluating on MNIST...")
            mnist_metrics = evaluate_detector(detector, test_loader, mnist_loader, args.device)
            visualize_results(
                mnist_metrics['id_scores'],
                mnist_metrics['ood_scores'],
                os.path.join(args.save_dir, 'dro_results_mnist.png')
            )
        
        # Save model and results
        save_dict = {
            'model_state_dict': detector.state_dict(),
            'feature_extractor': feature_extractor.state_dict(),
            'svhn_metrics': svhn_metrics,
            'args': vars(args)
        }
        if args.ood_dataset in ['mnist', 'both']:
            save_dict['mnist_metrics'] = mnist_metrics
        
        torch.save(save_dict, os.path.join(args.save_dir, 'fixed_dro_detector.pth'))
        print(f"\nResults saved to {args.save_dir}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()