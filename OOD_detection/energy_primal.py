import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cvxpy as cp
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
from collections import defaultdict

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
        self.feature_extractor = feature_extractor
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_hierarchical_features = use_hierarchical_features
        
        # Initialize parameters
        self.register_buffer('theta', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('feature_mean', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('feature_std', torch.ones(feature_dim, dtype=torch.float32))
        self.register_buffer('precision_matrix', torch.eye(feature_dim, dtype=torch.float32))
        
        # Energy-based parameters
        self.energy_temperature = energy_temperature
        self.energy_weight = energy_weight
        
        # Create classifier head for energy scoring
        self.classifier = nn.Linear(feature_dim, num_classes).to(device)
        
        # Energy calibration parameters
        self.register_buffer('energy_mean_id', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('energy_std_id', torch.tensor(1.0, dtype=torch.float32))
        
    def extract_hierarchical_features(self, x):
        """Extract hierarchical features from different network layers"""
        if not self.use_hierarchical_features:
            features = self.feature_extractor(x.to(self.device))
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
            return self.feature_extractor(x.to(self.device)).to(dtype=torch.float32)
    
    def compute_energy(self, features, temperature=None):
        """
        Compute free energy from features
        E(x) = -T * log(sum_i(exp(f_i(x)/T)))
        """
        if temperature is None:
            temperature = self.energy_temperature
            
        logits = self.classifier(features)
        return -temperature * torch.logsumexp(logits / temperature, dim=1)
    
    def solve_dro_primal_optimization(self, features, radius=10.0, reg_param=1.0):
        """
        Solve DRO optimization using the primal form.
        """
        # Convert features to numpy for cvxpy
        features_np = features.detach().cpu().numpy().astype(np.float32)
        n_samples, n_features = features_np.shape

        # Define variables
        theta = cp.Variable(n_features)
        bias = cp.Variable()

        # Define the nominal loss (hinge loss)
        predictions = features_np @ theta + bias
        nominal_loss = cp.sum(cp.pos(1.0 - predictions)) / n_samples

        # Define the Wasserstein constraint
        # The Wasserstein distance is approximated by the norm of theta
        wasserstein_constraint = cp.norm(theta, 2) <= radius

        # Define the regularization term
        regularization = reg_param * cp.sum_squares(theta)

    # Total objective
        objective = nominal_loss + regularization

        # Create problem with the Wasserstein constraint
        problem = cp.Problem(cp.Minimize(objective), [wasserstein_constraint])

        try:
            # Solve with cvxpy
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
    
    def train_energy_classifier(self, train_loader, val_loader=None, num_epochs=5, lr=0.001):
        """
        Train the energy classifier head
        """
        print("\nTraining energy-based classifier...")
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.classifier.train()
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    features = self.extract_hierarchical_features(images)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.classifier(features)
                loss = criterion(logits, labels)
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            # Print epoch results
            avg_loss = total_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
            
            # Validation, if provided
            if val_loader is not None and (epoch+1) % 2 == 0:
                self.evaluate_classifier(val_loader)
        
        # Calibrate energy scores after training
        self.calibrate_energy_scores(train_loader)
        
        return self.classifier
    
    def evaluate_classifier(self, loader):
        """Evaluate classifier performance"""
        self.classifier.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                features = self.extract_hierarchical_features(images)
                logits = self.classifier(features)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        return accuracy
    
    def calibrate_energy_scores(self, train_loader):
        """Calibrate energy scores using in-distribution data"""
        print("\nCalibrating energy scores...")
        all_energies = []
        
        self.classifier.eval()
        with torch.no_grad():
            for images, _ in tqdm(train_loader, desc="Computing energy statistics"):
                images = images.to(self.device)
                features = self.extract_hierarchical_features(images)
                energies = self.compute_energy(features)
                all_energies.append(energies.cpu())
        
        all_energies = torch.cat(all_energies)
        self.energy_mean_id = all_energies.mean()
        self.energy_std_id = all_energies.std()
        
        print(f"Energy calibration: Mean={self.energy_mean_id.item():.4f}, Std={self.energy_std_id.item():.4f}")
    
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
                        
                        theta, bias, loss = self.solve_dro_primal_optimization(norm_features, radius, reg)
                        
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
        raw_energy = self.compute_energy(features)
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
    axes[0, 0].hist(id_scores, bins=50, alpha=0.5, label='ID (CIFAR10)', density=True)
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

def create_feature_extractor(model_name='resnet18', num_classes=10):
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
            num_classes=num_classes,
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced DRO-Energy-based OOD Detection")
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='/Users/tanmoy/research/data', 
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'wideresnet'], 
                        help='Model architecture')
    parser.add_argument('--hierarchical', action='store_true', 
                        help='Use hierarchical features')
    
    # Training parameters
    parser.add_argument('--energy-epochs', type=int, default=5,
                        help='Number of epochs to train energy classifier')
    parser.add_argument('--energy-weight', type=float, default=0.5,
                        help='Weight for energy score in combined score')
    parser.add_argument('--energy-temperature', type=float, default=1.0,
                        help='Temperature for energy score calculation')
    
    # Component weights
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
    parser.add_argument('--detailed-eval', action='store_true',
                        help='Perform detailed evaluation of components')
    
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

def main():
    """Main function to run DRO-Energy OOD detection"""
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
    
    # Create detector
    detector = DROEnergyDetector(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=10,
        batch_size=args.batch_size,
        device=args.device,
        use_hierarchical_features=args.hierarchical,
        energy_temperature=args.energy_temperature,
        energy_weight=args.energy_weight
    ).to(args.device)
    
    # Train detector
    print("\nTraining DRO-Energy detector...")
    success = detector.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        energy_epochs=args.energy_epochs
    )
    
    if success:
        # Evaluate on SVHN
        print("\nEvaluating on SVHN...")
        svhn_results = evaluate_detector(
            detector, test_loader, svhn_loader, args.device, 
            detailed=args.detailed_eval
        )
        
        # Save visualization
        visualize_results(
            svhn_results,
            os.path.join(args.save_dir, 'dro_energy_results_svhn.png'),
            title="CIFAR10 vs SVHN Detection Performance"
        )
        
        # Evaluate on MNIST if requested
        if args.ood_dataset in ['mnist', 'both']:
            print("\nEvaluating on MNIST...")
            mnist_results = evaluate_detector(
                detector, test_loader, mnist_loader, args.device,
                detailed=args.detailed_eval
            )
            
            # Save visualization
            visualize_results(
                mnist_results,
                os.path.join(args.save_dir, 'dro_energy_results_mnist.png'),
                title="CIFAR10 vs MNIST Detection Performance"
            )
        
        # Save model and results
        save_dict = {
            'model_state_dict': detector.state_dict(),
            'feature_extractor': feature_extractor.state_dict(),
            'svhn_results': svhn_results,
            'args': vars(args)
        }
        
        if args.ood_dataset in ['mnist', 'both']:
            save_dict['mnist_results'] = mnist_results
        
        torch.save(save_dict, os.path.join(args.save_dir, 'dro_energy_detector.pth'))
        print(f"\nResults saved to {args.save_dir}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()

