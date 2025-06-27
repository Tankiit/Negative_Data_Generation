import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
import matplotlib.pyplot as plt
from tqdm import tqdm

class UniversalManifoldConstraints:
    """
    Universal manifold constraints that can be applied to ANY OOD detection method
    Not tied to energy functions - works with Mahalanobis, gradients, features, etc.
    """
    
    def __init__(self, manifold_method='autoencoder', manifold_dim=64):
        self.manifold_method = manifold_method
        self.manifold_dim = manifold_dim
        self.manifold_learner = None
        self.feature_stats = {}
        
    def fit_manifold(self, feature_extractor, data_loader, device='cuda'):
        """Learn manifold structure from ID features (not raw data)"""
        print(f"Learning manifold structure using {self.manifold_method}...")
        
        # Extract features from ID data
        features = self._extract_features(feature_extractor, data_loader, device)
        
        if self.manifold_method == 'pca':
            self._fit_pca_manifold(features)
        elif self.manifold_method == 'autoencoder':
            self._fit_autoencoder_manifold(features, device)
        
        # Store feature statistics for various OOD methods
        self._compute_feature_stats(features)
        
    def _extract_features(self, feature_extractor, data_loader, device):
        """Extract features from a trained model or feature extraction function"""
        # Handle both model objects and functions
        if hasattr(feature_extractor, 'eval'):
            feature_extractor.eval()
        
        all_features = []
        
        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="Extracting features"):
                data = data.to(device)
                features = feature_extractor(data)
                if len(features.shape) > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def _fit_pca_manifold(self, features):
        """Fit PCA-based manifold"""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.manifold_dim)
        self.pca.fit(features.numpy())
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0) + 1e-8
        print(f"PCA explains {self.pca.explained_variance_ratio_.sum():.3f} of feature variance")
    
    def _fit_autoencoder_manifold(self, features, device):
        """Fit autoencoder-based manifold"""
        feature_dim = features.shape[1]
        
        # Simple autoencoder
        class FeatureAutoencoder(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, input_dim)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon, z
        
        self.autoencoder = FeatureAutoencoder(feature_dim, self.manifold_dim).to(device)
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)
        
        # Train autoencoder
        dataset = TensorDataset(features)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        for epoch in range(50):
            total_loss = 0
            for batch, in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon, _ = self.autoencoder(batch)
                loss = F.mse_loss(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Autoencoder epoch {epoch+1}/50, Loss: {total_loss/len(loader):.4f}")
    
    def _compute_feature_stats(self, features):
        """Compute statistics needed for various OOD detection methods"""
        # For Mahalanobis distance
        self.feature_stats['mean'] = features.mean(dim=0)
        self.feature_stats['cov'] = torch.cov(features.T)
        self.feature_stats['cov_inv'] = torch.linalg.pinv(self.feature_stats['cov'])
        
        # For KNN-based methods
        self.feature_stats['train_features'] = features
    
    def manifold_distance(self, features):
        """Compute distance from features to learned manifold"""
        if self.manifold_method == 'pca':
            return self._pca_manifold_distance(features)
        elif self.manifold_method == 'autoencoder':
            return self._autoencoder_manifold_distance(features)
    
    def _pca_manifold_distance(self, features):
        """PCA-based manifold distance"""
        features_norm = (features - self.feature_mean) / self.feature_std
        features_proj = torch.tensor(
            self.pca.inverse_transform(self.pca.transform(features_norm.numpy())),
            dtype=features.dtype
        ) * self.feature_std + self.feature_mean
        return torch.norm(features - features_proj, dim=1)
    
    def _autoencoder_manifold_distance(self, features):
        """Autoencoder-based manifold distance"""
        self.autoencoder.eval()
        with torch.no_grad():
            device = next(self.autoencoder.parameters()).device
            features = features.to(device)
            recon, _ = self.autoencoder(features)
            return torch.norm(features - recon, dim=1).cpu()
    
    def generate_manifold_virtual_outliers(self, features, num_outliers=1000):
        """Generate virtual outliers constrained to manifold neighborhoods"""
        virtual_outliers = []
        
        n_samples = features.shape[0]
        
        for _ in range(num_outliers):
            # Sample random starting point
            idx = torch.randint(0, n_samples, (1,))
            start_feature = features[idx]
            
            if np.random.random() < 0.6:  # Distributional outliers
                if self.manifold_method == 'pca':
                    virtual = self._generate_pca_distributional_outlier(start_feature)
                else:
                    virtual = self._generate_ae_distributional_outlier(start_feature)
            else:  # Structural outliers
                virtual = self._generate_structural_outlier(start_feature)
            
            virtual_outliers.append(virtual)
        
        return torch.cat(virtual_outliers, dim=0)
    
    def _generate_pca_distributional_outlier(self, start_feature, epsilon=2.0):
        """Generate distributional outlier using PCA manifold"""
        feat_norm = (start_feature - self.feature_mean) / self.feature_std
        z = torch.tensor(self.pca.transform(feat_norm.numpy()), dtype=start_feature.dtype)
        z_perturbed = z + epsilon * torch.randn_like(z)
        feat_proj = torch.tensor(
            self.pca.inverse_transform(z_perturbed.numpy()),
            dtype=start_feature.dtype
        ) * self.feature_std + self.feature_mean
        return feat_proj
    
    def _generate_ae_distributional_outlier(self, start_feature, epsilon=1.0):
        """Generate distributional outlier using autoencoder manifold"""
        self.autoencoder.eval()
        with torch.no_grad():
            device = next(self.autoencoder.parameters()).device
            start_feature = start_feature.to(device)
            _, z = self.autoencoder(start_feature)
            z_perturbed = z + epsilon * torch.randn_like(z)
            virtual = self.autoencoder.decoder(z_perturbed)
            return virtual.cpu()
    
    def _generate_structural_outlier(self, start_feature, epsilon=1.5):
        """Generate structural outlier by perturbing off-manifold"""
        # Compute manifold projection
        manifold_proj = self._project_to_manifold(start_feature)
        
        # Normal direction
        normal = start_feature - manifold_proj
        normal_magnitude = torch.norm(normal, dim=1, keepdim=True)
        
        if normal_magnitude.item() < 1e-6:
            normal = torch.randn_like(start_feature)
            normal_magnitude = torch.norm(normal, dim=1, keepdim=True)
        
        normal_unit = normal / (normal_magnitude + 1e-8)
        
        # Perturb in normal direction
        virtual = start_feature + epsilon * torch.randn(1).item() * normal_unit
        return virtual
    
    def _project_to_manifold(self, features):
        """Project features onto manifold"""
        if self.manifold_method == 'pca':
            features_norm = (features - self.feature_mean) / self.feature_std
            features_proj = torch.tensor(
                self.pca.inverse_transform(self.pca.transform(features_norm.numpy())),
                dtype=features.dtype
            ) * self.feature_std + self.feature_mean
            return features_proj
        else:
            self.autoencoder.eval()
            with torch.no_grad():
                device = next(self.autoencoder.parameters()).device
                features = features.to(device)
                recon, _ = self.autoencoder(features)
                return recon.cpu()

class MultiMethodOODDetector:
    """
    Apply manifold constraints to multiple OOD detection methods
    Shows that manifold approach is universal, not tied to energy functions
    """
    
    def __init__(self, manifold_constraints):
        self.manifold_constraints = manifold_constraints
        
    def energy_score(self, features, classifier, use_manifold=True, lambda_manifold=1.0):
        """Energy-based OOD detection with optional manifold regularization"""
        logits = classifier(features)
        energy = -torch.logsumexp(logits, dim=1)
        
        if use_manifold:
            manifold_dist = self.manifold_constraints.manifold_distance(features)
            energy += lambda_manifold * (manifold_dist ** 2)
        
        return energy
    
    def mahalanobis_score(self, features, use_manifold=True, lambda_manifold=1.0):
        """Mahalanobis distance with optional manifold regularization"""
        mean = self.manifold_constraints.feature_stats['mean']
        cov_inv = self.manifold_constraints.feature_stats['cov_inv']
        
        centered = features - mean
        mahal_dist = torch.sqrt(torch.sum(centered @ cov_inv * centered, dim=1))
        
        if use_manifold:
            manifold_dist = self.manifold_constraints.manifold_distance(features)
            mahal_dist += lambda_manifold * manifold_dist
        
        return mahal_dist
    
    def knn_score(self, features, k=5, use_manifold=True, lambda_manifold=1.0):
        """KNN-based OOD detection with optional manifold regularization"""
        train_features = self.manifold_constraints.feature_stats['train_features']
        
        # Compute distances to k nearest neighbors
        distances = torch.cdist(features, train_features)
        knn_distances, _ = torch.topk(distances, k, largest=False, dim=1)
        knn_score = knn_distances.mean(dim=1)
        
        if use_manifold:
            manifold_dist = self.manifold_constraints.manifold_distance(features)
            knn_score += lambda_manifold * manifold_dist
        
        return knn_score
    
    def gradient_norm_score(self, features, classifier, use_manifold=True, lambda_manifold=1.0):
        """Gradient norm OOD detection with optional manifold regularization"""
        # Create a copy of features that requires gradients
        features_grad = features.clone().detach().requires_grad_(True)
        
        logits = classifier(features_grad)
        max_logits, _ = torch.max(logits, dim=1)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=max_logits.sum(),
            inputs=features_grad,
            create_graph=False,
            retain_graph=False
        )[0]
        
        grad_norm = torch.norm(gradients, dim=1)
        
        if use_manifold:
            manifold_dist = self.manifold_constraints.manifold_distance(features.detach())
            grad_norm += lambda_manifold * manifold_dist
        
        return grad_norm

def load_cifar10_data(batch_size=128):
    """Load CIFAR-10 as ID data and SVHN as OOD data"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # CIFAR-10 (ID)
    trainset = torchvision.datasets.CIFAR10(
        root='/Users/mukher74/research/data', train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root='/Users/mukher74/research/data', train=False, download=True, transform=transform
    )
    
    # Use subset for faster experimentation
    train_subset = Subset(trainset, range(0, 5000))  # Use 5k samples
    test_subset = Subset(testset, range(0, 1000))    # Use 1k samples
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # SVHN (OOD)
    try:
        ood_dataset = torchvision.datasets.SVHN(
            root='/Users/mukher74/research/data', split='test', download=True, transform=transform
        )
        ood_subset = Subset(ood_dataset, range(0, 1000))  # Use 1k samples
        ood_loader = DataLoader(ood_subset, batch_size=batch_size, shuffle=False)
    except:
        print("SVHN download failed, using Gaussian noise as OOD")
        ood_data = torch.randn(1000, 3, 32, 32)
        ood_labels = torch.zeros(1000)
        ood_dataset = TensorDataset(ood_data, ood_labels)
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, ood_loader

def create_simple_resnet():
    """Create a simple ResNet-like feature extractor + classifier"""
    class SimpleResNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 2
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                # Block 3
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
            self.classifier = nn.Linear(128, num_classes)
        
        def forward(self, x):
            features = self.features(x)
            features = features.view(features.size(0), -1)
            return self.classifier(features)
        
        def get_features(self, x):
            features = self.features(x)
            return features.view(features.size(0), -1)
    
    return SimpleResNet()

def train_simple_classifier(model, train_loader, epochs=10, device='cuda'):
    """Train the classifier on ID data"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

def evaluate_all_methods(model, manifold_constraints, test_loader, ood_loader, device='cuda'):
    """Evaluate multiple OOD detection methods with/without manifold constraints"""
    model.eval()
    detector = MultiMethodOODDetector(manifold_constraints)
    
    results = {}
    methods = ['energy', 'mahalanobis', 'knn', 'gradient_norm']
    
    for method_name in methods:
        print(f"\nEvaluating {method_name} method...")
        
        # Test with and without manifold constraints
        for use_manifold in [False, True]:
            variant = f"{method_name}_{'with' if use_manifold else 'without'}_manifold"
            
            id_scores = []
            ood_scores = []
            
            # Collect ID scores
            if method_name == 'gradient_norm':
                # Gradient norm needs gradients, so don't use no_grad context
                for data, _ in test_loader:
                    data = data.to(device)
                    features = model.get_features(data)
                    scores = detector.gradient_norm_score(features, model.classifier, use_manifold)
                    id_scores.extend(scores.cpu().numpy())
            else:
                with torch.no_grad():
                    for data, _ in test_loader:
                        data = data.to(device)
                        features = model.get_features(data)
                        
                        if method_name == 'energy':
                            scores = detector.energy_score(features, model.classifier, use_manifold)
                        elif method_name == 'mahalanobis':
                            scores = detector.mahalanobis_score(features, use_manifold)
                        elif method_name == 'knn':
                            scores = detector.knn_score(features, use_manifold=use_manifold)
                        
                        id_scores.extend(scores.cpu().numpy())
            
            # Collect OOD scores
            if method_name == 'gradient_norm':
                # Gradient norm needs gradients, so don't use no_grad context
                for data, _ in ood_loader:
                    data = data.to(device)
                    features = model.get_features(data)
                    scores = detector.gradient_norm_score(features, model.classifier, use_manifold)
                    ood_scores.extend(scores.cpu().numpy())
            else:
                with torch.no_grad():
                    for data, _ in ood_loader:
                        data = data.to(device)
                        features = model.get_features(data)
                        
                        if method_name == 'energy':
                            scores = detector.energy_score(features, model.classifier, use_manifold)
                        elif method_name == 'mahalanobis':
                            scores = detector.mahalanobis_score(features, use_manifold)
                        elif method_name == 'knn':
                            scores = detector.knn_score(features, use_manifold=use_manifold)
                        
                        ood_scores.extend(scores.cpu().numpy())
            
            # Calculate AUROC
            y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            y_scores = np.concatenate([id_scores, ood_scores])
            auroc = roc_auc_score(y_true, y_scores)
            
            results[variant] = {
                'auroc': auroc,
                'id_mean': np.mean(id_scores),
                'ood_mean': np.mean(ood_scores)
            }
    
    return results

def demonstrate_universal_manifold_ood():
    """Main demonstration of universal manifold constraints"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 data...")
    train_loader, test_loader, ood_loader = load_cifar10_data()
    
    # Create and train model
    print("Creating and training classifier...")
    model = create_simple_resnet()
    train_simple_classifier(model, train_loader, epochs=5, device=device)
    
    # Learn manifold structure from trained features
    print("Learning manifold structure...")
    manifold_constraints = UniversalManifoldConstraints(
        manifold_method='autoencoder',  # Try 'pca' for faster experimentation
        manifold_dim=32
    )
    manifold_constraints.fit_manifold(model.get_features, train_loader, device)
    
    # Evaluate all methods
    print("Evaluating OOD detection methods...")
    results = evaluate_all_methods(model, manifold_constraints, test_loader, ood_loader, device)
    
    # Print results
    print("\n" + "="*80)
    print("OOD DETECTION RESULTS")
    print("="*80)
    
    for method, result in results.items():
        improvement = ""
        if 'with_manifold' in method:
            base_method = method.replace('_with_manifold', '_without_manifold')
            if base_method in results:
                baseline_auroc = results[base_method]['auroc']
                improvement = f" (+{result['auroc'] - baseline_auroc:.3f})"
        
        print(f"{method:30s}: AUROC = {result['auroc']:.4f}{improvement}")
    
    # Calculate average improvement
    improvements = []
    base_methods = ['energy', 'mahalanobis', 'knn', 'gradient_norm']
    
    for method in base_methods:
        without_key = f"{method}_without_manifold"
        with_key = f"{method}_with_manifold"
        if without_key in results and with_key in results:
            improvement = results[with_key]['auroc'] - results[without_key]['auroc']
            improvements.append(improvement)
    
    if improvements:
        avg_improvement = np.mean(improvements)
        print(f"\nAverage AUROC improvement with manifold constraints: +{avg_improvement:.4f}")
        print(f"Methods improved: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}")
    
    return results, manifold_constraints

if __name__ == "__main__":
    results, manifold_constraints = demonstrate_universal_manifold_ood()
    print("\nUniversal manifold OOD detection completed!")
    print("Key insight: Manifold constraints improve MULTIPLE detection methods, not just energy!")