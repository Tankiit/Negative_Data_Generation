import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics
from pytorch_ood.model import WideResNet
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class FeatureManipulationPipeline:
    """
    Pipeline for feature manipulation using WideResNet and CIFAR-10
    """
    def __init__(self, num_classes=10, epsilon=0.1):
        # Initialize WideResNet model
        self.model = WideResNet(num_classes=num_classes, 
                              pretrained="er-cifar10-tune").eval().cuda()
        self.preprocess = WideResNet.transform_for("er-cifar10-tune")
        self.detector = EnergyBased(self.model)
        self.epsilon = epsilon
        
        # Statistical models for feature distribution
        self.pca = None
        self.gmm = None
        self.feature_stats = {}
        
    def setup_data(self, batch_size=128):
        """Setup CIFAR-10 data loaders"""
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # CIFAR-10 datasets
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=2
        )
        
    def extract_features(self, x):
        """Extract features from WideResNet"""
        x = self.preprocess(x).cuda()
        with torch.no_grad():
            # Get features from the penultimate layer
            features = []
            def hook(model, input, output):
                features.append(output.detach())
            
            # Register hook for the layer before classification
            handle = list(self.model.modules())[-2].register_forward_hook(hook)
            _ = self.model(x)
            handle.remove()
            
            return features[0]
    
    def fit_feature_distribution(self):
        """Fit statistical models to feature distribution"""
        print("Extracting features for distribution fitting...")
        features = []
        labels = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx}/{len(self.train_loader)}")
                feat = self.extract_features(x)
                features.append(feat.cpu())
                labels.append(y)
                
        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)
        
        print("Fitting PCA...")
        # Fit PCA
        self.pca = PCA(n_components=min(features.shape[1], 128))
        pca_features = self.pca.fit_transform(features.numpy())
        
        print("Fitting GMM...")
        # Fit GMM in PCA space
        self.gmm = GaussianMixture(n_components=10, covariance_type='full')
        self.gmm.fit(pca_features)
        
        # Store statistics
        self.feature_stats = {
            'mean': torch.tensor(features.mean(axis=0)),
            'std': torch.tensor(features.std(axis=0)),
            'pca_components': torch.tensor(self.pca.components_),
            'explained_var': torch.tensor(self.pca.explained_variance_ratio_)
        }
        
        return self.feature_stats
    
    def generate_perturbed_features(self, x, method='pca', scale=1.0):
        """Generate perturbed features using various methods"""
        original_features = self.extract_features(x)
        
        if method == 'pca':
            # PCA-based perturbation
            pca_coords = torch.matmul(
                original_features - self.feature_stats['mean'],
                self.feature_stats['pca_components'].t().cuda()
            )
            perturbation = torch.randn_like(pca_coords) * scale * self.epsilon
            perturbation *= torch.sqrt(self.feature_stats['explained_var'].cuda())
            
            perturbed_coords = pca_coords + perturbation
            perturbed_features = torch.matmul(
                perturbed_coords,
                self.feature_stats['pca_components'].cuda()
            ) + self.feature_stats['mean'].cuda()
            
        elif method == 'gaussian':
            # Direct Gaussian perturbation
            noise = torch.randn_like(original_features) * self.epsilon * scale
            perturbed_features = original_features + noise
            
        elif method == 'adversarial':
            # Generate adversarial perturbation
            perturbed_features = original_features.clone().detach().requires_grad_(True)
            
            # Maximize energy score
            energy_scores = self.detector(self.model.features_to_logits(perturbed_features))
            loss = -energy_scores.mean()
            
            grad = torch.autograd.grad(loss, perturbed_features)[0]
            perturbed_features = original_features + scale * self.epsilon * grad.sign()
            
        return perturbed_features
    
    def evaluate_ood_detection(self, features, labels, metrics=None):
        """Evaluate OOD detection performance"""
        if metrics is None:
            metrics = OODMetrics()
        
        # Convert features back to logits
        logits = self.model.features_to_logits(features)
        
        # Compute energy scores
        scores = self.detector(logits)
        metrics.update(scores, labels)
        
        return metrics.compute()

def run_pipeline():
    """Run the complete pipeline"""
    # Initialize pipeline
    pipeline = FeatureManipulationPipeline()
    pipeline.setup_data()
    
    # Fit feature distribution
    feature_stats = pipeline.fit_feature_distribution()
    
    # Initialize metrics
    metrics = OODMetrics()
    
    # Evaluate different perturbation methods
    perturbation_methods = ['pca', 'gaussian', 'adversarial']
    results = {}
    
    for method in perturbation_methods:
        print(f"\nEvaluating {method} perturbation...")
        batch_metrics = OODMetrics()
        
        for x, y in pipeline.test_loader:
            # Generate perturbed features
            perturbed_features = pipeline.generate_perturbed_features(x, method=method)
            
            # Evaluate
            results[method] = pipeline.evaluate_ood_detection(
                perturbed_features, y, batch_metrics
            )
    
    return results

if __name__ == "__main__":
    # Run the pipeline
    results = run_pipeline()
    print("\nResults for different perturbation methods:")
    for method, metrics in results.items():
        print(f"\n{method.upper()} Perturbation:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")