import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Uniform
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

class FeatureManipulator:
    """
    Implements various feature-space manipulations for uncertainty set construction
    """
    def __init__(self, model, feature_dim):
        self.model = model
        self.feature_dim = feature_dim
        self.pca = None
        self.gmm = None
        self.feature_stats = {}
        
    def extract_features(self, x):
        """Extract intermediate features from model"""
        with torch.no_grad():
            features = self.model.extract_features(x)
        return features
    
    def fit_feature_distribution(self, dataloader):
        """Fit statistical models to feature distribution"""
        features = []
        for x, _ in dataloader:
            x = x.cuda()
            feat = self.extract_features(x)
            features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        
        # Fit PCA
        self.pca = PCA(n_components=min(features.shape[1], 32))
        pca_features = self.pca.fit_transform(features)
        
        # Fit GMM in PCA space
        self.gmm = GaussianMixture(n_components=5, covariance_type='full')
        self.gmm.fit(pca_features)
        
        # Store statistics
        self.feature_stats = {
            'mean': torch.tensor(features.mean(axis=0)),
            'std': torch.tensor(features.std(axis=0)),
            'pca_components': torch.tensor(self.pca.components_),
            'principal_vars': torch.tensor(self.pca.explained_variance_)
        }

class FeatureSpaceUncertaintySet:
    """
    Creates uncertainty sets through feature-space manipulations
    """
    def __init__(self, feature_manipulator, epsilon=0.1):
        self.feature_manipulator = feature_manipulator
        self.epsilon = epsilon
        
    def generate_principal_perturbations(self, features, scale=1.0):
        """Generate perturbations along principal components"""
        batch_size = features.size(0)
        pca_components = self.feature_manipulator.feature_stats['pca_components']
        principal_vars = self.feature_manipulator.feature_stats['principal_vars']
        
        # Sample perturbation magnitudes
        perturbation_scale = torch.sqrt(principal_vars) * scale * self.epsilon
        perturbations = torch.randn(batch_size, len(principal_vars)) * perturbation_scale
        
        # Project perturbations back to feature space
        feature_perturbations = torch.matmul(perturbations, pca_components)
        return features + feature_perturbations.to(features.device)
    
    def generate_adversarial_features(self, features, target_features):
        """Generate adversarial feature perturbations"""
        delta = torch.zeros_like(features, requires_grad=True)
        
        optimizer = torch.optim.Adam([delta], lr=0.01)
        
        for _ in range(10):  # Optimization steps
            optimizer.zero_grad()
            
            perturbed_features = features + delta
            loss = F.mse_loss(perturbed_features, target_features)
            
            # Add norm constraint
            norm_penalty = torch.norm(delta, p=2, dim=1).mean()
            total_loss = loss + 0.1 * norm_penalty
            
            total_loss.backward()
            optimizer.step()
            
            # Project to epsilon ball
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
        
        return features + delta.detach()
    
    def interpolate_features(self, features, num_samples=2):
        """Generate interpolated feature vectors"""
        batch_size = features.size(0)
        
        # Random pairs for interpolation
        idx1 = torch.randperm(batch_size)
        idx2 = torch.randperm(batch_size)
        
        # Generate random interpolation coefficients
        alphas = torch.rand(batch_size, 1).to(features.device)
        
        # Interpolate
        interpolated = alphas * features[idx1] + (1 - alphas) * features[idx2]
        return interpolated

class FeatureAwareDRO:
    """
    DRO training incorporating feature-space manipulations
    """
    def __init__(self, model, feature_manipulator, uncertainty_set):
        self.model = model
        self.feature_manipulator = feature_manipulator
        self.uncertainty_set = uncertainty_set
        
    def compute_feature_robust_loss(self, x, y):
        """Compute robust loss considering feature-space uncertainty"""
        # Extract original features
        original_features = self.feature_manipulator.extract_features(x)
        
        # Generate perturbed features
        perturbed_features = []
        
        # 1. Principal component perturbations
        pca_perturbed = self.uncertainty_set.generate_principal_perturbations(
            original_features
        )
        perturbed_features.append(pca_perturbed)
        
        # 2. Adversarial features
        target_features = original_features + torch.randn_like(original_features) * 0.1
        adv_perturbed = self.uncertainty_set.generate_adversarial_features(
            original_features, target_features
        )
        perturbed_features.append(adv_perturbed)
        
        # 3. Interpolated features
        interp_features = self.uncertainty_set.interpolate_features(original_features)
        perturbed_features.append(interp_features)
        
        # Combine all perturbed features
        all_features = torch.cat([original_features] + perturbed_features, dim=0)
        
        # Forward pass through remaining layers
        logits = self.model.features_to_logits(all_features)
        
        # Compute worst-case loss
        expanded_y = y.repeat(len(perturbed_features) + 1)
        losses = F.cross_entropy(logits, expanded_y, reduction='none')
        losses = losses.view(len(perturbed_features) + 1, -1)
        
        return torch.max(losses, dim=0)[0].mean()

class CombinedUncertaintyDRO:
    """
    Combines input-space transformations with feature-space manipulations
    """
    def __init__(self, model, transform_uncertainty, feature_uncertainty):
        self.model = model
        self.transform_uncertainty = transform_uncertainty
        self.feature_uncertainty = feature_uncertainty
        
    def train_step(self, x, y, optimizer):
        optimizer.zero_grad()
        
        # 1. Generate transformed samples
        transformed_x = self.transform_uncertainty.generate_uncertain_batch(x)
        
        # 2. Extract and manipulate features
        features = self.model.extract_features(transformed_x)
        perturbed_features = self.feature_uncertainty.generate_uncertain_features(features)
        
        # 3. Forward pass and loss computation
        logits = self.model.features_to_logits(perturbed_features)
        expanded_y = y.repeat(logits.size(0) // y.size(0))
        
        # 4. Compute robust loss
        loss = F.cross_entropy(logits, expanded_y)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

def demonstrate_feature_manipulation():
    """
    Demonstrate feature manipulation effects
    """
    model = YourModel()
    feature_dim = 512  # Adjust based on your model
    
    # Initialize components
    feature_manipulator = FeatureManipulator(model, feature_dim)
    uncertainty_set = FeatureSpaceUncertaintySet(feature_manipulator)
    
    # Fit feature distribution
    feature_manipulator.fit_feature_distribution(train_loader)
    
    # Example usage
    x, y = next(iter(train_loader))
    x = x.cuda()
    
    # Extract and manipulate features
    original_features = feature_manipulator.extract_features(x)
    
    # Generate different types of perturbed features
    pca_perturbed = uncertainty_set.generate_principal_perturbations(original_features)
    interp_features = uncertainty_set.interpolate_features(original_features)
    
    # Visualize or analyze the differences
    feature_differences = {
        'pca_diff': (pca_perturbed - original_features).norm(dim=1).mean(),
        'interp_diff': (interp_features - original_features).norm(dim=1).mean(),
    }
    
    return feature_differences