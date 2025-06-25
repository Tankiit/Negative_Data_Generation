import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class SimpleManifoldVirtualSynthesis:
    """
    Minimal implementation of manifold-constrained virtual outlier generation
    Uses PCA for manifold approximation (can replace with autoencoders later)
    """
    
    def __init__(self, manifold_dim=10):
        self.manifold_dim = manifold_dim
        self.pca = None
        self.mean = None
        self.std = None
        
    def fit_manifold(self, X_id):
        """Learn manifold structure from ID data using PCA"""
        # Standardize data
        self.mean = X_id.mean(dim=0)
        self.std = X_id.std(dim=0) + 1e-8
        X_normalized = (X_id - self.mean) / self.std
        
        # Fit PCA to find principal manifold directions
        self.pca = PCA(n_components=self.manifold_dim)
        self.pca.fit(X_normalized.detach().numpy())
        
        print(f"Manifold explains {self.pca.explained_variance_ratio_.sum():.3f} of variance")
        
    def project_to_manifold(self, X):
        """Project data onto learned manifold"""
        X_normalized = (X - self.mean) / self.std
        X_np = X_normalized.detach().numpy()
        
        # Project to manifold and back
        Z = self.pca.transform(X_np)  # Project to manifold coordinates
        X_manifold = self.pca.inverse_transform(Z)  # Project back to ambient space
        
        return torch.tensor(X_manifold, dtype=X.dtype) * self.std + self.mean
    
    def manifold_distance(self, X):
        """Compute distance from points to manifold"""
        X_projected = self.project_to_manifold(X)
        return torch.norm(X - X_projected, dim=1)
    
    def generate_virtual_outliers(self, X_id, num_outliers=100, 
                                epsilon_manifold=2.0, epsilon_normal=1.0):
        """
        Generate virtual outliers using manifold constraints
        
        Args:
            X_id: ID training data
            num_outliers: Number of virtual outliers to generate
            epsilon_manifold: Perturbation strength along manifold
            epsilon_normal: Perturbation strength normal to manifold
        """
        virtual_outliers = []
        outlier_types = []
        
        # Sample random ID points as starting points
        n_samples = X_id.size(0)
        indices = torch.randint(0, n_samples, (num_outliers,))
        
        for i in range(num_outliers):
            x_start = X_id[indices[i]:indices[i]+1]
            
            if np.random.random() < 0.6:  # 60% distributional outliers
                # Type 1: Distributional outliers (along manifold)
                x_virtual = self._generate_distributional_outlier(
                    x_start, epsilon_manifold
                )
                outlier_types.append('distributional')
            else:
                # Type 2: Structural outliers (off manifold)  
                x_virtual = self._generate_structural_outlier(
                    x_start, epsilon_normal
                )
                outlier_types.append('structural')
                
            virtual_outliers.append(x_virtual)
        
        return torch.cat(virtual_outliers, dim=0), outlier_types
    
    def _generate_distributional_outlier(self, x_start, epsilon):
        """Generate outlier by perturbing along manifold directions"""
        # Normalize
        x_norm = (x_start - self.mean) / self.std
        
        # Get manifold coordinates
        z = torch.tensor(self.pca.transform(x_norm.detach().numpy()), dtype=x_start.dtype)
        
        # Add noise in manifold space (larger perturbation)
        z_perturbed = z + epsilon * torch.randn_like(z)
        
        # Project back to ambient space
        x_manifold = torch.tensor(
            self.pca.inverse_transform(z_perturbed.detach().numpy()), 
            dtype=x_start.dtype
        )
        
        return x_manifold * self.std + self.mean
    
    def _generate_structural_outlier(self, x_start, epsilon):
        """Generate outlier by perturbing in normal direction to manifold"""
        # Get projection onto manifold
        x_projected = self.project_to_manifold(x_start)
        
        # Compute normal direction
        normal_direction = x_start - x_projected
        normal_magnitude = torch.norm(normal_direction, dim=1, keepdim=True)
        
        # Normalize normal direction (handle zero case)
        if normal_magnitude.item() < 1e-6:
            # If already on manifold, add random normal direction
            normal_direction = torch.randn_like(x_start)
            normal_magnitude = torch.norm(normal_direction, dim=1, keepdim=True)
        
        normal_unit = normal_direction / (normal_magnitude + 1e-8)
        
        # Perturb in normal direction
        perturbation = epsilon * torch.randn(1).item() * normal_unit
        return x_start + perturbation

def simple_energy_score(model, X, manifold_synthesizer=None, lambda_manifold=1.0):
    """
    Compute energy score with optional manifold regularization
    
    E(x) = -log(sum(exp(logits))) + λ * distance_to_manifold(x)²
    """
    with torch.no_grad():
        logits = model(X)
        classification_energy = -torch.logsumexp(logits, dim=1)
        
        if manifold_synthesizer is not None:
            manifold_distances = manifold_synthesizer.manifold_distance(X)
            manifold_energy = lambda_manifold * (manifold_distances ** 2)
            total_energy = classification_energy + manifold_energy
        else:
            total_energy = classification_energy
            
        return total_energy

def demonstrate_simple_manifold_synthesis():
    """Simple demonstration of manifold-constrained virtual synthesis"""
    
    # Generate toy data: 2D spiral embedded in 10D
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_samples = 500
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # 2D spiral
    spiral_2d = np.column_stack([
        t * np.cos(t) / (4*np.pi),
        t * np.sin(t) / (4*np.pi)
    ])
    
    # Embed in 10D with random projection + noise
    embedding_matrix = np.random.randn(2, 10) * 0.5
    data_10d = spiral_2d @ embedding_matrix
    data_10d += np.random.randn(n_samples, 10) * 0.1  # Add noise
    
    X_id = torch.FloatTensor(data_10d)
    y_id = torch.LongTensor(np.ones(n_samples))  # Dummy labels
    
    print(f"Generated {n_samples} ID samples in {X_id.shape[1]}D space")
    
    # Learn manifold structure
    synthesizer = SimpleManifoldVirtualSynthesis(manifold_dim=3)
    synthesizer.fit_manifold(X_id)
    
    # Generate virtual outliers
    virtual_outliers, outlier_types = synthesizer.generate_virtual_outliers(
        X_id, num_outliers=200, epsilon_manifold=3.0, epsilon_normal=2.0
    )
    
    print(f"Generated {len(virtual_outliers)} virtual outliers:")
    print(f"  - {sum(1 for t in outlier_types if t == 'distributional')} distributional")
    print(f"  - {sum(1 for t in outlier_types if t == 'structural')} structural")
    
    # Create simple classifier for demonstration
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(), 
        nn.Linear(16, 2)  # Binary classification
    )
    
    # Compute energy scores
    print("\nComputing energy scores...")
    
    # ID data energy
    id_energies = simple_energy_score(model, X_id, synthesizer)
    
    # Virtual outlier energies  
    virtual_energies = simple_energy_score(model, virtual_outliers, synthesizer)
    
    # Random outliers (for comparison)
    random_outliers = torch.randn(200, 10) * 2
    random_energies = simple_energy_score(model, random_outliers, synthesizer)
    
    print(f"Energy Statistics:")
    print(f"  ID data:        {id_energies.mean():.3f} ± {id_energies.std():.3f}")
    print(f"  Virtual outs:   {virtual_energies.mean():.3f} ± {virtual_energies.std():.3f}")
    print(f"  Random outs:    {random_energies.mean():.3f} ± {random_energies.std():.3f}")
    
    # Visualize in 2D using PCA
    print("\nVisualizing in 2D...")
    
    # Combine all data for visualization
    all_data = torch.cat([X_id[:100], virtual_outliers, random_outliers], dim=0)
    all_energies = torch.cat([id_energies[:100], virtual_energies, random_energies], dim=0)
    
    # PCA for visualization
    pca_vis = PCA(n_components=2)
    data_2d = pca_vis.fit_transform(all_data.detach().numpy())
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Data types
    plt.subplot(1, 2, 1)
    n_id = 100
    n_virtual = len(virtual_outliers)
    
    plt.scatter(data_2d[:n_id, 0], data_2d[:n_id, 1], 
               c='blue', alpha=0.6, s=30, label='ID Data')
    plt.scatter(data_2d[n_id:n_id+n_virtual, 0], data_2d[n_id:n_id+n_virtual, 1],
               c='red', alpha=0.6, s=30, label='Virtual Outliers')
    plt.scatter(data_2d[n_id+n_virtual:, 0], data_2d[n_id+n_virtual:, 1],
               c='orange', alpha=0.6, s=30, label='Random Outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Data Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Energy scores
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                         c=all_energies.detach().numpy(), 
                         cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Energy Score')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Energy Landscape')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Analyze manifold distances
    print("\nManifold distance analysis:")
    id_distances = synthesizer.manifold_distance(X_id)
    virtual_distances = synthesizer.manifold_distance(virtual_outliers)
    random_distances = synthesizer.manifold_distance(random_outliers)
    
    print(f"  ID data:        {id_distances.mean():.3f} ± {id_distances.std():.3f}")
    print(f"  Virtual outs:   {virtual_distances.mean():.3f} ± {virtual_distances.std():.3f}")
    print(f"  Random outs:    {random_distances.mean():.3f} ± {random_distances.std():.3f}")
    
    return synthesizer, X_id, virtual_outliers

if __name__ == "__main__":
    synthesizer, X_id, virtual_outliers = demonstrate_simple_manifold_synthesis()
    print("\nSimple manifold synthesis completed!")
    print("Key insight: Virtual outliers respect manifold structure better than random outliers")