import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class ManifoldLearner(nn.Module):
    """Learn data manifold structure using differentiable autoencoders"""
    
    def __init__(self, input_dim, manifold_dim=32, hidden_dims=[256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.manifold_dim = manifold_dim
        
        # Encoder: data -> manifold
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, manifold_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder: manifold -> data  
        decoder_layers = []
        prev_dim = manifold_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Map data to manifold coordinates"""
        return self.encoder(x)
    
    def decode(self, z):
        """Map manifold coordinates back to data space"""
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def manifold_distance(self, x):
        """Compute distance from point to learned manifold"""
        x_recon, z = self.forward(x)
        return torch.norm(x - x_recon, dim=1)

class ManifoldConstrainedVirtualSynthesis:
    """Generate virtual outliers constrained to manifold neighborhoods"""
    
    def __init__(self, manifold_learner, epsilon_manifold=0.5, epsilon_normal=2.0):
        self.manifold_learner = manifold_learner
        self.epsilon_manifold = epsilon_manifold  # How far along manifold
        self.epsilon_normal = epsilon_normal      # How far off manifold
        
    def generate_virtual_outliers(self, x_id, num_outliers_per_sample=2):
        """
        Generate virtual outliers using manifold-constrained sampling
        
        Strategy:
        1. For each ID sample, encode to manifold space
        2. Perturb in manifold space (distributional outliers)
        3. Perturb in normal direction (structural outliers)
        4. Decode back to data space
        """
        virtual_outliers = []
        outlier_types = []
        
        with torch.no_grad():
            # Get manifold embeddings
            x_recon, z_id = self.manifold_learner(x_id)
            
            for i in range(x_id.size(0)):
                x_sample = x_id[i:i+1]
                z_sample = z_id[i:i+1]
                x_recon_sample = x_recon[i:i+1]
                
                for _ in range(num_outliers_per_sample):
                    if np.random.random() < 0.5:
                        # Type 1: Distributional outliers (perturb in manifold space)
                        z_perturbed = self._perturb_on_manifold(z_sample)
                        x_virtual = self.manifold_learner.decode(z_perturbed)
                        outlier_types.append('distributional')
                    else:
                        # Type 2: Structural outliers (perturb in normal direction)
                        x_virtual = self._perturb_off_manifold(x_sample, x_recon_sample)
                        outlier_types.append('structural')
                    
                    virtual_outliers.append(x_virtual)
        
        return torch.cat(virtual_outliers, dim=0), outlier_types
    
    def _perturb_on_manifold(self, z):
        """Perturb in manifold (latent) space - distributional outliers"""
        # Add Gaussian noise in manifold coordinates
        noise = torch.randn_like(z) * self.epsilon_manifold
        return z + noise
    
    def _perturb_off_manifold(self, x_original, x_reconstructed):
        """Perturb in normal direction to manifold - structural outliers"""
        # Normal direction = (original - reconstruction)
        normal_direction = x_original - x_reconstructed
        normal_magnitude = torch.norm(normal_direction, dim=1, keepdim=True)
        
        # Avoid division by zero
        normal_direction = normal_direction / (normal_magnitude + 1e-8)
        
        # Perturb in normal direction
        perturbation = normal_direction * self.epsilon_normal * torch.randn(1).item()
        return x_original + perturbation

class GeometricEnergyFunction(nn.Module):
    """Energy function that respects manifold geometry"""
    
    def __init__(self, classifier, manifold_learner, lambda_manifold=1.0):
        super().__init__()
        self.classifier = classifier
        self.manifold_learner = manifold_learner
        self.lambda_manifold = lambda_manifold
        
    def forward(self, x):
        """
        Compute geometric energy: E(x) = E_classification(x) + λ * E_manifold(x)
        """
        # Standard classification energy
        logits = self.classifier(x)
        E_classification = -torch.logsumexp(logits, dim=1)
        
        # Manifold energy (distance to manifold)
        manifold_distances = self.manifold_learner.manifold_distance(x)
        E_manifold = manifold_distances ** 2
        
        # Combined energy
        total_energy = E_classification + self.lambda_manifold * E_manifold
        return total_energy

class ManifoldDROTrainer:
    """DRO trainer with manifold-constrained virtual synthesis"""
    
    def __init__(self, classifier, manifold_learner, device='cuda'):
        self.classifier = classifier.to(device)
        self.manifold_learner = manifold_learner.to(device)
        self.virtual_synthesizer = ManifoldConstrainedVirtualSynthesis(manifold_learner)
        self.geometric_energy = GeometricEnergyFunction(classifier, manifold_learner)
        self.device = device
        
    def train_manifold_structure(self, train_loader, epochs=50, lr=1e-3):
        """First phase: Learn manifold structure from ID data"""
        print("Phase 1: Learning manifold structure...")
        
        optimizer = torch.optim.Adam(self.manifold_learner.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)  # Flatten if needed
                
                optimizer.zero_grad()
                
                # Reconstruction loss
                x_recon, z = self.manifold_learner(data)
                recon_loss = F.mse_loss(x_recon, data)
                
                # Manifold regularization (smoothness)
                if batch_idx % 2 == 0:  # Compute manifold loss every other batch
                    z_noise = z + 0.1 * torch.randn_like(z)
                    x_noise_recon = self.manifold_learner.decode(z_noise)
                    manifold_loss = F.mse_loss(x_noise_recon, x_recon)
                else:
                    manifold_loss = 0
                
                total_loss_batch = recon_loss + 0.1 * manifold_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, Manifold Loss: {avg_loss:.4f}")
    
    def train_robust_classifier(self, train_loader, epochs=100, lr=1e-3, dro_weight=1.0):
        """Second phase: Train classifier with manifold-constrained DRO"""
        print("Phase 2: Training robust classifier with manifold DRO...")
        
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_ce_loss = 0
            total_dro_loss = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                optimizer.zero_grad()
                
                # Standard classification loss
                logits_id = self.classifier(data)
                ce_loss = F.cross_entropy(logits_id, targets)
                
                # Generate virtual outliers constrained to manifold
                virtual_outliers, outlier_types = self.virtual_synthesizer.generate_virtual_outliers(
                    data, num_outliers_per_sample=1
                )
                
                # Compute geometric energy for virtual outliers
                virtual_energies = self.geometric_energy(virtual_outliers)
                
                # DRO loss: encourage high energy (low likelihood) for virtual outliers
                dro_loss = -virtual_energies.mean()  # Negative because we want high energy
                
                # Total loss
                total_loss = ce_loss + dro_weight * dro_loss
                total_loss.backward()
                optimizer.step()
                
                total_ce_loss += ce_loss.item()
                total_dro_loss += dro_loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_ce = total_ce_loss / len(train_loader)
                avg_dro = total_dro_loss / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs}, CE Loss: {avg_ce:.4f}, DRO Loss: {avg_dro:.4f}")
    
    def evaluate_ood_detection(self, id_loader, ood_loader):
        """Evaluate OOD detection using geometric energy"""
        self.classifier.eval()
        self.manifold_learner.eval()
        
        id_energies = []
        ood_energies = []
        
        # Collect ID energies
        with torch.no_grad():
            for data, _ in id_loader:
                data = data.to(self.device)
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                energies = self.geometric_energy(data)
                id_energies.extend(energies.cpu().numpy())
        
        # Collect OOD energies
        with torch.no_grad():
            for data, _ in ood_loader:
                data = data.to(self.device)
                if len(data.shape) > 2:
                    data = data.view(data.size(0), -1)
                
                energies = self.geometric_energy(data)
                ood_energies.extend(energies.cpu().numpy())
        
        return np.array(id_energies), np.array(ood_energies)
    
    def visualize_manifold_outliers(self, data_sample, save_path=None):
        """Visualize the different types of virtual outliers"""
        self.manifold_learner.eval()
        
        with torch.no_grad():
            if len(data_sample.shape) > 2:
                data_flat = data_sample.view(data_sample.size(0), -1)
            else:
                data_flat = data_sample
            
            # Generate virtual outliers
            virtual_outliers, outlier_types = self.virtual_synthesizer.generate_virtual_outliers(
                data_flat[:5], num_outliers_per_sample=3
            )
            
            # Get manifold embeddings for visualization (2D projection)
            original_z = self.manifold_learner.encode(data_flat[:5])
            virtual_z = self.manifold_learner.encode(virtual_outliers)
            
            # Project to 2D for visualization
            all_z = torch.cat([original_z, virtual_z], dim=0)
            
            # Simple 2D projection (take first 2 dimensions)
            z_2d = all_z[:, :2].cpu().numpy()
            
            plt.figure(figsize=(10, 6))
            
            # Plot original data
            plt.scatter(z_2d[:5, 0], z_2d[:5, 1], c='blue', s=100, alpha=0.7, label='ID Data')
            
            # Plot virtual outliers by type
            distributional_indices = [i for i, t in enumerate(outlier_types) if t == 'distributional']
            structural_indices = [i for i, t in enumerate(outlier_types) if t == 'structural']
            
            if distributional_indices:
                dist_z = z_2d[5:][distributional_indices]
                plt.scatter(dist_z[:, 0], dist_z[:, 1], c='red', s=60, alpha=0.7, 
                           label='Distributional Outliers', marker='^')
            
            if structural_indices:
                struct_z = z_2d[5:][structural_indices]
                plt.scatter(struct_z[:, 0], struct_z[:, 1], c='orange', s=60, alpha=0.7,
                           label='Structural Outliers', marker='s')
            
            plt.xlabel('Manifold Dimension 1')
            plt.ylabel('Manifold Dimension 2') 
            plt.title('Manifold-Constrained Virtual Outliers')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.show()

# Demonstration function
def demonstrate_manifold_dro():
    """Demonstrate manifold-constrained DRO on dummy data"""
    
    # Generate dummy high-dimensional data lying on a lower-dimensional manifold
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a 2D manifold embedded in 50D space
    n_samples = 1000
    n_classes = 3
    manifold_dim = 2
    ambient_dim = 50
    
    # Generate 2D manifold data
    theta = np.linspace(0, 4*np.pi, n_samples)
    r = 1 + 0.3 * np.sin(3*theta)
    manifold_data = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    
    # Embed in high-dimensional space with random projection
    embedding_matrix = np.random.randn(manifold_dim, ambient_dim) * 0.1
    high_dim_data = manifold_data @ embedding_matrix
    
    # Add small noise
    high_dim_data += np.random.randn(n_samples, ambient_dim) * 0.05
    
    # Create labels
    labels = (theta / (2*np.pi) * n_classes).astype(int) % n_classes
    
    # Convert to tensors
    X = torch.FloatTensor(high_dim_data)
    y = torch.LongTensor(labels)
    
    # Create data loaders
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Create OOD data (samples far from manifold)
    ood_data = np.random.randn(200, ambient_dim) * 2  # Much larger variance
    ood_labels = np.zeros(200)  # Dummy labels
    ood_dataset = TensorDataset(torch.FloatTensor(ood_data), torch.LongTensor(ood_labels))
    ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=False)
    
    # Create models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Simple classifier
    classifier = nn.Sequential(
        nn.Linear(ambient_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes)
    )
    
    # Manifold learner
    manifold_learner = ManifoldLearner(
        input_dim=ambient_dim,
        manifold_dim=8,  # Overestimate the true manifold dimension
        hidden_dims=[128, 64]
    )
    
    # Create trainer
    trainer = ManifoldDROTrainer(classifier, manifold_learner, device)
    
    # Phase 1: Learn manifold structure
    trainer.train_manifold_structure(train_loader, epochs=30)
    
    # Phase 2: Train robust classifier
    trainer.train_robust_classifier(train_loader, epochs=50, dro_weight=0.5)
    
    # Evaluate OOD detection
    print("\nEvaluating OOD detection...")
    id_energies, ood_energies = trainer.evaluate_ood_detection(test_loader, ood_loader)
    
    # Calculate AUROC
    from sklearn.metrics import roc_auc_score
    y_true = np.concatenate([np.zeros(len(id_energies)), np.ones(len(ood_energies))])
    y_scores = np.concatenate([id_energies, ood_energies])
    auroc = roc_auc_score(y_true, y_scores)
    
    print(f"OOD Detection AUROC: {auroc:.4f}")
    print(f"ID Energy - Mean: {id_energies.mean():.4f}, Std: {id_energies.std():.4f}")
    print(f"OOD Energy - Mean: {ood_energies.mean():.4f}, Std: {ood_energies.std():.4f}")
    
    # Visualize manifold and virtual outliers
    print("\nGenerating visualization...")
    sample_data = X[:10]
    trainer.visualize_manifold_outliers(sample_data)
    
    return trainer, auroc

if __name__ == "__main__":
    trainer, auroc = demonstrate_manifold_dro()
    print(f"\nManifold-constrained DRO completed! AUROC: {auroc:.4f}")