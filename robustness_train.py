import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_ood.detector import EnergyBased
from pytorch_ood.utils import OODMetrics
from pytorch_ood.model import WideResNet
from torchvision.datasets import CIFAR10
from robustness.train import train_model
from robustness.tools import helpers
from robustness import attacker
import cox.store

from feature_manipulator import FeatureManipulator
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import uuid


class RobustFeatureAttacker:
    """
    Feature space attacks using robustness library
    """
    def __init__(self, model, epsilon=8/255, step_size=2/255, num_steps=10):
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.energy_detector = EnergyBased(model)
        
    def extract_features(self, x):
        """Extract penultimate layer features"""
        features = []
        def hook(module, input, output):
            features.append(output.detach())
        
        # Register hook for feature extraction
        handle = list(self.model.modules())[-2].register_forward_hook(hook)
        _ = self.model(x)
        handle.remove()
        
        return features[0]

    def perturb_features(self, x, y, maximize_energy=True):
        """
        Generate adversarial perturbations in feature space using PGD
        """
        # First extract original features
        orig_features = self.extract_features(x)
        
        # Setup PGD attack using robustness library
        attack = attacker.PGDAttack(
            self.model, 
            self.epsilon,
            self.step_size,
            self.num_steps,
            random_start=True,
            targeted=False
        )
        
        # Custom loss function for energy maximization/minimization
        def custom_criterion(outputs, labels):
            energy_scores = self.energy_detector(outputs)
            return energy_scores.mean() if maximize_energy else -energy_scores.mean()
        
        # Perform attack with custom criterion
        perturbed_x, _ = attack(x, y, custom_criterion)
        
        # Extract features from perturbed inputs
        pert_features = self.extract_features(perturbed_x)
        
        return pert_features

    def generate_uncertainty_set(self, x, y, num_samples=5):
        """
        Generate uncertainty set around features using multiple attack iterations
        """
        uncertainty_features = []
        
        for _ in range(num_samples):
            # Generate perturbed features with different random starts
            pert_features = self.perturb_features(x, y, maximize_energy=True)
            uncertainty_features.append(pert_features)
            
        return torch.stack(uncertainty_features, dim=0)

class RobustFeatureTrainer:
    def __init__(self, model, train_loader, test_loader, device=None):
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_robust_features(self, num_epochs=100):
        """
        Train the model with robust feature learning
        """
        # Create output directory for logging
        log_dir = os.path.join('train_out', str(uuid.uuid4()))
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging in: {log_dir}")
        
        # Training setup
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs
        )
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate training metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()
                
                # Print batch progress
                if (batch_idx + 1) % 100 == 0:
                    print(f'Epoch: [{epoch + 1}/{num_epochs}] | Batch: [{batch_idx + 1}/{len(self.train_loader)}] | '
                          f'Loss: {train_loss/(batch_idx + 1):.3f} | '
                          f'Acc: {100.*train_correct/train_total:.2f}%')
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            # Calculate epoch metrics
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss = val_loss / len(self.test_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step()
            
            # Print epoch results
            print(f'\nEpoch: {epoch + 1}/{num_epochs}')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%\n')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(log_dir, 'best_model.pth'))
                print(f'Best model saved with accuracy: {best_acc:.2f}%')

def evaluate_robustness(model, test_loader):
    """
    Evaluate model robustness using energy scores
    """
    attacker = RobustFeatureAttacker(model)
    metrics = OODMetrics()
    
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        
        # Get original and perturbed features
        orig_features = attacker.extract_features(x)
        pert_features = attacker.perturb_features(x, y)
        
        # Compute energy scores
        orig_energy = attacker.energy_detector(model.features_to_logits(orig_features))
        pert_energy = attacker.energy_detector(model.features_to_logits(pert_features))
        
        # Update metrics
        metrics.update(orig_energy, torch.zeros_like(y))
        metrics.update(pert_energy, torch.ones_like(y))
    
    return metrics.compute()

if __name__ == "__main__":
    # Initialize model
    model = WideResNet(num_classes=10, pretrained="er-cifar10-tune").to('mps')
    
    # Define the transforms
    train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 stats
        std=[0.2023, 0.1994, 0.2010]
    )
])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    ])
    train_dataset=CIFAR10(root='/Users/cril/tanmoy/research/data/',train=True,transform=train_transform,download=True)
    test_dataset=CIFAR10(root='/Users/cril/tanmoy/research/data/',train=False,transform=test_transform,download=True)
    train_loader=DataLoader(train_dataset,batch_size=128,shuffle=True,num_workers=4)
    test_loader=DataLoader(test_dataset,batch_size=128,shuffle=False,num_workers=4)
    # Create trainer
    trainer = RobustFeatureTrainer(model, train_loader, test_loader)
    
    # Train robust model
    trainer.train_robust_features(num_epochs=100)
    
    # Evaluate
    results = evaluate_robustness(model, test_loader)
    print("Robustness Results:", results)