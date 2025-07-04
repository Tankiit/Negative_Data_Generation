import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from torchvision import transforms
from openood.networks import ResNet18_32x32
from torchvision.models import resnet18


class Trainer:
    def __init__(self, 
                 model: nn.Module,
                 feature_extractor: nn.Module,
                 num_classes: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Trainer that handles both ID classification and outlier detection
        
        Args:
            model: Base classification model
            feature_extractor: Feature extraction module
            num_classes: Number of ID classes
            device: Device to use for computation
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.num_classes = num_classes
        self.device = device
        
        # Get feature dimension dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust size if needed
            features = self.feature_extractor(dummy_input)
            feature_dim = features.shape[1]
        
        # Add OOD head to model with correct input dimension
        self.ood_head = nn.Linear(feature_dim, 1).to(device)
        
    def generate_outliers(self, 
                         batch: torch.Tensor,
                         method: str = 'feature_space',
                         **kwargs) -> torch.Tensor:
        """
        Generate outlier samples from ID data
        
        Args:
            batch: Input batch of images
            method: Method to generate outliers
            kwargs: Additional parameters for generation
        """
        with torch.no_grad():
            features = self.feature_extractor(batch)
            
        if method == 'feature_space':
            # Generate outliers by perturbing in feature space
            noise_scale = kwargs.get('noise_scale', 0.1)
            feature_noise = torch.randn_like(features) * noise_scale
            noisy_features = features + feature_noise
            
            # Optional: Add feature space constraints
            if kwargs.get('enforce_distance', True):
                # Ensure minimum distance from original features
                distances = torch.norm(noisy_features - features, dim=1)
                min_distance = kwargs.get('min_distance', 0.5)
                scale_factor = torch.clamp(distances / min_distance, min=1.0)
                noisy_features = features + (noisy_features - features) * scale_factor.unsqueeze(1)
                
        elif method == 'mixture':
            # Generate outliers by mixing features of different classes
            alpha = kwargs.get('alpha', 0.3)
            permuted_features = features[torch.randperm(len(features))]
            noisy_features = (1 - alpha) * features + alpha * permuted_features
            
        elif method == 'adversarial':
            # Generate adversarial features
            epsilon = kwargs.get('epsilon', 0.1)
            features.requires_grad_(True)
            
            # Maximize distance from original features
            logits = self.model(features)
            loss = -F.cross_entropy(logits, logits.argmax(dim=1))
            loss.backward()
            
            noisy_features = features + epsilon * features.grad.sign()
            features.requires_grad_(False)
            
        return noisy_features.detach()
    
    def train_step(self,
                   batch: Tuple[torch.Tensor, torch.Tensor],
                   optimizer: torch.optim.Optimizer,
                   outlier_params: Dict = None) -> Dict[str, float]:
        """
        Single training step with generated outliers
        """
        if outlier_params is None:
            outlier_params = {'method': 'feature_space', 'noise_scale': 0.1}
            
        self.model.train()
        images, labels = [x.to(self.device) for x in batch]
        batch_size = images.size(0)
        
        # Extract features
        features = self.feature_extractor(images)
        
        # Generate outlier features
        outlier_features = self.generate_outliers(images, **outlier_params)
        
        # Combine ID and outlier features
        combined_features = torch.cat([features, outlier_features], dim=0)
        
        # Forward pass through classifier
        logits = self.model(combined_features)
        ood_scores = self.ood_head(combined_features).squeeze()
        
        # Create labels for outlier detection
        ood_labels = torch.cat([
            torch.zeros(batch_size),
            torch.ones(batch_size)
        ]).to(self.device)
        
        # Calculate losses
        # Classification loss for ID samples
        clf_loss = F.cross_entropy(logits[:batch_size], labels)
        
        # OOD detection loss
        ood_loss = F.binary_cross_entropy_with_logits(ood_scores, ood_labels)
        
        # Confidence penalty for outlier samples
        outlier_confidence = F.softmax(logits[batch_size:], dim=1).max(dim=1)[0]
        confidence_penalty = outlier_confidence.mean()
        
        # Total loss
        total_loss = clf_loss + ood_loss + 0.1 * confidence_penalty
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'clf_loss': clf_loss.item(),
            'ood_loss': ood_loss.item(),
            'confidence_penalty': confidence_penalty.item()
        }
        
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        self.model.eval()
        metrics = defaultdict(list)
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            batch_size = images.size(0)
            
            # Get model predictions
            features = self.feature_extractor(images)
            logits = self.model(features)
            ood_scores = torch.sigmoid(self.ood_head(features))
            
            # Generate outliers for evaluation
            outlier_features = self.generate_outliers(images)
            outlier_logits = self.model(outlier_features)
            outlier_ood_scores = torch.sigmoid(self.ood_head(outlier_features))
            
            # Calculate metrics
            predictions = logits.argmax(dim=1)
            metrics['accuracy'].append(
                (predictions == labels).float().mean().item()
            )
            
            # OOD detection metrics
            all_scores = torch.cat([ood_scores, outlier_ood_scores]).cpu()
            all_labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(batch_size)
            ])
            
            metrics['ood_scores'].extend(all_scores.tolist())
            metrics['ood_labels'].extend(all_labels.tolist())
            
        # Calculate final metrics
        return {
            'accuracy': np.mean(metrics['accuracy']),
            'ood_auroc': roc_auc_score(
                metrics['ood_labels'],
                metrics['ood_scores']
            )
        }
class FeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, target_layers=None):
        super().__init__()
        # Use torchvision's resnet18
        base_model = resnet18(pretrained=pretrained)
        
        # Remove the final fully connected layer
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        
        # Default target layers if none specified - for ResNet we'll use the final features
        self.target_layers = target_layers if target_layers else [-1]
        
    def forward(self, x):
        x = self.model(x)
        # ResNet's output is [B, C, 1, 1], so we need to flatten it
        x = torch.flatten(x, 1)
        return x

def train_model():
    import argparse
    import torchvision
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/', required=False)
    parser.add_argument('--layer_name', default='layer4.1.bn2', required=False)
    args = parser.parse_args()

    # Replace timm transforms with standard torchvision transforms
    train_transform = transforms.Compose([
        transforms.Resize(224),  # ResNet18 expects 224x224 images
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=28),  # Adjust padding as needed
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # CIFAR-10 as ID data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )
    
    # SVHN as OOD data
    oodset = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=train_transform
    )
    oodloader = torch.utils.data.DataLoader(
        oodset, batch_size=64, shuffle=False, num_workers=2
    )


    
    # Load the checkpoint
    # Initialize model and trainer
    model = ResNet18_32x32()
    
    checkpoint_path = './cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    model.eval()

    # # Initialize model and trainer
    # model = ResNet18_32x32()
    feature_extractor = FeatureExtractor(
        model_name='resnet18',
        pretrained=True
    )
    trainer = Trainer(model, feature_extractor)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': trainer.ood_head.parameters(), 'lr': 0.001}
    ])
    
    # Training loop
    num_epochs = 10
    outlier_params = {
        'method': 'feature_space',
        'noise_scale': 0.1,
        'enforce_distance': True,
        'min_distance': 0.5
    }
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch in trainloader:
            losses = trainer.train_step(batch, optimizer, outlier_params)
            epoch_losses.append(losses)
            
        # Evaluate
        metrics = trainer.evaluate(testloader)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"OOD AUROC: {metrics['ood_auroc']:.3f}")
        
    return model, trainer

if __name__ == "__main__":
    model, trainer = train_model()
