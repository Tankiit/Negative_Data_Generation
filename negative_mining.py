import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cvxpy as cp
import torchvision

def get_device(device_name=None):
    """
    Get the appropriate device based on availability and user preference
    
    Args:
        device_name: Optional device specification ('cpu', 'cuda', 'mps', or None)
        
    Returns:
        torch.device: Selected device
    """
    if device_name is None:
        # First try CUDA
        if torch.cuda.is_available():
            return torch.device('cuda')
        # Then try MPS (Metal Performance Shaders for Mac)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        # Fall back to CPU
        else:
            return torch.device('cpu')
    
    # If device is specified, verify it's available
    device_name = device_name.lower()
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA device requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    elif device_name == 'mps' and (not hasattr(torch.backends, 'mps') or 
                                 not torch.backends.mps.is_available()):
        print("Warning: MPS device requested but not available. Falling back to CPU.")
        return torch.device('cpu')
    
    return torch.device(device_name)


class SyntheticNegativeGenerator:
    """
    Class for generating synthetic negative data through various augmentation techniques
    """
    def __init__(
        self,
        device=None,
        texture_path=None,  # Path to textures for PixMix
        num_synthetic_per_real=1,  # Number of synthetic samples per real sample
        severity=3,  # Severity of augmentations (1-5)
        img_size=224,  # Image size for augmentations
    ):
        self.device = device if device else get_device()
        self.num_synthetic_per_real = num_synthetic_per_real
        self.severity = severity
        self.img_size = img_size
        
        # Define augmentation pipelines with Albumentations
        self.strong_aug = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.4),
            A.OneOf([
                A.MotionBlur(p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.4),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.3),
                A.ElasticTransform(p=0.3),
            ], p=0.4),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.4),
            A.HueSaturationValue(p=0.4),
            A.RGBShift(p=0.4),
            A.RandomGamma(p=0.4),
            A.Posterize(p=0.4),
            A.Solarize(p=0.4),
            A.ToGray(p=0.2),
            A.JpegCompression(quality_lower=50, quality_upper=100, p=0.4),
            ToTensorV2(),
        ])
        
        # Load textures for PixMix if path provided
        self.textures = None
        if texture_path and os.path.exists(texture_path):
            self.textures = self._load_textures(texture_path)
        else:
            print("Warning: No texture path provided. PixMix will use random noise.")
    
    def _load_textures(self, texture_path):
        """Load texture images for PixMix"""
        textures = []
        if os.path.isdir(texture_path):
            for filename in os.listdir(texture_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(texture_path, filename)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        textures.append(img)
                    except:
                        print(f"Couldn't load texture: {img_path}")
        return textures
    
    def _apply_augmentation(self, image):
        """Apply strong augmentation using Albumentations"""
        # Convert tensor to numpy array for Albumentations
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy().transpose(1, 2, 0)
            
        # Apply augmentation
        augmented = self.strong_aug(image=image)['image']
        return augmented.to(self.device)
    
    def generate_synthetic_negatives(self, dataloader, methods=None):
        """
        Generate synthetic negative samples from a dataloader
        
        Args:
            dataloader: DataLoader with original samples
            methods: List of augmentation methods to use. Options:
                     ['cutmix', 'pixmix', 'mixup', 'cutout', 'albumentations', 'all']
        
        Returns:
            synthetic_data: List of (tensor, target) tuples with synthetic data
        """
        if methods is None:
            methods = ['cutmix', 'pixmix', 'albumentations']
        elif 'all' in methods:
            methods = ['cutmix', 'pixmix', 'mixup', 'cutout', 'albumentations']
        
        synthetic_data = []
        
        # Keep a buffer of images to use for mixing
        image_buffer = []
        
        # Process dataloader
        for inputs, targets in tqdm(dataloader, desc="Generating synthetic negatives"):
            inputs = inputs.to(self.device)
            
            # Add current batch to buffer
            image_buffer.append(inputs.cpu())
            if len(image_buffer) > 10:  # Keep buffer size reasonable
                image_buffer.pop(0)
            
            # Create mixing images
            if len(image_buffer) > 1:
                mixing_images = torch.cat(image_buffer, dim=0)
            else:
                mixing_images = inputs.cpu()
            
            # Generate synthetic samples for each image
            for i in range(inputs.size(0)):
                img = inputs[i].unsqueeze(0)  # [1, C, H, W]
                
                # Determine how many of each type to generate
                num_methods = len(methods)
                samples_per_method = max(1, self.num_synthetic_per_real // num_methods)
                
                # Generate samples using each method
                for method in methods:
                    for _ in range(samples_per_method):
                        if method == 'cutmix':
                            synthetic = self._apply_cutmix(img, mixing_images)
                        elif method == 'pixmix':
                            synthetic = self._apply_pixmix(img)
                        elif method == 'mixup':
                            synthetic = self._apply_mixup(img, mixing_images)
                        elif method == 'cutout':
                            synthetic = self._apply_cutout(img)
                        elif method == 'albumentations':
                            synthetic = self._apply_augmentation(img)
                        
                        # Add to synthetic data with original target
                        # (we're generating synthetic negatives, not changing classes)
                        synthetic_data.append((synthetic.cpu(), torch.tensor(-1)))  # -1 for synthetic negative class
        
        print(f"Generated {len(synthetic_data)} synthetic negative samples")
        return synthetic_data
    
    def _apply_cutmix(self, img, mixing_images):
        """Apply CutMix augmentation"""
        batch_size = mixing_images.size(0)
        mix_idx = random.randint(0, batch_size-1)
        mix_img = mixing_images[mix_idx].to(self.device)
        
        # Generate random box
        lam = np.random.beta(1.0, 1.0)
        
        img_h, img_w = img.size(2), img.size(3)
        cut_ratio = np.sqrt(1. - lam)
        cut_h = int(img_h * cut_ratio)
        cut_w = int(img_w * cut_ratio)
        
        # Generate random box position
        cy = np.random.randint(0, img_h)
        cx = np.random.randint(0, img_w)
        
        # Calculate box boundaries
        y1 = max(0, cy - cut_h // 2)
        y2 = min(img_h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(img_w, cx + cut_w // 2)
        
        # Apply cutmix
        mixed = img.clone()
        mixed[:, :, y1:y2, x1:x2] = mix_img[:, :, y1:y2, x1:x2]
        
        return mixed
    
    def _apply_pixmix(self, img):
        """Apply PixMix augmentation"""
        # Choose mixing image: either texture or random noise
        if self.textures and random.random() < 0.5:
            # Use texture image
            texture_idx = random.randint(0, len(self.textures)-1)
            texture = self.textures[texture_idx]
            
            # Resize to match input
            h, w = img.size(2), img.size(3)
            texture = transforms.Resize((h, w))(texture)
            texture = transforms.ToTensor()(texture).to(self.device)
            
            # Match channels
            if texture.size(0) != img.size(1):
                texture = texture.repeat(1, 1, 1, 1)[:, :img.size(1)]
        else:
            # Use random noise
            texture = torch.rand_like(img).to(self.device)
        
        # Apply mixing operation with varying strengths
        severity = self.severity
        mix_ops = ['add', 'mul']
        op = random.choice(mix_ops)
        
        # Apply operation
        alpha = random.uniform(0.1, 0.7)  # Mixing parameter
        
        if op == 'add':
            mixed = (1 - alpha) * img + alpha * texture * severity
        else:  # mul
            mixed = (1 - alpha) * img + alpha * img * texture * severity
        
        # Ensure valid range
        mixed = torch.clamp(mixed, 0, 1)
        
        return mixed
    
    def _apply_mixup(self, img, mixing_images):
        """Apply Mixup augmentation"""
        batch_size = mixing_images.size(0)
        mix_idx = random.randint(0, batch_size-1)
        mix_img = mixing_images[mix_idx].to(self.device)
        
        # Generate mixup coefficient
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        
        # Apply mixup
        mixed = lam * img + (1 - lam) * mix_img
        
        return mixed
    
    def _apply_cutout(self, img):
        """Apply Cutout augmentation"""
        img_h, img_w = img.size(2), img.size(3)
        
        # Generate random cutout size
        cut_ratio = random.uniform(0.1, 0.5)
        cut_h = int(img_h * cut_ratio)
        cut_w = int(img_w * cut_ratio)
        
        # Generate random cutout position
        cy = np.random.randint(0, img_h)
        cx = np.random.randint(0, img_w)
        
        # Calculate box boundaries
        y1 = max(0, cy - cut_h // 2)
        y2 = min(img_h, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(img_w, cx + cut_w // 2)
        
        # Apply cutout
        mixed = img.clone()
        mixed[:, :, y1:y2, x1:x2] = 0
        
        return mixed
    
    def _normalize_features(self, features):
        """Normalize feature vectors to unit length"""
        return F.normalize(features, p=2, dim=1)
    
    def find_hard_negatives(self, id_features, ood_features, theta, bias):
        """
        Find hard negative examples based on current decision boundary
        
        Args:
            id_features: Features from in-distribution samples (N x D)
            ood_features: Features from out-of-distribution samples (M x D)
            theta: Current decision boundary normal vector
            bias: Current decision boundary bias term
            
        Returns:
            hard_id_features: Features of hard negative ID samples
            hard_id_indices: Indices of hard negative ID samples
            hard_ood_features: Features of hard negative OOD samples
            hard_ood_indices: Indices of hard negative OOD samples
        """
        # Ensure inputs are on the correct device
        id_features = id_features.to(self.device)
        ood_features = ood_features.to(self.device)
        theta = torch.as_tensor(theta, device=self.device)
        bias = torch.as_tensor(bias, device=self.device)
        
        # Normalize features
        id_norm = self._normalize_features(id_features)
        ood_norm = self._normalize_features(ood_features)
        
        # Calculate scores
        id_scores = torch.matmul(id_norm, theta) + bias
        ood_scores = torch.matmul(ood_norm, theta) + bias
        
        # Find hard negatives (ID samples with low scores, OOD with high scores)
        hard_id_indices = torch.where(id_scores < 0)[0]
        hard_ood_indices = torch.where(ood_scores > 0)[0]
        
        # Extract hard negative features
        hard_id_features = id_features[hard_id_indices]
        hard_ood_features = ood_features[hard_ood_indices]
        
        return hard_id_features, hard_id_indices, hard_ood_features, hard_ood_indices
    
    def generate_hard_synthetic_negatives(self, id_features, ood_features, theta, bias):
        """
        Generate synthetic negatives focusing on hard examples
        
        Args:
            id_features: Features from in-distribution samples
            ood_features: Features from out-of-distribution samples
            theta: Current decision boundary normal vector
            bias: Current decision boundary bias term
            
        Returns:
            synthetic_features: Generated synthetic negative features
        """
        # Find hard negative examples
        hard_id_features, _, hard_ood_features, _ = self.find_hard_negatives(
            id_features, ood_features, theta, bias
        )
        
        # Combine hard negatives
        all_hard_features = torch.cat([hard_id_features, hard_ood_features], dim=0)
        
        # Generate synthetic samples from hard negatives
        num_synthetic = int(len(all_hard_features) * self.num_synthetic_per_real)
        if num_synthetic == 0:
            return all_hard_features  # Return original hard negatives if no synthetics needed
        
        synthetic_features = []
        for _ in range(num_synthetic):
            # Randomly select pairs of hard negatives
            idx1 = torch.randint(0, len(all_hard_features), (1,))
            idx2 = torch.randint(0, len(all_hard_features), (1,))
            
            # Interpolate between them
            alpha = torch.rand(1).to(self.device)
            synthetic = alpha * all_hard_features[idx1] + (1 - alpha) * all_hard_features[idx2]
            synthetic_features.append(synthetic)
        
        synthetic_features = torch.cat(synthetic_features, dim=0)
        return torch.cat([all_hard_features, synthetic_features], dim=0)
    
    def visualize_synthetic_samples(self, original_loader, synthetic_data, num_samples=5, save_path=None):
        """Visualize original and synthetic samples"""
        # Get a few original samples
        original_samples = []
        for inputs, _ in original_loader:
            original_samples.append(inputs[:num_samples])
            break
        
        original_samples = original_samples[0].cpu()
        
        # Get corresponding synthetic samples
        synthetic_samples = []
        for i in range(min(num_samples, len(synthetic_data))):
            synthetic_samples.append(synthetic_data[i][0])
        
        synthetic_samples = torch.cat(synthetic_samples, dim=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples*3, 6))
        
        # Define normalization inverse transform
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        
        # Plot original samples
        for i in range(num_samples):
            # Denormalize
            img = inv_normalize(original_samples[i])
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
        
        # Plot synthetic samples
        for i in range(num_samples):
            # Denormalize
            img = inv_normalize(synthetic_samples[i])
            img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
            
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"Synthetic {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        plt.show()
        
        return fig


class SyntheticNegativeDataset(Dataset):
    """Dataset combining original and synthetic negative data"""
    def __init__(self, original_dataset, synthetic_data, transform=None):
        self.original_dataset = original_dataset
        self.synthetic_data = synthetic_data
        self.transform = transform
        
        # Compute total length
        self.total_len = len(original_dataset) + len(synthetic_data)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # Original data
        if idx < len(self.original_dataset):
            data, target = self.original_dataset[idx]
            source = 0  # 0 indicates original
        # Synthetic data
        else:
            synthetic_idx = idx - len(self.original_dataset)
            data, target = self.synthetic_data[synthetic_idx]
            source = 1  # 1 indicates synthetic
        
        # Apply additional transforms if needed
        if self.transform:
            data = self.transform(data)
        
        return data, target, source


class DROSolver:
    """
    Distributionally Robust Optimization (DRO) solver using CVXPY
    """
    def __init__(self, epsilon=0.1, reg_param=0.01):
        self.epsilon = epsilon  # Wasserstein ball radius
        self.reg_param = reg_param  # Regularization parameter
        
    def solve_dro_problem(self, features, labels, epsilon=None, reg_param=None,
                         sample_weights=None):
        """
        Solve DRO optimization problem with CVXPY
        
        Args:
            features: Input features tensor
            labels: Target labels tensor
            epsilon: Wasserstein ball radius (optional)
            reg_param: Regularization parameter (optional)
            sample_weights: Sample importance weights (optional)
            
        Returns:
            theta: Optimal decision boundary normal vector
            bias: Optimal bias term
            obj_value: Optimal objective value
        """
        # Set default parameters if not provided
        if epsilon is None:
            epsilon = self.epsilon
        if reg_param is None:
            reg_param = self.reg_param
        
        # Convert to numpy for CVXPY
        features_np = features.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        n_samples, n_features = features_np.shape
        
        # Set sample weights (default to uniform if not provided)
        if sample_weights is None:
            weights_np = np.ones(n_samples) / n_samples
        else:
            weights_np = sample_weights.detach().cpu().numpy()
            weights_np = weights_np / weights_np.sum()  # Normalize
        
        # Define CVXPY variables
        theta = cp.Variable(n_features)  # DRO decision boundary normal vector
        bias = cp.Variable()  # DRO decision boundary bias term
        
        # Slack variables for hinge loss
        xi = cp.Variable(n_samples, nonneg=True)
        
        # Predictions
        predictions = features_np @ theta + bias
        
        # Hinge loss with sample weights
        hinge_loss = cp.sum(cp.multiply(weights_np, xi))
        
        # Constraints: y_i * (x_i · θ + b) ≥ 1 - ξ_i
        constraints = [
            labels_np[i] * predictions[i] >= 1 - xi[i]
            for i in range(n_samples)
        ]
        
        # Wasserstein robustness term
        wasserstein_term = epsilon * cp.norm(theta, 2)
        
        # Regularization for better generalization
        regularization = reg_param * cp.sum_squares(theta)
        
        # Total objective
        objective = hinge_loss + wasserstein_term + regularization
        
        # Define and solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            # Try different solvers for robustness
            solvers = [cp.SCS, cp.ECOS, cp.OSQP]
            solver_names = ["SCS", "ECOS", "OSQP"]
            
            for solver, name in zip(solvers, solver_names):
                try:
                    print(f"Trying solver: {name}")
                    problem.solve(solver=solver, verbose=True, eps=1e-4, max_iters=5000)
                    if problem.status in ["optimal", "optimal_inaccurate"]:
                        print(f"Solved with {name}")
                        break
                except Exception as e:
                    print(f"Solver {name} failed: {e}")
                    continue
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                theta_val = theta.value.astype(np.float32)
                bias_val = float(bias.value)
                
                # Check solution quality
                theta_norm = np.linalg.norm(theta_val)
                if theta_norm < 1e-6:
                    print(f"Warning: Very small theta norm: {theta_norm}")
                
                return theta_val, bias_val, float(problem.value)
            else:
                print(f"All solvers failed with status: {problem.status}")
                return None, None, float('inf')
        except Exception as e:
            print(f"CVXPY optimization error: {e}")
            return None, None, float('inf')


class Evaluator:
    """Class for evaluating OOD detection performance"""
    def __init__(self, writer=None):
        self.writer = writer
        self.reset()
    
    def reset(self):
        """Reset all tracking variables"""
        self.all_scores = []
        self.all_labels = []
        self.running_loss = 0
        self.num_samples = 0
    
    def update(self, scores, labels, loss=None, step=None):
        """
        Update metrics with new batch of predictions
        
        Args:
            scores: OOD detection scores (higher = more likely OOD)
            labels: True labels (1 for OOD, 0 for ID)
            loss: Optional loss value for logging
            step: Optional step number for TensorBoard logging
        """
        self.all_scores.extend(scores.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        
        if loss is not None:
            self.running_loss += loss.item() * len(scores)
            self.num_samples += len(scores)
            
            if self.writer is not None and step is not None:
                self.writer.add_scalar('Loss/train', loss.item(), step)
    
    def compute_metrics(self, prefix='', step=None):
        """
        Compute evaluation metrics
        
        Args:
            prefix: Prefix for metric names in TensorBoard (e.g., 'train/', 'val/')
            step: Optional step number for TensorBoard logging
            
        Returns:
            metrics: Dictionary containing computed metrics
        """
        scores = np.array(self.all_scores)
        labels = np.array(self.all_labels)
        
        metrics = {}
        metrics['auroc'] = roc_auc_score(labels, scores)
        metrics['aupr'] = average_precision_score(labels, scores)
        
        if len(np.unique(labels)) > 1:
            metrics['fpr_at_95_tpr'] = self._fpr_at_fixed_tpr(scores, labels, 0.95)
        
        if self.num_samples > 0:
            metrics['loss'] = self.running_loss / self.num_samples
        
        if self.writer is not None and step is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(f'{prefix}{name}', value, step)
        
        return metrics

    
    
    @staticmethod
    def _fpr_at_fixed_tpr(scores, labels, tpr_level=0.95):
        """
        Calculate FPR at a fixed TPR level
        
        Args:
            scores: Array of model scores
            labels: Array of true labels
            tpr_level: Target TPR level (default: 0.95)
            
        Returns:
            fpr: False Positive Rate at the specified TPR level
        """
        sorted_scores = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        total_p = sum(1 for _, l in sorted_scores if l == 1)
        total_n = sum(1 for _, l in sorted_scores if l == 0)
        
        target_tp = total_p * tpr_level
        current_tp = current_fp = 0
        
        for score, label in sorted_scores:
            if label == 1:
                current_tp += 1
            else:
                current_fp += 1
            if current_tp >= target_tp:
                return current_fp / total_n
        return 1.0


def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Train DRO-based OOD detector with synthetic negative mining')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='/Users/cril/tanmoy/research/data',
                        help='Path to dataset directory containing both ID and OOD data')
    parser.add_argument('--id-dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet', 'custom'],
                        help='Name of the in-distribution dataset')
    parser.add_argument('--ood-dataset', type=str, default='svhn',
                        choices=['svhn', 'cifar100', 'imagenet', 'custom'],
                        help='Name of the out-of-distribution dataset')
    parser.add_argument('--texture-path', type=str, default=None,
                        help='Path to textures for PixMix augmentation')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Device selection
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default=None,
                        help='Device to use (cpu, cuda, mps). If not specified, will use the best available device')
    
    # DRO parameters
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Wasserstein ball radius for DRO')
    parser.add_argument('--reg-param', type=float, default=0.01,
                        help='Regularization parameter for DRO')
    
    # Synthetic negative generation parameters
    parser.add_argument('--num-synthetic-per-real', type=float, default=0.5,
                        help='Number of synthetic samples to generate per real sample')
    parser.add_argument('--aug-severity', type=int, default=3,
                        help='Severity of augmentations (1-5)')
    parser.add_argument('--aug-methods', type=str, nargs='+',
                        default=['cutmix', 'pixmix', 'albumentations'],
                        help='Augmentation methods to use')
    
    # Logging parameters
    parser.add_argument('--output-dir', type=str, default='dro_synthetic_results',
                        help='Directory to save results and tensorboard logs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='How often to log training metrics (iterations)')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='How often to run evaluation (epochs)')
    parser.add_argument('--save-model', action='store_true',
                        help='Whether to save model checkpoints')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    return args


def get_dataset_loader(data_dir, dataset_name, batch_size, is_training=True):
    """
    Create data loader for a specified dataset
    
    Args:
        data_dir: Root directory containing datasets
        dataset_name: Name of the dataset to load ('cifar10', 'cifar100', 'svhn', 'imagenet', 'custom')
        batch_size: Batch size for the data loader
        is_training: Whether to use training transforms and shuffle data
        
    Returns:
        data_loader: DataLoader for the specified dataset
    """
    # Define standard transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_training:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    
    # Load the appropriate dataset
    dataset_path = os.path.join(data_dir, dataset_name)
    
    if dataset_name == 'cifar10':
        from torchvision.datasets import CIFAR10
        dataset = CIFAR10(root=data_dir, train=is_training, 
                         download=True, transform=transform)
    
    elif dataset_name == 'cifar100':
        from torchvision.datasets import CIFAR100
        dataset = CIFAR100(root=data_dir, train=is_training, 
                          download=True, transform=transform)
    
    elif dataset_name == 'svhn':
        from torchvision.datasets import SVHN
        split = 'train' if is_training else 'test'
        dataset = SVHN(root=data_dir, split=split, 
                      download=True, transform=transform)
    
    elif dataset_name == 'imagenet':
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root=dataset_path, transform=transform)
    
    elif dataset_name == 'custom':
        # Assuming custom dataset follows ImageFolder structure
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(root=dataset_path, transform=transform)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def train_dro_with_synthetic_negatives(
    id_loader,
    ood_loader,
    feature_extractor,
    feature_dim,
    device=None,
    output_dir="dro_synthetic_results",
    texture_path=None,
    num_synthetic_per_real=0.5,  # Generate 50% synthetic samples relative to real
    log_interval=10,
    eval_interval=1
):
    """
    Train DRO-based OOD detector with synthetic negative data generation
    
    Args:
        id_loader: DataLoader for ID data
        ood_loader: DataLoader for OOD data
        feature_extractor: Feature extraction model
        feature_dim: Feature dimension
        device: Device to use
        output_dir: Directory to save results
        texture_path: Path to textures for PixMix
        num_synthetic_per_real: Number of synthetic samples to generate per real sample
        log_interval: Interval for logging training metrics
        eval_interval: Interval for evaluating on validation set
    """
    
    if device is None:
        device = get_device()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create synthetic negative generator
    generator = SyntheticNegativeGenerator(
        device=device,
        texture_path=texture_path,
        num_synthetic_per_real=num_synthetic_per_real,
        severity=3
    )
    
    # Generate synthetic negatives from both ID and OOD data
    print("\nGenerating synthetic negatives from ID data...")
    id_synthetic = generator.generate_synthetic_negatives(
        id_loader,
        methods=['cutmix', 'pixmix', 'mixup']
    )
    
    print("\nGenerating synthetic negatives from OOD data...")
    ood_synthetic = generator.generate_synthetic_negatives(
        ood_loader,
        methods=['cutmix', 'pixmix', 'mixup']
    )
    
    # Visualize synthetic samples
    generator.visualize_synthetic_samples(
        id_loader,
        id_synthetic,
        num_samples=5,
        save_path=os.path.join(output_dir, "id_synthetic_samples.png")
    )
    
    generator.visualize_synthetic_samples(
        ood_loader,
        ood_synthetic,
        num_samples=5,
        save_path=os.path.join(output_dir, "ood_synthetic_samples.png")
    )
    
    # Create datasets with synthetic samples
    # We'll extract features from synthetic data
    print("\nExtracting features from synthetic samples...")
    
    # Extract features from synthetic ID samples
    id_synthetic_features = []
    id_synthetic_targets = []
    
    with torch.no_grad():
        for img, target in tqdm(id_synthetic, desc="ID synthetic"):
            img = img.to(device)
            features = feature_extractor(img)
            id_synthetic_features.append(features.cpu())
            id_synthetic_targets.append(target)
    
    # Extract features from synthetic OOD samples
    ood_synthetic_features = []
    ood_synthetic_targets = []
    
    with torch.no_grad():
        for img, target in tqdm(ood_synthetic, desc="OOD synthetic"):
            img = img.to(device)
            features = feature_extractor(img)
            ood_synthetic_features.append(features.cpu())
            ood_synthetic_targets.append(target)
    
    # Create feature dataset for DRO training
    id_synthetic_features = torch.cat(id_synthetic_features, dim=0)
    ood_synthetic_features = torch.cat(ood_synthetic_features, dim=0)
    
    print(f"Generated {id_synthetic_features.size(0)} ID synthetic features")
    print(f"Generated {ood_synthetic_features.size(0)} OOD synthetic features")
    
    # Now create DRO trainer with hard negative mining
    print("\nTraining DRO detector with synthetic negatives and hard mining...")
    detector = DROWithHardNegativeMining(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        device=device,
        initial_epsilon=2.0,
        reg_param=0.1,
        mining_rounds=3
    )
    
    # Create a customized fit method that uses synthetic data
    def fit_with_synthetic_data(detector, id_loader, ood_loader, 
                               id_synthetic_features, ood_synthetic_features):
        """Modified fit method that incorporates synthetic features"""
        # Extract initial features
        print("Extracting initial features...")
        id_features, id_labels = detector._extract_dataset(id_loader)
        ood_features, ood_labels = detector._extract_dataset(ood_loader)
        
        # Add synthetic features
        id_all_features = torch.cat([id_features, id_synthetic_features], dim=0)
        ood_all_features = torch.cat([ood_features, ood_synthetic_features], dim=0)
        
        # Create synthetic labels
        id_synthetic_labels = torch.tensor(id_synthetic_targets, device=detector.device)
        ood_synthetic_labels = torch.tensor(ood_synthetic_targets, device=detector.device)
        
        id_all_labels = torch.cat([id_labels, id_synthetic_labels], dim=0)
        ood_all_labels = torch.cat([ood_labels, ood_synthetic_labels], dim=0)
        
        # Compute feature statistics for normalization (using all data)
        all_features = torch.cat([id_all_features, ood_all_features], dim=0)
        detector.feature_mean = all_features.mean(dim=0)
        detector.feature_std = all_features.std(dim=0) + 1e-8
        
        # Normalize features
        id_all_features = (id_all_features - detector.feature_mean) / detector.feature_std
        ood_all_features = (ood_all_features - detector.feature_mean) / detector.feature_std
        
        # Initialize synthetic sample weights (higher for synthetic samples)
        id_weights = torch.ones(id_features.size(0), device=detector.device)
        id_synthetic_weights = torch.ones(id_synthetic_features.size(0), device=detector.device) * 2.0
        
        ood_weights = torch.ones(ood_features.size(0), device=detector.device)
        ood_synthetic_weights = torch.ones(ood_synthetic_features.size(0), device=detector.device) * 2.0
        
        id_all_weights = torch.cat([id_weights, id_synthetic_weights], dim=0)
        ood_all_weights = torch.cat([ood_weights, ood_synthetic_weights], dim=0)
        
        # Normalize weights
        id_all_weights = id_all_weights / id_all_weights.sum()
        ood_all_weights = ood_all_weights / ood_all_weights.sum()
        
        # Multiple rounds of hard negative mining
        for round_idx in range(detector.mining_rounds):
            print(f"\n=== Hard Negative Mining Round {round_idx+1}/{detector.mining_rounds} ===")
            
            # Combine features and labels
            combined_features = torch.cat([id_all_features, ood_all_features], dim=0)
            combined_labels = torch.cat([
                torch.ones(id_all_features.size(0), device=detector.device),
                -torch.ones(ood_all_features.size(0), device=detector.device)
            ])
            combined_weights = torch.cat([id_all_weights, ood_all_weights], dim=0)
            
            # Solve DRO problem
            print(f"Solving DRO with epsilon={detector.epsilon:.4f}, reg_param={detector.reg_param:.4f}")
            theta, bias, obj_value = detector.solve_dro_problem(
                combined_features, combined_labels, 
                epsilon=detector.epsilon, 
                sample_weights=combined_weights
            )
            
            if theta is None:
                print("DRO optimization failed! Trying with simpler parameters...")
                # Try with simpler parameters
                detector.epsilon *= 0.5
                detector.reg_param *= 10
                
                theta, bias, obj_value = detector.solve_dro_problem(
                    combined_features, combined_labels,
                    epsilon=detector.epsilon,
                    reg_param=detector.reg_param
                )
                
                if theta is None:
                    print("Optimization still failed. Breaking.")
                    break
            
            # Update model parameters
            detector.theta = torch.tensor(theta, device=detector.device)
            detector.bias = torch.tensor(bias, device=detector.device)
            
            # Evaluate current model
            metrics = detector.evaluate(id_loader, ood_loader)
            
            # Store history
            detector.history.append({
                'round': round_idx,
                'epsilon': detector.epsilon,
                'reg_param': detector.reg_param,
                'theta_norm': np.linalg.norm(theta),
                'obj_value': obj_value,
                'metrics': metrics
            })
            
            # Find hard negatives for next round (from original data only)
            hard_id_features, hard_id_indices, hard_ood_features, hard_ood_indices = \
                detector.find_hard_negatives(id_loader, ood_loader, detector.theta, detector.bias)
            
            detector.hard_id_indices = hard_id_indices
            detector.hard_ood_indices = hard_ood_indices
            
            # If no hard negatives found, we're done
            if len(hard_id_indices) == 0 and len(hard_ood_indices) == 0:
                print("No hard negatives found. Early stopping.")
                break
            
            # Update sample weights to focus on hard negatives
            # Reset weights
            id_weights = torch.ones(id_features.size(0), device=detector.device)
            ood_weights = torch.ones(ood_features.size(0), device=detector.device)
            
            # Increase weights for hard negatives
            for idx in hard_id_indices:
                if idx < id_weights.size(0):
                    id_weights[idx] *= 3.0
            
            for idx in hard_ood_indices:
                if idx < ood_weights.size(0):
                    ood_weights[idx] *= 3.0
            
            # Re-combine weights
            id_all_weights = torch.cat([id_weights, id_synthetic_weights], dim=0)
            ood_all_weights = torch.cat([ood_weights, ood_synthetic_weights], dim=0)
            
            # Normalize weights
            id_all_weights = id_all_weights / id_all_weights.sum()
            ood_all_weights = ood_all_weights / ood_all_weights.sum()
            
            # Adjust Wasserstein radius based on hard negative rate
            hard_negative_rate = (len(hard_id_indices) + len(hard_ood_indices)) / \
                                (id_features.size(0) + ood_features.size(0))
            
            print(f"Hard negative rate: {hard_negative_rate:.4f}")
            
            # Increase epsilon if too many hard negatives
            if hard_negative_rate > 0.2:
                detector.epsilon *= 1.5
                print(f"Increasing epsilon to {detector.epsilon:.4f}")
            
            # Small regularization increase each round for stability
            detector.reg_param *= 1.1
        
        return detector
    
    # Use our custom fit method
    detector = fit_with_synthetic_data(
        detector, 
        id_loader, 
        ood_loader,
        id_synthetic_features, 
        ood_synthetic_features
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = detector.evaluate(id_loader, ood_loader)
    
    # Visualize results
    detector.visualize_results(
        id_loader, ood_loader,
        save_path=os.path.join(output_dir, "dro_synthetic_results.png")
    )
    
    # Print final DRO parameters
    print("\nFinal DRO Parameters:")
    print(f"Wasserstein radius (epsilon): {detector.epsilon:.4f}")
    print(f"Regularization parameter: {detector.reg_param:.4f}")
    print(f"Decision boundary norm: {torch.norm(detector.theta).item():.4f}")
    
    # Save results
    results = {
        'metrics': final_metrics,
        'epsilon': detector.epsilon,
        'reg_param': detector.reg_param,
        'theta_norm': torch.norm(detector.theta).item(),
        'history': detector.history,
        'num_id_synthetic': id_synthetic_features.size(0),
        'num_ood_synthetic': ood_synthetic_features.size(0)
    }
    
    import pickle
    with open(os.path.join(output_dir, "dro_synthetic_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")
    
    return detector, final_metrics


class DROWithHardNegativeMining:
    """DRO with hard negative mining for OOD detection"""
    
    def __init__(self, epsilon=0.1, reg_param=0.01):
        self.epsilon = epsilon
        self.reg_param = reg_param
    
    def solve_dro_problem(self, features, labels):
        """
        Solve the DRO optimization problem with hard negative mining
        
        Args:
            features: Feature vectors (N x D)
            labels: Binary labels (N)
            
        Returns:
            theta: Normal vector of the decision boundary
            bias: Bias term of the decision boundary
            obj_value: Objective value of the optimization
        """
        try:
            # Convert to numpy for CVXPY
            features_np = features.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            # Problem dimensions
            n_samples, n_features = features_np.shape
            
            # Define variables
            theta = cp.Variable(n_features)
            bias = cp.Variable()
            
            # Calculate margins
            margins = features_np @ theta + bias
            
            # Define objective: maximize margin with regularization
            obj = cp.sum(margins[labels_np == 1]) - cp.sum(margins[labels_np == 0])
            obj = obj - self.reg_param * cp.norm(theta, 2)
            
            # DRO constraints
            constraints = [
                cp.norm(theta, 2) <= 1,  # Unit norm constraint
                margins >= -self.epsilon,  # Margin constraints for positive samples
                margins <= self.epsilon   # Margin constraints for negative samples
            ]
            
            # Solve the problem
            prob = cp.Problem(cp.Maximize(obj), constraints)
            prob.solve()
            
            if prob.status != cp.OPTIMAL:
                print(f"Warning: Problem status is {prob.status}")
                return None, None, None
            
            return theta.value, float(bias.value), float(prob.value)
            
        except Exception as e:
            print(f"Error in DRO optimization: {e}")
            return None, None, None


def main():
    """Main training function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print(f"Loading ID dataset: {args.id_dataset}")
    id_loader = get_dataset_loader(
        args.data_dir,
        args.id_dataset,
        args.batch_size,
        is_training=True
    )
    
    print(f"Loading OOD dataset: {args.ood_dataset}")
    ood_loader = get_dataset_loader(
        args.data_dir,
        args.ood_dataset,
        args.batch_size,
        is_training=False  # Use test set for OOD
    )
    
    # Create feature extractor (example with ResNet18)
    print("Creating feature extractor...")
    feature_extractor = torchvision.models.resnet18(pretrained=True)
    feature_extractor.fc = nn.Identity()  # Remove final FC layer
    feature_dim = 512  # ResNet18's feature dimension
    
    # Train the model
    print("\nStarting training...")
    feature_extractor, theta, bias, metrics = train_dro_with_synthetic_negatives(
        id_loader=id_loader,
        ood_loader=ood_loader,
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        device=device,
        output_dir=args.output_dir,
        texture_path=args.texture_path,
        num_synthetic_per_real=args.num_synthetic_per_real
    )
    
    # Print final metrics
    print("\nFinal Results:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    return feature_extractor, metrics

if __name__ == "__main__":
    feature_extractor, metrics = main()
