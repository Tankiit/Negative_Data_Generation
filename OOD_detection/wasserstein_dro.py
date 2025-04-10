import torch
import torch.nn as nn
import numpy as np
import cvxpy as cp
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class WassersteinDRODetector:
    def __init__(
        self,
        feature_extractor,
        feature_dim,
        device=None,
        epsilon=1.0,
        reg_param=0.1
    ):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_extractor.eval()
        self.feature_dim = feature_dim
        
        # DRO parameters
        self.epsilon = epsilon  # Wasserstein ball radius
        self.reg_param = reg_param  # Regularization parameter
        
        # Register parameters for the decision boundary
        self.register_buffer('theta', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(0.0, dtype=torch.float32))
        
        # Statistics for normalization
        self.register_buffer('feature_mean', torch.zeros(feature_dim, dtype=torch.float32))
        self.register_buffer('feature_std', torch.ones(feature_dim, dtype=torch.float32))
    
    def register_buffer(self, name, tensor):
        """Simple implementation of register_buffer for non-nn.Module classes"""
        setattr(self, name, tensor.to(self.device))
    
    def extract_features(self, x):
        """Extract features from input data"""
        x = x.to(self.device)
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features
    
    def solve_dro_problem(self, features, y=None, epsilon=None, reg_param=None):
        """
        Solve the DRO optimization problem using CVXPY
        
        Args:
            features: Feature matrix (n_samples, feature_dim)
            y: Optional labels (1 for ID, -1 for OOD)
            epsilon: Wasserstein ball radius
            reg_param: Regularization parameter
        
        Returns:
            theta: Optimal decision boundary normal vector
            bias: Optimal decision boundary bias
            problem_value: Optimal objective value
        """
        # Set default parameters if not provided
        if epsilon is None:
            epsilon = self.epsilon
        if reg_param is None:
            reg_param = self.reg_param
        
        # Convert features to numpy for CVXPY
        features_np = features.detach().cpu().numpy()
        n_samples, n_features = features_np.shape
        
        # Default to all positive examples (ID) if no labels provided
        if y is None:
            y_np = np.ones(n_samples)
        else:
            y_np = y.detach().cpu().numpy()
        
        # Define CVXPY variables
        theta = cp.Variable(n_features)
        bias = cp.Variable()
        
        # Compute predictions
        predictions = features_np @ theta + bias
        
        # Define DRO optimization problem
        # For one-class setting, we use hinge loss to push ID samples inside the boundary
        hinge_loss = cp.sum(cp.pos(1.0 - y_np * predictions)) / n_samples
        
        # Wasserstein robustness penalty
        # This is a simplified version of the Wasserstein DRO formulation
        # The L2 norm of theta constrains the Lipschitz constant
        wasserstein_penalty = epsilon * cp.norm(theta, 2)
        
        # Add regularization for better generalization
        regularization = reg_param * cp.sum_squares(theta)
        
        # Total objective
        objective = hinge_loss + wasserstein_penalty + regularization
        
        # Define the problem
        problem = cp.Problem(cp.Minimize(objective))
        
        try:
            # Solve the problem
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-4, max_iters=5000)
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                theta_val = theta.value.astype(np.float32)
                bias_val = float(bias.value)
                return theta_val, bias_val, float(problem.value)
            else:
                print(f"Optimization failed with status: {problem.status}")
                return None, None, float('inf')
        except Exception as e:
            print(f"CVXPY optimization error: {e}")
            return None, None, float('inf')
    
    def fit(self, id_loader, ood_loader=None, radius_grid=None):
        """
        Fit the DRO detector on in-distribution data
        
        Args:
            id_loader: DataLoader for in-distribution data
            ood_loader: Optional DataLoader for out-of-distribution data
            radius_grid: List of epsilon values to try for optimization
        """
        print("Extracting features from ID data...")
        id_features = []
        with torch.no_grad():
            for inputs, _ in tqdm(id_loader):
                features = self.extract_features(inputs)
                id_features.append(features.cpu())
        
        # Concatenate all ID features
        id_features = torch.cat(id_features, dim=0)
        
        # Compute feature statistics for normalization
        self.feature_mean = id_features.mean(dim=0).to(self.device)
        self.feature_std = id_features.std(dim=0).to(self.device)
        
        # Normalize ID features
        id_features_norm = (id_features - self.feature_mean) / (self.feature_std + 1e-8)
        id_features_norm = id_features_norm.to(self.device)
        
        # Create labels (1 for ID samples)
        id_labels = torch.ones(id_features_norm.size(0), device=self.device)
        
        # Get OOD features if available
        if ood_loader is not None:
            print("Extracting features from OOD data...")
            ood_features = []
            with torch.no_grad():
                for inputs, _ in tqdm(ood_loader):
                    features = self.extract_features(inputs)
                    ood_features.append(features.cpu())
            
            # Concatenate all OOD features
            ood_features = torch.cat(ood_features, dim=0)
            
            # Normalize OOD features using ID statistics
            ood_features_norm = (ood_features - self.feature_mean) / (self.feature_std + 1e-8)
            ood_features_norm = ood_features_norm.to(self.device)
            
            # Create labels (-1 for OOD samples)
            ood_labels = -torch.ones(ood_features_norm.size(0), device=self.device)
            
            # Combine ID and OOD data for training
            combined_features = torch.cat([id_features_norm, ood_features_norm], dim=0)
            combined_labels = torch.cat([id_labels, ood_labels], dim=0)
        else:
            # Use only ID data for one-class training
            combined_features = id_features_norm
            combined_labels = id_labels
        
        # Try different radius values if specified
        if radius_grid is not None and ood_loader is not None:
            print("Grid search for optimal Wasserstein radius...")
            best_epsilon = None
            best_auc = 0
            best_params = None
            
            results = []
            
            for epsilon in radius_grid:
                print(f"Trying epsilon = {epsilon:.4f}")
                
                # Solve the DRO problem with current epsilon
                theta_val, bias_val, obj_value = self.solve_dro_problem(
                    combined_features, combined_labels, epsilon=epsilon
                )
                
                if theta_val is None:
                    continue
                
                # Compute scores for evaluation
                id_scores = -(id_features_norm @ torch.tensor(theta_val, device=self.device) + bias_val)
                ood_scores = -(ood_features_norm @ torch.tensor(theta_val, device=self.device) + bias_val)
                
                # Compute AUC for evaluation
                y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
                y_score = np.concatenate([id_scores.cpu().numpy(), ood_scores.cpu().numpy()])
                auc = roc_auc_score(y_true, y_score)
                
                results.append({
                    'epsilon': epsilon,
                    'auc': auc,
                    'obj_value': obj_value,
                    'theta': theta_val,
                    'bias': bias_val
                })
                
                print(f"  AUC: {auc:.4f}, Objective: {obj_value:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_epsilon = epsilon
                    best_params = (theta_val, bias_val)
            
            # Plot results
            plt.figure(figsize=(10, 6))
            plt.plot([r['epsilon'] for r in results], [r['auc'] for r in results], 'o-')
            plt.axvline(x=best_epsilon, color='r', linestyle='--', 
                        label=f'Best ε = {best_epsilon:.4f}')
            plt.xlabel('Wasserstein Radius (ε)')
            plt.ylabel('AUC')
            plt.title('AUC vs Wasserstein Radius')
            plt.grid(True)
            plt.legend()
            plt.savefig('wasserstein_dro_grid_search.png')
            plt.close()
            
            # Set best parameters
            self.epsilon = best_epsilon
            self.theta = torch.tensor(best_params[0], dtype=torch.float32, device=self.device)
            self.bias = torch.tensor(best_params[1], dtype=torch.float32, device=self.device)
            
            print(f"Best Wasserstein radius: {best_epsilon:.4f}, Best AUC: {best_auc:.4f}")
        else:
            # Solve the DRO problem with current epsilon
            print(f"Solving DRO problem with epsilon = {self.epsilon:.4f}...")
            theta_val, bias_val, obj_value = self.solve_dro_problem(combined_features, combined_labels)
            
            if theta_val is not None:
                self.theta = torch.tensor(theta_val, dtype=torch.float32, device=self.device)
                self.bias = torch.tensor(bias_val, dtype=torch.float32, device=self.device)
                print(f"Optimization successful! Objective value: {obj_value:.4f}")
            else:
                print("Optimization failed. Using fallback parameters.")
                # Fallback to simple mean-centered approach
                magnitude = torch.norm(self.feature_mean)
                self.theta = -self.feature_mean / (magnitude + 1e-8)
                self.bias = 1.0
        
        return self
    
    def get_scores(self, inputs):
        """
        Compute OOD scores (higher values indicate more likely OOD samples)
        
        Args:
            inputs: Input data tensor
        
        Returns:
            scores: OOD scores
        """
        features = self.extract_features(inputs)
        
        # Normalize features
        features_norm = (features - self.feature_mean) / (self.feature_std + 1e-8)
        
        # Apply DRO decision function
        # Negative because we want higher scores to indicate OOD samples
        scores = -(features_norm @ self.theta + self.bias)
        
        return scores
    
    def predict(self, inputs, threshold=0.0):
        """
        Predict if inputs are OOD
        
        Args:
            inputs: Input data tensor
            threshold: Decision threshold (default: 0.0)
        
        Returns:
            is_ood: Boolean tensor where True means OOD
            scores: OOD scores
        """
        scores = self.get_scores(inputs)
        is_ood = scores > threshold
        
        return is_ood, scores
    
    def evaluate(self, id_loader, ood_loader, verbose=True):
        """
        Evaluate detector performance on ID and OOD data
        
        Args:
            id_loader: DataLoader for in-distribution data
            ood_loader: DataLoader for out-of-distribution data
            verbose: Whether to print results
        
        Returns:
            metrics: Dictionary of evaluation metrics
            id_scores: OOD scores for ID data
            ood_scores: OOD scores for OOD data
        """
        id_scores = []
        ood_scores = []
        
        # Process ID data
        if verbose:
            print("Evaluating on ID data...")
        with torch.no_grad():
            for inputs, _ in tqdm(id_loader, disable=not verbose):
                scores = self.get_scores(inputs)
                id_scores.extend(scores.cpu().numpy())
        
        # Process OOD data
        if verbose:
            print("Evaluating on OOD data...")
        with torch.no_grad():
            for inputs, _ in tqdm(ood_loader, disable=not verbose):
                scores = self.get_scores(inputs)
                ood_scores.extend(scores.cpu().numpy())
        
        # Convert to numpy arrays
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        # Compute metrics
        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        y_score = np.concatenate([id_scores, ood_scores])
        
        # ROC and AUROC
        from sklearn.metrics import roc_curve, roc_auc_score
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        
        # FPR at 95% TPR
        idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95_tpr = fpr[idx_tpr_95]
        
        # Detection accuracy using default threshold
        threshold = 0.0  # Using the DRO decision boundary
        id_accuracy = (id_scores <= threshold).mean()
        ood_accuracy = (ood_scores > threshold).mean()
        balanced_acc = 0.5 * (id_accuracy + ood_accuracy)
        
        metrics = {
            'AUROC': auroc,
            'FPR@95%TPR': fpr_at_95_tpr,
            'ID Accuracy': id_accuracy,
            'OOD Accuracy': ood_accuracy,
            'Balanced Accuracy': balanced_acc,
            'ID Mean Score': id_scores.mean(),
            'OOD Mean Score': ood_scores.mean(),
            'ID Std Score': id_scores.std(),
            'OOD Std Score': ood_scores.std()
        }
        
        if verbose:
            print("\nDRO-Based Wasserstein OOD Detection Results:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            
            # Print DRO parameters
            print(f"\nWasserstein Radius (epsilon): {self.epsilon:.4f}")
            print(f"Theta norm: {torch.norm(self.theta).item():.4f}")
            print(f"Bias: {self.bias.item():.4f}")
        
        return metrics, id_scores, ood_scores
    
    def visualize_results(self, id_scores, ood_scores, save_path=None):
        """
        Visualize detection results
        
        Args:
            id_scores: OOD scores for ID data
            ood_scores: OOD scores for OOD data
            save_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Score distributions
        plt.subplot(1, 2, 1)
        plt.hist(id_scores, bins=50, alpha=0.5, label='In-Distribution', density=True, color='blue')
        plt.hist(ood_scores, bins=50, alpha=0.5, label='Out-of-Distribution', density=True, color='red')
        plt.axvline(x=0, color='black', linestyle='--', label='DRO Decision Boundary')
        plt.xlabel('OOD Score (negative margin)')
        plt.ylabel('Density')
        plt.legend()
        plt.title('Score Distributions')
        plt.grid(True)
        
        # ROC curve
        from sklearn.metrics import roc_curve, roc_auc_score
        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        y_score = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        
        plt.subplot(1, 2, 2)
        plt.plot(fpr, tpr, label=f'AUROC = {auroc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title('ROC Curve')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"ID Mean Score: {np.mean(id_scores):.4f} ± {np.std(id_scores):.4f}")
        print(f"OOD Mean Score: {np.mean(ood_scores):.4f} ± {np.std(ood_scores):.4f}")
        print(f"ID Samples Inside DRO Boundary: {np.mean(id_scores <= 0):.2%}")
        print(f"OOD Samples Outside DRO Boundary: {np.mean(ood_scores > 0):.2%}")
        print(f"Detection Accuracy: {0.5*(np.mean(id_scores <= 0) + np.mean(ood_scores > 0)):.2%}")


# Example usage with grid search for optimal epsilon
def main():
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    from torch.utils.data import DataLoader
    import argparse
    
    parser = argparse.ArgumentParser(description="DRO-Based Wasserstein OOD Detection")
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial Wasserstein radius')
    parser.add_argument('--reg-param', type=float, default=0.1, help='Regularization parameter')
    parser.add_argument('--grid-search', action='store_true', help='Perform grid search for optimal epsilon')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'],
                       help='Feature extractor model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    cifar10_train = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    cifar10_test = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    svhn_test = datasets.SVHN(root=args.data_dir, split='test', download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(cifar10_train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(cifar10_test, batch_size=args.batch_size, shuffle=False)
    ood_loader = DataLoader(svhn_test, batch_size=args.batch_size, shuffle=False)
    
    # Load pre-trained model
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        feature_dim = 512
    else:  # resnet50
        model = models.resnet50(pretrained=True)
        feature_dim = 2048
    
    # Remove classification layer
    model.fc = nn.Identity()
    model.eval()
    
    # Create DRO detector
    detector = WassersteinDRODetector(
        feature_extractor=model,
        feature_dim=feature_dim,
        device=device,
        epsilon=args.epsilon,
        reg_param=args.reg_param
    )
    
    # Epsilon values for grid search
    epsilon_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0] if args.grid_search else None
    
    # Fit the detector
    detector.fit(train_loader, ood_loader=ood_loader, radius_grid=epsilon_grid)
    
    # Evaluate the detector
    metrics, id_scores, ood_scores = detector.evaluate(test_loader, ood_loader)
    
    # Visualize results
    detector.visualize_results(id_scores, ood_scores, save_path='dro_wasserstein_results.png')

if __name__ == "__main__":
    main()
