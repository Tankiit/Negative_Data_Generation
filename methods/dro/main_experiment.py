import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score

def get_device():
    """Helper function to get the best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class CalibratedDROEnergy(nn.Module):
    """
    An improved implementation of alternating DRO and Energy-based optimization
    for OOD detection with proper calibration techniques for better performance.
    """
    def __init__(
        self,
        feature_dim=512,
        num_classes=10,
        temperature=1.0,
        dro_reg=0.01,
        dropout_rate=0.3,
        device=None
    ):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store configuration
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dro_reg = dro_reg
        self.dropout_rate = dropout_rate
        
        # Create feature extractor
        self.feature_extractor = self._create_feature_extractor()
        
        # Create classifier with dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, num_classes)
        ).to(self.device)
        
        # Initialize temperature parameter for energy scoring
        self.temperature = temperature
        self.temperature_param = nn.Parameter(torch.tensor([temperature], device=self.device))
        
        # Initialize DRO parameters
        # Instead of using separate w and b parameters, use a proper linear layer
        self.dro_layer = nn.Linear(feature_dim, 1).to(self.device)
        # Initialize with zeros
        nn.init.zeros_(self.dro_layer.weight)
        nn.init.zeros_(self.dro_layer.bias)
        
        # Feature statistics for normalization
        self.register_buffer('feature_mean', torch.zeros(feature_dim, device=self.device))
        self.register_buffer('feature_std', torch.ones(feature_dim, device=self.device))
        
        # Energy calibration parameters
        self.register_buffer('energy_mean', torch.tensor(0.0, device=self.device))
        self.register_buffer('energy_std', torch.tensor(1.0, device=self.device))
        
        # DRO calibration parameters
        self.register_buffer('dro_mean', torch.tensor(0.0, device=self.device))
        self.register_buffer('dro_std', torch.tensor(1.0, device=self.device))
        
        # Combination weight for the two methods
        self.alpha = 0.5
        
        # OOD threshold (calibrated later)
        self.threshold = 0.0
        
        # Calibration tracking
        self.is_calibrated = False
        
    def _create_feature_extractor(self):
        """Create a simple feature extractor using ResNet18"""
        # Use a pre-trained model with the classifier removed
        model = torchvision.models.resnet18(pretrained=True)
        # Remove the final classification layer
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        feature_extractor.to(self.device)
        return feature_extractor
    
    def extract_features(self, x):
        """Extract features from input"""
        self.feature_extractor.eval()  # Always use eval mode for feature extraction
        with torch.no_grad():
            # Handle different input sizes and channels
            if x.dim() == 3:
                x = x.unsqueeze(0)  # Add batch dimension
            
            # Handle grayscale images by repeating channels
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
                
            # Extract features
            features = self.feature_extractor(x)
            
            # Flatten features
            features = features.view(features.size(0), -1)
            
            return features
    
    def normalize_features(self, features):
        """Normalize features using stored statistics"""
        return (features - self.feature_mean) / (self.feature_std + 1e-6)
    
    def compute_energy(self, logits):
        """Compute energy scores from logits"""
        # Ensure positive temperature
        temp = torch.abs(self.temperature_param) + 1e-6
        
        # Energy function: -T * log(sum(exp(logits/T)))
        return -temp * torch.logsumexp(logits / temp, dim=1)
    
    def get_dro_score(self, features):
        """Compute DRO-based OOD score"""
        # Normalize features
        normalized = self.normalize_features(features)
        
        # Compute DRO score (linear decision boundary)
        return self.dro_layer(normalized).squeeze(-1)
    
    def forward(self, x):
        """Forward pass for both classification and OOD detection"""
        # Extract features
        features = self.extract_features(x)
        
        # Get classifier logits
        logits = self.classifier(features)
        
        # Compute raw energy score
        energy_score = self.compute_energy(logits)
        
        # Compute raw DRO score
        dro_score = self.get_dro_score(features)
        
        # Apply calibration if available
        if self.is_calibrated:
            # Normalize both scores
            norm_energy = (energy_score - self.energy_mean) / (self.energy_std + 1e-6)
            norm_dro = (dro_score - self.dro_mean) / (self.dro_std + 1e-6)
            
            # Combine scores with calibrated alpha (lower score = more likely ID)
            combined_score = self.alpha * norm_energy + (1 - self.alpha) * norm_dro
        else:
            # Without calibration, just use raw scores with default alpha
            combined_score = self.alpha * energy_score + (1 - self.alpha) * dro_score
        
        return logits, combined_score
    
    def predict_ood(self, x):
        """Predict whether samples are OOD using calibrated threshold"""
        _, scores = self.forward(x)
        predictions = (scores > self.threshold).int()
        return predictions, scores
    
    def generate_synthetic_ood(self, x):
        """Generate synthetic OOD samples for training with improved diversity"""
        batch_size = x.size(0)
        channels = x.size(1)
        h, w = x.size(2), x.size(3)
        
        # Create a list of transformation functions
        transforms = [
            # Noise-based transformations
            lambda x: torch.randn_like(x).to(self.device),
            lambda x: x + 0.5 * torch.randn_like(x).to(self.device),
            
            # Color-based transformations
            lambda x: x.roll(shifts=1, dims=1),  # Channel shift
            lambda x: 1.0 - x,  # Inversion
            
            # Geometric transformations
            lambda x: x.flip(2),  # Horizontal flip
            lambda x: x.flip(3),  # Vertical flip
            
            # Blending transformations
            lambda x: 0.5 * x + 0.5 * x[torch.randperm(batch_size)].to(self.device),  # Mix samples
            lambda x: 0.7 * x + 0.3 * torch.rand_like(x).to(self.device)  # Add uniform noise
        ]
        
        # Select random transformations for each sample to increase diversity
        synthetic_samples = []
        for i in range(batch_size):
            # Apply 1-3 random transformations
            num_transforms = torch.randint(1, 4, (1,)).item()
            transform_indices = torch.randperm(len(transforms))[:num_transforms]
            
            sample = x[i:i+1]
            for idx in transform_indices:
                sample = transforms[idx](sample)
            
            # Ensure values are in valid range (0-1)
            sample = torch.clamp(sample, 0, 1)
            synthetic_samples.append(sample)
        
        return torch.cat(synthetic_samples, dim=0)
    
    def train_energy_component(self, train_loader, val_loader=None, epochs=1, lr=0.001):
        """Train the energy-based component with validation-based early stopping"""
        print("\n=== Training Energy Component ===")
        self.train()
        
        # Setup optimizer
        optimizer = optim.Adam([
            {'params': self.classifier.parameters()},
            {'params': [self.temperature_param], 'lr': lr * 0.1}
        ], lr=lr)
        
        # Classification loss
        ce_criterion = nn.CrossEntropyLoss()
        
        # For early stopping
        best_val_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        energy_values = []
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Progress bar
            pbar = tqdm(train_loader, desc=f"Energy Epoch {epoch+1}/{epochs}")
            
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Generate synthetic OOD samples
                synthetic_ood = self.generate_synthetic_ood(inputs)
                
                optimizer.zero_grad()
                
                # Extract features
                features = self.extract_features(inputs)
                ood_features = self.extract_features(synthetic_ood)
                
                # Forward pass for ID samples
                logits = self.classifier(features)
                
                # Classification loss
                ce_loss = ce_criterion(logits, targets)
                
                # Energy scores
                energy_id = self.compute_energy(logits)
                energy_values.extend(energy_id.detach().cpu().numpy())
                
                # OOD logits
                ood_logits = self.classifier(ood_features)
                energy_ood = self.compute_energy(ood_logits)
                
                # Energy loss: ID should have lower energy than OOD
                # Using margin-based contrastive loss
                margin = 10.0
                energy_loss = torch.mean(F.relu(energy_id - energy_ood.mean() + margin))
                
                # Temperature regularization
                temp_reg = 0.1 * (self.temperature_param - 1.0).abs()
                
                # Total loss
                loss = ce_loss + 0.1 * energy_loss + temp_reg
                
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item() * batch_size
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss/total:.3f}",
                    'acc': f"{100.*correct/total:.2f}%",
                    'T': f"{self.temperature_param.item():.2f}"
                })
            
            # Validation phase if validation loader is provided
            if val_loader is not None:
                val_loss, val_acc = self._validate_energy(val_loader)
                print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Update energy calibration using training data
        self.energy_mean = torch.tensor(np.mean(energy_values), device=self.device)
        self.energy_std = torch.tensor(np.std(energy_values), device=self.device)
        print(f"Updated energy stats - Mean: {self.energy_mean.item():.3f}, Std: {self.energy_std.item():.3f}")
        
        return self
    
    def _validate_energy(self, val_loader):
        """Validate energy model performance"""
        self.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        ce_criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Extract features and get logits
                features = self.extract_features(inputs)
                logits = self.classifier(features)
                
                # Classification loss
                loss = ce_criterion(logits, targets)
                
                # Update metrics
                total_loss += loss.item() * batch_size
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += batch_size
                
        val_loss = total_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train_dro_component(self, train_loader, val_loader=None, epochs=1, lr=0.001):
        """Train the DRO component using PyTorch optimizers with validation"""
        print("\n=== Training DRO Component ===")
        self.eval()  # Set model to eval mode for feature extraction
        
        # First compute feature statistics
        print("Computing feature statistics...")
        features_list = []
        ood_features_list = []
        
        # Extract features from ID samples
        with torch.no_grad():
            for inputs, _ in tqdm(train_loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
                
                # Extract features
                features = self.extract_features(inputs)
                features_list.append(features)
                
                # Generate synthetic OOD samples
                synthetic_ood = self.generate_synthetic_ood(inputs)
                ood_features = self.extract_features(synthetic_ood)
                ood_features_list.append(ood_features)
                
                # Limit memory usage
                if len(features_list) >= 20:
                    break
        
        # Concatenate all features
        id_features = torch.cat(features_list, dim=0)
        ood_features = torch.cat(ood_features_list, dim=0)
        
        # Update feature statistics
        self.feature_mean = id_features.mean(dim=0)
        self.feature_std = id_features.std(dim=0) + 1e-6
        
        # Now normalize the features
        id_features = self.normalize_features(id_features)
        ood_features = self.normalize_features(ood_features)
        
        # Create labels: 1 for ID, -1 for OOD
        id_labels = torch.ones(id_features.size(0), device=self.device)
        ood_labels = -torch.ones(ood_features.size(0), device=self.device)
        
        # Combine data
        all_features = torch.cat([id_features, ood_features], dim=0)
        all_labels = torch.cat([id_labels, ood_labels], dim=0)
        
        # Create balanced weights
        id_weight = 1.0 / id_features.size(0)
        ood_weight = 1.0 / ood_features.size(0)
        total_weight = id_weight + ood_weight
        weights = torch.cat([
            torch.ones(id_features.size(0), device=self.device) * (id_weight / total_weight),
            torch.ones(ood_features.size(0), device=self.device) * (ood_weight / total_weight)
        ])
        
        # Create dataset and loader for DRO training
        dro_dataset = TensorDataset(all_features, all_labels, weights)
        dro_loader = DataLoader(dro_dataset, batch_size=128, shuffle=True)
        
        # For early stopping
        best_val_accuracy = 0
        patience = 3
        patience_counter = 0
        
        # Collect DRO scores for calibration
        dro_scores = []
        
        # Optimize DRO parameters
        optimizer = optim.Adam(self.dro_layer.parameters(), lr=lr, weight_decay=self.dro_reg)
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(dro_loader, desc=f"DRO Epoch {epoch+1}/{epochs}")
            
            for features, labels, sample_weights in pbar:
                features = features.to(self.device)
                labels = labels.to(self.device)
                sample_weights = sample_weights.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                scores = self.dro_layer(features).squeeze(-1)
                dro_scores.extend(scores.detach().cpu().numpy())
                
                # Hinge loss for classification
                hinge_loss = torch.mean(
                    sample_weights * F.relu(1.0 - labels * scores)
                )
                
                # L2 regularization is handled by optimizer
                
                # Wasserstein DRO regularization (approximated)
                wasserstein_reg = self.dro_reg * torch.norm(self.dro_layer.weight)
                
                # Total loss
                loss = hinge_loss + wasserstein_reg
                
                loss.backward()
                optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                predictions = (scores > 0).float() * 2 - 1  # Convert to -1/+1
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss/(pbar.n+1):.3f}",
                    'acc': f"{100.*correct/total:.2f}%",
                    'w_norm': f"{torch.norm(self.dro_layer.weight).item():.3f}"
                })
            
            # Validation if provided
            if val_loader is not None:
                val_accuracy = self._validate_dro(val_loader)
                print(f"Validation Accuracy: {val_accuracy:.2f}%")
                
                # Early stopping
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    patience_counter = 0
                    print(f"New best validation accuracy: {best_val_accuracy:.2f}%")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
        
        # Update DRO calibration using collected scores
        self.dro_mean = torch.tensor(np.mean(dro_scores), device=self.device)
        self.dro_std = torch.tensor(np.std(dro_scores), device=self.device)
        print(f"Updated DRO stats - Mean: {self.dro_mean.item():.3f}, Std: {self.dro_std.item():.3f}")
        
        print(f"DRO training complete. W norm: {torch.norm(self.dro_layer.weight).item():.4f}, b: {self.dro_layer.bias.item():.4f}")
        return self
    
    def _validate_dro(self, val_loader):
        """Validate DRO component on a validation set"""
        self.eval()
        correct = 0
        total = 0
        
        # Generate OOD samples for validation
        val_features_list = []
        val_ood_features_list = []
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                
                # Extract features
                features = self.extract_features(inputs)
                val_features_list.append(features)
                
                # Generate synthetic OOD for validation
                synthetic_ood = self.generate_synthetic_ood(inputs)
                ood_features = self.extract_features(synthetic_ood)
                val_ood_features_list.append(ood_features)
                
                # Limit to reduce computation
                if len(val_features_list) >= 10:
                    break
        
        # Process collected features
        val_id_features = torch.cat(val_features_list, dim=0)
        val_ood_features = torch.cat(val_ood_features_list, dim=0)
        
        # Normalize features
        val_id_features = self.normalize_features(val_id_features)
        val_ood_features = self.normalize_features(val_ood_features)
        
        # Create labels: 1 for ID, -1 for OOD
        id_labels = torch.ones(val_id_features.size(0), device=self.device)
        ood_labels = -torch.ones(val_ood_features.size(0), device=self.device)
        
        # Score ID samples
        id_scores = self.dro_layer(val_id_features).squeeze(-1)
        id_preds = (id_scores > 0).float() * 2 - 1
        correct += (id_preds == id_labels).sum().item()
        total += id_labels.size(0)
        
        # Score OOD samples
        ood_scores = self.dro_layer(val_ood_features).squeeze(-1)
        ood_preds = (ood_scores > 0).float() * 2 - 1
        correct += (ood_preds == ood_labels).sum().item()
        total += ood_labels.size(0)
        
        # Calculate accuracy
        accuracy = 100. * correct / total
        
        return accuracy
    
    def evaluate(self, id_loader, ood_loader=None):
        """Evaluate OOD detection performance with detailed metrics"""
        if not self.is_calibrated:
            print("\nWARNING: Model is not calibrated. Running evaluation with default parameters.")
            print("For better results, calibrate the model first with model.calibrate().")
        
        self.eval()
    
        id_scores = []
        ood_scores = []
        correct = 0
        total = 0
    
        # Process ID data
        print("Evaluating on ID data...")
        with torch.no_grad():
            for inputs, targets in tqdm(id_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                logits, scores = self.forward(inputs)
                id_scores.extend(scores.cpu().numpy())
                
                # Count correct predictions
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
    
        id_accuracy = 100. * correct / total
        print(f"ID Classification Accuracy: {id_accuracy:.2f}%")
    
        # Process OOD data if provided
        if ood_loader is not None:
            print("Evaluating on OOD data...")
            with torch.no_grad():
                for inputs, _ in tqdm(ood_loader):
                    inputs = inputs.to(self.device)
                    
                    # Forward pass
                    _, scores = self.forward(inputs)
                    ood_scores.extend(scores.cpu().numpy())
        else:
            print("No OOD loader provided. Generating synthetic OOD data...")
            # Generate synthetic OOD data for evaluation
            synthetic_ood_scores = []
            with torch.no_grad():
                for inputs, _ in tqdm(id_loader):
                    inputs = inputs.to(self.device)
                    synthetic_ood = self.generate_synthetic_ood(inputs)
                    _, scores = self.forward(synthetic_ood)
                    synthetic_ood_scores.extend(scores.cpu().numpy())
            
            ood_scores = synthetic_ood_scores
        
        # Convert to numpy arrays
        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        
        # Higher score = more likely OOD
        y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
        y_score = np.concatenate([id_scores, ood_scores])
        
        # ROC curve and AUROC
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auroc = roc_auc_score(y_true, y_score)
        
        # PR curve and AUPR
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        aupr = auc(recall, precision)
        
        # Find threshold for 95% TPR
        tpr_95_idx = np.argmin(np.abs(tpr - 0.95))
        fpr_at_95tpr = fpr[tpr_95_idx]
        threshold_95tpr = thresholds[tpr_95_idx]
        
        # Also compute FPR at 90% and 99% TPR
        tpr_90_idx = np.argmin(np.abs(tpr - 0.90))
        fpr_at_90tpr = fpr[tpr_90_idx]
        
        tpr_99_idx = np.argmin(np.abs(tpr - 0.99))
        fpr_at_99tpr = fpr[tpr_99_idx]
        
        # Compute detection metrics using calibrated threshold
        threshold = self.threshold if self.is_calibrated else thresholds[np.argmax(tpr - fpr)]
        y_pred = (y_score >= threshold).astype(int)
        detection_accuracy = (y_pred == y_true).mean() * 100
        
        # Compute TPR and FPR at the calibrated threshold
        idx_thresh = np.argmin(np.abs(thresholds - threshold))
        tpr_at_thresh = tpr[idx_thresh]
        fpr_at_thresh = fpr[idx_thresh]
        
        # Print results
        print("\n=== OOD Detection Results ===")
        print(f"AUROC: {auroc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        print(f"FPR@90%TPR: {fpr_at_90tpr:.4f}")
        print(f"FPR@95%TPR: {fpr_at_95tpr:.4f}")
        print(f"FPR@99%TPR: {fpr_at_99tpr:.4f}")
        print(f"TPR@Threshold({threshold:.4f}): {tpr_at_thresh:.4f}")
        print(f"FPR@Threshold({threshold:.4f}): {fpr_at_thresh:.4f}")
        print(f"OOD Detection Accuracy: {detection_accuracy:.2f}%")
        
        # Plot results if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 5))
            
            # Plot ROC curve
            plt.subplot(1, 3, 1)
            plt.plot(fpr, tpr, label=f'AUROC = {auroc:.4f}')
            plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
            plt.scatter(fpr_at_95tpr, 0.95, color='red', marker='o', label=f'FPR@95%TPR = {fpr_at_95tpr:.4f}')
            plt.scatter(fpr_at_thresh, tpr_at_thresh, color='green', marker='x', 
                      label=f'Threshold = {threshold:.4f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            
            # Plot PR curve
            plt.subplot(1, 3, 2)
            plt.plot(recall, precision, label=f'AUPR = {aupr:.4f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            
            # Plot score distributions
            plt.subplot(1, 3, 3)
            bins = np.linspace(min(np.min(id_scores), np.min(ood_scores)),
                              max(np.max(id_scores), np.max(ood_scores)), 50)
            plt.hist(id_scores, bins=bins, alpha=0.5, label='ID', density=True)
            plt.hist(ood_scores, bins=bins, alpha=0.5, label='OOD', density=True)
            plt.axvline(x=threshold, color='k', linestyle='--', label=f'Threshold = {threshold:.4f}')
            plt.xlabel('OOD Score')
            plt.ylabel('Density')
            plt.title('Score Distributions')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('ood_evaluation.png')
            print("Visualization saved to 'ood_evaluation.png'")
            plt.close()
        except Exception as e:
            print(f"Could not create visualization: {e}")
        
        # Return results as dictionary
        return {
            'id_accuracy': id_accuracy,
            'auroc': auroc,
            'aupr': aupr,
            'fpr_at_90tpr': fpr_at_90tpr,
            'fpr_at_95tpr': fpr_at_95tpr,
            'fpr_at_99tpr': fpr_at_99tpr,
            'threshold': threshold,
            'tpr_at_threshold': tpr_at_thresh,
            'fpr_at_threshold': fpr_at_thresh,
            'detection_accuracy': detection_accuracy,
            'alpha': self.alpha
        }
    
    def train_alternating(self, train_loader, val_loader=None, num_iterations=3, energy_epochs=1, dro_epochs=1, 
                        energy_lr=0.001, dro_lr=0.001):
        """Train using alternating optimization with validation-based tuning"""
        print("\n=== Starting Alternating Optimization ===")
        
        # Create validation set if not provided
        if val_loader is None and train_loader is not None:
            # Use 10% of training data for validation
            train_data = train_loader.dataset
            val_size = int(0.1 * len(train_data))
            train_size = len(train_data) - val_size
            
            train_subset, val_subset = random_split(
                train_data, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            # Create new data loaders
            train_loader_new = DataLoader(
                train_subset, 
                batch_size=train_loader.batch_size,
                shuffle=True,
                num_workers=train_loader.num_workers
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=train_loader.batch_size,
                shuffle=False,
                num_workers=train_loader.num_workers
            )
            
            # Use the new training loader
            train_loader = train_loader_new
        
        for i in range(num_iterations):
            print(f"\n--- Alternation {i+1}/{num_iterations} ---")
            
            # Step 1: Train energy component
            self.train_energy_component(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=energy_epochs,
                lr=energy_lr
            )
            
            # Step 2: Train DRO component
            self.train_dro_component(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=dro_epochs,
                lr=dro_lr
            )
        
        print("\nAlternating optimization completed.")
        return self
    
    def calibrate(self, val_loader, ood_val_loader=None, alpha_grid=None):
        """
        Calibrate the detector using validation data:
        1. Recalibrate energy and DRO scores
        2. Find optimal alpha weight
        3. Find optimal threshold
        """
        print("\n=== Calibrating OOD Detector ===")
        self.eval()
        
        # If OOD validation loader not provided, create synthetic OOD
        if ood_val_loader is None:
            print("Creating synthetic OOD data for calibration...")
            # Extract validation features and create synthetic OOD
            val_features_list = []
            ood_features_list = []
            val_logits_list = []
            ood_logits_list = []
            
            with torch.no_grad():
                for inputs, _ in tqdm(val_loader, desc="Generating calibration data"):
                    inputs = inputs.to(self.device)
                    
                    # Get ID features and logits
                    features = self.extract_features(inputs)
                    logits = self.classifier(features)
                    
                    val_features_list.append(features)
                    val_logits_list.append(logits)
                    
                    # Generate synthetic OOD samples
                    synthetic_ood = self.generate_synthetic_ood(inputs)
                    ood_features = self.extract_features(synthetic_ood)
                    ood_logits = self.classifier(ood_features)
                    
                    ood_features_list.append(ood_features)
                    ood_logits_list.append(ood_logits)
            
            # Process ID data
            val_features = torch.cat(val_features_list, dim=0)
            val_logits = torch.cat(val_logits_list, dim=0)
            
            # Process synthetic OOD data
            ood_features = torch.cat(ood_features_list, dim=0)
            ood_logits = torch.cat(ood_logits_list, dim=0)
        else:
            # Use real OOD validation data
            print("Using provided OOD validation data for calibration...")
            val_features_list = []
            ood_features_list = []
            val_logits_list = []
            ood_logits_list = []
            
            # Process ID validation data
            with torch.no_grad():
                for inputs, _ in tqdm(val_loader, desc="Processing ID validation data"):
                    inputs = inputs.to(self.device)
                    features = self.extract_features(inputs)
                    logits = self.classifier(features)
                    
                    val_features_list.append(features)
                    val_logits_list.append(logits)
            
            # Process OOD validation data
            with torch.no_grad():
                for inputs, _ in tqdm(ood_val_loader, desc="Processing OOD validation data"):
                    inputs = inputs.to(self.device)
                    features = self.extract_features(inputs)
                    logits = self.classifier(features)
                    
                    ood_features_list.append(features)
                    ood_logits_list.append(logits)
            
            # Concatenate data
            val_features = torch.cat(val_features_list, dim=0)
            val_logits = torch.cat(val_logits_list, dim=0)
            ood_features = torch.cat(ood_features_list, dim=0)
            ood_logits = torch.cat(ood_logits_list, dim=0)
        
        # 1. Recalibrate energy scores
        id_energy = self.compute_energy(val_logits)
        ood_energy = self.compute_energy(ood_logits)
        
        self.energy_mean = torch.mean(id_energy)
        self.energy_std = torch.std(id_energy)
        
        # 2. Recalibrate DRO scores
        id_dro = self.dro_layer(self.normalize_features(val_features)).squeeze(-1)
        ood_dro = self.dro_layer(self.normalize_features(ood_features)).squeeze(-1)
        
        self.dro_mean = torch.mean(id_dro)
        self.dro_std = torch.std(id_dro)
        
        print(f"Calibrated Energy - Mean: {self.energy_mean.item():.4f}, Std: {self.energy_std.item():.4f}")
        print(f"Calibrated DRO - Mean: {self.dro_mean.item():.4f}, Std: {self.dro_std.item():.4f}")
        
        # 3. Find optimal alpha weight
        # Default grid if not provided
        if alpha_grid is None:
            alpha_grid = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
            
        best_alpha = 0.5  # Default
        best_auroc = 0.0
        
        # Normalize scores with calibration values
        norm_id_energy = (id_energy - self.energy_mean) / (self.energy_std + 1e-6)
        norm_ood_energy = (ood_energy - self.energy_mean) / (self.energy_std + 1e-6)
        
        norm_id_dro = (id_dro - self.dro_mean) / (self.dro_std + 1e-6)
        norm_ood_dro = (ood_dro - self.dro_mean) / (self.dro_std + 1e-6)
        
        print("\nFinding optimal alpha weight...")
        for alpha in alpha_grid:
            # Combine scores
            combined_id = alpha * norm_id_energy + (1 - alpha) * norm_id_dro
            combined_ood = alpha * norm_ood_energy + (1 - alpha) * norm_ood_dro
            
            # Create labels: 0 for ID, 1 for OOD
            y_true = np.concatenate([
                np.zeros(len(combined_id)),
                np.ones(len(combined_ood))
            ])
            
            # Combine scores
            y_score = np.concatenate([
                combined_id.cpu().numpy(),
                combined_ood.cpu().numpy()
            ])
            
            # Compute AUROC
            auroc = roc_auc_score(y_true, y_score)
            
            print(f"Alpha = {alpha:.2f}: AUROC = {auroc:.4f}")
            
            if auroc > best_auroc:
                best_auroc = auroc
                best_alpha = alpha
        
        # Set best alpha
        self.alpha = best_alpha
        print(f"\nOptimal alpha = {self.alpha:.4f} with AUROC = {best_auroc:.4f}")
        
        # 4. Find optimal threshold using best alpha
        # Combine scores with best alpha
        combined_id = self.alpha * norm_id_energy + (1 - self.alpha) * norm_id_dro
        combined_ood = self.alpha * norm_ood_energy + (1 - self.alpha) * norm_ood_dro
        
        # Create labels and scores
        y_true = np.concatenate([
            np.zeros(len(combined_id)),
            np.ones(len(combined_ood))
        ])
        
        y_score = np.concatenate([
            combined_id.cpu().numpy(),
            combined_ood.cpu().numpy()
        ])
        
        # Find threshold that maximizes F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        
        # Calculate F1 scores for each threshold
        f1_scores = []
        for prec, rec in zip(precision[:-1], recall[:-1]):  # precision has one more element
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
            f1_scores.append(f1)
        
        # Find threshold with best F1 score
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        self.threshold = optimal_threshold
        
        print(f"Optimal threshold = {self.threshold:.4f} with F1 score = {f1_scores[best_idx]:.4f}")
        
        # Mark as calibrated
        self.is_calibrated = True
        
        return self
def get_dataset(name, root, train=True, download=True):
    """Get dataset by name"""
    if name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return torchvision.datasets.CIFAR10(root=root, train=train, 
                                          download=download, transform=transform)
    
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return torchvision.datasets.CIFAR100(root=root, train=train,
                                           download=download, transform=transform)
    
    elif name == 'tinyimagenet':
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        return TinyImageNet(root=root, train=train, transform=transform)
    
    elif name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ])
        split = 'train' if train else 'test'
        return torchvision.datasets.SVHN(root=root, split=split,
                                       download=download, transform=transform)
    
    else:
        raise ValueError(f"Unsupported dataset: {name}")

def main():
    """Example usage with different models and datasets"""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    data_root = './data'
    model_configs = [
        {'name': 'resnet18', 'num_classes': 200},
        {'name': 'resnet50', 'num_classes': 200},
        {'name': 'wide_resnet50_2', 'num_classes': 200}
    ]
    
    # Dataset configuration
    id_dataset = 'tinyimagenet'
    ood_datasets = ['cifar100', 'svhn']  # Datasets to use as OOD
    batch_size = 128
    num_workers = 4
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load datasets
    print(f"\nLoading ID dataset: {id_dataset}")
    train_dataset = get_dataset(id_dataset, data_root, train=True)
    test_dataset = get_dataset(id_dataset, data_root, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    
    # Load OOD datasets
    ood_loaders = {}
    for ood_dataset in ood_datasets:
        print(f"Loading OOD dataset: {ood_dataset}")
        ood_data = get_dataset(ood_dataset, data_root, train=False)
        ood_loaders[ood_dataset] = DataLoader(ood_data, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    
    # Train and evaluate each model
    for config in model_configs:
        model_name = config['name']
        num_classes = config['num_classes']
        
        print(f"\n=== Training {model_name} ===")
        
        # Create model
        model = CalibratedDROEnergy(
            model_name=model_name,
            num_classes=num_classes,
            device=device
        )
        
        # Train model
        model.train_alternating(
            train_loader=train_loader,
            num_iterations=3,
            energy_epochs=5,
            dro_epochs=5
        )
        
        # Calibrate model
        model.calibrate(
            val_loader=test_loader,
            ood_val_loader=ood_loaders['cifar100'],
            alpha_grid=np.linspace(0.0, 1.0, 11)
        )
        
        # Evaluate on each OOD dataset
        print("\nEvaluating model:")
        for ood_name, ood_loader in ood_loaders.items():
            print(f"\nTesting against {ood_name}:")
            results = model.evaluate(test_loader, ood_loader)
            
            print(f"AUROC: {results['auroc']:.4f}")
            print(f"AUPR: {results['aupr']:.4f}")
            print(f"FPR@95TPR: {results['fpr_at_95tpr']:.4f}")
            print(f"ID Accuracy: {results['id_accuracy']:.2f}%")

if __name__ == "__main__":
    main()
