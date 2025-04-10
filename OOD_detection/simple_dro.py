import cvxpy as cp
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_ood.model import WideResNet
from tqdm import tqdm

def dro_ood_detection_formulation(X_train, radius=0.1, reg_param=1.0, batch_size=1000):
    """
    Simplified DRO formulation using batched processing for better memory management.
    """
    n, d = X_train.shape
    print(f"Problem size: {n} samples, {d} features")
    
    # Variables
    theta = cp.Variable(d)
    b = cp.Variable()
    
    # Process in batches
    n_batches = (n + batch_size - 1) // batch_size
    objectives = []
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        
        # Create parameter matrix for current batch
        X_batch = cp.Parameter(shape=(end_idx - start_idx, d))
        X_batch.value = X_train[start_idx:end_idx]
        
        # Compute loss for current batch
        predictions = X_batch @ theta + b
        batch_losses = cp.sum(cp.logistic(-predictions))
        objectives.append(batch_losses)
    
    # Average loss across all batches
    avg_loss = sum(objectives) / n
    
    # Add regularization (use quad_form for better numerical stability)
    reg_term = cp.quad_form(theta, np.eye(d))
    
    # Total objective
    objective = avg_loss + reg_param * reg_term
    
    # Create problem
    prob = cp.Problem(cp.Minimize(objective))
    
    return prob, theta, b

def learn_dro_ood_model(X_train, radius=0.1, reg_param=1.0):
    """
    Learn the DRO model with improved numerical stability and error handling.
    """
    # Normalize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    
    try:
        # Try with different batch sizes
        for batch_size in [1000, 500, 200]:
            print(f"Attempting optimization with batch size {batch_size}")
            prob, theta, b = dro_ood_detection_formulation(
                X_train_norm, radius, reg_param, batch_size=batch_size
            )
            
            # Conservative solver settings
            solver_opts = {
                'max_iters': 1000,
                'eps': 1e-3,
                'verbose': True,
                'use_indirect': True,  # Use indirect solver
                'normalize': True      # Enable normalization
            }
            
            try:
                result = prob.solve(solver=cp.SCS, **solver_opts)
                if prob.status in ["optimal", "optimal_inaccurate"]:
                    break
            except Exception as e:
                print(f"Failed with batch size {batch_size}: {str(e)}")
                continue
        
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            print("Optimization failed with all batch sizes")
            return np.zeros(X_train.shape[1]), 0.0
        
        # Rescale parameters back
        theta_value = theta.value / X_std
        b_value = b.value + np.sum(theta_value * X_mean)
        
        print(f"Optimization successful with status: {prob.status}")
        print(f"Final objective value: {prob.value}")
        
        return theta_value, b_value
        
    except Exception as e:
        print(f"Unexpected error in optimization: {str(e)}")
        return np.zeros(X_train.shape[1]), 0.0

def batch_dro_training(X_train, radius=0.1, reg_param=1.0, batch_size=1000):
    """
    Train DRO model in batches with more efficient parameter handling and improved stability
    """
    n, d = X_train.shape
    print(f"Problem size: {n} samples, {d} features")
    
    # Normalize features
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    
    # Define variables once (reused across batches)
    theta = cp.Variable(d)
    b = cp.Variable()
    
    # Initialize theta and b with small random values
    theta_value = np.random.randn(d) * 0.01
    b_value = 0.0
    
    # Number of batches
    num_batches = (n + batch_size - 1) // batch_size
    print(f"Processing {num_batches} batches of size {batch_size}")
    
    # Track best solution
    best_loss = float('inf')
    best_theta = None
    best_b = None
    
    # Create solver options
    solver_opts = {
        'max_iters': 2000,  # Reduced for faster per-batch solving
        'eps': 1e-4,       # Relaxed tolerance
        'verbose': False,   # Reduce output noise
        'normalize': True,  # Enable normalization
        'use_indirect': True  # Use indirect solver
    }
    
    for epoch in range(3):  # Multiple epochs for better convergence
        print(f"\nEpoch {epoch+1}")
        epoch_losses = []
        
        # Shuffle data for each epoch
        perm = np.random.permutation(n)
        X_shuffled = X_train_norm[perm]
        
        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n)
            X_batch = X_shuffled[start_idx:end_idx]
            batch_n = end_idx - start_idx
            
            # Reset variables with current values for warm start
            theta.value = theta_value
            b.value = b_value
            
            # Define batch-specific parameters
            X = cp.Parameter((batch_n, d))
            X.value = X_batch
            
            # Batch loss calculation (vectorized)
            predictions = X @ theta + b
            losses = cp.logistic(-predictions)
            
            # Simplified Wasserstein penalty
            wasserstein_penalty = radius * cp.norm(theta, 2)
            
            # Total objective with normalized losses
            objective = cp.sum(losses)/batch_n + wasserstein_penalty + reg_param * cp.sum_squares(theta)
            
            # Solve problem for this batch
            prob = cp.Problem(cp.Minimize(objective))
            
            try:
                result = prob.solve(solver=cp.SCS, ignore_dpp=True, **solver_opts)
                epoch_losses.append(result)
                
                if prob.status not in ["optimal", "optimal_inaccurate"]:
                    solver_opts['eps'] = 1e-3  # Relax tolerance further
                    result = prob.solve(solver=cp.ECOS, ignore_dpp=True, **solver_opts)
            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                continue
            
            # Update parameters with momentum
            if theta.value is not None:
                # Momentum update
                momentum = 0.9
                theta_value = momentum * theta_value + (1 - momentum) * theta.value
                b_value = momentum * b_value + (1 - momentum) * b.value
                
                # Track best solution
                if result < best_loss and not np.any(np.isnan(theta.value)):
                    best_loss = result
                    best_theta = theta_value.copy()
                    best_b = b_value
        
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
    
    # Use best found solution
    if best_theta is not None:
        print("Using best solution found")
        theta_value = best_theta
        b_value = best_b
    
    # Rescale parameters back
    theta_final = theta_value / X_std
    b_final = b_value + np.sum(theta_final * X_mean)
    
    print(f"Training complete. Final theta norm: {np.linalg.norm(theta_final):.4f}, bias: {b_final:.4f}")
    return theta_final, b_final

def score_ood(X_test, theta, b, feature_extractor=None):
    """
    Score test samples with improved numerical stability.
    """
    if theta is None or b is None:
        print("Warning: Invalid model parameters, returning default scores")
        return np.zeros(len(X_test))
    
    try:
        # Compute scores with clipping for numerical stability
        scores = X_test @ theta + b
        scores = np.clip(scores, -100, 100)  # Prevent numerical overflow
        return -scores  # Higher score indicates more likely to be OOD
    except Exception as e:
        print(f"Error in scoring: {str(e)}")
        return np.zeros(len(X_test))

def feature_based_dro_formulation(X_train, transformations, radius=0.1, 
                                 consistency_weight=0.5, reg_param=1.0):
    """
    Formulate the feature-based DRO-OOD detection with transformation consistency.
    
    Args:
        X_train: Training data matrix, samples from the normal distribution
        transformations: List of transformation functions to apply to data
        radius: Uncertainty radius for the Wasserstein ball
        consistency_weight: Weight for the consistency regularization term
        reg_param: Regularization parameter
    
    Returns:
        prob: CVXPY problem formulation
        theta: Learned parameters
    """
    n, d = X_train.shape  # n samples, d dimensions
    num_transforms = len(transformations)
    
    # Decision variables
    theta = cp.Variable(d)  # Model parameters
    b = cp.Variable()       # Bias term
    
    # For each data point, we need a dual variable for the Wasserstein constraint
    lambdas = cp.Variable(n, nonneg=True)  # Dual variables
    
    # Base robust loss
    robust_loss = 0
    for i in range(n):
        x_i = X_train[i]
        orig_loss = cp.logistic(-(theta.T @ x_i + b))
        adv_term = radius * cp.norm(theta, 2) * lambdas[i]
        robust_loss += orig_loss + adv_term
    
    # Transformation consistency loss
    consistency_loss = 0
    for i in range(n):
        x_i = X_train[i]
        # Apply each transformation
        for t in range(num_transforms):
            # In practice, we'd precompute these transformations
            # Here we just represent it abstractly
            x_i_transformed = transformations[t](x_i)
            
            # Compute difference in outputs between original and transformed
            consistency_term = cp.square(theta.T @ x_i + b - theta.T @ x_i_transformed - b)
            consistency_loss += consistency_term
    
    # Regularization term
    regularizer = reg_param * cp.norm(theta, 2)**2
    
    # Total objective with consistency regularization
    objective = (robust_loss / n) + (consistency_weight * consistency_loss / (n * num_transforms)) + regularizer
    
    # Create the optimization problem
    prob = cp.Problem(cp.Minimize(objective))
    
    return prob, theta, b

# Implementation of the Mahalanobis distance-based extension (DROCC-LF)
def mahalanobis_dro_formulation(X_train, y_train, feature_weights=None, radius=0.1, reg_param=1.0):
    """
    Formulate the DRO-OOD detection with Mahalanobis distance.
    
    Args:
        X_train: Training data matrix
        y_train: Labels (1 for ID, -1 for known OOD if available)
        feature_weights: Feature importance weights
        radius: Uncertainty radius
        reg_param: Regularization parameter
    
    Returns:
        prob: CVXPY problem
        theta: Parameters
    """
    n, d = X_train.shape
    
    # If feature weights not provided, use uniform weights
    if feature_weights is None:
        feature_weights = np.ones(d)
    
    # Create weight matrix (diagonal matrix with feature weights)
    W = np.diag(feature_weights)
    
    # Decision variables
    theta = cp.Variable(d)
    b = cp.Variable()
    lambdas = cp.Variable(n, nonneg=True)
    
    # Robust loss with Mahalanobis distance
    robust_loss = 0
    for i in range(n):
        x_i = X_train[i]
        # Original loss
        orig_loss = cp.logistic(-(theta.T @ x_i + b) * y_train[i])
        
        # Adversarial perturbation with Mahalanobis distance
        # This is a modified norm incorporating the weight matrix W
        weighted_norm = cp.norm(W @ theta, 2)
        adv_term = radius * weighted_norm * lambdas[i]
        
        robust_loss += orig_loss + adv_term
    
    # Regularization
    regularizer = reg_param * cp.norm(theta, 2)**2
    
    # Total objective
    objective = robust_loss / n + regularizer
    
    # Create problem
    prob = cp.Problem(cp.Minimize(objective))
    
    return prob, theta, b

def lagrangian_dual_dro(X_train, radius=0.1, reg_param=1.0, max_iter=100, batch_size=1000):
    """
    Solve the DRO-OOD detection problem using Lagrangian dual approach with batching
    """
    n, d = X_train.shape
    print(f"Starting dual optimization with {n} samples, {d} features")
    
    # Normalize features for better numerical stability
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0) + 1e-8
    X_train_norm = (X_train - X_mean) / X_std
    
    # Initialize variables
    theta = cp.Variable(d)
    b = cp.Variable()
    
    # Initialize perturbations with small random values
    perturbations = np.random.randn(n, d) * 1e-4
    
    # Process in batches
    n_batches = (n + batch_size - 1) // batch_size
    best_objective = float('inf')
    best_theta = None
    best_b = None
    
    for iteration in tqdm(range(max_iter), desc="Dual optimization"):
        # Shuffle data for each iteration
        perm = np.random.permutation(n)
        X_shuffled = X_train_norm[perm]
        pert_shuffled = perturbations[perm]
        
        epoch_losses = []
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n)
            
            X_batch = X_shuffled[start_idx:end_idx]
            pert_batch = pert_shuffled[start_idx:end_idx]
            
            # Create parameters for current batch
            X_param = cp.Parameter(shape=(end_idx - start_idx, d))
            X_param.value = X_batch
            X_pert_param = cp.Parameter(shape=(end_idx - start_idx, d))
            X_pert_param.value = X_batch + pert_batch
            
            # Compute losses
            orig_loss = cp.sum(cp.logistic(-(X_param @ theta + b)))
            pert_loss = cp.sum(cp.logistic(X_pert_param @ theta + b))
            reg_loss = reg_param * cp.sum_squares(theta)
            
            # Total objective for this batch
            batch_size = end_idx - start_idx
            objective = (orig_loss + pert_loss) / batch_size + reg_loss
            
            # Solve the subproblem
            prob = cp.Problem(cp.Minimize(objective))
            try:
                prob.solve(solver=cp.SCS, eps=1e-3, max_iters=500, verbose=False)
                epoch_losses.append(prob.value)
                
                # Update best solution if we found a better one
                if prob.value < best_objective and prob.status == "optimal":
                    best_objective = prob.value
                    best_theta = theta.value.copy()
                    best_b = b.value
            except Exception as e:
                print(f"Batch optimization failed: {e}")
                continue
        
        # Update perturbations for all samples
        if theta.value is not None:
            grad_norms = np.zeros(n)
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n)
                X_batch = X_train_norm[start_idx:end_idx]
                
                # Compute gradients
                z = -(X_batch @ theta.value + b.value)
                sigmoid_z = 1 / (1 + np.exp(-z))
                grads = -np.outer(sigmoid_z, theta.value)
                
                # Update perturbations
                grad_norms_batch = np.linalg.norm(grads, axis=1)
                mask = grad_norms_batch > 1e-10
                grads[mask] = grads[mask] / grad_norms_batch[mask, np.newaxis]
                perturbations[start_idx:end_idx] = radius * grads
                grad_norms[start_idx:end_idx] = grad_norms_batch
            
            max_change = np.max(grad_norms)
            avg_loss = np.mean(epoch_losses)
            print(f"Iteration {iteration+1}, Avg Loss: {avg_loss:.4f}, Max grad norm: {max_change:.4f}")
            
            # Check convergence
            if max_change < 1e-4:
                print("Converged!")
                break
    
    # Rescale parameters back
    if best_theta is not None:
        theta_value = best_theta / X_std
        b_value = best_b + np.sum(theta_value * X_mean)
    else:
        print("Warning: No valid solution found")
        theta_value = np.zeros(d)
        b_value = 0.0
    
    return theta_value, b_value

def main():
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR10 stats
    ])

    # Load datasets with tqdm
    print("Loading datasets...")
    cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(cifar10_train, batch_size=32, shuffle=True)
    
    svhn_test = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, 
                              transform=transforms.Compose([
                                  transforms.Resize(32),
                                  transforms.Grayscale(3),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                              ]))
    
    svhn_loader = DataLoader(svhn_test, batch_size=32, shuffle=False)
    mnist_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    # Initialize WideResNet
    model = WideResNet(
        num_classes=10,
        depth=40,
        widen_factor=2,
        drop_rate=0.3,
        in_channels=3,
        pretrained='oe-cifar10-tune'
    )
    model = model.to(device)
    model.eval()

    # Extract features with tqdm
    X_train = []
    print("Extracting features from CIFAR10 training data...")
    with torch.no_grad():
        for images, _ in tqdm(train_loader, desc="Processing CIFAR10"):
            images = images.to(device)
            features = model.features(images)
            X_train.append(features.cpu().numpy())
    
    X_train = np.concatenate(X_train, axis=0)
    
    # Train DRO model
    print("Training DRO model...")
    theta, b = batch_dro_training(
        X_train,
        radius=0.1,
        reg_param=1.0
    )
    
    # Test on SVHN with tqdm
    print("Testing on SVHN...")
    svhn_scores = []
    with torch.no_grad():
        for images, _ in tqdm(svhn_loader, desc="Processing SVHN"):
            images = images.to(device)
            features = model.features(images)
            scores = score_ood(features.cpu().numpy(), theta, b)
            svhn_scores.extend(scores)
    
    # Test on MNIST with tqdm
    print("Testing on MNIST...")
    mnist_scores = []
    with torch.no_grad():
        for images, _ in tqdm(mnist_loader, desc="Processing MNIST"):
            images = images.to(device)
            features = model.features(images)
            scores = score_ood(features.cpu().numpy(), theta, b)
            mnist_scores.extend(scores)
    
    # Calculate and print results
    print("\nResults:")
    print(f"Average OOD score for SVHN: {np.mean(svhn_scores):.4f}")
    print(f"Average OOD score for MNIST: {np.mean(mnist_scores):.4f}")
    
    return theta, b, model, svhn_scores, mnist_scores

if __name__ == "__main__":
    main()
