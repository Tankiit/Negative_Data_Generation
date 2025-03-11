import torch
import numpy as np
import argparse
import itertools
import json
import os
from tqdm import tqdm
from Energy_Detector import create_feature_extractor, DROEnergyDetector, evaluate_detector
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="DRO-Energy Hyperparameter Tuning")
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='/Users/tanmoy/research/data', 
                        help='Data directory')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'resnet50', 'wideresnet'], 
                        help='Model architecture')
    parser.add_argument('--hierarchical', action='store_true', 
                        help='Use hierarchical features')
    
    # Device
    parser.add_argument('--device', type=str, default='auto', 
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use')
    
    # Hyperparameters to tune
    parser.add_argument('--alpha-values', type=float, nargs='+', 
                        default=[0.2, 0.3, 0.4, 0.5], 
                        help='Weight values for DRO score')
    parser.add_argument('--beta-values', type=float, nargs='+', 
                        default=[0.3, 0.4, 0.5, 0.6], 
                        help='Weight values for energy score')
    parser.add_argument('--gamma-values', type=float, nargs='+', 
                        default=[0.1, 0.2, 0.3], 
                        help='Weight values for Mahalanobis score')
    parser.add_argument('--temperature-values', type=float, nargs='+', 
                        default=[0.5, 1.0, 1.5, 2.0], 
                        help='Temperature values for energy score')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./tuning_results', 
                        help='Directory to save tuning results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == 'auto':
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    
    return args

def setup_data_loaders(args):
    """Set up data loaders for training and evaluation"""
    # Set up transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    cifar10_train = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    cifar10_val = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    svhn_test = torchvision.datasets.SVHN(
        root=args.data_dir, split='test', download=True, transform=transform
    )
    
    # Create smaller datasets for faster tuning
    train_indices = np.random.choice(len(cifar10_train), size=10000, replace=False)
    val_indices = np.random.choice(len(cifar10_val), size=1000, replace=False)
    svhn_indices = np.random.choice(len(svhn_test), size=1000, replace=False)
    
    train_subset = torch.utils.data.Subset(cifar10_train, train_indices)
    val_subset = torch.utils.data.Subset(cifar10_val, val_indices)
    svhn_subset = torch.utils.data.Subset(svhn_test, svhn_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    ood_loader = DataLoader(
        svhn_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, ood_loader

def tune_hyperparameters(args):
    """Run hyperparameter tuning"""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data loaders
    train_loader, val_loader, ood_loader = setup_data_loaders(args)
    
    # Create feature extractor
    feature_extractor, feature_dim = create_feature_extractor(args.model)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.eval()
    
    # Generate parameter combinations with weight normalization
    param_combinations = []
    
    for alpha in args.alpha_values:
        for beta in args.beta_values:
            for gamma in args.gamma_values:
                for temperature in args.temperature_values:
                    # Normalize weights to sum to 1
                    total = alpha + beta + gamma
                    norm_alpha = alpha / total
                    norm_beta = beta / total
                    norm_gamma = gamma / total
                    
                    param_combinations.append({
                        'alpha': norm_alpha,
                        'beta': norm_beta,
                        'gamma': norm_gamma,
                        'temperature': temperature
                    })
    
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    # Create detector
    detector = DROEnergyDetector(
        feature_extractor=feature_extractor,
        feature_dim=feature_dim,
        num_classes=10,
        batch_size=args.batch_size,
        device=args.device,
        use_hierarchical_features=args.hierarchical
    ).to(args.device)
    
    # Train detector once
    print("\nTraining base detector...")
    success = detector.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        energy_epochs=3  # Fewer epochs for tuning
    )
    
    if not success:
        print("Training failed!")
        return
    
    # Results storage
    results = []
    
    # Evaluate with different hyperparameters
    print("\nEvaluating hyperparameter combinations...")
    for params in tqdm(param_combinations):
        alpha = params['alpha']
        beta = params['beta']
        gamma = params['gamma']
        temperature = params['temperature']
        
        # Update detector temperature
        detector.energy_temperature = temperature
        
        # Custom forward function for this specific parameter set
        def custom_score_fn(x):
            with torch.no_grad():
                features = detector.extract_hierarchical_features(x)
                
                # DRO score
                dro_score = detector.get_dro_score(features)
                
                # Energy score
                energy_score = detector.get_energy_score(features)
                
                # Mahalanobis score
                mahal_score = detector.get_mahalanobis_score(features)
                
                # Normalize scores to similar ranges
                dro_score = dro_score / (dro_score.abs().max() + 1e-8)
                energy_score = energy_score / (energy_score.abs().max() + 1e-8)
                mahal_score = mahal_score / (mahal_score.max() + 1e-8)
                
                # Combine scores with current parameters
                combined_score = alpha * dro_score + beta * energy_score + gamma * mahal_score
                
                return combined_score
        
        # Temporarily replace forward method
        original_forward = detector.forward
        detector.forward = custom_score_fn
        
        # Evaluate
        metrics = evaluate_detector(
            detector, val_loader, ood_loader, args.device, detailed=False
        )
        
        # Restore original forward method
        detector.forward = original_forward
        
        # Store results
        param_results = {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'temperature': temperature,
            'auroc': metrics['auroc'],
            'aupr': metrics['aupr'],
            'fpr': metrics['fpr']
        }
        
        results.append(param_results)
        
        print(f"\nParameters: alpha={alpha:.2f}, beta={beta:.2f}, gamma={gamma:.2f}, temp={temperature:.1f}")
        print(f"AUROC: {metrics['auroc']:.4f}, AUPR: {metrics['aupr']:.4f}, FPR@95: {metrics['fpr']:.4f}")
    
    # Find best parameters
    best_result = max(results, key=lambda x: x['auroc'])
    print("\nBest parameters:")
    print(f"Alpha: {best_result['alpha']:.4f}")
    print(f"Beta: {best_result['beta']:.4f}")
    print(f"Gamma: {best_result['gamma']:.4f}")
    print(f"Temperature: {best_result['temperature']:.4f}")
    print(f"AUROC: {best_result['auroc']:.4f}")
    print(f"AUPR: {best_result['aupr']:.4f}")
    print(f"FPR@95: {best_result['fpr']:.4f}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'tuning_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save best parameters
    with open(os.path.join(args.output_dir, 'best_params.json'), 'w') as f:
        json.dump(best_result, f, indent=4)
    
    return best_result

if __name__ == "__main__":
    args = parse_args()
    best_params = tune_hyperparameters(args)