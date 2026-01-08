"""
Witness OOD Detection: Realistic Benchmark Comparison

Standard benchmark: CIFAR-10 (ID) vs SVHN (OOD)
Compare against established methods: MSP, ODIN, Energy, Mahalanobis

Key: Show witness consistently outperforms on realistic task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm


class WideResNet(nn.Module):
    """
    Stronger backbone for realistic performance.
    Based on standard OOD detection benchmarks.
    """
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Block 3
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, return_features=False):
        # Block 1
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out, 2)
        
        # Block 2
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.max_pool2d(out, 2)
        
        # Block 3
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.max_pool2d(out, 2)
        
        features = self.pool(out).view(out.size(0), -1)
        logits = self.fc(self.dropout(features))
        
        if return_features:
            return logits, features
        return logits


class WitnessWideResNet(nn.Module):
    """WideResNet with witness head for epistemic uncertainty."""
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)
        
        # Witness head: monitors features from all blocks
        self.witness = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(self.bn4(self.conv4(out)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.max_pool2d(out, 2)
        
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.max_pool2d(out, 2)
        
        features = self.pool(out).view(out.size(0), -1)
        logits = self.fc(self.dropout(features))
        uncertainty = self.witness(features)
        
        if return_features:
            return logits, features, uncertainty
        return logits, uncertainty


def train_model(model, loader, device, epochs, is_witness=False):
    """Train with optional witness head."""
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if is_witness:
                logits, uncertainty = model(imgs)
                loss_class = criterion(logits, labels)
                
                # Witness loss: predict entropy from multiple passes
                with torch.no_grad():
                    entropies = []
                    for _ in range(3):
                        l, _ = model(imgs)
                        p = F.softmax(l, dim=1)
                        entropies.append(p)
                    mean_p = torch.stack(entropies).mean(0)
                    target_unc = -torch.sum(mean_p * torch.log(mean_p + 1e-8), dim=1) / np.log(10)
                
                loss_witness = F.mse_loss(torch.sigmoid(uncertainty.squeeze()), target_unc)
                loss = loss_class + 0.5 * loss_witness
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})
        
        scheduler.step()


def extract_features(model, loader, device):
    """Extract features for Mahalanobis."""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, features = model(imgs, return_features=True)
            features_list.append(features.cpu())
            labels_list.append(labels)
    
    return torch.cat(features_list), torch.cat(labels_list)


def fit_mahalanobis(features, labels, num_classes=10):
    """Fit class-conditional Gaussian for Mahalanobis distance."""
    class_means = []
    class_covs = []
    
    for c in range(num_classes):
        class_features = features[labels == c].numpy()
        class_means.append(class_features.mean(axis=0))
        
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(class_features)
        class_covs.append(cov_estimator.covariance_)
    
    # Tied covariance (average across classes)
    tied_cov = np.mean(class_covs, axis=0)
    precision = np.linalg.pinv(tied_cov)
    
    return np.array(class_means), precision


# OOD Detection Methods

def score_msp(model, loader, device):
    """Maximum Softmax Probability (baseline)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(device))
            probs = F.softmax(logits, dim=1)
            scores.extend((1 - probs.max(dim=1)[0]).cpu().numpy())
    return np.array(scores)


def score_odin(model, loader, device, temperature=1000, epsilon=0.0014):
    """ODIN: temperature scaling + input perturbation."""
    model.eval()
    scores = []
    
    for imgs, _ in loader:
        imgs = imgs.to(device).requires_grad_()
        
        logits = model(imgs)
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)
        
        # Input perturbation
        max_probs, _ = probs.max(dim=1)
        loss = -max_probs.sum()
        loss.backward()
        
        gradient = imgs.grad.sign()
        perturbed = imgs - epsilon * gradient
        
        with torch.no_grad():
            logits_pert = model(perturbed)
            probs_pert = F.softmax(logits_pert / temperature, dim=1)
            scores.extend((1 - probs_pert.max(dim=1)[0]).cpu().numpy())
    
    return np.array(scores)


def score_energy(model, loader, device, temperature=1.0):
    """Energy-based OOD detection."""
    model.eval()
    scores = []
    with torch.no_grad():
        for imgs, _ in loader:
            logits = model(imgs.to(device))
            energy = -temperature * torch.logsumexp(logits / temperature, dim=1)
            scores.extend(energy.cpu().numpy())
    return np.array(scores)


def score_mahalanobis(model, loader, device, class_means, precision):
    """Mahalanobis distance in feature space."""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            _, features = model(imgs.to(device), return_features=True)
            features = features.cpu().numpy()
            
            # Minimum Mahalanobis distance to any class
            for feat in features:
                dists = []
                for mean in class_means:
                    diff = feat - mean
                    dist = np.sqrt(diff @ precision @ diff.T)
                    dists.append(dist)
                scores.append(min(dists))
    
    return np.array(scores)


def score_witness(model, loader, device, num_passes=10):
    """Witness: predictive entropy."""
    model.eval()
    scores = []
    
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            model.train()  # Keep dropout
            
            all_probs = []
            for _ in range(num_passes):
                logits, _ = model(imgs)
                all_probs.append(F.softmax(logits, dim=1))
            
            mean_probs = torch.stack(all_probs).mean(0)
            entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
            scores.extend(entropy.cpu().numpy())
    
    return np.array(scores)


def run_experiment():
    print("="*80)
    print("WITNESS OOD DETECTION: BENCHMARK COMPARISON")
    print("="*80)
    print("\nTask: CIFAR-10 (ID) vs SVHN (OOD)")
    print("Methods: MSP, ODIN, Energy, Mahalanobis, Witness")
    print("Goal: Show witness outperforms established baselines\n")
    
    results_dir = Path("examples/hallucinations/experiment_11/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cache_dir = Path.home() / ".scrapbook" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    print("Loading datasets...")
    cifar_train = datasets.CIFAR10(root=str(cache_dir), train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=str(cache_dir), train=False, download=True, transform=transform)
    svhn_test = datasets.SVHN(root=str(cache_dir), split='test', download=True, transform=transform)
    
    # Subsample for speed
    train_size, test_size = 10000, 2000
    train_idx = np.random.RandomState(42).choice(len(cifar_train), train_size, replace=False)
    id_test_idx = np.random.RandomState(43).choice(len(cifar_test), test_size, replace=False)
    ood_test_idx = np.random.RandomState(44).choice(len(svhn_test), test_size, replace=False)
    
    train_loader = DataLoader(Subset(cifar_train, train_idx), batch_size=128, shuffle=True)
    id_loader = DataLoader(Subset(cifar_test, id_test_idx), batch_size=128, shuffle=False)
    ood_loader = DataLoader(Subset(svhn_test, ood_test_idx), batch_size=128, shuffle=False)
    
    print(f"  Train: {train_size}, Test: {test_size} ID + {test_size} OOD\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Train models
    print("Training WideResNet (for MSP, ODIN, Energy, Mahalanobis)...")
    torch.manual_seed(42)
    model_standard = WideResNet()
    train_model(model_standard, train_loader, device, epochs=20, is_witness=False)
    
    print("\nTraining Witness WideResNet...")
    torch.manual_seed(42)
    model_witness = WitnessWideResNet()
    train_model(model_witness, train_loader, device, epochs=20, is_witness=True)
    
    # Fit Mahalanobis
    print("\nFitting Mahalanobis statistics...")
    features, labels = extract_features(model_standard, train_loader, device)
    class_means, precision = fit_mahalanobis(features, labels)
    
    # Evaluate all methods
    print("\nEvaluating OOD detection methods...")
    results = {}
    
    print("  MSP...")
    id_scores_msp = score_msp(model_standard, id_loader, device)
    ood_scores_msp = score_msp(model_standard, ood_loader, device)
    results['MSP'] = {
        'auroc': roc_auc_score(
            np.concatenate([[0]*len(id_scores_msp), [1]*len(ood_scores_msp)]),
            np.concatenate([id_scores_msp, ood_scores_msp])
        )
    }
    
    print("  ODIN...")
    id_scores_odin = score_odin(model_standard, id_loader, device)
    ood_scores_odin = score_odin(model_standard, ood_loader, device)
    results['ODIN'] = {
        'auroc': roc_auc_score(
            np.concatenate([[0]*len(id_scores_odin), [1]*len(ood_scores_odin)]),
            np.concatenate([id_scores_odin, ood_scores_odin])
        )
    }
    
    print("  Energy...")
    id_scores_energy = score_energy(model_standard, id_loader, device)
    ood_scores_energy = score_energy(model_standard, ood_loader, device)
    results['Energy'] = {
        'auroc': roc_auc_score(
            np.concatenate([[0]*len(id_scores_energy), [1]*len(ood_scores_energy)]),
            np.concatenate([id_scores_energy, ood_scores_energy])
        )
    }
    
    print("  Mahalanobis...")
    id_scores_maha = score_mahalanobis(model_standard, id_loader, device, class_means, precision)
    ood_scores_maha = score_mahalanobis(model_standard, ood_loader, device, class_means, precision)
    results['Mahalanobis'] = {
        'auroc': roc_auc_score(
            np.concatenate([[0]*len(id_scores_maha), [1]*len(ood_scores_maha)]),
            np.concatenate([id_scores_maha, ood_scores_maha])
        )
    }
    
    print("  Witness...")
    id_scores_witness = score_witness(model_witness, id_loader, device)
    ood_scores_witness = score_witness(model_witness, ood_loader, device)
    results['Witness'] = {
        'auroc': roc_auc_score(
            np.concatenate([[0]*len(id_scores_witness), [1]*len(ood_scores_witness)]),
            np.concatenate([id_scores_witness, ood_scores_witness])
        )
    }
    
    # Visualize
    print("\nCreating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    methods = list(results.keys())
    aurocs = [results[m]['auroc'] for m in methods]
    colors = ['red' if m != 'Witness' else 'blue' for m in methods]
    
    bars = axes[0].bar(methods, aurocs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('AUROC', fontsize=12)
    axes[0].set_title('OOD Detection: CIFAR-10 vs SVHN', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=1)
    
    for bar, auroc in zip(bars, aurocs):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{auroc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Rankings
    sorted_results = sorted(results.items(), key=lambda x: x[1]['auroc'], reverse=True)
    axes[1].axis('off')
    axes[1].text(0.5, 0.9, 'Method Rankings', ha='center', fontsize=14, fontweight='bold',
                transform=axes[1].transAxes)
    
    y_pos = 0.75
    for rank, (method, res) in enumerate(sorted_results, 1):
        color = 'blue' if method == 'Witness' else 'black'
        weight = 'bold' if method == 'Witness' else 'normal'
        axes[1].text(0.5, y_pos, f"{rank}. {method}: {res['auroc']:.3f}",
                    ha='center', fontsize=12, color=color, weight=weight,
                    transform=axes[1].transAxes)
        y_pos -= 0.12
    
    plt.tight_layout()
    plt.savefig(results_dir / 'benchmark_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'benchmark_comparison.png'}")
    
    # Save results
    with open(results_dir / 'benchmark_results.json', 'w') as f:
        json.dump({k: {'auroc': float(v['auroc'])} for k, v in results.items()}, f, indent=2)
    print(f"  Saved: {results_dir / 'benchmark_results.json'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for method, res in sorted_results:
        marker = "★" if method == "Witness" else " "
        print(f"{marker} {method:15s}: AUROC = {res['auroc']:.3f}")
    print()


if __name__ == "__main__":
    run_experiment()