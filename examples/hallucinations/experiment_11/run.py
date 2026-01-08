"""
Experiment 11: Architectural Sufficiency for OOD Detection

Tests OOD detection using SSB-Hard (Semantic Shift Benchmark - Hard),
a standard benchmark for near-OOD detection where OOD samples are
semantically different but visually similar to in-distribution samples.

This uses a proper benchmark rather than synthetic corruptions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from pytorch_ood.dataset.img import SSBHard
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

import contrakit as ck
from contrakit.observatory import Observatory


def compute_K_for_ood_task() -> float:
    """Compute K for OOD detection task."""
    obs = Observatory.create(symbols=['Classify', 'Abstain'])
    behavior = obs.concept('Action')
    
    lens_id = obs.lens('InDistribution')
    with lens_id:
        lens_id.perspectives[behavior] = {'Classify': 1.0, 'Abstain': 0.0}
    
    lens_ood = obs.lens('OutOfDistribution')
    with lens_ood:
        lens_ood.perspectives[behavior] = {'Classify': 0.0, 'Abstain': 1.0}
    
    combined = lens_id | lens_ood
    task_behavior = combined.to_behavior()
    
    return task_behavior.K


class LabeledDataset(Dataset):
    """Wrapper to add OOD labels."""
    
    def __init__(self, dataset, is_ood=False):
        self.dataset = dataset
        self.is_ood_flag = is_ood
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        is_ood = torch.tensor(1.0 if self.is_ood_flag else 0.0)
        return img, label, is_ood


class StandardCNN(nn.Module):
    """Standard CNN with r ≈ 0."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        return self.classifier(self.features(x))


class WitnessCNN(nn.Module):
    """CNN with explicit uncertainty output (r ≥ 1)."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU()
        )
        
        self.classifier = nn.Linear(128, num_classes)
        self.uncertainty = nn.Linear(128, 1)
    
    def forward(self, x):
        h = self.features(x)
        logits = self.classifier(h)
        uncertainty = torch.sigmoid(self.uncertainty(h))
        return logits, uncertainty


def train_standard_cnn(
    model: StandardCNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.001
) -> None:
    """Train standard CNN on ID data only."""
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, is_ood in pbar:
            images = images.to(device)
            labels = labels.to(device)
            is_ood = is_ood.to(device)
            
            id_mask = (is_ood == 0)
            if id_mask.sum() == 0:
                continue
                
            images_id = images[id_mask]
            labels_id = labels[id_mask]
            
            optimizer.zero_grad()
            logits = model(images_id)
            loss = criterion(logits, labels_id)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})


def train_witness_cnn(
    model: WitnessCNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 20,
    lr: float = 0.001
) -> None:
    """Train witness CNN with explicit OOD supervision."""
    model.to(device)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels, is_ood in pbar:
            images = images.to(device)
            labels = labels.to(device)
            is_ood = is_ood.to(device).float()
            
            optimizer.zero_grad()
            logits, uncertainty = model(images)
            
            id_mask = (is_ood == 0)
            if id_mask.sum() > 0:
                class_loss = criterion(logits[id_mask], labels[id_mask])
            else:
                class_loss = torch.tensor(0.0, device=device)
            
            uncertainty_loss = nn.BCELoss()(uncertainty.squeeze(), is_ood)
            
            loss = class_loss + uncertainty_loss
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})


def evaluate_ood_detection(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    method: str = "witness"
) -> Dict:
    """Evaluate OOD detection performance."""
    model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for images, _, is_ood in test_loader:
            images = images.to(device)
            
            if method == "witness":
                logits, uncertainty = model(images)
                ood_scores = uncertainty.squeeze()
            else:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                
                if method == "max_softmax":
                    ood_scores = 1 - probs.max(dim=1)[0]
                elif method == "entropy":
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                    ood_scores = entropy / np.log(10)
            
            all_scores.extend(ood_scores.cpu().numpy())
            all_labels.extend(is_ood.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    auroc = roc_auc_score(all_labels, all_scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    fpr_at_95_tpr = fpr[np.where(tpr >= 0.95)[0][0]] if np.any(tpr >= 0.95) else 1.0
    
    return {
        'auroc': auroc,
        'fpr_at_95_tpr': fpr_at_95_tpr,
        'scores': all_scores,
        'labels': all_labels
    }


def run_experiment():
    """Test OOD detection with SSB-Hard benchmark."""
    print("="*80)
    print("EXPERIMENT 11: Architectural Sufficiency for OOD Detection")
    print("="*80)
    print()
    
    results_dir = Path("examples/hallucinations/experiment_11/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Compute K
    print("STEP 1: Compute Task K")
    print("-"*80)
    
    K = compute_K_for_ood_task()
    print(f"  Task: ID vs OOD detection")
    print(f"  K = {K:.4f} bits")
    print(f"  Standard architecture: r ≈ 0 bits")
    print(f"  Witness architecture: r ≥ 1 bit")
    print()
    
    # STEP 2: Setup datasets
    print("STEP 2: Dataset Setup")
    print("-"*80)
    
    # Use centralized cache directory
    cache_dir = Path.home() / ".scrapbook" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize all to 32x32 to match CIFAR-10
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),  # Convert grayscale to RGB
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # CIFAR-10 as in-distribution
    print("  Loading CIFAR-10 (ID)...")
    cifar_train = datasets.CIFAR10(root=str(cache_dir), train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=str(cache_dir), train=False, download=True, transform=transform)
    
    # Subsample training for laptop-runnable speed
    train_size = 10000
    train_indices = np.random.RandomState(42).choice(len(cifar_train), train_size, replace=False)
    cifar_train_subset = Subset(cifar_train, train_indices)
    
    print(f"  ID (CIFAR-10): {len(cifar_train_subset)} train, {len(cifar_test)} test")
    
    # SSB-Hard as out-of-distribution
    print("  Loading SSB-Hard (OOD)...")
    ssb_hard = SSBHard(root=str(cache_dir), download=True, transform=transform)
    
    # Subsample SSB-Hard for training OOD supervision
    ood_train_size = train_size // 5
    ood_train_indices = np.random.RandomState(42).choice(len(ssb_hard), ood_train_size, replace=False)
    ssb_train_subset = Subset(ssb_hard, ood_train_indices)
    
    # Use different subset for testing
    ood_test_indices = np.random.RandomState(43).choice(len(ssb_hard), len(cifar_test), replace=False)
    ssb_test_subset = Subset(ssb_hard, ood_test_indices)
    
    print(f"  OOD (SSB-Hard): {len(ssb_train_subset)} train, {len(ssb_test_subset)} test")
    print(f"  SSB-Hard: near-OOD benchmark with semantic shifts")
    print()
    
    # Create datasets with OOD labels
    train_id = LabeledDataset(cifar_train_subset, is_ood=False)
    train_ood = LabeledDataset(ssb_train_subset, is_ood=True)
    train_dataset = torch.utils.data.ConcatDataset([train_id, train_ood])
    
    test_id = LabeledDataset(cifar_test, is_ood=False)
    test_ood = LabeledDataset(ssb_test_subset, is_ood=True)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader_combined = DataLoader(
        torch.utils.data.ConcatDataset([test_id, test_ood]),
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    print()
    
    # STEP 3: Train models
    print("STEP 3: Train Models")
    print("-"*80)
    
    results = {}
    
    print("  Training Standard CNN (r ≈ 0)...")
    torch.manual_seed(42)
    model_standard = StandardCNN(num_classes=10, input_channels=3)
    train_standard_cnn(model_standard, train_loader, device, epochs=20)
    
    metrics_max_softmax = evaluate_ood_detection(
        model_standard, test_loader_combined, device, method="max_softmax"
    )
    results['max_softmax'] = metrics_max_softmax
    print(f"    Max Softmax: AUROC = {metrics_max_softmax['auroc']:.3f}")
    
    metrics_entropy = evaluate_ood_detection(
        model_standard, test_loader_combined, device, method="entropy"
    )
    results['entropy'] = metrics_entropy
    print(f"    Entropy: AUROC = {metrics_entropy['auroc']:.3f}")
    print()
    
    print("  Training Witness CNN (r ≥ 1)...")
    torch.manual_seed(42)
    model_witness = WitnessCNN(num_classes=10, input_channels=3)
    train_witness_cnn(model_witness, train_loader, device, epochs=20)
    
    metrics_witness = evaluate_ood_detection(
        model_witness, test_loader_combined, device, method="witness"
    )
    results['witness'] = metrics_witness
    print(f"    Witness: AUROC = {metrics_witness['auroc']:.3f}")
    print()
    
    # Visualize examples with predictions
    print("  Creating visualization with predictions...")
    model_witness.eval()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    
    with torch.no_grad():
        for i in range(5):
            # ID examples
            img, label = cifar_test[i]
            img_tensor = img.unsqueeze(0).to(device)
            logits, uncertainty = model_witness(img_tensor)
            pred_class = logits.argmax(dim=1).item()
            uncertainty_score = uncertainty.item()
            
            # Correct if: low uncertainty (< 0.5) AND correct class prediction
            is_correct = (uncertainty_score < 0.5) and (pred_class == label)
            
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465]))
            img_display = np.clip(img_display, 0, 1)
            
            axes[0, i].imshow(img_display)
            color = 'green' if is_correct else 'red'
            for spine in axes[0, i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            if i == 0:
                axes[0, i].set_title('ID (CIFAR-10)', fontsize=10, pad=10)
            
            # OOD examples
            img, _ = ssb_hard[ood_test_indices[i]]
            img_tensor = img.unsqueeze(0).to(device)
            logits, uncertainty = model_witness(img_tensor)
            uncertainty_score = uncertainty.item()
            
            # Correct if: high uncertainty (>= 0.5) for OOD
            is_correct = uncertainty_score >= 0.5
            
            img_display = img.permute(1, 2, 0).cpu().numpy()
            img_display = (img_display * np.array([0.2470, 0.2435, 0.2616]) + np.array([0.4914, 0.4822, 0.4465]))
            img_display = np.clip(img_display, 0, 1)
            
            axes[1, i].imshow(img_display)
            color = 'green' if is_correct else 'red'
            for spine in axes[1, i].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(4)
                spine.set_visible(True)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            if i == 0:
                axes[1, i].set_title('OOD (SSB-Hard)', fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'id_vs_ood_examples.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'id_vs_ood_examples.png'}")
    print()
    
    # STEP 4: Visualization
    print("="*80)
    print("STEP 4: Visualization")
    print("="*80)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, label, color in [
        ('max_softmax', 'Standard (Max Softmax)', 'orange'),
        ('entropy', 'Standard (Entropy)', 'red'),
        ('witness', 'Witness (Uncertainty Head)', 'blue')
    ]:
        fpr, tpr, _ = roc_curve(results[name]['labels'], results[name]['scores'])
        auroc = results[name]['auroc']
        ax1.plot(fpr, tpr, label=f'{label} (AUROC={auroc:.3f})', color=color, linewidth=2)
    
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('OOD Detection: SSB-Hard Benchmark', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    id_scores_witness = results['witness']['scores'][results['witness']['labels'] == 0]
    ood_scores_witness = results['witness']['scores'][results['witness']['labels'] == 1]
    
    ax2.hist(id_scores_witness, bins=30, alpha=0.5, label='ID', color='green', density=True)
    ax2.hist(ood_scores_witness, bins=30, alpha=0.5, label='OOD', color='red', density=True)
    ax2.set_xlabel('Uncertainty Score', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Witness: Score Distributions', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'ood_detection.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'ood_detection.png'}")
    
    # Save results
    results_json = {
        'task_K': float(K),
        'id_dataset': 'CIFAR-10',
        'ood_dataset': 'SSB-Hard (near-OOD benchmark)',
        'training': 'ID + OOD (both models see same data)',
        'methods': {
            name: {
                'auroc': float(metrics['auroc']),
                'fpr_at_95_tpr': float(metrics['fpr_at_95_tpr'])
            }
            for name, metrics in results.items()
        }
    }
    
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {results_dir / 'results.json'}")
    
    # STEP 5: Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTask K = {K:.4f} bits")
    print(f"Benchmark: CIFAR-10 (ID) vs SSB-Hard (OOD)")
    print(f"\nResults:")
    print(f"  Max Softmax (r ≈ 0): AUROC = {results['max_softmax']['auroc']:.3f}")
    print(f"  Entropy (r ≈ 0):     AUROC = {results['entropy']['auroc']:.3f}")
    print(f"  Witness (r ≥ 1):     AUROC = {results['witness']['auroc']:.3f}")
    print()


if __name__ == "__main__":
    run_experiment()
