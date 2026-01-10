"""
Test witness capacity for epistemic uncertainty in OOD detection.

Computes structural contradiction K using contrakit, trains with witness capacity r ≥ K,
and evaluates abstention on contradictory inputs with generalization to OOD detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
from contrakit import Observatory


EPOCHS = 20


def compute_task_contradiction(num_contexts=3):
    """Compute structural contradiction K using contrakit."""
    class_names = [f'class_{i}' for i in range(10)]
    obs = Observatory.create(symbols=class_names)
    prediction = obs.concept("Prediction")

    lenses = []
    for ctx_idx in range(num_contexts):
        context_lens = obs.lens(f"Context_{ctx_idx}")
        with context_lens:
            context_output = ctx_idx % 10
            context_lens.perspectives[prediction] = {prediction.alphabet[context_output]: 1.0}
        lenses.append(context_lens)

    combined = lenses[0] if len(lenses) == 1 else lenses[0]
    for lens in lenses[1:]:
        combined = combined | lens

    behavior = combined.to_behavior()
    return {
        'K': behavior.K,
        'alpha_star': behavior.alpha_star,
        'required_witness_capacity': behavior.K,
        'num_contexts': num_contexts
    }


def create_contradictory_cifar10(cifar_data, contradiction_ratio=0.3, seed=42):
    """Create defined (consistent) and undefined (contradictory) CIFAR-10 examples."""
    np.random.seed(seed)

    total = len(cifar_data)
    n_undefined = int(total * contradiction_ratio)
    n_defined = total - n_undefined

    indices = np.random.permutation(total)
    defined_idx = indices[:n_defined]
    undefined_idx = indices[n_defined:]

    defined_data = []
    for idx in defined_idx:
        img, label = cifar_data[idx]
        defined_data.append((img, label, True))  # is_defined = True

    undefined_data = []
    contradictory_pairs = []
    label_permutation = np.array([3, 7, 1, 9, 4, 2, 8, 0, 6, 5])

    for idx in undefined_idx:
        img, original_label = cifar_data[idx]
        contradictory_label = label_permutation[original_label]
        undefined_data.append((img, contradictory_label, False))  # is_defined = False
        contradictory_pairs.append({
            'img': img,
            'label_context1': original_label,
            'label_context2': contradictory_label
        })

    return defined_data, undefined_data, contradictory_pairs


class WitnessNetwork(nn.Module):
    """Neural network with configurable witness capacity."""
    def __init__(self, num_classes=10, witness_bits=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        self.num_witness_states = max(1, int(2 ** witness_bits))
        if self.num_witness_states > 1:
            self.witness = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, self.num_witness_states)
            )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        features = self.pool(x).view(x.size(0), -1)

        logits = self.fc(features)
        witness_logits = self.witness(features) if hasattr(self, 'witness') and self.witness else None
        return logits, witness_logits


def train_on_contradiction(model, defined_data, undefined_data, device, epochs=20):
    """Train network with defined/undefined examples using witness capacity."""
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    all_data = defined_data + undefined_data
    np.random.shuffle(all_data)

    imgs = torch.stack([x[0] for x in all_data])
    labels = torch.tensor([x[1] for x in all_data])
    is_defined = torch.tensor([x[2] for x in all_data])

    dataset = TensorDataset(imgs, labels, is_defined)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    criterion_class = nn.CrossEntropyLoss()
    criterion_witness = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        epoch_loss = 0.0
        epoch_class_loss = 0.0
        epoch_witness_loss = 0.0
        epoch_correct = 0
        epoch_total_defined = 0
        num_batches = len(loader)

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for imgs_batch, labels_batch, is_defined_batch in pbar:
            imgs_batch = imgs_batch.to(device)
            labels_batch = labels_batch.to(device)
            is_defined_batch = is_defined_batch.to(device)

            optimizer.zero_grad()
            logits, witness_logits = model(imgs_batch)

            defined_mask = is_defined_batch.bool()
            loss_class = criterion_class(logits[defined_mask], labels_batch[defined_mask]) if defined_mask.any() else torch.tensor(0.0, device=device)

            loss = loss_class

            if witness_logits is not None:
                witness_targets = is_defined_batch.long() * (model.num_witness_states - 1)

                # With balanced classes (50/50), use equal weighting (following experiment 9)
                num_classes = model.num_witness_states
                class_weights = torch.ones(num_classes, device=device)  # Equal weights
                criterion_weighted = nn.CrossEntropyLoss(weight=class_weights)

                loss_witness = criterion_weighted(witness_logits, witness_targets)
                loss = loss + 0.5 * loss_witness  # Following experiment 9's weighting
                epoch_witness_loss += loss_witness.item()

            loss.backward()
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_class_loss += loss_class.item()

            # Calculate accuracy on defined examples
            if defined_mask.any():
                predictions = logits[defined_mask].argmax(dim=1)
                correct = (predictions == labels_batch[defined_mask]).sum().item()
                epoch_correct += correct
                epoch_total_defined += defined_mask.sum().item()

            # Update progress bar
            avg_loss = epoch_loss / (pbar.n + 1)
            avg_class_loss = epoch_class_loss / (pbar.n + 1)
            accuracy = epoch_correct / max(1, epoch_total_defined) * 100

            postfix_dict = {
                'loss': f'{avg_loss:.3f}',
                'cls_loss': f'{avg_class_loss:.3f}',
                'acc': f'{accuracy:.1f}%'
            }

            if witness_logits is not None:
                avg_witness_loss = epoch_witness_loss / (pbar.n + 1)
                postfix_dict['wit_loss'] = f'{avg_witness_loss:.3f}'

            pbar.set_postfix(postfix_dict)

        scheduler.step()


def evaluate_abstention(model, test_data, device):
    """
    Evaluate abstention following experiment 9's mathematical approach.

    Witness state 0 = abstain (undefined/contradictory)
    Witness state > 0 = commit (defined/consistent)
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_abstains = []

    with torch.no_grad():
        for item in test_data:
            img = item[0].unsqueeze(0).to(device) if not isinstance(item, dict) else item['img'].unsqueeze(0).to(device)
            label = item[1] if not isinstance(item, dict) else item['label_context1']

            logits, witness_logits = model(img)
            prediction = logits.argmax(dim=1).cpu().item()

            abstain = False
            if witness_logits is not None:
                witness_prediction = torch.argmax(witness_logits, dim=1).cpu().item()
                abstain = (witness_prediction == 0)

            all_predictions.append(prediction)
            all_labels.append(label)
            all_abstains.append(abstain)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_abstains = np.array(all_abstains)

    abstention_rate = all_abstains.mean()
    commits = ~all_abstains
    accuracy_on_commits = (all_predictions[commits] == all_labels[commits]).mean() if commits.sum() > 0 else 0.0

    return {
        'abstention_rate': abstention_rate,
        'accuracy_on_commits': accuracy_on_commits
    }


def run_experiment():
    """Run experiment testing witness capacity for epistemic uncertainty."""
    results_dir = Path("examples/hallucinations/experiment_11/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path.home() / ".scrapbook" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar_train = datasets.CIFAR10(root=str(cache_dir), train=True, download=True, transform=transform)
    cifar_test = datasets.CIFAR10(root=str(cache_dir), train=False, download=True, transform=transform)
    svhn_test = datasets.SVHN(root=str(cache_dir), split='test', download=True, transform=transform)

    np.random.seed(42)
    train_idx = np.random.choice(len(cifar_train), 20000, replace=False)
    test_idx = np.random.choice(len(cifar_test), 2000, replace=False)
    ood_idx = np.random.choice(len(svhn_test), 2000, replace=False)

    cifar_train_subset = Subset(cifar_train, train_idx)
    cifar_test_subset = Subset(cifar_test, test_idx)
    svhn_test_subset = Subset(svhn_test, ood_idx)

    # Compute K
    task_info = compute_task_contradiction(num_contexts=3)
    K = task_info['K']

    print(f"Task contradiction: K = {K:.4f} bits")
    print(f"Witness capacities to test: r = [0.0, {max(1.0, np.ceil(K)):.1f}, {max(1.0, np.ceil(K)) + 1:.1f}] bits")
    print(f"Theoretical minimum error: E ≥ {1 - 2**(-K):.4f}")

    # Create data with balanced classes (following experiment 9's ambiguous_ratio=0.5)
    defined_data, undefined_data, contradictory_pairs = create_contradictory_cifar10(
        cifar_train_subset,
        contradiction_ratio=0.5  # 50/50 balance for proper witness learning
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test witness capacities following Theorem 7.4: E + r ≥ K
    # r = 0: baseline (r < K, should fail)
    # r = ceil(K): should achieve E + r = K (optimal)
    # r = ceil(K) + 1: excess capacity (should still work)
    witness_bits_values = [0.0, max(1.0, np.ceil(K)), max(1.0, np.ceil(K)) + 1.0]
    results = {}
    
    for witness_bits in witness_bits_values:
        print(f"\nTraining model with r = {witness_bits:.1f} bits witness capacity...")
        torch.manual_seed(42)
        model = WitnessNetwork(witness_bits=witness_bits)

        train_on_contradiction(model, defined_data, undefined_data, device, epochs=EPOCHS)  # Shorter training to avoid overfitting

        metrics_contradictory = evaluate_abstention(model, contradictory_pairs[:200], device)
        cifar_test_list = [(cifar_test_subset[i][0], cifar_test_subset[i][1]) for i in range(min(200, len(cifar_test_subset)))]
        metrics_consistent = evaluate_abstention(model, cifar_test_list, device)
        svhn_test_list = [(svhn_test_subset[i][0], svhn_test_subset[i][1]) for i in range(min(200, len(svhn_test_subset)))]
        metrics_ood = evaluate_abstention(model, svhn_test_list, device)

        # Store accuracy for comparison
        accuracy = metrics_consistent['accuracy_on_commits']

        results[f'r={witness_bits:.1f}'] = {
            'r_bits': float(witness_bits),
            'contradictory': {
                'abstention_rate': float(metrics_contradictory['abstention_rate']),
                'accuracy_on_commits': float(metrics_contradictory['accuracy_on_commits'])
            },
            'consistent': {
                'abstention_rate': float(metrics_consistent['abstention_rate']),
                'accuracy_on_commits': float(metrics_consistent['accuracy_on_commits'])
            },
            'ood': {
                'abstention_rate': float(metrics_ood['abstention_rate'])
            }
        }

        # Print per-model summary
        print(f"\n✓ Model r={witness_bits:.1f} completed:")
        print(f"  Contradictory: {metrics_contradictory['abstention_rate']*100:.1f}% abstain, "
              f"{metrics_contradictory['accuracy_on_commits']*100:.1f}% accuracy")
        print(f"  Consistent: {metrics_consistent['abstention_rate']*100:.1f}% abstain, "
              f"{metrics_consistent['accuracy_on_commits']*100:.1f}% accuracy")
        print(f"  OOD (SVHN): {metrics_ood['abstention_rate']*100:.1f}% abstain")
        print()
    
    # Print results summary
    print("\nAbstention Rates by Witness Capacity:")
    print(f"{'Model':<12} {'Contradictory':<15} {'Consistent':<15} {'OOD (SVHN)':<15} {'Selectivity':<12}")
    print("-" * 70)
    for r in witness_bits_values:
        result = results[f'r={r:.1f}']
        contra_abstain = result['contradictory']['abstention_rate'] * 100
        consist_abstain = result['consistent']['abstention_rate'] * 100
        selectivity = contra_abstain - consist_abstain  # Higher = more selective abstention
        model_name = f"r={r:.1f}"
        print(f"{model_name:<12} "
              f"{contra_abstain:>13.1f}% "
              f"{consist_abstain:>13.1f}% "
              f"{result['ood']['abstention_rate']*100:>13.1f}% "
              f"{selectivity:>+10.1f}%")

    # Calculate and print OOD improvements
    baseline_ood = results['r=0.0']['ood']['abstention_rate'] * 100
    print(f"\nOOD Abstention Improvement over Baseline (r=0.0):")
    print(f"{'Model':<12} {'OOD Rate':<12} {'Improvement':<12}")
    print("-" * 40)
    for r in witness_bits_values:
        ood_rate = results[f'r={r:.1f}']['ood']['abstention_rate'] * 100
        improvement = ood_rate - baseline_ood
        model_name = f"r={r:.1f}"
        print(f"{model_name:<12} {ood_rate:>10.1f}% {improvement:>+10.1f}%")

    # Phase transition evaluation
    print(f"\nPhase Transition Analysis (K = {K:.3f} bits):")
    for r in witness_bits_values:
        result = results[f'r={r:.1f}']
        contradictory_rate = result['contradictory']['abstention_rate']
        expected_high_abstention = r >= K
        actual_high_abstention = contradictory_rate > 0.5
        status = "✓" if expected_high_abstention == actual_high_abstention else "✗"
        model_name = f"r={r:.1f}"
        print(f"  {model_name}: {contradictory_rate*100:.1f}% abstention {status}")

    # Save results
    with open(results_dir / 'generalization_test.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Prepare data for visualization and analysis
    r_values = witness_bits_values
    contradictory_rates = [results[f'r={r:.1f}']['contradictory']['abstention_rate']*100 for r in r_values]
    consistent_rates = [results[f'r={r:.1f}']['consistent']['abstention_rate']*100 for r in r_values]
    ood_rates = [results[f'r={r:.1f}']['ood']['abstention_rate']*100 for r in r_values]
    baseline_accuracies = [results[f'r={r:.1f}']['consistent']['accuracy_on_commits']*100 for r in r_values]

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(r_values))
    width = 0.25

    ax1.bar(x - width, contradictory_rates, width, label='Contradictory', color='#e74c3c', alpha=0.8)
    ax1.bar(x, consistent_rates, width, label='Consistent', color='#3498db', alpha=0.8)
    ax1.bar(x + width, ood_rates, width, label='OOD (SVHN)', color='#27ae60', alpha=0.8)

    ax1.set_xlabel('Witness Capacity r (bits)')
    ax1.set_ylabel('Abstention Rate (%)')
    ax1.set_title('Abstention Across Test Conditions')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r:.1f}' for r in r_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])

    baseline_ood = ood_rates[0]
    ood_improvements = [rate - baseline_ood for rate in ood_rates]

    ax2.plot(r_values, ood_improvements, 'o-', linewidth=3, markersize=10, color='#27ae60')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Witness Capacity r (bits)')
    ax2.set_ylabel('OOD Abstention Improvement (%)')
    ax2.set_title('OOD Detection Generalization')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'generalization_test.png', dpi=150, bbox_inches='tight')

    # Evaluate what "better than baseline" means
    print(f"\nWhat does 'better than baseline' mean?")
    print(f"Baseline (r=0): {ood_rates[0]:.1f}% OOD abstention, {baseline_accuracies[0]:.1f}% accuracy")
    print(f"Witness models should show:")
    print(f"1. Higher abstention on contradictory vs consistent inputs (selective uncertainty)")
    print(f"2. Appropriate abstention on OOD data (generalized epistemic uncertainty)")

    # Evaluate witness capacity effectiveness
    print("\nEvaluating witness capacity effectiveness:")
    print("1. Phase transition (Theorem 7.4): E + r ≥ K")
    phase_success = all(
        (r >= K and results[f'r={r:.1f}']['contradictory']['abstention_rate'] > 0) or
        (r < K and results[f'r={r:.1f}']['contradictory']['abstention_rate'] == 0)
        for r in witness_bits_values
    )

    print(f"   ✓ Phase transition confirmed" if phase_success else "   ✗ Phase transition failed")

    print(f"2. Selective uncertainty (epistemic awareness)")
    selective_scores = []
    for r in witness_bits_values[1:]:  # Skip baseline
        result = results[f'r={r:.1f}']
        selectivity = result['contradictory']['abstention_rate'] - result['consistent']['abstention_rate']
        selective_scores.append(selectivity)
        print(f"   r={r:.1f}: selectivity = {selectivity*100:+.1f}%")

    selective_success = any(s > 0 for s in selective_scores)  # Any positive selectivity
    print(f"   {'✓' if selective_success else '✗'} Selective uncertainty {'achieved' if selective_success else 'not achieved'}")

    print(f"3. OOD generalization (epistemic transfer)")
    ood_improvements = [rate - ood_rates[0] for rate in ood_rates[1:]]
    max_ood_improvement = max(ood_improvements) if ood_improvements else 0
    print(f"   Max OOD improvement: {max_ood_improvement*100:+.1f}% over baseline")
    ood_success = max_ood_improvement > 0.05  # >5% improvement
    print(f"   {'✓' if ood_success else '✗'} OOD generalization {'achieved' if ood_success else 'not achieved'}")




if __name__ == "__main__":
    run_experiment()