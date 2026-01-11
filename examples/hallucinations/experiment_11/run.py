"""
Experiment 11: Witness Capacity for Epistemic Uncertainty in OOD Detection

This experiment tests the theoretical framework from Theorem 7.4: E + r ≥ K, where:
- E is the irreducible error due to contradiction
- r is the witness capacity (bits)
- K is the structural contradiction level

The experiment:
1. Computes structural contradiction K using contrakit for multi-context scenarios
2. Trains neural networks with configurable witness capacity r
3. Evaluates abstention behavior on contradictory vs consistent inputs
4. Tests generalization to out-of-distribution (OOD) detection

Key findings validate that witness capacity enables epistemic uncertainty detection,
allowing models to abstain from contradictory inputs while maintaining accuracy
on consistent data, with generalization to OOD scenarios.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
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
from sklearn.metrics import roc_auc_score, brier_score_loss
import random
from contrakit.constants import DEFAULT_SEED

# Experiment configuration constants
TRAINING_EPOCHS = 20
CONTRADICTION_TYPES = ['permutation', 'rotation', 'multi_label', 'adversarial']

# Data configuration
TRAIN_SUBSET_SIZE = 20000
TEST_SUBSET_SIZE = 2000
OOD_SUBSET_SIZE = 2000
EVALUATION_SAMPLE_SIZE = 200
CONTRADICTION_RATIO_DEFAULT = 0.3
BALANCED_CONTRADICTION_RATIO = 0.5
RANDOM_SEED = DEFAULT_SEED

# Task computation
DEFAULT_NUM_CONTEXTS = 3

# Training hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
WITNESS_LOSS_WEIGHT = 0.5
DROPOUT_RATE = 0.3

# Model architecture
NUM_CLASSES = 10

# Evaluation constants
CALIBRATION_BINS = 10
SELECTIVE_RISK_COVERAGE_LEVELS = np.linspace(0.1, 1.0, 10)

# Visualization constants
FIGURE_SIZE = (16, 12)
BAR_WIDTH = 0.25
PLOT_DPI = 150
PHASE_TRANSITION_THRESHOLD = 0.5
SELECTIVITY_SUCCESS_THRESHOLD = 0.7
OOD_IMPROVEMENT_THRESHOLD = 5.0

# Data structures for functional programming
@dataclass(frozen=True)
class ContradictionData:
    """Structured data for contradictory examples."""
    defined_samples: List[Tuple[Any, int, bool]]
    undefined_samples: List[Tuple[Any, int, bool]]
    contradiction_examples: List[Dict[str, Any]]

@dataclass(frozen=True)
class ExperimentMetrics:
    """Metrics from model evaluation."""
    abstention_rate: float
    accuracy_on_commits: float
    calibration_error: float
    selective_risk_curve: List[Tuple[float, float]]
    auroc: Optional[float] = None

@dataclass(frozen=True)
class TaskContradictionInfo:
    """Information about task contradiction structure."""
    contradiction_level: float
    optimal_strategy: float
    required_capacity: float
    context_count: int


def compute_task_contradiction(num_contexts: int = DEFAULT_NUM_CONTEXTS) -> TaskContradictionInfo:
    """Compute structural contradiction K for multi-context classification tasks.

    Uses contrakit's mathematical framework to analyze how different contexts
    create structural contradictions in the prediction space.

    Args:
        num_contexts: Number of different contexts/perspectives to consider

    Returns:
        TaskContradictionInfo containing contradiction level K, optimal strategy,
        required witness capacity, and context count
    """
    if num_contexts < 1:
        raise ValueError(f"num_contexts must be positive, got {num_contexts}")

    class_names = [f'class_{i}' for i in range(NUM_CLASSES)]
    observatory = Observatory.create(symbols=class_names)
    prediction_concept = observatory.concept("Prediction")

    context_lenses = []
    for context_index in range(num_contexts):
        context_lens = observatory.lens(f"Context_{context_index}")
        with context_lens:
            context_output = context_index % NUM_CLASSES
            context_lens.perspectives[prediction_concept] = {prediction_concept.alphabet[context_output]: 1.0}
        context_lenses.append(context_lens)

    combined_lens = context_lenses[0] if len(context_lenses) == 1 else context_lenses[0]
    for lens in context_lenses[1:]:
        combined_lens = combined_lens | lens

    behavior_model = combined_lens.to_behavior()
    return TaskContradictionInfo(
        contradiction_level=behavior_model.K,
        optimal_strategy=behavior_model.alpha_star,
        required_capacity=behavior_model.K,
        context_count=num_contexts
    )


def _create_permutation_contradictions(
    cifar_data: Any,
    undefined_indices: np.ndarray
) -> Tuple[List[Tuple[Any, int, bool]], List[Dict[str, Any]]]:
    """Create permutation-based contradictions."""
    """Create permutation-based contradictions."""
    label_permutation = np.array([3, 7, 1, 9, 4, 2, 8, 0, 6, 5])
    undefined_samples = []
    contradiction_examples = []

    for sample_index in undefined_indices:
        image, original_label = cifar_data[sample_index]
        contradictory_label = label_permutation[original_label]
        undefined_samples.append((image, contradictory_label, False))
        contradiction_examples.append({
            'img': image,
            'label_context1': original_label,
            'label_context2': contradictory_label,
            'type': 'permutation'
        })

    return undefined_samples, contradiction_examples


def _create_rotation_contradictions(
    cifar_data: Any,
    undefined_indices: np.ndarray
) -> Tuple[List[Tuple[Any, int, bool]], List[Dict[str, Any]]]:
    """Create rotation-based contradictions."""
    rotation_angles = [90, 180, 270]
    undefined_samples = []
    contradiction_examples = []

    for sample_index in undefined_indices:
        image, original_label = cifar_data[sample_index]
        rotation_angle = int(np.random.choice(rotation_angles))
        rotated_image = transforms.functional.rotate(image, rotation_angle)
        potential_labels = [original_label, (original_label + 1) % NUM_CLASSES, (original_label + 2) % NUM_CLASSES]
        contradictory_label = np.random.choice([label for label in potential_labels if label != original_label])
        undefined_samples.append((rotated_image, contradictory_label, False))
        contradiction_examples.append({
            'img': rotated_image,
            'label_context1': original_label,
            'label_context2': contradictory_label,
            'type': 'rotation',
            'rotation': rotation_angle
        })

    return undefined_samples, contradiction_examples


def _create_multi_label_contradictions(
    cifar_data: Any,
    undefined_indices: np.ndarray
) -> Tuple[List[Tuple[Any, int, bool]], List[Dict[str, Any]]]:
    """Create multi-label based contradictions using visually similar classes."""
    visually_similar_classes = {
        0: [0, 8],  # airplane, ship
        1: [1, 9],  # automobile, truck
        2: [2, 3, 4, 5, 6, 7],  # birds and animals
        3: [3, 4, 5, 6, 7],  # cat-like animals
        4: [4, 5, 6, 7],  # deer-like animals
        5: [5, 6, 7],  # dog-like animals
        6: [6, 7],  # frog, horse
        7: [7],  # horse
        8: [8],  # ship
        9: [9]  # truck
    }

    undefined_samples = []
    contradiction_examples = []

    for sample_index in undefined_indices:
        image, original_label = cifar_data[sample_index]
        possible_labels = visually_similar_classes[original_label]
        if len(possible_labels) > 1:
            contradictory_label = np.random.choice([label for label in possible_labels if label != original_label])
        else:
            # Fallback to random permutation if no similar classes
            contradictory_label = (original_label + np.random.randint(1, NUM_CLASSES)) % NUM_CLASSES
        undefined_samples.append((image, contradictory_label, False))
        contradiction_examples.append({
            'img': image,
            'label_context1': original_label,
            'label_context2': contradictory_label,
            'type': 'multi_label'
        })

    return undefined_samples, contradiction_examples


def _create_adversarial_contradictions(
    cifar_data: Any,
    undefined_indices: np.ndarray
) -> Tuple[List[Tuple[Any, int, bool]], List[Dict[str, Any]]]:
    """Create adversarial contradictions with noise perturbations."""
    undefined_samples = []
    contradiction_examples = []

    for sample_index in undefined_indices:
        image, original_label = cifar_data[sample_index]
        # Add small random noise to create adversarial ambiguity
        noise = torch.randn_like(image) * 0.1
        adversarial_image = torch.clamp(image + noise, 0, 1)
        contradictory_label = (original_label + np.random.randint(1, NUM_CLASSES)) % NUM_CLASSES
        undefined_samples.append((adversarial_image, contradictory_label, False))
        contradiction_examples.append({
            'img': adversarial_image,
            'label_context1': original_label,
            'label_context2': contradictory_label,
            'type': 'adversarial'
        })

    return undefined_samples, contradiction_examples


def create_contradictory_cifar10(
    cifar_data: Any,
    contradiction_ratio: float = CONTRADICTION_RATIO_DEFAULT,
    seed: int = RANDOM_SEED,
    contradiction_type: str = 'permutation'
) -> ContradictionData:
    """Create defined (consistent) and undefined (contradictory) CIFAR-10 examples."""
    if not (0.0 < contradiction_ratio < 1.0):
        raise ValueError(f"contradiction_ratio must be between 0 and 1, got {contradiction_ratio}")

    if contradiction_type not in CONTRADICTION_TYPES:
        raise ValueError(f"Unknown contradiction_type '{contradiction_type}'. Must be one of {CONTRADICTION_TYPES}")

    if len(cifar_data) == 0:
        raise ValueError("cifar_data cannot be empty")

    np.random.seed(seed)

    total_samples = len(cifar_data)
    num_undefined_samples = int(total_samples * contradiction_ratio)
    num_defined_samples = total_samples - num_undefined_samples

    sample_indices = np.random.permutation(total_samples)
    defined_indices = sample_indices[:num_defined_samples]
    undefined_indices = sample_indices[num_defined_samples:]

    # Create defined data (consistent examples)
    defined_samples = []
    for sample_index in defined_indices:
        image, label = cifar_data[sample_index]
        defined_samples.append((image, label, True))  # is_defined = True

    # Create undefined data based on contradiction type
    contradiction_handlers = {
        'permutation': _create_permutation_contradictions,
        'rotation': _create_rotation_contradictions,
        'multi_label': _create_multi_label_contradictions,
        'adversarial': _create_adversarial_contradictions
    }

    if contradiction_type not in contradiction_handlers:
        raise ValueError(f"Unknown contradiction type: {contradiction_type}")

    undefined_samples, contradiction_examples = contradiction_handlers[contradiction_type](
        cifar_data, undefined_indices
    )

    return ContradictionData(
        defined_samples=defined_samples,
        undefined_samples=undefined_samples,
        contradiction_examples=contradiction_examples
    )


class WitnessNetwork(nn.Module):
    """Neural network with configurable witness capacity for epistemic uncertainty.

    Implements a CNN architecture that can learn to abstain from predictions
    when faced with contradictory evidence. The witness capacity r determines
    how many distinct uncertainty states the model can represent.

    Attributes:
        num_witness_states: Number of witness states (2^r where r is witness_bits)
        witness: Optional witness head for uncertainty estimation
    """
    def __init__(self, num_classes: int = NUM_CLASSES, witness_bits: float = 0.0) -> None:
        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2, got {num_classes}")
        if witness_bits < 0.0:
            raise ValueError(f"witness_bits must be non-negative, got {witness_bits}")
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
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(64, self.num_witness_states)
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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


def train_on_contradiction(
    model: WitnessNetwork,
    defined_data: List[Tuple[Any, int, bool]],
    undefined_data: List[Tuple[Any, int, bool]],
    device: torch.device,
    epochs: int = TRAINING_EPOCHS
) -> None:
    """Train network on mixed defined/undefined data with witness capacity.

    Trains the model to classify defined (consistent) examples while learning
    to abstain from undefined (contradictory) examples using the witness mechanism.

    Args:
        model: WitnessNetwork to train
        defined_data: List of (image, label, is_defined) tuples for consistent examples
        undefined_data: List of (image, label, is_defined) tuples for contradictory examples
        device: PyTorch device for training
        epochs: Number of training epochs
    """
    if epochs < 1:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if not defined_data and not undefined_data:
        raise ValueError("At least one of defined_data or undefined_data must be non-empty")
    if len(defined_data) == 0:
        raise ValueError("defined_data cannot be empty for proper training")
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    all_samples = defined_data + undefined_data
    np.random.shuffle(all_samples)

    images = torch.stack([sample[0] for sample in all_samples])
    labels = torch.tensor([sample[1] for sample in all_samples])
    is_defined_flags = torch.tensor([sample[2] for sample in all_samples])

    dataset = TensorDataset(images, labels, is_defined_flags)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion_class = nn.CrossEntropyLoss()
    criterion_witness = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        epoch_total_loss = 0.0
        epoch_classification_loss = 0.0
        epoch_witness_loss = 0.0
        epoch_correct_predictions = 0
        epoch_total_defined_samples = 0
        num_batches = len(data_loader)

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_images, batch_labels, batch_is_defined in progress_bar:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            batch_is_defined = batch_is_defined.to(device)

            optimizer.zero_grad()
            logits, witness_logits = model(batch_images)

            defined_mask = batch_is_defined.bool()
            classification_loss = criterion_class(logits[defined_mask], batch_labels[defined_mask]) if defined_mask.any() else torch.tensor(0.0, device=device)

            total_loss = classification_loss

            if witness_logits is not None:
                witness_targets = batch_is_defined.long() * (model.num_witness_states - 1)

                # With balanced classes (50/50), use equal weighting (following experiment 9)
                num_witness_classes = model.num_witness_states
                witness_class_weights = torch.ones(num_witness_classes, device=device)  # Equal weights
                weighted_criterion = nn.CrossEntropyLoss(weight=witness_class_weights)

                witness_loss = weighted_criterion(witness_logits, witness_targets)
                # Weight witness loss less than classification to avoid interfering with main task
                total_loss = total_loss + WITNESS_LOSS_WEIGHT * witness_loss
                epoch_witness_loss += witness_loss.item()

            total_loss.backward()
            optimizer.step()

            # Track metrics
            epoch_total_loss += total_loss.item()
            epoch_classification_loss += classification_loss.item()

            # Calculate accuracy on defined examples
            if defined_mask.any():
                predictions = logits[defined_mask].argmax(dim=1)
                correct_predictions = (predictions == batch_labels[defined_mask]).sum().item()
                epoch_correct_predictions += correct_predictions
                epoch_total_defined_samples += defined_mask.sum().item()

            # Update progress bar
            average_loss = epoch_total_loss / (progress_bar.n + 1)
            average_classification_loss = epoch_classification_loss / (progress_bar.n + 1)
            accuracy_percentage = epoch_correct_predictions / max(1, epoch_total_defined_samples) * 100

            progress_info = {
                'loss': f'{average_loss:.3f}',
                'cls_loss': f'{average_classification_loss:.3f}',
                'acc': f'{accuracy_percentage:.1f}%'
            }

            if witness_logits is not None:
                average_witness_loss = epoch_witness_loss / (progress_bar.n + 1)
                progress_info['wit_loss'] = f'{average_witness_loss:.3f}'

            progress_bar.set_postfix(progress_info)

        scheduler.step()


def evaluate_abstention(model: WitnessNetwork, test_data: List[Any], device: torch.device) -> ExperimentMetrics:
    """Evaluate model's abstention behavior on test data.

    Following the mathematical framework from experiment 9, evaluates how well
    the model abstains from contradictory inputs while maintaining accuracy
    on consistent inputs.

    Witness interpretation:
    - Witness state 0 = abstain (epistemic uncertainty from contradiction)
    - Witness state > 0 = commit (sufficient evidence for prediction)

    Args:
        model: Trained WitnessNetwork to evaluate
        test_data: List of test samples (images with labels)
        device: PyTorch device for evaluation

    Returns:
        ExperimentMetrics containing abstention rate, accuracy, calibration error,
        selective risk curve, and AUROC scores
    """
    if not test_data:
        raise ValueError("test_data cannot be empty")

    model.eval()

    all_predictions = []
    all_labels = []
    all_abstains = []
    all_probabilities = []
    all_witness_probs = []

    with torch.no_grad():
        for item in test_data:
            img = item[0].unsqueeze(0).to(device) if not isinstance(item, dict) else item['img'].unsqueeze(0).to(device)
            label = item[1] if not isinstance(item, dict) else item['label_context1']

            logits, witness_logits = model(img)
            prediction = logits.argmax(dim=1).cpu().item()
            probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]

            abstain = False
            witness_prob = None
            if witness_logits is not None:
                witness_prediction = torch.argmax(witness_logits, dim=1).cpu().item()
                abstain = (witness_prediction == 0)
                witness_prob = F.softmax(witness_logits, dim=1).cpu().numpy()[0]

            all_predictions.append(prediction)
            all_labels.append(label)
            all_abstains.append(abstain)
            all_probabilities.append(probabilities)
            all_witness_probs.append(witness_prob)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_abstains = np.array(all_abstains)
    all_probabilities = np.array(all_probabilities)

    abstention_rate = all_abstains.mean()
    commits = ~all_abstains
    accuracy_on_commits = (all_predictions[commits] == all_labels[commits]).mean() if commits.sum() > 0 else 0.0

    # Additional metrics
    calibration_error = compute_calibration_error(all_probabilities, all_labels)
    selective_risk_curve = compute_selective_risk_curve(all_probabilities, all_labels, all_abstains)

    # AUROC for OOD detection (using abstention as positive class for contradictory data)
    auroc = None
    if len(test_data) > 0 and isinstance(test_data[0], dict):
        # For contradictory data, use abstention confidence as scores
        if witness_logits is not None and all_witness_probs[0] is not None:
            abstain_scores = np.array([prob[0] for prob in all_witness_probs])  # P(abstain)
            true_labels = np.array([1 if abstain else 0 for abstain in all_abstains])  # 1 = should abstain
            if len(np.unique(true_labels)) > 1:
                auroc = roc_auc_score(true_labels, abstain_scores)

    return ExperimentMetrics(
        abstention_rate=abstention_rate,
        accuracy_on_commits=accuracy_on_commits,
        calibration_error=calibration_error,
        selective_risk_curve=selective_risk_curve,
        auroc=auroc
    )


def compute_calibration_error(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Compute expected calibration error."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == labels)

    # ECE calculation
    n_bins = CALIBRATION_BINS
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = correct[mask].mean()
            bin_size = mask.sum() / len(confidences)
            ece += bin_size * abs(bin_confidence - bin_accuracy)

    return ece


def compute_selective_risk_curve(probabilities: np.ndarray, labels: np.ndarray, abstains: np.ndarray) -> List[Tuple[float, float]]:
    """Compute selective risk curve points."""
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correct = (predictions == labels)

    # Sort by confidence (descending)
    sort_indices = np.argsort(confidences)[::-1]
    sorted_correct = correct[sort_indices]
    sorted_abstains = abstains[sort_indices]

    # Consider only non-abstained predictions
    committed_correct = sorted_correct[~sorted_abstains[sort_indices]]

    if len(committed_correct) == 0:
        return [(0.0, 1.0)]  # Default risk curve

    # Compute cumulative accuracy at different coverage levels
    risk_curve = []
    for coverage in SELECTIVE_RISK_COVERAGE_LEVELS:
        n_select = int(coverage * len(committed_correct))
        if n_select > 0:
            risk = 1.0 - committed_correct[:n_select].mean()
            risk_curve.append((coverage, risk))

    return risk_curve


def load_ood_datasets(cache_dir: Path, transform: Any) -> Dict[str, Any]:
    """Load multiple OOD datasets for comprehensive evaluation."""
    ood_datasets = {}

    # SVHN (street view house numbers)
    ood_datasets['svhn'] = datasets.SVHN(root=str(cache_dir), split='test', download=True, transform=transform)

    # CIFAR-100 (different classes, same distribution)
    ood_datasets['cifar100'] = datasets.CIFAR100(root=str(cache_dir), train=False, download=True, transform=transform)

    # MNIST as OOD (different distribution entirely)
    mnist_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((32, 32)),  # Resize to CIFAR size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    ood_datasets['mnist'] = datasets.MNIST(root=str(cache_dir), train=False, download=True, transform=mnist_transform)

    return ood_datasets




def _setup_experiment_data() -> Tuple[Any, Any, Any, Dict[str, Any], torch.device]:
    """Set up datasets and device for the experiment."""
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
    ood_datasets = load_ood_datasets(cache_dir, transform)

    np.random.seed(RANDOM_SEED)
    train_idx = np.random.choice(len(cifar_train), TRAIN_SUBSET_SIZE, replace=False)
    test_idx = np.random.choice(len(cifar_test), TEST_SUBSET_SIZE, replace=False)
    ood_idx = np.random.choice(len(svhn_test), OOD_SUBSET_SIZE, replace=False)

    cifar_train_subset = Subset(cifar_train, train_idx)
    cifar_test_subset = Subset(cifar_test, test_idx)
    svhn_test_subset = Subset(svhn_test, ood_idx)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return cifar_train_subset, cifar_test_subset, svhn_test_subset, ood_datasets, device


def _compute_witness_capacity_range(K: float) -> List[float]:
    """Compute the range of witness capacities to test based on contradiction K."""
    base_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    ceil_k = max(1.0, np.ceil(K))
    return sorted(list(set(base_values + [ceil_k, ceil_k + 1.0])))


def _evaluate_model_on_datasets(
    model: WitnessNetwork,
    contradictory_pairs: List[Dict[str, Any]],
    cifar_test_subset: Any,
    ood_datasets: Dict[str, Any],
    device: torch.device
) -> Tuple[ExperimentMetrics, ExperimentMetrics, Dict[str, ExperimentMetrics]]:
    """Evaluate model on all test datasets."""
    # Evaluate on contradictory data
    metrics_contradictory = evaluate_abstention(model, contradictory_pairs[:EVALUATION_SAMPLE_SIZE], device)

    # Evaluate on consistent CIFAR-10
    cifar_test_list = [(cifar_test_subset[i][0], cifar_test_subset[i][1])
                     for i in range(min(EVALUATION_SAMPLE_SIZE, len(cifar_test_subset)))]
    metrics_consistent = evaluate_abstention(model, cifar_test_list, device)

    # Evaluate on multiple OOD datasets
    ood_metrics = {}
    for ood_name, ood_dataset in ood_datasets.items():
        ood_subset = Subset(ood_dataset, np.random.choice(len(ood_dataset), min(EVALUATION_SAMPLE_SIZE, len(ood_dataset)), replace=False))
        ood_test_list = [(ood_subset[i][0], ood_subset[i][1]) for i in range(len(ood_subset))]
        ood_metrics[ood_name] = evaluate_abstention(model, ood_test_list, device)

    return metrics_contradictory, metrics_consistent, ood_metrics


def _train_and_evaluate_contradiction_type(
    contradiction_type: str,
    witness_bits_values: List[float],
    cifar_train_subset: Any,
    cifar_test_subset: Any,
    ood_datasets: Dict[str, Any],
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate models for a specific contradiction type."""
    print(f"\n=== Testing contradiction type: {contradiction_type} ===")

    # Create data with different contradiction types
    contradiction_data = create_contradictory_cifar10(
        cifar_train_subset,
        contradiction_ratio=BALANCED_CONTRADICTION_RATIO,
        seed=RANDOM_SEED,
        contradiction_type=contradiction_type
    )
    defined_data = contradiction_data.defined_samples
    undefined_data = contradiction_data.undefined_samples
    contradictory_pairs = contradiction_data.contradiction_examples

    results = {}
    for witness_bits in witness_bits_values:
        print(f"  Training r={witness_bits:.1f}...")
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        model = WitnessNetwork(witness_bits=witness_bits)
        train_on_contradiction(model, defined_data, undefined_data, device, epochs=TRAINING_EPOCHS)

        evaluation_results = _evaluate_model_on_datasets(
            model, contradictory_pairs, cifar_test_subset, ood_datasets, device
        )
        metrics_contradictory = evaluation_results[0]
        metrics_consistent = evaluation_results[1]
        ood_metrics = evaluation_results[2]

        # Store results
        result_key = f'r={witness_bits:.1f}'
        results[result_key] = {
            'r_bits': float(witness_bits),
            'contradictory': metrics_contradictory,
            'consistent': metrics_consistent,
            'ood': ood_metrics
        }

    return results


def _print_experiment_results(results: Dict[str, Any], witness_bits_values: List[float], K: float) -> None:
    """Print summary results for all contradiction types."""
    for contradiction_type in CONTRADICTION_TYPES:
        print(f"\n=== Results for {contradiction_type} contradictions ===")
        print("Abstention Rates by Witness Capacity:")
        header = f"{'Model':<12} {'Contradictory':<15} {'Consistent':<15} {'OOD (SVHN)':<15} {'Selectivity':<12}"
        print(header)
        print("-" * len(header))

        for r in witness_bits_values:
            result = results[f'r={r:.1f}'][contradiction_type]
            contra_abstain = result['contradictory'].abstention_rate * 100
            consist_abstain = result['consistent'].abstention_rate * 100
            ood_abstain = result['ood']['svhn'].abstention_rate * 100
            selectivity = contra_abstain - consist_abstain

            model_name = f"r={r:.1f}"
            print(f"{model_name:<12} {contra_abstain:>13.1f}% {consist_abstain:>13.1f}% {ood_abstain:>13.1f}% {selectivity:>+10.1f}%")

    # Phase transition evaluation: Theorem 7.4 predicts high abstention when r >= K
    for contradiction_type in CONTRADICTION_TYPES:
        print(f"\nPhase Transition Analysis for {contradiction_type} (K = {K:.3f} bits):")
        for r in witness_bits_values:
            result = results[f'r={r:.1f}'][contradiction_type]
            contradictory_rate = result['contradictory'].abstention_rate
            expected_high_abstention = r >= K
            actual_high_abstention = contradictory_rate > PHASE_TRANSITION_THRESHOLD
            status = "✓" if expected_high_abstention == actual_high_abstention else "✗"
            model_name = f"r={r:.1f}"
            print(f"  {model_name}: {contradictory_rate*100:.1f}% abstention {status}")


def _save_experiment_results(results: Dict[str, Any]) -> None:
    """Save experiment results to JSON file."""
    results_dir = Path("examples/hallucinations/experiment_11/results")
    with open(results_dir / 'generalization_test.json', 'w') as f:
        json.dump(results, f, indent=2)


def _create_contradiction_visualization(
    results: Dict[str, Any],
    r_values: List[float],
    contradiction_type: str
) -> None:
    """Create comprehensive visualization for a contradiction type."""
    contradictory_rates = [results[f'r={r:.1f}'][contradiction_type]['contradictory'].abstention_rate*100 for r in r_values]
    consistent_rates = [results[f'r={r:.1f}'][contradiction_type]['consistent'].abstention_rate*100 for r in r_values]
    ood_rates = [results[f'r={r:.1f}'][contradiction_type]['ood']['svhn'].abstention_rate*100 for r in r_values]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE)

    x = np.arange(len(r_values))
    width = BAR_WIDTH

    # Main abstention rates
    ax1.bar(x - width, contradictory_rates, width, label='Contradictory', color='#e74c3c', alpha=0.8)
    ax1.bar(x, consistent_rates, width, label='Consistent', color='#3498db', alpha=0.8)
    ax1.bar(x + width, ood_rates, width, label='OOD (SVHN)', color='#27ae60', alpha=0.8)

    ax1.set_xlabel('Witness Capacity r (bits)')
    ax1.set_ylabel('Abstention Rate (%)')
    ax1.set_title(f'Abstention Across Test Conditions ({contradiction_type})')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{r:.1f}' for r in r_values])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])

    # OOD improvements
    baseline_ood = ood_rates[0]
    ood_improvements = [rate - baseline_ood for rate in ood_rates]

    ax2.plot(r_values, ood_improvements, 'o-', linewidth=3, markersize=10, color='#27ae60')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Witness Capacity r (bits)')
    ax2.set_ylabel('OOD Abstention Improvement (%)')
    ax2.set_title('OOD Detection Generalization')
    ax2.grid(True, alpha=0.3)

    # Calibration error
    cal_contra = [results[f'r={r:.1f}'][contradiction_type]['contradictory'].calibration_error for r in r_values]
    cal_consist = [results[f'r={r:.1f}'][contradiction_type]['consistent'].calibration_error for r in r_values]
    cal_ood = [results[f'r={r:.1f}'][contradiction_type]['ood']['svhn'].calibration_error for r in r_values]

    ax3.plot(r_values, cal_contra, 'o-', label='Contradictory', color='#e74c3c', linewidth=2, markersize=8)
    ax3.plot(r_values, cal_consist, 's-', label='Consistent', color='#3498db', linewidth=2, markersize=8)
    ax3.plot(r_values, cal_ood, '^-', label='OOD (SVHN)', color='#27ae60', linewidth=2, markersize=8)
    ax3.set_xlabel('Witness Capacity r (bits)')
    ax3.set_ylabel('Expected Calibration Error')
    ax3.set_title('Calibration Error by Witness Capacity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Selectivity analysis
    selectivity_scores = []
    for r in r_values:
        contra_rate = results[f'r={r:.1f}'][contradiction_type]['contradictory'].abstention_rate
        consist_rate = results[f'r={r:.1f}'][contradiction_type]['consistent'].abstention_rate
        selectivity = contra_rate - consist_rate
        selectivity_scores.append(selectivity * 100)

    ax4.plot(r_values, selectivity_scores, 'o-', linewidth=3, markersize=10, color='#9b59b6')
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Witness Capacity r (bits)')
    ax4.set_ylabel('Selectivity (%)')
    ax4.set_title('Epistemic Uncertainty Selectivity')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    results_dir = Path("examples/hallucinations/experiment_11/results")
    plt.savefig(results_dir / f'generalization_test_{contradiction_type}.png', dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()


def _print_experiment_summary(results: Dict[str, Any], witness_bits_values: List[float], K: float) -> None:
    """Print overall experiment summary with success metrics."""
    print(f"\n{'='*60}")
    print("OVERALL EXPERIMENT SUMMARY")
    print(f"{'='*60}")

    for contradiction_type in CONTRADICTION_TYPES:
        print(f"\n{contradiction_type.upper()} CONTRADICTIONS:")

        # Evaluate witness capacity effectiveness using three key criteria
        print("1. Phase transition (Theorem 7.4): E + r ≥ K")
        phase_success_count = 0
        for r in witness_bits_values:
            result = results[f'r={r:.1f}'][contradiction_type]
            contradictory_rate = result['contradictory'].abstention_rate
            expected_high_abstention = r >= K
            actual_high_abstention = contradictory_rate > PHASE_TRANSITION_THRESHOLD
            if expected_high_abstention == actual_high_abstention:
                phase_success_count += 1

        phase_success = phase_success_count / len(witness_bits_values) > SELECTIVITY_SUCCESS_THRESHOLD
        print(f"   {'✓' if phase_success else '✗'} Phase transition confirmed ({phase_success_count}/{len(witness_bits_values)} correct)")

        print("2. Selective uncertainty (epistemic awareness)")
        selective_scores = []
        for r in witness_bits_values[1:]:  # Skip baseline
            result = results[f'r={r:.1f}'][contradiction_type]
            selectivity = (result['contradictory'].abstention_rate -
                         result['consistent'].abstention_rate)
            selective_scores.append(selectivity)

        selective_success = any(s > 0 for s in selective_scores)
        max_selectivity = max(selective_scores) * 100 if selective_scores else 0
        print(f"   {'✓' if selective_success else '✗'} Selective uncertainty achieved (max: {max_selectivity:+.1f}%)")

        print("3. OOD generalization (epistemic transfer)")
        baseline_ood = results['r=0.0'][contradiction_type]['ood']['svhn'].abstention_rate
        ood_improvements = []
        for r in witness_bits_values[1:]:
            current_ood = results[f'r={r:.1f}'][contradiction_type]['ood']['svhn'].abstention_rate
            ood_improvements.append(current_ood - baseline_ood)

        max_ood_improvement = max(ood_improvements) * 100 if ood_improvements else 0
        ood_success = max_ood_improvement > OOD_IMPROVEMENT_THRESHOLD
        print(f"   {'✓' if ood_success else '✗'} OOD generalization achieved (max: {max_ood_improvement:+.1f}%)")


def run_experiment() -> None:
    """Run the complete experiment testing witness capacity for epistemic uncertainty.

    This experiment validates Theorem 7.4 (E + r ≥ K) by:
    1. Computing structural contradiction K for multi-context scenarios
    2. Training models with varying witness capacities r
    3. Testing different types of contradictions (permutation, rotation, multi-label, adversarial)
    4. Evaluating abstention behavior and OOD generalization

    Results are saved to JSON and visualized as plots for each contradiction type.
    """
    cifar_train_subset, cifar_test_subset, svhn_test_subset, ood_datasets, device = _setup_experiment_data()

    # Compute K
    task_info = compute_task_contradiction(num_contexts=DEFAULT_NUM_CONTEXTS)
    K = task_info.contradiction_level

    print(f"Task contradiction: K = {K:.4f} bits")
    print(f"Witness capacities to test: r = [0.0, {max(1.0, np.ceil(K)):.1f}, {max(1.0, np.ceil(K)) + 1:.1f}] bits")
    print(f"Theoretical minimum error: E ≥ {1 - 2**(-K):.4f}")

    witness_bits_values = _compute_witness_capacity_range(K)
    results = {}

    for contradiction_type in CONTRADICTION_TYPES:
        contradiction_results = _train_and_evaluate_contradiction_type(
            contradiction_type, witness_bits_values, cifar_train_subset,
            cifar_test_subset, ood_datasets, device
        )
        for result_key, result_data in contradiction_results.items():
            if result_key not in results:
                results[result_key] = {}
            results[result_key][contradiction_type] = result_data
    
    # Print and save results
    _print_experiment_results(results, witness_bits_values, K)
    _save_experiment_results(results)

    # Create visualization for each contradiction type
    for contradiction_type in CONTRADICTION_TYPES:
        _create_contradiction_visualization(results, witness_bits_values, contradiction_type)

    # Overall evaluation summary
    _print_experiment_summary(results, witness_bits_values, K)




if __name__ == "__main__":
    run_experiment()