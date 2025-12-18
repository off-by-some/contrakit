"""
Shared utilities for testing hallucination inevitability in neural networks.

This module provides common dataset generation, training, and evaluation functions
for experiments testing whether neural networks hallucinate when given inputs
outside their training domain, and whether architectural modifications can
mitigate this behavior.

The experiments probe mathematical hypotheses about partial functions and learning:
- Standard neural networks hallucinate confidently on undefined inputs
- Definedness heads can learn to abstain but show limited generalization
- Training data composition affects hallucination rates systematically
- Hallucination rates increase with imbalance toward structured outputs

Tests examine:
- How fraction of defined inputs affects hallucination probability
- Whether definedness supervision reduces fabrication rates
- Statistical significance of hallucination behavior across random seeds
- Architectural differences between abstention-capable and forced-choice models

Assumptions:
- Input domain is discrete (0 to INPUT_SIZE-1) with partial function definition
- Neural networks use standard architectures (embeddings + MLPs)
- Training uses cross-entropy loss with optional definedness supervision
- Evaluation assumes undefined inputs should map to abstention (⊥) class

Typical usage:
- Generate partial functions: generate_partial_function(INPUT_SIZE, OUTPUT_CLASSES, ...)
- Train models with or without definedness heads: HallucinationNet(..., use_definedness_head=bool)
- Run complete experiments: run_experiment(defined_ratio, use_definedness_head, ...)
- Evaluate hallucination rates: calculate_hallucination_rate(predictions)
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, TypedDict
from contrakit.constants import DEFAULT_SEED


# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class PredictionAnalysis(TypedDict):
    """Type definition for prediction analysis results."""
    accuracy: float
    undefined_rate: float
    prediction_distribution: Dict[str, int]
    total_predictions: int


class ExperimentResult(TypedDict):
    """Type definition for experiment run results."""
    defined_ratio: float
    use_definedness_head: bool
    defined_accuracy: float
    hallucination_rate: float
    abstention_rate: float
    avg_confidence_defined: float
    avg_confidence_undefined: float
    function_map: Dict[int, str]


# ==============================================================================
# CONFIGURATION
# ==============================================================================
INPUT_SIZE = 128  # Number of possible inputs (0 to 127)
OUTPUT_CLASSES = ['A', 'B', 'C', 'D', '⊥']  # ⊥ represents "I don't know"
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 16


# ==============================================================================
# DATASET GENERATION
# ==============================================================================
def generate_partial_function(input_size: int, output_classes: List[str],
                            defined_ratio: float, undefined_supervision: float,
                            seed: int = DEFAULT_SEED) -> Tuple[Dict[int, str], List[int]]:
    """
    Create a partial function with some inputs defined and others undefined.

    Args:
        input_size: Total number of possible inputs
        output_classes: List of possible output labels
        defined_ratio: Fraction of inputs that have defined outputs
        undefined_supervision: Fraction of undefined inputs to label with ⊥ during training
        seed: Random seed for reproducibility

    Returns:
        function_map: Dictionary mapping input indices to output labels (or ⊥)
        defined_inputs: List of inputs that have defined outputs
    """
    np.random.seed(seed)

    # Determine which inputs are defined
    n_defined = int(input_size * defined_ratio)
    defined_inputs = np.random.choice(input_size, n_defined, replace=False)

    # Assign random labels to defined inputs (excluding ⊥)
    structured_classes = [c for c in output_classes if c != '⊥']
    function_map = {}

    for x in defined_inputs:
        function_map[x] = np.random.choice(structured_classes)

    # Add ⊥ labels to some undefined inputs for training supervision
    undefined_inputs = [x for x in range(input_size) if x not in defined_inputs]
    if undefined_supervision > 0:
        n_undefined_labeled = int(len(undefined_inputs) * undefined_supervision)
        undefined_supervised = np.random.choice(undefined_inputs, n_undefined_labeled,
                                              replace=False)
        for x in undefined_supervised:
            function_map[x] = '⊥'

    return function_map, defined_inputs


def create_datasets(function_map: Dict[int, str], input_size: int) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],  # train_data
    Tuple[np.ndarray, np.ndarray],            # test_defined
    Tuple[np.ndarray, np.ndarray]             # test_undefined
]:
    """
    Split data into training and test sets.

    Args:
        function_map: Mapping from inputs to outputs (including ⊥ labels)
        input_size: Total number of possible inputs

    Returns:
        train_data: (inputs, labels, definedness_flags) for training
        test_defined: (inputs, labels) for testing on defined inputs
        test_undefined: (inputs, labels) for testing on undefined inputs
    """
    all_inputs = list(range(input_size))
    undefined_idx = OUTPUT_CLASSES.index('⊥')

    # Training set: all explicitly labeled inputs
    train_inputs = list(function_map.keys())
    train_x = np.array(train_inputs)
    train_y = np.array([OUTPUT_CLASSES.index(function_map[x]) for x in train_inputs])

    # Track which training examples are actually defined vs ⊥-labeled
    train_defined = np.array([1.0 if function_map[x] != '⊥' else 0.0
                             for x in train_inputs])

    # Test sets
    defined_inputs = [x for x, y in function_map.items() if y != '⊥']
    test_defined_x = np.array(defined_inputs)
    test_defined_y = np.array([OUTPUT_CLASSES.index(function_map[x])
                               for x in defined_inputs])

    # Undefined inputs (not in function_map or explicitly ⊥)
    undefined_inputs = [x for x in all_inputs if x not in defined_inputs]
    test_undefined_x = np.array(undefined_inputs)
    test_undefined_y = np.array([undefined_idx] * len(undefined_inputs))

    return ((train_x, train_y, train_defined),
            (test_defined_x, test_defined_y),
            (test_undefined_x, test_undefined_y))


# ==============================================================================
# MODEL
# ==============================================================================
class HallucinationNet(nn.Module):
    """
    Neural network that may hallucinate when given undefined inputs.

    Without definedness head: Standard classifier that always produces confident outputs
    With definedness head: Can learn to abstain when uncertain
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 use_definedness_head: bool = False):
        super().__init__()
        self.use_definedness_head = use_definedness_head

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        if use_definedness_head:
            self.defined_head = nn.Linear(hidden_size, 1)

    def forward(self, x) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.embedding(x)
        h = self.relu(self.fc1(h))
        logits = self.fc2(h)

        if self.use_definedness_head:
            definedness = torch.sigmoid(self.defined_head(h))
            return logits, definedness

        return logits


# ==============================================================================
# TRAINING
# ==============================================================================
def train_model(model: nn.Module, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                epochs: int, lr: float, batch_size: int, verbose: bool = True) -> None:
    """
    Train the model on the provided dataset.

    Args:
        model: Neural network to train
        train_data: Tuple of (inputs, labels, definedness_flags)
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        verbose: Whether to print progress
    """
    train_x, train_y, train_defined = train_data
    train_x = torch.LongTensor(train_x)
    train_y = torch.LongTensor(train_y)
    train_defined = torch.FloatTensor(train_defined)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()

    dataset = torch.utils.data.TensorDataset(train_x, train_y, train_defined)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y, batch_def in loader:
            optimizer.zero_grad()

            if model.use_definedness_head:
                logits, definedness = model(batch_x)
                class_loss = criterion(logits, batch_y)
                def_loss = bce_loss(definedness.view(-1), batch_def)
                loss = class_loss + def_loss
            else:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


# ==============================================================================
# EVALUATION
# ==============================================================================
def evaluate_predictions(model: nn.Module, inputs: np.ndarray,
                        use_definedness_head: bool = False,
                        threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Get model predictions and confidence scores.

    Args:
        model: Trained model
        inputs: Input data to evaluate
        use_definedness_head: Whether model has definedness head
        threshold: Threshold for definedness head (if used)

    Returns:
        predictions: Predicted class indices
        confidences: Confidence scores for predictions
        abstention_rate: Fraction of inputs where model abstained (0 if no definedness head)
    """
    model.eval()
    undefined_idx = OUTPUT_CLASSES.index('⊥')
    inputs = torch.LongTensor(inputs)

    with torch.no_grad():
        output = model(inputs)

        if use_definedness_head:
            logits, definedness = output
            # Apply definedness threshold
            preds_raw = torch.argmax(logits, dim=1)
            preds = torch.where(
                definedness.squeeze() < threshold,
                torch.tensor(undefined_idx),
                preds_raw
            ).numpy()
            probs = torch.softmax(logits, dim=1)
            confidences = torch.max(probs, dim=1)[0].numpy()
            abstention_rate = (definedness.squeeze() < threshold).float().mean().item()
        else:
            logits = output
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).numpy()
            confidences = torch.max(probs, dim=1)[0].numpy()
            abstention_rate = 0.0

    return preds, confidences, abstention_rate


def calculate_hallucination_rate(predictions: np.ndarray) -> float:
    """
    Calculate what fraction of predictions are hallucinations (not ⊥).

    Args:
        predictions: Array of predicted class indices

    Returns:
        hallucination_rate: Fraction of predictions that are not ⊥
    """
    undefined_idx = OUTPUT_CLASSES.index('⊥')
    return (predictions != undefined_idx).mean()


def analyze_predictions(predictions: np.ndarray, true_labels: np.ndarray,
                       label: str = "") -> PredictionAnalysis:
    """
    Analyze prediction results and return summary statistics.

    Args:
        predictions: Predicted class indices
        true_labels: True class indices
        label: Descriptive label for this analysis

    Returns:
        Dictionary with accuracy, confidence, and prediction distribution
    """
    accuracy = np.mean(predictions == true_labels)
    confidences = np.array([])  # Would need to be passed in if available

    # Analyze prediction distribution
    pred_dist = defaultdict(int)
    for p in predictions:
        pred_dist[OUTPUT_CLASSES[p]] += 1

    undefined_idx = OUTPUT_CLASSES.index('⊥')
    undefined_rate = pred_dist['⊥'] / len(predictions) if len(predictions) > 0 else 0

    return {
        'accuracy': accuracy,
        'undefined_rate': undefined_rate,
        'prediction_distribution': dict(pred_dist),
        'total_predictions': len(predictions)
    }


def print_prediction_analysis(predictions: np.ndarray, true_labels: np.ndarray,
                            confidences: Optional[np.ndarray] = None,
                            abstention_rate: float = 0.0,
                            label: str = "") -> None:
    """
    Print a formatted analysis of prediction results.

    Args:
        predictions: Predicted class indices
        true_labels: True class indices
        confidences: Confidence scores (optional)
        abstention_rate: Fraction of abstentions (for definedness head)
        label: Descriptive label for this analysis
    """
    analysis = analyze_predictions(predictions, true_labels, label)

    print(f"\n{label}:")
    print(f"  Accuracy: {analysis['accuracy']:.2%}")
    if confidences is not None and len(confidences) > 0:
        print(f"  Average Confidence: {confidences.mean():.2%}")
    if abstention_rate > 0:
        print(f"  Abstention Rate: {abstention_rate:.2%}")
    print(f"  Prediction Distribution:")
    for cls in OUTPUT_CLASSES:
        count = analysis['prediction_distribution'].get(cls, 0)
        pct = count / analysis['total_predictions'] * 100 if analysis['total_predictions'] > 0 else 0
        print(f"    {cls}: {count:3d} ({pct:5.1f}%)")


# ==============================================================================
# EXPERIMENT UTILITIES
# ==============================================================================
def run_experiment(defined_ratio: float, use_definedness_head: bool = False,
                  undefined_supervision: float = 0.05, seed: int = DEFAULT_SEED) -> ExperimentResult:
    """
    Run a complete experiment: generate data, train model, evaluate.

    Args:
        defined_ratio: Fraction of inputs with defined outputs
        use_definedness_head: Whether to use definedness head
        undefined_supervision: Fraction of undefined inputs to supervise with ⊥
        seed: Random seed

    Returns:
        Dictionary with experiment results
    """
    # Generate data
    function_map, _ = generate_partial_function(
        INPUT_SIZE, OUTPUT_CLASSES, defined_ratio, undefined_supervision, seed
    )
    train_data, test_defined, test_undefined = create_datasets(function_map, INPUT_SIZE)

    # Train model
    torch.manual_seed(seed)
    model = HallucinationNet(INPUT_SIZE, HIDDEN_SIZE, len(OUTPUT_CLASSES),
                           use_definedness_head=use_definedness_head)
    train_model(model, train_data, EPOCHS, LEARNING_RATE, BATCH_SIZE, verbose=False)

    # Evaluate
    test_defined_x, test_defined_y = test_defined
    test_undefined_x, test_undefined_y = test_undefined

    # Evaluate on defined inputs
    preds_defined, conf_defined, _ = evaluate_predictions(
        model, test_defined_x, use_definedness_head
    )

    # Evaluate on undefined inputs
    preds_undefined, conf_undefined, abstention_rate = evaluate_predictions(
        model, test_undefined_x, use_definedness_head
    )

    hallucination_rate = calculate_hallucination_rate(preds_undefined)

    return {
        'defined_ratio': defined_ratio,
        'use_definedness_head': use_definedness_head,
        'defined_accuracy': np.mean(preds_defined == test_defined_y),
        'hallucination_rate': hallucination_rate,
        'abstention_rate': abstention_rate,
        'avg_confidence_defined': conf_defined.mean() if len(conf_defined) > 0 else 0,
        'avg_confidence_undefined': conf_undefined.mean() if len(conf_undefined) > 0 else 0,
        'function_map': function_map
    }
