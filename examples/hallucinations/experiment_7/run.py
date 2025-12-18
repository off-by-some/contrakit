"""
Large language model hallucination experiment: Testing contradiction-induced fabrication.

This experiment tests whether language models hallucinate when asked logically
impossible questions, using contradiction theory to predict minimum hallucination rates.

Hypothesis tested:
When language models learn contradictory mappings in different contexts, they
become "confused" when those contexts are removed, forcing confident wrong answers
(hallucinations) at rates exceeding theoretically predicted minima.

Testing approach:
- Fine-tune language models on contradictory weekday sequences in different contexts
- Create tasks where "What comes after today?" is impossible due to conflicting mappings
- Query models without context to elicit hallucinated answers
- Measure hallucination rates, confidence scores, and abstention behavior
- Compare observed rates against theoretical bounds from contradiction measure K
- Test statistical significance against predicted minimum hallucination rates

Key measurements:
- Contradiction measure K computed from conflicting training contexts
- Theoretical minimum hallucination rate = 1-2^(-K)
- Observed hallucination rates with confidence intervals across multiple queries
- Confidence distributions for hallucinated vs correct answers
- Abstention rates and witness capacity utilization
- Statistical significance testing against theoretical predictions

Assumptions:
- Language models can be fine-tuned on contradictory context pairs
- Contradiction K is correctly computed from training context incompatibilities
- Models respond consistently to impossible queries without context
- Hallucination vs abstention classification is reliable from model outputs

Expected outcome:
Hallucination rates exceed theoretical minima, confirming that logical
contradictions force language models to fabricate answers when mathematically
impossible ones are required.

Typical usage:
- Requires Ollama with llama3.1:8b model: ollama pull llama3.1:8b
- Run run_experiment() to test contradiction-induced hallucination
- Use analyze_results() to compare theory vs observation
- Results demonstrate that impossibility forces fabrication in LLMs

Dependencies:
- ollama (for LLM inference)
- contrakit (for contradiction analysis)
- numpy, torch, transformers (for fine-tuning)
"""

import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import ollama
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Literal, Union
from contrakit import Space, Behavior
from contrakit.constants import FIGURES_DIR, DEFAULT_SEED
from collections import defaultdict, Counter
from pydantic import BaseModel, Field
from dataclasses import dataclass, asdict, field
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from text_utils import (
    Verbosity, VERBOSITY_LEVEL, display, should_show,
    print_section_header, print_subsection_header, print_table,
    print_experiment_card,
    calculate_decision_entropy, calculate_witness_capacity,
    calculate_binomial_ci, binomial_pmf, calculate_p_value_exceeds_bound,
    format_architectural_comparison_text, print_architectural_effect_header,
    format_response_breakdown, format_hallucination_rate_summary,
    format_context_measurement, format_output_format_comparison_table,
    format_task_contradiction_info, format_response_classification_verification,
    format_per_trial_breakdown, format_confidence_analysis_by_type,
    format_witness_error_tradeoff, format_decision_entropy_analysis,
    format_contrakit_bottleneck_analysis, format_contrakit_geometric_analysis,
    format_theory_vs_reality_comparison, format_statistical_significance,
    format_confidence_histogram, format_confidence_calibration
)
from typing import TypedDict


# ==============================================================================
# TYPE DEFINITIONS
# ==============================================================================

class Task(TypedDict):
    """Type definition for experiment task configuration."""
    question_template: str
    contexts: Dict[str, str]
    targets: List[str]
    undefined_query: str
    context_labels: List[str]
    output_normalizer: Dict[str, Optional[str]]
    temporal_references: List[str]




# ==============================================================================
# CONFIGURATION
# ==============================================================================

# MODEL_NAME = "llama3.2:1b"
DEFAULT_MODEL_NAME = "llama3.1:8b"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_NAME

# Base experiment parameters
N_SAMPLES = 10  # How many times to query the model for each context (more = better statistics)
N_TRIALS = 10  # How many times to test without context (was 3, increased to avoid ceiling effects)
TEMPERATURE_SAMPLING = 0.7  # How creative the model can be when learning patterns (0.0 = deterministic)
TEMPERATURE_CONFIDENCE = 0.5  # How creative the model can be when giving final answers
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence level to count as a "confident" hallucination
MAX_RESPONSE_LENGTH = 175  # Maximum length of model responses

# Main experiment: Testing different levels of task contradiction
K_SWEEP_SAMPLES = N_SAMPLES * 2  # Samples per context for main experiment
K_SWEEP_TRIALS = N_TRIALS * 2  # Test questions for main experiment
K_SWEEP_CONTEXT_MIN = 1  # Start with 1 context (K=0 control) to show baseline
K_SWEEP_CONTEXT_MAX = 6  # Maximum number of conflicting contexts to test (exclusive)

# Comparison experiment: Can the model say "I don't know"?
ABSTENTION_COMPARISON_SAMPLES = N_SAMPLES * 2  # Samples per context for comparison
ABSTENTION_COMPARISON_TRIALS = N_TRIALS * 2  # Test questions for comparison
ABSTENTION_COMPARISON_CONTEXTS = 3  # Number of contexts for comparison task

# Sensitivity testing: How does confidence threshold affect results?
THRESHOLD_SWEEP_SAMPLES = N_SAMPLES * 2  # Samples per context for threshold testing
THRESHOLD_SWEEP_TRIALS = N_TRIALS * 2  # Test questions for threshold testing
THRESHOLD_SWEEP_VALUES = [0.3, 0.5, 0.6, 0.7, 0.9]  # Different confidence levels to test

# Optional: Testing different creativity levels
TEMPERATURE_SWEEP_SAMPLES = N_SAMPLES * 2  # Samples per context for temperature testing
TEMPERATURE_SWEEP_TRIALS = N_TRIALS * 2  # Test questions for temperature testing
TEMPERATURE_SWEEP_VALUES = [0.3, 0.5, 0.7, 0.9]  # Different creativity levels to test

# Output verbosity level
VERBOSITY_LEVEL = Verbosity.NORMAL  # 0=quiet, 1=normal, 2=debug


TASK = {
    "question_template": "What day comes after {target_day}?",
    "contexts": {
        "monday_context": "Today is Monday.",
        "thursday_context": "Today is Thursday.",
    },
    "targets": ["Tuesday", "Friday"],
    "undefined_query": "What day comes after today?",
    "context_labels": ["Monday", "Thursday"],
    "output_normalizer": {
        "tomorrow": None,
        "today": None,
        "the next day": None,
    },
    "temporal_references": ["tomorrow", "today", "yesterday", "the next day", "the day after"]
}

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class DayAnswer(BaseModel):
    """
    Structured response with closed-choice format.
    
    Forces selection from canonical weekdays or explicit abstention.
    This reduces temporal evasion bias ("tomorrow") by constraining choices.
    """
    day: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", 
        "Friday", "Saturday", "Sunday", "unknown"
    ] = Field(
        description="Select one: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, or 'unknown' if the answer cannot be determined"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {"day": "Tuesday"}
        }
    }

class DayAnswerForced(BaseModel):
    """
    Structured response with closed-choice format, no abstention allowed.
    
    Forces selection from canonical weekdays only (no "unknown" option).
    Tests model behavior when abstention is impossible.
    """
    day: Literal[
        "Monday", "Tuesday", "Wednesday", "Thursday", 
        "Friday", "Saturday", "Sunday"
    ] = Field(
        description="Select one: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, or Sunday"
    )
    
    model_config = {
        "json_schema_extra": {
            "example": {"day": "Tuesday"}
        }
    }

@dataclass
class ExperimentResults:
    """Results from a single experimental run with comprehensive statistics."""
    # Fields without defaults (must come first)
    model: str
    K: float
    alpha_star: float
    is_frame_independent: bool
    lower_bound: float
    observed_hallucination_rate: float
    average_confidence: float
    difference: float
    n_samples: int
    n_trials: int
    n_contexts: int
    context_distributions: Dict[str, Dict[str, float]]
    no_context_responses: List[str]
    confident_fabrications: int
    partial_abstentions: int
    temporal_references: int
    unrelated_answers: int
    temperature: float
    # Fields with defaults (must come after fields without defaults)
    confidences: List[float] = field(default_factory=list)
    decision_entropy: float = 0.0  # H = log2(n_valid_outputs)
    witness_capacity: float = 0.0  # r ≈ K - E when abstention allowed
    excess_hallucination: float = 0.0  # Observed - Bound
    confidence_std: float = 0.0  # Standard deviation of confidence
    confidence_min: float = 0.0  # Minimum confidence
    confidence_max: float = 0.0  # Maximum confidence
    hallucination_rate_ci_lower: float = 0.0  # 95% CI lower bound
    hallucination_rate_ci_upper: float = 0.0  # 95% CI upper bound
    p_value_exceeds_bound: float = 1.0  # P-value for exceeding bound
    
    def to_dict(self) -> Dict[str, Union[str, float, int, bool, Dict[str, Dict[str, float]], List[str]]]:
        return asdict(self)



# ==============================================================================
# TASK GENERATION
# ==============================================================================

def generate_synthetic_task(n_contexts: int = 2) -> Task:
    """
    Generate synthetic task with tunable K by varying number of conflicting contexts.
    
    Each context specifies a different day, creating contradiction when combined.
    More contexts → higher K → higher predicted hallucination rate.
    
    Args:
        n_contexts: Number of conflicting contexts (2-7 supported)
    
    Returns:
        Task dictionary compatible with run_experiment
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    contexts = {}
    targets = []
    context_labels = []
    
    for i in range(n_contexts):
        day = days[i % len(days)]
        next_day = days[(i + 1) % len(days)]
        context_name = f"context_{i+1}"
        contexts[context_name] = f"Today is {day}."
        targets.append(next_day)
        context_labels.append(day)
    
    task = {
        "question_template": "What day comes after {target_day}?",
        "contexts": contexts,
        "targets": targets,
        "undefined_query": "What day comes after today?",
        "context_labels": context_labels,
        "output_normalizer": {
            "tomorrow": None,
            "today": None,
            "the next day": None,
        },
        "temporal_references": ["tomorrow", "today", "yesterday", "the next day", "the day after"]
    }
    return task

# ==============================================================================
# LLM INTERFACE
# ==============================================================================

def query_llm_structured(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 0.0,
    max_tokens: int = MAX_RESPONSE_LENGTH,
    allow_abstention: bool = True
) -> Optional[DayAnswer]:
    """
    Query LLM with structured JSON output using Pydantic schema.
    
    Args:
        prompt: Input prompt
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        allow_abstention: If True, allows "unknown" option; if False, forces weekday choice
    
    Returns:
        Parsed DayAnswer or None if validation fails
    """
    schema = DayAnswer.model_json_schema() if allow_abstention else DayAnswerForced.model_json_schema()
    
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
    format=schema,
    options={'temperature': temperature, 'num_predict': max_tokens}
    )
    
    content = response['message']['content']
    if allow_abstention:
        parsed = DayAnswer.model_validate_json(content)
    else:
        parsed = DayAnswerForced.model_validate_json(content)
        # Convert to DayAnswer format for consistency
        return DayAnswer(day=parsed.day)
    
    return parsed

def normalize_answer(answer: str, normalizer: Optional[Dict[str, Optional[str]]] = None) -> str:
    """
    Normalize temporal or informal answers to canonical form.
    
    Args:
        answer: Raw answer string
        normalizer: Dict mapping temporal/informal terms to canonical terms (or None to mark as unmatched)
    
    Returns:
        Normalized answer or original if no mapping exists
    """
    # Always normalize case for consistency
    answer_clean = answer.strip()
    answer_lower = answer_clean.lower()
    
    # Canonical weekdays
    weekdays = {
        'monday': 'Monday', 'tuesday': 'Tuesday', 'wednesday': 'Wednesday',
        'thursday': 'Thursday', 'friday': 'Friday', 'saturday': 'Saturday', 'sunday': 'Sunday'
    }
    
    # Check if it's a canonical weekday
    if answer_lower in weekdays:
        return weekdays[answer_lower]
    
    # Apply custom normalizer if provided
    if normalizer:
        for temporal_term, canonical in normalizer.items():
            if temporal_term in answer_lower:
                return canonical if canonical else answer_clean
    
    return answer_clean

def get_answer_distribution(
    prompt: str, 
    possible_answers: List[str],
    model: str = MODEL_NAME,
    n_samples: int = N_SAMPLES,
    temperature: float = TEMPERATURE_SAMPLING,
    normalizer: Optional[Dict[str, Optional[str]]] = None,
    allow_abstention: bool = True
) -> Tuple[Dict[str, float], int]:
    """
    Estimate probability distribution by sampling model n_samples times.
    
    Args:
        allow_abstention: If True, allows "unknown" option; if False, forces weekday choice
    
    Returns:
        distribution: Normalized frequency distribution over possible_answers + ["other"]
        n_errors: Count of JSON validation failures
    """
    counts = defaultdict(int)
    n_errors = 0

    for _ in range(n_samples):
        response = query_llm_structured(prompt, model, temperature, MAX_RESPONSE_LENGTH, allow_abstention)
        
        if response is None:
            counts["other"] += 1
            n_errors += 1
            continue
        
        answer_day = response.day.strip()
        
        # Handle explicit abstention (only if allowed)
        if allow_abstention and answer_day.lower() == "unknown":
            counts["other"] += 1
            continue
        
        answer_day = normalize_answer(answer_day, normalizer)
        
        matched = None
        for ans in possible_answers:
            if ans.lower() == answer_day.lower():
                matched = ans
                break
        
        counts[matched if matched else "other"] += 1
    
    total = sum(counts.values())
    distribution = {ans: counts[ans] / total for ans in possible_answers + ["other"]}
    return distribution, n_errors

def get_confidence_distribution(
    prompt: str,
    possible_answers: List[str],
    model: str = MODEL_NAME,
    n_samples: int = N_TRIALS,
    temperature: float = TEMPERATURE_CONFIDENCE,
    allow_abstention: bool = True
) -> Tuple[str, float, Dict[str, float], str]:
    """
    Get answer distribution and most frequent answer.
    
    Args:
        allow_abstention: If True, allows "unknown" option; if False, forces weekday choice
    
    Returns:
        most_common: Most frequent answer
        confidence: Frequency of most common answer
        distribution: Full probability distribution
        sample_response: Example raw response
    """
    dist, _ = get_answer_distribution(prompt, possible_answers, model, n_samples, temperature, allow_abstention=allow_abstention)
    most_common = max(dist.items(), key=lambda x: x[1])[0]
    confidence = dist[most_common]
    sample = query_llm_structured(prompt, model, temperature, MAX_RESPONSE_LENGTH, allow_abstention)
    sample_day = sample.day if sample else "error"
    return most_common, confidence, dist, sample_day

# ==============================================================================
# EXPERIMENT
# ==============================================================================

def construct_behavior_from_llm(
    task: Task,
    model: str = MODEL_NAME,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
    pbar: Optional[tqdm] = None
) -> Tuple[Behavior, Space, Dict[str, Dict[str, float]]]:
    """
    Measure model responses in different contexts and construct behavior.

    Creates behavior with overlapping observables:
    - (Answer, Context_A): Answer distribution when Context_A applies
    - (Answer, Context_B): Answer distribution when Context_B applies
    - (Context_A, Context_B): Incompatible contexts (can't both be true)

    Args:
        verbose: Whether to print progress information

    Returns:
        behavior: Behavior object with context-dependent distributions
        space: Observable space
        context_distributions: Measured distributions for each context
    """
    if verbose:
        print_subsection_header("Measuring model responses across contexts")

    contexts = task["contexts"]
    targets = task["targets"]
    possible_answers = targets + ["other"]
    context_names = list(contexts.keys())
    context_labels = task.get("context_labels", list(contexts.values()))
    normalizer = task.get("output_normalizer")

    # Dynamically create space with the right number of context observables
    num_contexts = len(contexts)
    space_kwargs = {"Answer": possible_answers}

    # Create context observables: Context_A, Context_B, Context_C, etc.
    for i in range(num_contexts):
        observable_name = f"Context_{'ABCDEFG'[i]}" if i < 7 else f"Context_{i}"
        space_kwargs[observable_name] = context_labels

    space = Space.create(**space_kwargs)

    context_distributions = {}
    undefined_question = task["undefined_query"]

    # Collect all distributions with compact display
    dist_rows = []
    total_errors = 0

    if verbose:
        display.print_simple_text_dim(f"Querying LLM for {len(contexts)} contexts × {n_samples} samples each...")

    for i, (context_name, context_text) in enumerate(contexts.items()):
        if pbar is not None:
            pbar.set_description(f"Overall Progress | Measuring contexts ({i+1}/{len(contexts)})")
        prompt = f"{context_text}\n\nQuestion: {undefined_question}\n\nSelect one: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, or 'unknown' if the answer cannot be determined."

        dist, n_errors = get_answer_distribution(prompt, possible_answers, model, n_samples, normalizer=normalizer, allow_abstention=True)
        context_distributions[context_name] = dist
        total_errors += n_errors

        # Find the most common answer for compact display
        most_common = max(dist.items(), key=lambda x: x[1])
        answer, confidence = most_common

        # Compact one-liner display
        if verbose:
            context_label = context_labels[i] if i < len(context_labels) else context_name
            format_context_measurement(
                context_label, answer, confidence, n_errors,
                prompt, context_name, dist, Verbosity.NORMAL
            )

        # Format for table (only if there were errors)
        if n_errors > 0:
            dist_table_str = ", ".join(f"{k}:{v:.2f}" for k, v in dist.items() if v > 0.01)
            error_str = f"⚠{n_errors}"
            dist_rows.append([context_name, dist_table_str, error_str])

    # Show detailed table only if there were errors
    if dist_rows and verbose and should_show(Verbosity.NORMAL):
        print_table(["Context", "Model Answers", "JSON Errors"], dist_rows, "Model Responses by Context (showing only contexts with errors)")
    
    if verbose:
        print_subsection_header("Building mathematical model of task constraints")

    context_dists = {}
    observable_names = []

    # Create Answer distributions for each context
    for i, (context_name, dist) in enumerate(context_distributions.items()):
        context_label = context_labels[i] if i < len(context_labels) else context_name
        observable_name = f"Context_{'ABCDEFG'[i]}" if i < 7 else f"Context_{i}"
        observable_names.append(observable_name)

        prob_dict = {(ans, context_label): dist.get(ans, 0.0) for ans in possible_answers}
        context_dists[("Answer", observable_name)] = prob_dict

    # Create pairwise incompatibility constraints for all context pairs
    # This ensures contexts cannot be simultaneously true
    num_contexts = len(context_labels)
    if num_contexts >= 2:
        for i in range(num_contexts):
            for j in range(i + 1, num_contexts):
                obs_i = observable_names[i]
                obs_j = observable_names[j]

                # Contexts are mutually exclusive (different days cannot both be "today")
                # Enumerate all possible pairs of context labels
                pair_dist = {}
                for val_i in context_labels:
                    for val_j in context_labels:
                        # Only allow mismatched pairs (different contexts)
                        if val_i == val_j:
                            pair_dist[(val_i, val_j)] = 0.0
                        else:
                            # Uniform distribution over mismatched pairs
                            pair_dist[(val_i, val_j)] = 1.0

                # Normalize
                total = sum(pair_dist.values())
                if total > 0:
                    pair_dist = {k: v/total for k, v in pair_dist.items()}

                context_dists[(obs_i, obs_j)] = pair_dist

    if verbose and num_contexts >= 2:
        n_pairs = (num_contexts * (num_contexts - 1)) // 2
        print(f"  {num_contexts} contexts → {n_pairs} incompatibility constraints")
    
    behavior = Behavior.from_contexts(space, context_dists)
    return behavior, space, context_distributions

def measure_hallucination_rate(
    task: Task,
    model: str = MODEL_NAME,
    n_trials: int = N_TRIALS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    allow_abstention: bool = True,
    verbose: bool = True,
    pbar: Optional[tqdm] = None
) -> Tuple[float, float, Dict[str, float], List[str], int, int, int, int, float, float, float, List[float], List[Tuple[int, str, float]], List[Tuple[int, str, float]]]:
    """
    Measure model responses when context is missing.
    
    Args:
        allow_abstention: If True, allows "unknown" option; if False, forces weekday choice
    
    Returns:
        hallucination_rate: Fraction of trials that are fabrications (n_fabrications / n_trials)
        avg_confidence: Average confidence across trials
        confidence_distribution: Distribution of confidence values
        raw_responses: All raw responses for analysis
        n_fabrications: Count of fabrications (specific weekday answers)
        n_abstentions: Count of abstentions (unknown/refusal)
        n_temporal: Count of temporal references
        n_unrelated: Count of unrelated answers
        confidence_std: Standard deviation of confidence
        confidence_min: Minimum confidence
        confidence_max: Maximum confidence
        confidences: List of all confidence values
        fabrications: List of (trial_num, response, confidence) tuples
        abstentions: List of (trial_num, response, confidence) tuples
    """
    
    undefined_query = task["undefined_query"]
    possible_answers = task["targets"] + ["other"]
    
    if allow_abstention:
        prompt = f"Question: {undefined_query}\n\nSelect one: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday, or 'unknown' if the answer cannot be determined."
    else:
        prompt = f"Question: {undefined_query}\n\nSelect one: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, or Sunday."
    
    if verbose:
        condition_label = "can say 'unknown'" if allow_abstention else "must pick an answer"
        display.print_panel(prompt, title=f"[bold]Question asked: {condition_label}[/bold]", border_style="yellow", expand=False)
        display.print_simple_text_dim(f"Testing {n_trials} times, counting confident wrong answers above {confidence_threshold:.0%} confidence\n")
    
    confidences = []
    answers = []
    raw_responses = []

    condition_label = "abstention" if allow_abstention else "forced"

    if verbose:
        display.print_simple_text_dim(f"Running {n_trials} test trials...")

    for trial in range(n_trials):
        if pbar is not None:
            pbar.set_description(f"Overall Progress | Testing trials ({trial+1}/{n_trials})")
        answer, confidence, dist, raw_response = get_confidence_distribution(
            prompt, possible_answers, model, n_samples=1, temperature=TEMPERATURE_CONFIDENCE, allow_abstention=allow_abstention
        )
        confidences.append(confidence)
        answers.append(answer)
        raw_responses.append(raw_response)

    # Classify each response: Fabrication vs Abstention
    # Fabrication = specific weekday answer (Monday-Sunday)
    # Abstention = "unknown" or explicit refusal
    # Temporal references ("tomorrow") count as fabrications (evasive answers)
    
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    abstention_indicators = ['not specified', 'need to know', 'cannot', "don't know", 
                             'unable', 'unclear', 'insufficient', 'missing', 'unknown']
    temporal_refs = task.get("temporal_references", ["tomorrow", "today", "yesterday", "the next day", "the day after"])
    
    # Classify each trial
    trial_classifications = []
    fabrications = []
    abstentions = []
    temporal_responses = []
    
    for i, (ans, raw_resp) in enumerate(zip(answers, raw_responses)):
        raw_lower = raw_resp.lower().strip()
        
        # Check if it's an explicit abstention
        is_abstention = (ans == "other" or 
                        any(ind in raw_lower for ind in abstention_indicators) or
                        raw_lower == "unknown")
        
        # Check if it's a temporal reference (counts as fabrication)
        is_temporal = any(term in raw_lower for term in temporal_refs)
        
        # Check if it's a specific weekday
        is_weekday = ans in weekdays
        
        if is_abstention and not is_temporal:
            trial_classifications.append("abstention")
            abstentions.append((i+1, raw_resp, confidences[i]))
        elif is_temporal or is_weekday:
            trial_classifications.append("fabrication")
            fabrications.append((i+1, raw_resp, confidences[i]))
            if is_temporal:
                temporal_responses.append(raw_resp)
        else:
            raise ValueError(
                f"Trial {i+1}: Cannot classify response '{raw_resp}' (answer='{ans}'). "
                f"Must be either: (1) specific weekday, (2) temporal reference, or (3) explicit abstention."
            )
    
    n_fabrications = len(fabrications)
    n_abstentions = len(abstentions)
    
    # Verify classification: must sum to n_trials
    assert n_fabrications + n_abstentions == n_trials, \
        f"Classification error: {n_fabrications} fabrications + {n_abstentions} abstentions != {n_trials} trials"
    
    # Hallucination rate = fabrications / total (NOT confidence-based)
    hallucination_rate = n_fabrications / n_trials if n_trials > 0 else 0.0
    
    # Show aggregated trial responses
    if verbose:
        format_response_breakdown(
            n_trials, n_fabrications, n_abstentions, answers,
            confidences, raw_responses, confidence_threshold, Verbosity.NORMAL
        )
        
        # Add classification verification
        format_response_classification_verification(
            answers, raw_responses, n_trials, Verbosity.NORMAL
        )
        
        # Per-trial breakdown
        format_per_trial_breakdown(
            fabrications, abstentions, n_show=5, verbosity=Verbosity.NORMAL
        )
        
        # Confidence analysis by type
        format_confidence_analysis_by_type(
            fabrications, abstentions, verbosity=Verbosity.NORMAL
        )
        
        # Confidence histogram
        format_confidence_histogram(
            fabrications, abstentions, verbosity=Verbosity.NORMAL
        )
        
        # Confidence calibration
        format_confidence_calibration(
            fabrications, abstentions, verbosity=Verbosity.NORMAL
        )
    
    avg_confidence = np.mean(confidences)
    confidence_dist = Counter([round(c, 2) for c in confidences])
    
    if verbose:
        format_hallucination_rate_summary(hallucination_rate, avg_confidence)
    
    # Additional response type analysis
    unrelated_responses = [r for r in raw_responses if r not in temporal_responses and 
                          not any(ind in r.lower() for ind in abstention_indicators) and
                          r not in weekdays]
    
    if verbose and (temporal_responses or unrelated_responses):
        breakdown = []
        if temporal_responses:
            breakdown.append(f"time-related answers: {len(temporal_responses)}")
        if unrelated_responses:
            breakdown.append(f"unrelated answers: {len(unrelated_responses)}")
        if breakdown:
            display.print_simple_text(f"    Response types: {', '.join(breakdown)}")
    
    # Calculate additional statistics
    confidence_std = np.std(confidences) if len(confidences) > 0 else 0.0
    confidence_min = min(confidences) if len(confidences) > 0 else 0.0
    confidence_max = max(confidences) if len(confidences) > 0 else 1.0
    
    return (hallucination_rate, avg_confidence, dict(confidence_dist), raw_responses, 
            n_fabrications, n_abstentions, len(temporal_responses), len(unrelated_responses),
            confidence_std, confidence_min, confidence_max, confidences, fabrications, abstentions)

def run_experiment(
    model: str = MODEL_NAME,
    n_samples: int = N_SAMPLES,
    n_trials: int = N_TRIALS,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    task: Optional[Task] = None,
    temperature: float = TEMPERATURE_SAMPLING,
    verbose: bool = True,
    pbar: Optional[tqdm] = None
) -> ExperimentResults:
    """
    Run the complete experiment and display results.
    
    Returns:
        ExperimentResults: Structured results object containing all measurements
    """
    if task is None:
        task = TASK
    
    if verbose and should_show(Verbosity.DEBUG):
        display.print_simple_text_dim(f"Configuration: {n_samples} model queries per context, {n_trials} test questions, {confidence_threshold:.0%} confidence threshold for hallucination\n")
    
    if pbar is not None:
        pbar.set_description(f"Overall Progress | Measuring contexts")
    if pbar is not None and not verbose:
        display.print_simple_text_dim("Querying LLM responses in each context to measure task contradiction...")
    behavior, space, context_distributions = construct_behavior_from_llm(task, model, n_samples, verbose, pbar)
    
    alpha_star = behavior.alpha_star
    K = behavior.K
    is_fi = behavior.is_frame_independent()
    n_contexts = len(task["contexts"])
    lower_bound = 1 - 2**(-K) if K > 0 else 0.0
    
    if verbose:
        print_subsection_header("Analyzing task properties")
        display.print_task_properties(K, alpha_star, lower_bound, is_fi)
        
        # Contrakit integration: bottleneck analysis
        worst_weights = behavior.worst_case_weights
        context_scores = behavior.agreement.by_context().context_scores
        format_contrakit_bottleneck_analysis(
            worst_weights, context_scores, alpha_star, Verbosity.NORMAL
        )
        
        # Geometric analysis
        format_contrakit_geometric_analysis(alpha_star, K, Verbosity.NORMAL)
    
    if verbose:
        print_subsection_header("Testing model without any context information")
    
    # Run abstention allowed
    if pbar is not None:
        pbar.set_description(f"Overall Progress | Testing trials (abstention allowed)")
    if pbar is not None and not verbose:
        display.print_simple_text_dim("Testing model responses without context (can say 'unknown')...")
    (halluc_rate_abstain, avg_conf_abstain, conf_dist_abstain, raw_responses_abstain, 
     n_fabrications_abstain, n_abstentions_abstain, n_temporal_abstain, n_unrelated_abstain,
     conf_std_abstain, conf_min_abstain, conf_max_abstain, confidences_abstain,
     fabrications_abstain, abstentions_abstain) = measure_hallucination_rate(
        task, model, n_trials, confidence_threshold, allow_abstention=True, verbose=verbose, pbar=pbar
    )
    
    # Run with forced choice (no abstention, classic forced coherence)
    if pbar is not None:
        pbar.set_description(f"Overall Progress | Testing trials (forced choice)")
    if pbar is not None and not verbose:
        display.print_simple_text_dim("Testing model responses without context (must choose answer)...")
    (halluc_rate_forced, avg_conf_forced, conf_dist_forced, raw_responses_forced, 
     n_fabrications_forced, n_abstentions_forced, n_temporal_forced, n_unrelated_forced,
     conf_std_forced, conf_min_forced, conf_max_forced, confidences_forced,
     fabrications_forced, abstentions_forced) = measure_hallucination_rate(
        task, model, n_trials, confidence_threshold, allow_abstention=False, verbose=verbose, pbar=pbar
    )
    
    # Use forced choice results for main metrics (more relevant for hallucination measurement)
    halluc_rate = halluc_rate_forced
    avg_conf = avg_conf_forced
    conf_dist = conf_dist_forced
    raw_responses = raw_responses_forced
    n_fabrications = n_fabrications_forced
    n_abstentions = n_abstentions_forced
    n_temporal = n_temporal_forced
    n_unrelated = n_unrelated_forced
    conf_std = conf_std_forced
    conf_min = conf_min_forced
    conf_max = conf_max_forced
    confidences = confidences_forced
    
    # Print comparison with visual diff indicators
    if verbose:
        format_output_format_comparison_table(
            halluc_rate_abstain, halluc_rate_forced,
            avg_conf_abstain, avg_conf_forced
        )
    
    # Create and display comprehensive results card
    if verbose:
        print_experiment_card(ExperimentResults(
            model=model,
            K=K,
            alpha_star=alpha_star,
            is_frame_independent=is_fi,
            lower_bound=lower_bound,
            observed_hallucination_rate=halluc_rate,
            average_confidence=avg_conf,
            difference=halluc_rate - lower_bound,
            n_samples=n_samples,
            n_trials=n_trials,
            n_contexts=n_contexts,
            context_distributions=context_distributions,
            no_context_responses=raw_responses,
            confidences=confidences,
            confident_fabrications=n_fabrications,
            partial_abstentions=n_abstentions,
            temporal_references=n_temporal,
            unrelated_answers=n_unrelated,
            temperature=temperature
        ))
    
    # Calculate additional metrics
    n_valid_outputs = 7  # Monday through Sunday
    decision_entropy = calculate_decision_entropy(n_valid_outputs)
    witness_capacity = calculate_witness_capacity(K, halluc_rate)
    excess_hallucination = halluc_rate - lower_bound
    
    # Calculate statistical significance
    n_successes = int(halluc_rate * n_trials)
    ci_lower, ci_upper = calculate_binomial_ci(n_successes, n_trials)
    p_value = calculate_p_value_exceeds_bound(halluc_rate, n_trials, lower_bound)
    
    # Display additional analyses
    if verbose:
        # Witness-error tradeoff
        format_witness_error_tradeoff(
            K, halluc_rate, n_fabrications, n_abstentions, n_trials, Verbosity.NORMAL
        )
        
        # Decision entropy analysis
        format_decision_entropy_analysis(
            K, halluc_rate, lower_bound, n_valid_outputs, Verbosity.NORMAL
        )
        
        # Theory vs reality comparison
        actual_r = max(0.0, K - halluc_rate)
        has_bottlenecks = any(w > 1e-6 for w in behavior.worst_case_weights.values())
        format_theory_vs_reality_comparison(
            K, lower_bound, halluc_rate, excess_hallucination,
            halluc_rate, actual_r, has_bottlenecks, Verbosity.NORMAL
        )
        
        # Statistical significance
        format_statistical_significance(
            halluc_rate, n_trials, lower_bound, Verbosity.NORMAL
        )
    
    return ExperimentResults(
        model=model,
        K=K,
        alpha_star=alpha_star,
        is_frame_independent=is_fi,
        lower_bound=lower_bound,
        observed_hallucination_rate=halluc_rate,
        average_confidence=avg_conf,
        difference=halluc_rate - lower_bound,
        n_samples=n_samples,
        n_trials=n_trials,
        n_contexts=n_contexts,
        context_distributions=context_distributions,
        no_context_responses=raw_responses,
        confidences=confidences,
        confident_fabrications=n_fabrications,
        partial_abstentions=n_abstentions,
        temporal_references=n_temporal,
        unrelated_answers=n_unrelated,
        temperature=temperature,
        decision_entropy=decision_entropy,
        witness_capacity=witness_capacity,
        excess_hallucination=excess_hallucination,
        confidence_std=conf_std,
        confidence_min=conf_min,
        confidence_max=conf_max,
        hallucination_rate_ci_lower=ci_lower,
        hallucination_rate_ci_upper=ci_upper,
        p_value_exceeds_bound=p_value
    )

def run_k_sweep(
    model: str = MODEL_NAME,
    context_range: range = range(K_SWEEP_CONTEXT_MIN, K_SWEEP_CONTEXT_MAX),
    n_samples: int = K_SWEEP_SAMPLES,
    n_trials: int = K_SWEEP_TRIALS
) -> List[ExperimentResults]:
    """
    Run experiment across multiple contexts to measure K vs hallucination curve.
    
    Args:
        model: Model name
        context_range: Range of context counts to test
        n_samples: Samples per context
        n_trials: Trials without context
    
    Returns:
        List of ExperimentResults for each context count
    """
    np.random.seed(DEFAULT_SEED)
    
    results_list = []

    context_list = list(context_range)
    has_control = min(context_list) == 1
    
    if has_control:
        print_section_header(f"TESTING DIFFERENT LEVELS OF CONTRADICTION: {len(context_list)} tasks (K=0 control + {len(context_list)-1} contradiction tasks)")
    else:
        print_section_header(f"TESTING DIFFERENT LEVELS OF CONTRADICTION: {len(context_list)} tasks")

    results_list = []

    # Collect all results first
    if should_show(Verbosity.NORMAL):
        display.print_simple_text_dim(f"Running experiments across {len(context_list)} different context levels...")
        display.print_simple_text_dim("Each experiment measures how task contradiction affects hallucination rates.\n")

    # Create single progress bar for all operations
    pbar = tqdm(total=len(context_list), desc="Overall Progress", disable=not should_show(Verbosity.NORMAL), unit="task")
    display.register_progress_bar(pbar)

    for i, n_ctx in enumerate(context_list):
        task = generate_synthetic_task(n_ctx)
        
        # Update progress bar with current task info
        if pbar is not None:
            pbar.set_description(f"Overall Progress | Testing {n_ctx} context{'s' if n_ctx > 1 else ''}")
        
        # Show what we're testing
        if should_show(Verbosity.NORMAL):
            display.print_simple_text_dim(f"Testing {n_ctx} contexts... ({i+1}/{len(context_list)})")

            # Show all contexts
            for ctx_idx, (ctx_id, ctx_text) in enumerate(task['contexts'].items()):
                display.print_simple_text_dim(f"  Context {ctx_idx+1}: \"{ctx_text}\"")

            # Show the undefined query
            display.print_simple_text_dim(f"  Query: \"{task['undefined_query']}\"")

        results = run_experiment(
            model=model,
            task=task,
            n_samples=n_samples,
            n_trials=n_trials,
            verbose=False,
            pbar=pbar
        )

        results_list.append(results)
        if pbar is not None:
            pbar.update(1)
    
    if pbar is not None:
        display.unregister_progress_bar(pbar)
        pbar.close()

    # Display results in a readable format
    display.print_newline()
    display.print_simple_text("[bold]EXPERIMENT RESULTS:[/bold]")
    display.print_simple_text_dim("Each task tests a different number of conflicting contexts that make the question impossible to answer consistently.")
    display.print_newline()

    for i, r in enumerate(results_list, 1):
        display.print_experiment_result_panel(r, task_index=i)
    
    # Show detailed table only in debug mode
    if should_show(Verbosity.DEBUG):
        # Create color-coded table rows with status indicators in observed column
        table_rows = []
        for r in results_list:
            # Color-code K by intensity (higher K = more red)
            k_intensity = min(255, int(255 * (r.K / max(r.K for r in results_list) if results_list else 1)))
            k_color = f"#{k_intensity:02x}0000"

            # Status indicator for observed rate
            status_icon = "✓" if r.observed_hallucination_rate >= r.lower_bound else "✗"
            status_color = "green" if r.observed_hallucination_rate >= r.lower_bound else "red"
            observed_display = f"[{status_color}]{status_icon}[/{status_color}] {r.observed_hallucination_rate:.1%}"

            table_rows.append([
                str(r.n_contexts),
                f"[{k_color}]{r.K:.3f}[/{k_color}]",
                f"[yellow]{r.lower_bound:.1%}[/yellow]",
                observed_display
            ])

        print_table(
            ["Contexts", "K (bits)", "Bound", "Observed"],
            table_rows,
            "K Sweep Summary (✓ = exceeds bound)"
        )
    
    return results_list

def plot_comprehensive_results(
    sweep_results: List[ExperimentResults],
    abstention_result: Optional[ExperimentResults] = None,
    forced_result: Optional[ExperimentResults] = None,
    filename: str = "contradiction_hallucination_analysis.png"
) -> Path:
    """
    Create comprehensive multi-panel visualization of the hallucination theory.
    
    Four panels tell the complete story:
    - Panel A: K vs hallucination rate (main theoretical result)
    - Panel B: Context distributions showing the contradiction structure
    - Panel C: Abstention vs forced choice (architectural effect)
    - Panel D: Confidence distribution by response type
    
    Args:
        sweep_results: Results from K sweep experiment
        abstention_result: Optional result with abstention allowed
        forced_result: Optional result with forced choice
        filename: Filename to save in FIGURES_DIR
    
    Returns:
        Path to saved figure
    """
    np.random.seed(DEFAULT_SEED)
    
    plt.style.use('default')
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: K vs Hallucination (main result)
    ax1 = fig.add_subplot(gs[0, 0])
    
    Ks = np.array([r.K for r in sweep_results])
    observed_rates = np.array([r.observed_hallucination_rate for r in sweep_results])
    n_contexts = np.array([r.n_contexts for r in sweep_results])
    ci_lowers = np.array([r.hallucination_rate_ci_lower for r in sweep_results])
    ci_uppers = np.array([r.hallucination_rate_ci_upper for r in sweep_results])
    
    K_max = max(Ks) if len(Ks) > 0 else 1.0
    K_theory = np.linspace(0, K_max * 1.15, 300)
    theory_curve = 1 - 2**(-K_theory)
    
    ax1.fill_between(K_theory, 0, theory_curve, alpha=0.08, color='gray', linewidth=0)
    ax1.plot(K_theory, theory_curve, 'k--', linewidth=2, 
            label=r'$1 - 2^{-K}$ (bound)', zorder=2)
    
    # Add saturation guide line at 95%
    ax1.axhline(y=0.95, color='gray', linestyle=':', linewidth=1.5, alpha=0.5, zorder=1)
    ax1.text(K_max * 1.10, 0.96, 'Saturation', fontsize=8, color='gray', ha='right', va='bottom')
    
    unique_contexts = sorted(set(n_contexts))
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(unique_contexts), vmax=max(unique_contexts))
    
    # Separate K=0 control from K>0 tasks
    k0_mask = Ks < 0.01
    k_pos_mask = Ks >= 0.01
    
    # Plot K=0 control with different marker (square, red)
    if np.any(k0_mask):
        ax1.errorbar(Ks[k0_mask], observed_rates[k0_mask],
                    yerr=[observed_rates[k0_mask] - ci_lowers[k0_mask],
                          ci_uppers[k0_mask] - observed_rates[k0_mask]],
                    fmt='s', markersize=10, color='red', 
                    ecolor='red', elinewidth=2, capsize=4, capthick=2,
                    markeredgecolor='black', markeredgewidth=1.5,
                    alpha=0.85, zorder=4, label='K=0 Baseline')
        
        # Annotate K=0 if unexpected (should be near 0%)
        if observed_rates[k0_mask][0] > 0.1:
            ax1.annotate('⚠ Unexpected\n(should be ~0%)', 
                        xy=(Ks[k0_mask][0], observed_rates[k0_mask][0]),
                        xytext=(Ks[k0_mask][0] + K_max * 0.15, observed_rates[k0_mask][0] - 0.15),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                        fontsize=8, color='red', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))
    
    # Plot K>0 tasks with error bars
    for n_ctx in unique_contexts:
        if n_ctx == 1 and np.any(k0_mask):
            continue  # Skip K=0 (already plotted)
        
        mask = (n_contexts == n_ctx) & k_pos_mask
        if not np.any(mask):
            continue
        
        ax1.errorbar(Ks[mask], observed_rates[mask],
                    yerr=[observed_rates[mask] - ci_lowers[mask],
                          ci_uppers[mask] - observed_rates[mask]],
                    fmt='o', markersize=10, color=cmap(norm(n_ctx)),
                    ecolor=cmap(norm(n_ctx)), elinewidth=1.5, capsize=3, capthick=1.5,
                    markeredgecolor='black', markeredgewidth=1.2,
                    alpha=0.85, zorder=3, label=f'{n_ctx} contexts')
    
    # Check for ceiling effect and annotate
    if np.mean(observed_rates[k_pos_mask]) > 0.95 if np.any(k_pos_mask) else False:
        ax1.text(K_max * 0.5, 0.88, '⚠ CEILING EFFECT\nAll tasks ≥95%', 
                fontsize=9, color='red', weight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                         edgecolor='red', alpha=0.3))
    
    ax1.set_xlabel('K (contradiction, bits)', fontweight='bold')
    ax1.set_ylabel('Hallucination rate', fontweight='bold')
    ax1.set_title('A. More Contradiction → More Hallucination', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.02, K_max * 1.15)
    ax1.set_ylim(-0.02, 1.05)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
    ax1.legend(loc='lower right', framealpha=0.95, edgecolor='gray')
    
    # Panel B: Context distributions showing contradiction structure
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Use a representative result (middle of sweep)
    mid_result = sweep_results[len(sweep_results)//2]
    ctx_dists = mid_result.context_distributions
    
    if ctx_dists:
        contexts = list(ctx_dists.keys())
        n_ctx = len(contexts)
        x = np.arange(len(contexts))
        
        # Stack the answer distributions
        answers = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'other']
        colors_answers = plt.cm.Set3(np.linspace(0, 1, len(answers)))
        
        bottom = np.zeros(n_ctx)
        for i, ans in enumerate(answers):
            values = [ctx_dists[ctx].get(ans, 0.0) for ctx in contexts]
            if sum(values) > 0:
                ax2.bar(x, values, bottom=bottom, label=ans, 
                       color=colors_answers[i], edgecolor='black', linewidth=0.8)
                bottom += values
        
        # Add visual contradiction indicators between bars
        for i in range(n_ctx - 1):
            y_pos = 0.5
            ax2.annotate('', xy=(x[i+1] - 0.3, y_pos), xytext=(x[i] + 0.3, y_pos),
                        arrowprops=dict(arrowstyle='<->', color='red', lw=2, alpha=0.6))
            ax2.text((x[i] + x[i+1]) / 2, y_pos + 0.05, '⚠', 
                    fontsize=14, color='red', ha='center', weight='bold')
        
        # Add annotation explaining incompatibility
        ax2.text(0.5, 0.98, 'No single answer works for all contexts!',
                transform=ax2.transAxes, ha='center', va='top',
                fontsize=9, style='italic', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         edgecolor='red', alpha=0.2))
        
        ax2.set_xlabel('Context', fontweight='bold')
        ax2.set_ylabel('Probability', fontweight='bold')
        ax2.set_title(f'B. Each Context Gives Different Answer', 
                     fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Ctx {i+1}' for i in range(n_ctx)])
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
        ax2.legend(loc='upper right', framealpha=0.95, edgecolor='gray', ncol=2)
        ax2.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    
    # Panel C: Abstention vs Forced Choice (stacked bar chart)
    ax3 = fig.add_subplot(gs[1, 0])
    
    if abstention_result and forced_result:
        conditions = ['Abstention\nAllowed', 'Forced\nChoice']
        x_pos = np.arange(len(conditions))
        
        # Calculate abstention and fabrication rates
        abstention_rates = [
            abstention_result.partial_abstentions / abstention_result.n_trials,
            forced_result.partial_abstentions / forced_result.n_trials
        ]
        fabrication_rates = [
            abstention_result.observed_hallucination_rate,
            forced_result.observed_hallucination_rate
        ]
        
        # Stacked bar chart: abstentions (blue) + fabrications (red)
        bars_abs = ax3.bar(x_pos, abstention_rates, 
                          color='#3498db', edgecolor='black', linewidth=1.5, 
                          alpha=0.85, label='Abstentions')
        bars_fab = ax3.bar(x_pos, fabrication_rates, bottom=abstention_rates,
                          color='#e74c3c', edgecolor='black', linewidth=1.5,
                          alpha=0.85, label='Fabrications')
        
        # Add numerical labels on bars
        for i, (abs_rate, fab_rate) in enumerate(zip(abstention_rates, fabrication_rates)):
            if abs_rate > 0.05:
                ax3.text(x_pos[i], abs_rate / 2, f'{abs_rate:.0%}',
                        ha='center', va='center', fontsize=10, weight='bold', color='white')
            if fab_rate > 0.05:
                ax3.text(x_pos[i], abs_rate + fab_rate / 2, f'{fab_rate:.0%}',
                        ha='center', va='center', fontsize=10, weight='bold', color='white')
        
        # Add witness capacity annotations above bars
        K_val = forced_result.K
        r_abstain = max(0.0, K_val - abstention_result.observed_hallucination_rate)
        r_forced = max(0.0, K_val - forced_result.observed_hallucination_rate)
        
        ax3.text(x_pos[0], 1.02, f'r≈{r_abstain:.2f}',
                ha='center', va='bottom', fontsize=9, color='blue', weight='bold')
        ax3.text(x_pos[1], 1.02, f'r={r_forced:.2f}',
                ha='center', va='bottom', fontsize=9, color='red', weight='bold')
        
        # Add K value as text
        ax3.text(0.5, 0.88, f'K = {K_val:.2f} bits', 
                transform=ax3.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor='black', linewidth=1))
        
        ax3.set_ylabel('Response composition', fontweight='bold')
        ax3.set_title('C. Output Format Affects Behavior', fontweight='bold', pad=10)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(conditions)
        ax3.set_ylim(0, 1.15)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{int(y*100)}%'))
        ax3.legend(loc='upper left', framealpha=0.95, edgecolor='gray')
        ax3.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    
    # Panel D: Confidence distribution histogram
    ax4 = fig.add_subplot(gs[1, 1])
    
    if abstention_result and abstention_result.confidences:
        # Collect all confidences and classify by response type
        fabrication_confidences = []
        abstention_confidences = []
        
        for i, (resp, conf) in enumerate(zip(abstention_result.no_context_responses, abstention_result.confidences)):
            resp_lower = resp.lower().strip()
            if 'unknown' in resp_lower or 'cannot' in resp_lower or 'not sure' in resp_lower:
                abstention_confidences.append(conf)
            else:
                fabrication_confidences.append(conf)
        
        # Define bins
        bins = [0, 0.5, 0.7, 0.9, 1.0]
        bin_labels = ['0-50%', '50-70%', '70-90%', '90-100%']
        
        # Histogram for fabrications (red)
        fab_counts, _ = np.histogram(fabrication_confidences, bins=bins)
        # Histogram for abstentions (blue)
        abs_counts, _ = np.histogram(abstention_confidences, bins=bins)
        
        x_pos = np.arange(len(bin_labels))
        width = 0.35
        
        bars_fab = ax4.bar(x_pos - width/2, fab_counts, width, 
                          color='#e74c3c', edgecolor='black', linewidth=1.2,
                          alpha=0.85, label='Fabrications')
        bars_abs = ax4.bar(x_pos + width/2, abs_counts, width,
                          color='#3498db', edgecolor='black', linewidth=1.2,
                          alpha=0.85, label='Abstentions')
        
        # Add count labels on bars
        for bars in [bars_fab, bars_abs]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2, height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=9, weight='bold')
        
        # Calculate and display mean confidences
        mean_fab = np.mean(fabrication_confidences) if fabrication_confidences else 0
        mean_abs = np.mean(abstention_confidences) if abstention_confidences else 0
        
        ax4.text(0.5, 0.98, 
                f'Mean: Fab={mean_fab:.0%}, Abs={mean_abs:.0%}',
                transform=ax4.transAxes, ha='center', va='top',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='white', edgecolor='black'))
        
        # Add interpretation
        if mean_fab > mean_abs + 0.1:
            interpretation = '⚠ MORE confident when wrong!'
            color = 'red'
        else:
            interpretation = '✓ Calibrated confidence'
            color = 'green'
        
        ax4.text(0.5, 0.88, interpretation,
                transform=ax4.transAxes, ha='center', va='top',
                fontsize=9, weight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                         edgecolor=color, alpha=0.2))
        
        ax4.set_xlabel('Confidence bin', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('D. Confidence by Response Type', fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(bin_labels, rotation=0)
        ax4.legend(loc='upper right', framealpha=0.95, edgecolor='gray')
        ax4.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
    
    plt.suptitle('How Task Contradiction Forces Model Hallucination', 
                fontsize=14, fontweight='bold', y=0.93)
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nComprehensive figure saved to: {save_path}")
    
    plt.close()
    
    return save_path

def plot_combined_alternative_views(
    sweep_results: List[ExperimentResults],
    abstention_result: Optional[ExperimentResults] = None,
    filename: str = "combined_alternative_views.png"
) -> Path:
    """
    Create a combined visualization with three alternative views:
    1. Excess hallucination beyond theoretical bound
    2. K vs hallucination on logit scale
    3. Context contradiction heatmap
    
    Args:
        sweep_results: Results from K sweep experiment
        abstention_result: Optional result with abstention allowed (for heatmap)
        filename: Filename to save in FIGURES_DIR
    
    Returns:
        Path to saved figure
    """
    np.random.seed(DEFAULT_SEED)
    
    plt.style.use('default')
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
    
    # ========================================================================
    # Panel 1: Excess Hallucination
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    Ks = np.array([r.K for r in sweep_results])
    observed_rates = np.array([r.observed_hallucination_rate for r in sweep_results])
    bounds = np.array([r.lower_bound for r in sweep_results])
    n_contexts = np.array([r.n_contexts for r in sweep_results])
    ci_lowers = np.array([r.hallucination_rate_ci_lower for r in sweep_results])
    ci_uppers = np.array([r.hallucination_rate_ci_upper for r in sweep_results])
    
    excess = observed_rates - bounds
    excess_ci_lower = ci_lowers - bounds
    excess_ci_upper = ci_uppers - bounds
    
    # Zero line (theoretical minimum)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, 
              label='Theoretical bound', zorder=2)
    
    # Separate K=0 control from K>0 tasks
    k0_mask = Ks < 0.01
    k_pos_mask = Ks >= 0.01
    
    unique_contexts = sorted(set(n_contexts))
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=min(unique_contexts), vmax=max(unique_contexts))
    
    # Plot K=0 control with different marker (square, red)
    if np.any(k0_mask):
        ax1.errorbar(Ks[k0_mask], excess[k0_mask],
                   yerr=[excess[k0_mask] - excess_ci_lower[k0_mask],
                         excess_ci_upper[k0_mask] - excess[k0_mask]],
                   fmt='s', markersize=8, color='red',
                   ecolor='red', elinewidth=1.5, capsize=3, capthick=1.5,
                   markeredgecolor='black', markeredgewidth=1.2,
                   alpha=0.85, zorder=4, label='K=0 Baseline')
    
    # Plot K>0 tasks with error bars
    for n_ctx in unique_contexts:
        if n_ctx == 1 and np.any(k0_mask):
            continue
        
        mask = (n_contexts == n_ctx) & k_pos_mask
        if not np.any(mask):
            continue
        
        ax1.errorbar(Ks[mask], excess[mask],
                   yerr=[excess[mask] - excess_ci_lower[mask],
                         excess_ci_upper[mask] - excess[mask]],
                   fmt='o', markersize=8, color=cmap(norm(n_ctx)),
                   ecolor=cmap(norm(n_ctx)), elinewidth=1.2, capsize=2.5, capthick=1.2,
                   markeredgecolor='black', markeredgewidth=1.0,
                   alpha=0.85, zorder=3, label=f'{n_ctx} contexts')
    
    # Add decision entropy reference
    H = np.log2(7)
    ax1.axhline(y=H, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
              label=f'H = {H:.2f} bits')
    
    ax1.text(max(Ks) * 0.95, H + 0.02, f'H = {H:.2f}',
           ha='right', va='bottom', fontsize=9, color='orange', weight='bold')
    
    ax1.set_xlabel('K (contradiction, bits)', fontweight='bold')
    ax1.set_ylabel('Excess hallucination', fontweight='bold')
    ax1.set_title('A. Excess Beyond Theoretical Bound', fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.legend(loc='upper left', framealpha=0.95, edgecolor='gray', fontsize=8)
    
    # Add annotation explaining excess - centered in the plot area (after limits are set)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    x_center = (xlim[0] + xlim[1]) / 2
    y_center = (ylim[0] + ylim[1]) / 2
    ax1.text(x_center, y_center, 
           'Excess = Observed - Bound\nShows H + architecture',
           ha='center', va='center',
           fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
                                 facecolor='white', edgecolor='black', alpha=0.9),
           zorder=10)
    
    # ========================================================================
    # Panel 2: Logit Scale
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Convert to logit scale, handling edge cases
    def safe_logit(p):
        p_clipped = np.clip(p, 0.001, 0.999)
        return np.log(p_clipped / (1 - p_clipped))
    
    observed_logit = safe_logit(observed_rates)
    
    # Theoretical curve on logit scale
    K_max = max(Ks) if len(Ks) > 0 else 1.0
    K_theory = np.linspace(0.01, K_max * 1.15, 300)
    theory_curve = 1 - 2**(-K_theory)
    theory_logit = safe_logit(theory_curve)
    
    ax2.plot(K_theory, theory_logit, 'k--', linewidth=2,
           label=r'$\mathrm{logit}(1 - 2^{-K})$', zorder=2)
    
    # Plot K=0 control with different marker (square, red)
    if np.any(k0_mask):
        ax2.scatter(Ks[k0_mask], observed_logit[k0_mask],
                  s=100, marker='s', color='red',
                  edgecolors='black', linewidth=1.2,
                  alpha=0.85, zorder=4, label='K=0 Baseline')
    
    # Plot K>0 tasks
    for n_ctx in unique_contexts:
        if n_ctx == 1 and np.any(k0_mask):
            continue
        
        mask = (n_contexts == n_ctx) & k_pos_mask
        if not np.any(mask):
            continue
        
        ax2.scatter(Ks[mask], observed_logit[mask],
                  s=100, c=[cmap(norm(n_ctx))], marker='o',
                  edgecolors='black', linewidth=1.0,
                  alpha=0.85, zorder=3, label=f'{n_ctx} contexts')
    
    # Add reference lines for common probabilities
    ref_probs = [0.5, 0.9, 0.95, 0.99]
    for p in ref_probs:
        logit_p = safe_logit(p)
        ax2.axhline(y=logit_p, color='gray', linestyle=':', linewidth=1, alpha=0.4)
        ax2.text(K_max * 1.12, logit_p, f'{p:.0%}',
               ha='left', va='center', fontsize=8, color='gray')
    
    ax2.set_xlabel('K (contradiction, bits)', fontweight='bold')
    ax2.set_ylabel('logit(Hallucination rate)', fontweight='bold')
    ax2.set_title('B. Logit Scale View', fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax2.legend(loc='lower right', framealpha=0.95, edgecolor='gray', fontsize=8)
    
    # Add annotation explaining logit scale
    ax2.text(0.5, 0.98,
           'Makes saturation visible\nLinear near 100%',
           transform=ax2.transAxes, ha='center', va='top',
           fontsize=9, bbox=dict(boxstyle='round,pad=0.4',
                                 facecolor='white', edgecolor='black'))
    
    # ========================================================================
    # Panel 3: Context Heatmap
    # ========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    if abstention_result and abstention_result.context_distributions:
        ctx_dists = abstention_result.context_distributions
        contexts = list(ctx_dists.keys())
        answers = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'other']
        
        # Build probability matrix
        prob_matrix = np.zeros((len(answers), len(contexts)))
        for j, ctx in enumerate(contexts):
            for i, ans in enumerate(answers):
                prob_matrix[i, j] = ctx_dists[ctx].get(ans, 0.0)
        
        # Create heatmap
        im = ax3.imshow(prob_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(contexts)))
        ax3.set_yticks(np.arange(len(answers)))
        ax3.set_xticklabels([f'Ctx {i+1}' for i in range(len(contexts))])
        ax3.set_yticklabels(answers)
        
        # Rotate x labels for readability
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(answers)):
            for j in range(len(contexts)):
                text = ax3.text(j, i, f'{prob_matrix[i, j]:.0%}',
                              ha="center", va="center", 
                              color="black" if prob_matrix[i, j] < 0.5 else "white",
                              fontsize=9, weight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Probability', rotation=270, labelpad=15, fontweight='bold', fontsize=9)
        
        # Add title and labels
        ax3.set_title(f'C. Context Contradiction (K = {abstention_result.K:.2f} bits)', 
                    fontweight='bold', pad=10)
        ax3.set_xlabel('Context', fontweight='bold')
        ax3.set_ylabel('Answer', fontweight='bold')
        
        # Add annotation explaining contradiction - with 20px margin top
        ax3.text(0.5, -0.20, 
               '⚠ Each column shows different answer → No single answer works!',
               transform=ax3.transAxes, ha='center', va='top',
               fontsize=9, style='italic', color='red',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', 
                        edgecolor='red', alpha=0.3))
    else:
        ax3.text(0.5, 0.5, 'No context data available',
                transform=ax3.transAxes, ha='center', va='center',
                fontsize=12, style='italic')
        ax3.set_title('C. Context Contradiction', fontweight='bold', pad=10)
    
    plt.suptitle('Alternative Views of Hallucination Analysis', 
                fontsize=14, fontweight='bold', y=0.88)
    
    save_path = FIGURES_DIR / filename
    plt.savefig(save_path, dpi=300, facecolor='white')
    print(f"\nCombined alternative views figure saved to: {save_path}")
    
    plt.close()
    
    return save_path

def export_results(results: List[ExperimentResults], output_path: str = "results.json") -> None:
    """
    Export experiment results to JSON for further analysis.
    
    Args:
        results: List of experiment results
        output_path: Path to save JSON file
    """
    data = [r.to_dict() for r in results]
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults exported to: {output_path}")
    
    # Also create CSV summary
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w') as f:
        f.write("n_contexts,task_contradiction_bits,optimum_strategy,minimum_hallucination,observed_hallucination,avg_confidence,time_related_answers,unrelated_answers,admitted_uncertainty\n")
        for r in results:
            f.write(f"{r.n_contexts},{r.K:.4f},{r.alpha_star:.4f},{r.lower_bound:.4f},"
                   f"{r.observed_hallucination_rate:.4f},{r.average_confidence:.4f},"
                   f"{r.temporal_references},{r.unrelated_answers},{r.partial_abstentions}\n")
    
    print(f"CSV summary saved to: {csv_path}")

def run_threshold_sweep(
    task: Task,
    model: str = MODEL_NAME,
    thresholds: List[float] = THRESHOLD_SWEEP_VALUES,
    n_samples: int = THRESHOLD_SWEEP_SAMPLES,
    n_trials: int = THRESHOLD_SWEEP_TRIALS
) -> Dict[float, float]:
    """
    Run experiment across multiple confidence thresholds.
    
    Tests sensitivity of hallucination rate to confidence threshold.
    Helps distinguish architectural (forced commitment) vs sampling stochasticity.
    
    Args:
        task: Task dictionary
        model: Model name
        thresholds: List of confidence thresholds to test
        n_samples: Samples per context
        n_trials: Trials without context
    
    Returns:
        Dict mapping threshold to hallucination rate
    """
    print_section_header(f"Testing Different Confidence Thresholds: {len(thresholds)} values")
    
    # First, construct behavior (only once)
    behavior, _, _ = construct_behavior_from_llm(task, model, n_samples, verbose=False)
    K = behavior.K
    lower_bound = 1 - 2**(-K) if K > 0 else 0.0
    
    format_task_contradiction_info(K, lower_bound)
    
    threshold_results = {}
    
    for threshold in tqdm(thresholds, desc="  Testing thresholds"):
        (halluc_rate, avg_conf, _, _, _, _, _, _, _, _, _, _) = measure_hallucination_rate(
            task, model, n_trials, threshold, allow_abstention=True, verbose=False
        )
        threshold_results[threshold] = halluc_rate
    
    display.print_table(
        ["Confidence Threshold", "Hallucination Rate", "Exceeds Theory?"],
        [[f"{thresh:.0%}", f"{rate:.1%}", "✓ Yes" if rate >= lower_bound else "✗ No"]
         for thresh, rate in sorted(threshold_results.items())],
        "How Hallucination Rate Changes with Confidence Threshold"
    )
    
    return threshold_results

if __name__ == "__main__":
    print("✓ Text utils import successful!")
    if should_show(Verbosity.NORMAL):
        print(__doc__)

    # ========================================================================
    # MAIN EXPERIMENT: K sweep + architectural comparison
    # ========================================================================

    print_section_header(f"LLM Hallucination Experiment ({MODEL_NAME})", verbosity=Verbosity.QUIET)
    
    # Run K sweep
    sweep_results = run_k_sweep(
        MODEL_NAME, 
        context_range=range(K_SWEEP_CONTEXT_MIN, K_SWEEP_CONTEXT_MAX), 
        n_samples=K_SWEEP_SAMPLES, 
        n_trials=K_SWEEP_TRIALS
    )
    
    # Run abstention comparison (reuse one task from sweep to avoid redundant computation)
    from dataclasses import replace
    mid_result = sweep_results[len(sweep_results)//2]
    k_pos_task = generate_synthetic_task(n_contexts=mid_result.n_contexts)
    
    # Build behavior once
    if should_show(Verbosity.NORMAL):
        display.print_simple_text_dim("\nPreparing architectural comparison experiment...")
    
    # Create progress bar for architectural comparison
    pbar_arch = tqdm(total=3, desc="Overall Progress | Preparing comparison", disable=not should_show(Verbosity.NORMAL), unit="step")
    display.register_progress_bar(pbar_arch)
    if should_show(Verbosity.NORMAL):
        display.print_simple_text_dim("Measuring model responses in different contexts...")
    behavior, _, ctx_dists = construct_behavior_from_llm(
        k_pos_task, MODEL_NAME, n_samples=ABSTENTION_COMPARISON_SAMPLES, verbose=False, pbar=pbar_arch
    )
    pbar_arch.update(1)
    
    print_section_header(f"TESTING OUTPUT FORMAT EFFECTS (Contradiction level: {behavior.K:.2f} bits)")
    if should_show(Verbosity.NORMAL):
        display.print_simple_text_dim("Does requiring the model to pick an answer (instead of allowing 'unknown') increase hallucination rates?\n")
    
    # Progress wrapper for the two measurement runs
    steps = [
        ("Abstention allowed", True),
        ("Forced choice", False)
    ]
    
    results_dict = {}
    for step_name, allow_abstention in steps:
        pbar_arch.set_description(f"Overall Progress | Testing {step_name.lower()}")
        if should_show(Verbosity.NORMAL):
            display.print_simple_text_dim(f"Testing with {step_name.lower()}...")
        (halluc, avg_conf, _, raw, n_fab, n_abs, n_temp, n_unrel,
         conf_std, conf_min, conf_max, confidences, _, _) = measure_hallucination_rate(
            k_pos_task, MODEL_NAME, n_trials=ABSTENTION_COMPARISON_TRIALS,
            allow_abstention=allow_abstention, verbose=False, pbar=pbar_arch
        )
        results_dict[step_name] = (halluc, avg_conf, raw, n_fab, n_abs, n_temp, n_unrel, conf_std, conf_min, conf_max, confidences)
        pbar_arch.update(1)
    
    display.unregister_progress_bar(pbar_arch)
    pbar_arch.close()
    
    # Unpack results
    halluc_abstain, avg_conf_abstain, raw_abstain, n_fab_abstain, n_abs_abstain, n_temp_abstain, n_unrel_abstain, conf_std_abs, conf_min_abs, conf_max_abs, confidences_abstain = results_dict["Abstention allowed"]
    halluc_forced, avg_conf_forced, raw_forced, n_fab_forced, n_abs_forced, n_temp_forced, n_unrel_forced, conf_std_frc, conf_min_frc, conf_max_frc, confidences_forced = results_dict["Forced choice"]
    
    # Display results in readable format
    print_architectural_effect_header()

    result_text = format_architectural_comparison_text(halluc_abstain, halluc_forced)
    display.print_comparison_panel(result_text, "Output Format Comparison", "blue")
    display.print_newline()
    
    # Create result objects for visualization
    lower_bound_abstain = 1 - 2**(-behavior.K)
    decision_entropy_abstain = calculate_decision_entropy(7)
    witness_capacity_abstain = calculate_witness_capacity(behavior.K, halluc_abstain)
    
    results_abstain = ExperimentResults(
        model=MODEL_NAME, K=behavior.K, alpha_star=behavior.alpha_star,
        is_frame_independent=behavior.is_frame_independent(),
        lower_bound=lower_bound_abstain, observed_hallucination_rate=halluc_abstain,
        average_confidence=avg_conf_abstain, difference=halluc_abstain - lower_bound_abstain,
        n_samples=ABSTENTION_COMPARISON_SAMPLES, n_trials=ABSTENTION_COMPARISON_TRIALS, 
        n_contexts=mid_result.n_contexts, context_distributions=ctx_dists,
        no_context_responses=raw_abstain, confidences=confidences_abstain,
        confident_fabrications=n_fab_abstain,
        partial_abstentions=n_abs_abstain, temporal_references=n_temp_abstain,
        unrelated_answers=n_unrel_abstain, temperature=TEMPERATURE_SAMPLING,
        decision_entropy=decision_entropy_abstain,
        witness_capacity=witness_capacity_abstain,
        excess_hallucination=halluc_abstain - lower_bound_abstain
    )
    
    # Recalculate metrics for forced choice
    witness_capacity_forced = calculate_witness_capacity(behavior.K, halluc_forced)
    excess_hallucination_forced = halluc_forced - lower_bound_abstain
    
    results_forced = replace(results_abstain, 
        observed_hallucination_rate=halluc_forced,
        average_confidence=avg_conf_forced,
        no_context_responses=raw_forced,
        confidences=confidences_forced,
        confident_fabrications=n_fab_forced,
        partial_abstentions=n_abs_forced,
        temporal_references=n_temp_forced,
        unrelated_answers=n_unrel_forced,
        witness_capacity=witness_capacity_forced,
        excess_hallucination=excess_hallucination_forced,
        difference=excess_hallucination_forced
    )
    
    # Generate comprehensive visualization
    if should_show(Verbosity.DEBUG):
        print_subsection_header("Generating visualizations")
    plot_comprehensive_results(
        sweep_results, 
        abstention_result=results_abstain,
        forced_result=results_forced,
        filename="contradiction_hallucination_analysis.png"
    )
    
    # Generate combined alternative visualization
    plot_combined_alternative_views(
        sweep_results,
        abstention_result=results_abstain,
        filename="combined_alternative_views.png"
    )
    
    # Export data
    export_results(sweep_results, "hallucination_results.json")
    
    # Executive Summary
    print_section_header("✓ EXPERIMENT COMPLETE", verbosity=Verbosity.QUIET)
    display.print_simple_text("Generating final experiment summary...")
    display.print_experiment_summary(sweep_results)
    
    display.print_simple_text(f"📊 Main visualization: {FIGURES_DIR / 'contradiction_hallucination_analysis.png'}")
    display.print_simple_text(f"📊 Combined alternative views: {FIGURES_DIR / 'combined_alternative_views.png'}")
    display.print_simple_text(f"💾 Raw data saved to: hallucination_results.json")
    
    # ========================================================================
    # OPTIONAL: Additional sensitivity analyses
    # ========================================================================
    # Uncomment below to run threshold and temperature sensitivity tests
    
    # threshold_results = run_threshold_sweep(
    #     TASK, MODEL_NAME, 
    #     thresholds=THRESHOLD_SWEEP_VALUES,
    #     n_samples=THRESHOLD_SWEEP_SAMPLES, 
    #     n_trials=THRESHOLD_SWEEP_TRIALS
    # )
    
    # for temp in TEMPERATURE_SWEEP_VALUES:
    #     print(f"\nTemperature = {temp}")
    #     result = run_experiment(
    #         MODEL_NAME, temperature=temp, 
    #         n_samples=TEMPERATURE_SWEEP_SAMPLES, 
    #         n_trials=TEMPERATURE_SWEEP_TRIALS, 
    #         verbose=False
    #     )
    #     print(f"  K = {result.K:.4f}, Hallucination = {result.observed_hallucination_rate:.1%}")
