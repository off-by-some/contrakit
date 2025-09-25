"""Pytest configuration for mathematical theory of contradiction tests."""

import pytest
import numpy as np


@pytest.fixture
def sample_space():
    """Create a sample space for testing."""
    return Space.make(
        ["A", "B", "C"],
        {"A": [0, 1], "B": ["x", "y"], "C": ["p", "q", "r"]}
    )


@pytest.fixture
def feasible_behavior(sample_space):
    """Create a feasible behavior for testing."""
    return Behavior.from_raw(sample_space, {
        ("A",): {(0,): 0.5, (1,): 0.5},
        ("B",): {("x",): 0.4, ("y",): 0.6},
        ("C",): {("p",): 0.3, ("q",): 0.4, ("r",): 0.3},
        ("A", "B"): {(0, "x"): 0.2, (0, "y"): 0.3, (1, "x"): 0.2, (1, "y"): 0.3},
        ("B", "C"): {("x", "p"): 0.12, ("x", "q"): 0.16, ("x", "r"): 0.12,
                     ("y", "p"): 0.18, ("y", "q"): 0.24, ("y", "r"): 0.18},
        ("A", "C"): {(0, "p"): 0.15, (0, "q"): 0.2, (0, "r"): 0.15,
                     (1, "p"): 0.15, (1, "q"): 0.2, (1, "r"): 0.15},
    })


@pytest.fixture
def infeasible_behavior(sample_space):
    """Create an infeasible behavior for testing."""
    return Behavior.from_raw(sample_space, {
        ("A", "B"): {(0, "x"): 0.6, (1, "y"): 0.4},
        ("B", "C"): {("x", "p"): 0.6, ("y", "q"): 0.4},
        ("C", "A"): {("p", "0"): 0.6, ("q", "1"): 0.4},
    })


# Import the core classes here for easy access in tests
from contrakit import Space, Context, Behavior, FrameIndependence
from contrakit.observatory import Observatory


# Observatory API Fixtures

@pytest.fixture
def basic_observatory():
    """Create a basic observatory."""
    return Observatory.create()


@pytest.fixture
def reviewer_observatory(basic_observatory):
    """Create an observatory with reviewer concepts defined."""
    candidate = basic_observatory.concept("Candidate", symbols=["Hire", "No_Hire"])
    reviewer_a = basic_observatory.concept("Reviewer_A", symbols=["Hire", "No_Hire"])
    reviewer_b = basic_observatory.concept("Reviewer_B", symbols=["Hire", "No_Hire"])
    reviewer_c = basic_observatory.concept("Reviewer_C", symbols=["Hire", "No_Hire"])
    return basic_observatory, candidate, reviewer_a, reviewer_b, reviewer_c


@pytest.fixture
def inconsistent_reviewer_observatory():
    """Create an observatory with inconsistent reviewer distributions."""
    obs = Observatory.create()
    candidate = obs.concept("Candidate", symbols=["Hire", "No_Hire"])
    reviewer_a = obs.concept("Reviewer_A", symbols=["Hire", "No_Hire"])
    reviewer_b = obs.concept("Reviewer_B", symbols=["Hire", "No_Hire"])
    reviewer_c = obs.concept("Reviewer_C", symbols=["Hire", "No_Hire"])

    hire, no_hire = reviewer_a.alphabet
    perspectives = obs.perspectives

    # Set up inconsistent distributions
    perspectives[reviewer_a] = {hire: 0.8}  # High probability
    perspectives[reviewer_b] = {hire: 0.2}  # Low probability

    # Joint that contradicts marginals
    perspectives[reviewer_a, reviewer_b] = {
        hire & hire: 0.1,       # Low probability for both
        hire & no_hire: 0.1,
        no_hire & hire: 0.1,
        no_hire & no_hire: 0.7  # High probability for neither
    }

    return obs, candidate, reviewer_a, reviewer_b, reviewer_c


@pytest.fixture
def empty_observatory():
    """Create an observatory with no concepts (for edge case testing)."""
    return Observatory.create()


@pytest.fixture
def single_value_observatory():
    """Create an observatory with a concept that has only one possible value."""
    obs = Observatory.create()
    concept = obs.concept("Single", symbols=["Only"])
    return obs, concept
