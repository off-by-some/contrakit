"""
Lens utilities for the Mathematical Theory of Contradiction.

This module provides utilities for creating and working with lens contexts,
which are used to distinguish different perspectives on the same underlying variables.
"""

from typing import Tuple, Any
from .space import Space


def lens_space(base: Space, **lens_singletons) -> Space:
    """
    Extend a space with singleton observables that serve only to distinguish contexts.

    This helper creates the standard "two different lenses on the same variables" pattern
    without users having to hand-roll singleton tags.

    Args:
        base: The base space containing the main observables
        **lens_singletons: Keyword arguments mapping lens names to singleton values

    Returns:
        Extended space with both base observables and lens tags

    Example:
        base = Space.create(Treatment=['A','B'], Outcome=[1,0])
        space = lens_space(base, S1Tag=[0], S2Tag=[0])
        # Creates space with Treatment, Outcome, S1Tag, S2Tag observables
    """
    spec = {}

    # Add all base observables, but allow lens singletons to override
    for name in base.names:
        if name in lens_singletons:
            # If this name is also a lens singleton, use the singleton value
            spec[name] = tuple(lens_singletons[name])
        else:
            # Otherwise use the original alphabet
            spec[name] = base.alphabets[name]

    # Add any additional lens singletons that weren't already in base space
    for name, value in lens_singletons.items():
        if name not in spec:
            spec[name] = tuple(value)

    return Space.create(**spec)


def as_lens_context(base_observables: Tuple[str, ...], lens: str) -> Tuple[str, ...]:
    """
    Return a context that 'looks at' base_observables but is distinguished by a lens tag.

    Args:
        base_observables: The main observables to include in the context
        lens: The name of the lens tag observable

    Returns:
        Context tuple including both base observables and the lens tag

    Example:
        ctx = as_lens_context(("Treatment", "Outcome"), "S1Tag")
        # Returns ("Treatment", "Outcome", "S1Tag")
    """
    return (*base_observables, lens)
