#!/usr/bin/env python3
"""
Common utilities for contrakit examples.

This module contains shared functionality used across different
contrakit examples and demonstrations.
"""

import shutil
import os


def print_header(text):
    """
    Print a header string that spans the full terminal width.

    This function gets the current terminal width and prints the provided
    text centered and padded to fill the entire width with '=' characters.

    Parameters
    ----------
    text : str
        The header text to display
    """
    # Get terminal width, default to 80 if can't determine
    try:
        width = shutil.get_terminal_size().columns
    except (OSError, AttributeError):
        # Fallback for environments where terminal size can't be determined
        width = int(os.environ.get('COLUMNS', 80))

    # Ensure minimum width
    width = max(width, 40)

    print("=" * width)
    print(text.center(width))
    print("=" * width)
