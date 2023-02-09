"""Utility functions broadly related to python as a whole

TODO determine if this should be separated into more specific files - e.g. debug.py
"""

from typing import Any


def print_red(message: Any):
    """Helper function for printing in red text

    Args:
        message (Any): The message to print out in red
    """
    print(f"\033[31m{message}\033[0m")
