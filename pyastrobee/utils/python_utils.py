"""Utility functions broadly related to python as a whole"""

from typing import Any


def print_red(message: Any):
    """Helper function for printing in red text

    Args:
        message (Any): The message to print out in red
    """
    print(f"\033[31m{message}\033[0m")


def print_green(message: Any):
    """Helper function for printing in green text

    Args:
        message (Any): The message to print out in green
    """
    print(f"\033[32m{message}\033[0m")


def flatten(l: list[list]) -> list:
    """Flatten a list of lists into a single list

    Args:
        l (list[list]): List of lists to flaten

    Returns:
        list: Flattened list
    """
    return [item for sublist in l for item in sublist]
