"""Common algorithms"""

from typing import Iterable, TypeVar, Optional

T = TypeVar("T")


def dfs(graph: dict[T, Iterable[T]], start: T, end: T) -> list[T] | None:
    """Depth first search

    Args:
        graph (dict[T, Iterable[T]]): Graph to search. Maps nodes to a list or other iterable of neighbors
        start (T): Starting node
        end (T): Ending node

    Returns:
        (list[T] | None): Sequence of nodes from start to end. None if the the path does not exist
    """
    if start not in graph or end not in graph:
        return None
    visited = set()
    path = [start]

    def dfs_rec(node: T) -> bool:
        if node == end:
            return True
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                path.append(neighbor)
                if dfs_rec(neighbor):
                    return True
                path.pop()
        return False

    found_path = dfs_rec(start)
    if not found_path:
        return None
    return path
