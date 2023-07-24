"""Common algorithms"""

from typing import Iterable, TypeVar

T = TypeVar("T")


def dfs(graph: dict[T, Iterable[T]], start: T, end: T) -> list[T] | None:
    """Depth first search

    Args:
        graph (dict[Any, Iterable]): Graph to search. Maps nodes to a list or other iterable of neighbors
        start (Any): Starting node
        end (Any): Ending node

    Returns:
        list[Any] | None: Sequence of nodes from start to end. None if the the path does not exist
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
