import time
from collections import deque
from Embeddings import SearchResult

def bfs(embeddings, start, goal, k=20, **kwargs):
    """
    Breadth-First Search from `start` to `goal`.

    Parameters
    embeddings : WordEmbeddings
    start      : str  —> source word
    goal       : str  —> target word
    k          : int  —> number of neighbors to expand at each step (5–50)

    Returns
    SearchResult
    """
    start, goal = start.lower(), goal.lower()
    t0 = time.time()
    nodes_expanded = 0
    # Quick check: if either word is missing, we can't find a path.
    if not embeddings.contains(start) or not embeddings.contains(goal):
        return SearchResult([], 0, time.time()-t0, False)
    # Quick check: if start and goal are the same, we're done.
    if start == goal:
        return SearchResult([start], 0, time.time()-t0, True)
    frontier = deque([(start, [start])])  # queue of paths to explore
    visited = {start}  # track visited words to avoid cycles and redundant expansions
    while frontier:
        current, path = frontier.popleft()
        nodes_expanded += 1
        for neighbor, _ in embeddings.get_neighbors(current, k):
            if neighbor in visited:
                continue
            new_path = path + [neighbor]
            if neighbor == goal:
                return SearchResult(new_path, nodes_expanded, time.time()-t0, True)
            
            visited.add(neighbor)
            frontier.append((neighbor, new_path))
    return SearchResult([], nodes_expanded, time.time()-t0, False)