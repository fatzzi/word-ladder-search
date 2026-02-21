import time
from Embeddings import SearchResult
def dfs(embeddings, start, goal, k=20, depth_limit=10, **kwargs):
    """
    Depth-First Search from `start` to `goal`.

    Parameters
    embeddings : WordEmbeddings
    start      : str  —> source word
    goal       : str  —> target word
    k          : int  —> number of neighbors to expand at each step (5–50)
    depth_limit : int —> maximum search depth to prevent infinite recursion

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

    # Stack holds (current_word, path_so_far, depth)
    frontier = [(start, [start], 0)]
    visited = set()

    while frontier:
        current, path, depth = frontier.pop()

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if current == goal:
            return SearchResult(path, nodes_expanded, time.time() - t0, True)

        if depth >= depth_limit:
            continue  # pruned by depth limit

        for neighbor, _ in embeddings.get_neighbors(current, k):
            if neighbor not in visited:
                frontier.append((neighbor, path + [neighbor], depth + 1))
                
    return SearchResult(path or [], nodes_expanded, time.time()-t0, False)