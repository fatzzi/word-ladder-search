"""

Strategy : Min-heap ordered by cumulative path cost g(n).
Cost     : Cosine distance between adjacent word vectors.
             edge_cost(u, v) = 1 -> cosine_similarity(embed(u), embed(v))
Heuristic: None.
Optimal  : Yes — finds the minimum total cosine-distance path.
Complete : Yes (finite vocabulary, revisit-tracking).

Why cosine distance as cost?
  - Captures semantic dissimilarity: small cost = semantically similar step.
  - Encourages paths that stay semantically coherent.
  - Gives UCS meaningful differentiation vs. BFS (which treats all hops equally).
"""
import time
import heapq   
from Embeddings import SearchResult, cosine_distance

def ucs(embeddings, start, goal, k=20, **kwargs):
    '''
    Uniform Cost Search in embedding space.
    Args:
        embeddings: WordEmbeddings instance for vector lookups.
        start: Starting word (string).
        goal: Target word (string).
        k: Number of nearest neighbors to consider at each step.
        Returns:
            SearchResult with path, nodes expanded, runtime, and found status.'''
            
    start, goal = start.lower(), goal.lower()
    t0 = time.time()
    nodes_expanded = 0
    
    # Vocabulary check
    if not embeddings.contains(start) or not embeddings.contains(goal):
        return SearchResult([], 0, time.time() - t0, False)

    # Trivial case
    if start == goal:
        return SearchResult([start], 0, time.time() - t0, True)
    
    # Min-heap priority queue: (cumulative_cost, current_word, path).
    frontier = [(0.0, 0, start, [start])]
    counter = 0  # tie-breaker for equal costs
    
    best_cost = {start: 0.0}  # best known cost to reach each word
    
    while frontier:
        cost, _, current, path = heapq.heappop(frontier)
        # Skip stale entries
        if cost > best_cost.get(current, float("inf")):
            continue
        nodes_expanded += 1
        
        if current == goal:
            return SearchResult(path, nodes_expanded, time.time() - t0, True)
        
        vec_current = embeddings.get_vector(current)
        
        for neighbor, _ in embeddings.get_neighbors(current, k):
            vec_neighbor = embeddings.get_vector(neighbor)
            edge_cost = cosine_distance(vec_current, vec_neighbor)
            new_cost = cost + edge_cost
            
            if new_cost < best_cost.get(neighbor, float("inf")):
                best_cost[neighbor] = new_cost
                counter += 1
                heapq.heappush(frontier, (new_cost, counter, neighbor, path + [neighbor]))
                
    return SearchResult([], nodes_expanded, time.time() - t0, False)