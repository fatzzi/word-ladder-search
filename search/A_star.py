import numpy as np
import heapq
import time
from Embeddings import SearchResult, cosine_distance

"""

Strategy : Min-heap ordered by f(n) = g(n) + h(n).
Cost     : g(n) = cumulative cosine distance from start to n.
Heuristic: h(n) = cosine_distance(embed(n), embed(goal))
             = 1 - cosine_similarity(embed(n), embed(goal))
Optimal  : Yes — provided the heuristic is admissible (never overestimates).
Complete : Yes (finite vocabulary, revisit-tracking).

"""
def a_star(embeddings, start, goal, k=20, **kwargs):
    start, goal = start.lower(), goal.lower()
    t0 = time.time()
    nodes_expanded = 0
    
    # Vocabulary check
    if not embeddings.contains(start) or not embeddings.contains(goal):
        return SearchResult([], 0, time.time() - t0, False)

    # Trivial case
    if start == goal:
        return SearchResult([start], 0, time.time() - t0, True)
    
    goal_vec = embeddings.get_vector(goal)
    
    def h(word):
         return cosine_distance(embeddings.get_vector(word), goal_vec)
     # Priority queue: (f(n), tie_break_counter, word, path, g(n))
    counter = 0
    frontier = [(h(start), counter, start, [start], 0.0)]
    visited = {}
    # g_costs[word] = best known cumulative cost to reach word
    g_costs = {start: 0.0}
    while frontier:
        f_n, _, current, path, g_n = heapq.heappop(frontier)
        
        if current in visited and visited[current] <= g_n:
            continue
        visited[current] = g_n
        nodes_expanded += 1
        
        if current == goal:
            return SearchResult(path, nodes_expanded, time.time() - t0, True)
        g_current = g_costs[current]
        vec_current = embeddings.get_vector(current)

        
        for neighbor,_  in embeddings.get_neighbors(current, k):
            if neighbor in visited and visited[neighbor] <= g_current:
                continue
            edge_cost = cosine_distance(vec_current, embeddings.get_vector(neighbor))
            g_new = g_current + edge_cost
            if g_new < g_costs.get(neighbor, float('inf')):
                g_costs[neighbor] = g_new
                f_new = g_new + h(neighbor)
                counter += 1
                heapq.heappush(frontier, (f_new, counter, neighbor, path + [neighbor], g_new))
        
    return SearchResult([], nodes_expanded, time.time() - t0, False)