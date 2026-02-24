"""
greedy.py — Greedy Best-First Search for Word Ladder in Embedding Space.

Strategy : Min-heap ordered purely by heuristic h(n).
Cost     : Not tracked.
Heuristic: h(n) = cosine_distance(embed(n), embed(goal))
             = 1 - cosine_similarity(embed(n), embed(goal))
Optimal  : No — ignores past cost; may find a suboptimal path.
Complete : Yes with revisit-tracking (finite vocabulary).

At each step, greedily jump to whichever unexplored neighbor is closest
(in cosine space) to the goal. Fast in practice but can be misled by
local geometry —> a word very close to the goal may sit in a
semantic "dead-end" cluster with no path outward.
"""
import time
import heapq
from Embeddings import SearchResult, cosine_distance

def greedy(embeddings, start, goal, k=20, **kwargs):
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
     # Priority queue: (h(n), tie_break_counter, word, path)
    counter = 0
    frontier = [(h(start), counter, start, [start])]
    visited = set()
    while frontier:
        _, _, current, path = heapq.heappop(frontier)
        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1
        
        if current == goal:
            return SearchResult(path, nodes_expanded, time.time() - t0, True)
        
        for neighbor, _ in embeddings.get_neighbors(current, k):
            if neighbor not in visited:
                counter += 1
                heapq.heappush(frontier, (h(neighbor), counter, neighbor, path + [neighbor]))
    return SearchResult([], nodes_expanded, time.time() - t0, False)