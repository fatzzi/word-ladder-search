
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity


def cosine_similarity(v1, v2):
    """Cosine similarity between two 1-D vectors using sklearn."""
    return float(_sk_cosine_similarity(v1.reshape(1, -1), v2.reshape(1, -1))[0, 0])


def cosine_distance(v1, v2):
    """Cosine distance = 1 - cosine similarity. Range [0, 2]."""
    return 1.0 - cosine_similarity(v1, v2)


class SearchResult:
    """Holds the result of any search algorithm."""

    def __init__(self, path, nodes_expanded, runtime, found):
        self.path = path
        self.nodes_expanded = nodes_expanded
        self.runtime = runtime
        self.found = found
        self.path_length = len(path) - 1 if path else 0

    def __repr__(self):
        if self.found:
            return (
                f"SearchResult(found=True, steps={self.path_length}, "
                f"nodes={self.nodes_expanded}, time={self.runtime:.4f}s)"
            )
        return (
            f"SearchResult(found=False, nodes={self.nodes_expanded}, "
            f"time={self.runtime:.4f}s)"
        )


class WordEmbeddings:
    """Loads and queries GloVe word embeddings."""

    def __init__(self, filepath):
        self.embeddings = {}
        self._neighbor_cache = {}
        self._load(filepath)
        self._build_matrix() 
        

    def _load(self, filepath):
        print(f"Loading embeddings from '{filepath}' ...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vector = np.array(parts[1:], dtype=np.float32)
                    self.embeddings[word] = vector
                except ValueError:
                    continue
        print(f"Loaded {len(self.embeddings):,} words.")

    def _build_matrix(self):
        """Build the full embedding matrix once for vectorised similarity queries."""
        self._words  = list(self.embeddings.keys())
        self._matrix = np.stack([self.embeddings[w] for w in self._words])  # (V, D)

    def get_vector(self, word):
        return self.embeddings.get(word.lower())

    def contains(self, word):
        return word.lower() in self.embeddings

    def get_neighbors(self, word, k=20):
        word = word.lower()
        cache_key = (word, k)
        if cache_key in self._neighbor_cache:
            return self._neighbor_cache[cache_key]

        query_vec = self.get_vector(word)
        # NumPy matrix multiplication is MUCH faster than a Python for-loop
        # (V, D) dot (D,) -> (V,)
        similarities = np.dot(self._matrix, query_vec) / (
            np.linalg.norm(self._matrix, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get indices of top k+1 (to exclude the word itself)
        idx = np.argpartition(similarities, -k-1)[-k-1:]
        idx = idx[np.argsort(similarities[idx])][::-1]
        
        result = []
        for i in idx:
            w = self._words[i]
            if w != word:
                result.append((w, similarities[i]))
                if len(result) == k: break

        self._neighbor_cache[cache_key] = result
        return result