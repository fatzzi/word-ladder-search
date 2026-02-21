
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
        self._matrix = None   # built lazily on first get_neighbors call
        self._words  = None
        self._load(filepath)

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
        """
        Return the k nearest neighbors of `word` by cosine similarity.
        Compares the query vector against each word vector individually.
        Returns list of (neighbor_word, similarity) sorted descending.
        """
        word = word.lower()
        if word not in self.embeddings:
            return []

        query_vec = self.embeddings[word]
        scores = [
            (cosine_similarity(query_vec, vec), w)
            for w, vec in self.embeddings.items()
            if w != word
        ]
        scores.sort(reverse=True)
        return [(w, sim) for sim, w in scores[:k]]