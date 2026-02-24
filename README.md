#  Semantic Word Ladder Search

A high-performance AI pathfinding system that navigates a high-dimensional semantic state space (GloVe embeddings) to find "meaningful" bridges between words.

##  Overview
Unlike traditional word ladder games that rely on character-level edits, this project uses **Natural Language Processing (NLP)** to treat words as vectors in a 100-dimensional space. The goal is to find a path between a start word and a goal word by moving through the most semantically similar neighbors.

### Key Features:
* **Vectorized Search:** Leverages NumPy matrix operations for near-instant neighbor discovery.
* **Multiple Search Strategies:** Compare **BFS, DFS, UCS, Greedy Best-First, and A***.
* **Optimized Pathfinding:** Uses Cosine Distance as an admissible and consistent heuristic for A*.
* **Interactive UI:** Built with **Streamlit** for real-time search visualization.

---

##  Technical Deep Dive: Matrix Vectorization
A major challenge was identifying the $k$-nearest neighbors among 20,000 words. Using standard Python loops was too slow for real-time use.

### The Solution:
During initialization, all word vectors are stacked into a single NumPy matrix $M \in \mathbb{R}^{20000 \times 100}$. Neighborhood discovery is reformulated as a linear algebra problem:

1.  **Dot Product:** Compute $D = M \cdot \vec{v}_{query}$ in a single BLAS-optimized operation.
2.  **Normalization:** Divide by pre-computed norms to get Cosine Similarities.
3.  **Partitioning:** Use `np.argpartition` to find the top $k$ candidates in $O(N)$ time instead of an $O(N \log N)$ sort.

This optimization reduced average node expansion time from several hundred milliseconds to the **order of tens of microseconds**.



---

##  Performance Benchmarks (k=20)
| Pair | Algorithm | Steps | Expanded | Time (s) |
| :--- | :--- | :--- | :--- | :--- |
| **treatment → muursepp** | BFS | 8 | 16,856 | 24.57 |
| **treatment → muursepp** | A* | 8 | 12,486 | 14.83 |
| **shindo → ramadhin** | UCS | 6 | 8,226 | 38.05 |
| **shindo → ramadhin** | A* | 6 | 3,964 | 5.84 |

---

##  Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/fatzzi/word-ladder-search.git](https://github.com/yourusername/semantic-word-ladder.git)
   cd word-ladder-search
2. **Run the Streamlit App:**
```bash
python -m streamlit run app.py
```


<img width="1863" height="876" alt="image" src="https://github.com/user-attachments/assets/651c360b-fec6-43a2-a4ff-3b3897eb5ef5" />
