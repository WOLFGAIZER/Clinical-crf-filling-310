"""
WTTS Retriever — Builds per-patient FAISS index and retrieves
relevant tuples per CRF item with weight-based re-ranking
and P_j temporal sorting.
"""

import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from src.rag.embedder import WTTSEmbedder


class WTTSRetriever:
    """Per-patient FAISS index for retrieving relevant WTTS tuples."""

    def __init__(self, embedder: WTTSEmbedder):
        self.embedder = embedder
        self.index: Optional[faiss.IndexFlatIP] = None  # Inner product (cosine on normalized vecs)
        self.tuples: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    #  Build index for one patient's tuples
    # ------------------------------------------------------------------
    def build_index(self, tuples: List[Dict], embeddings: np.ndarray):
        """
        Build a FAISS index from pre-computed tuple embeddings.

        Args:
            tuples: Parsed WTTS tuple dicts
            embeddings: np.ndarray of shape (n_tuples, embedding_dim)
        """
        self.tuples = tuples
        self.embeddings = embeddings

        if len(tuples) == 0:
            self.index = None
            return

        dim = embeddings.shape[1]
        # Use Inner Product (IP) since embeddings are L2-normalized
        # This makes IP equivalent to cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    # ------------------------------------------------------------------
    #  Raw retrieval (top-k by cosine similarity)
    # ------------------------------------------------------------------
    def retrieve(self, query_embedding: np.ndarray, top_k: int = 15) -> List[Dict]:
        """
        Retrieve top-k most similar tuples to the query.

        Args:
            query_embedding: 1D vector of shape (embedding_dim,)
            top_k: Number of tuples to retrieve

        Returns:
            List of tuple dicts with added 'similarity_score' field
        """
        if self.index is None or len(self.tuples) == 0:
            return []

        # Clamp top_k to available tuples
        top_k = min(top_k, len(self.tuples))

        # FAISS expects 2D input
        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            result = self.tuples[idx].copy()
            result["similarity_score"] = float(score)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    #  Retrieve + re-rank by weight + sort by P_j
    # ------------------------------------------------------------------
    def retrieve_and_rerank(
        self,
        query_embedding: np.ndarray,
        top_k: int = 15,
        weight_boost: float = 0.3,
        fetch_multiplier: int = 3,
    ) -> List[Dict]:
        """
        Retrieve, re-rank using weight W, then sort by P_j for temporal order.

        Strategy:
            1. Over-fetch (top_k * fetch_multiplier) candidates from FAISS
            2. Re-score: final_score = similarity + weight_boost * W
            3. Take top_k by final_score
            4. Sort the final set by P_j (ascending) to preserve temporal order

        Args:
            query_embedding: 1D vector
            top_k: Final number of tuples to return
            weight_boost: How much to boost critical events (W=1.0 gets +0.3)
            fetch_multiplier: How many extra candidates to fetch for re-ranking

        Returns:
            List of tuple dicts sorted by P_j (temporal order),
            each with 'similarity_score', 'rerank_score' fields
        """
        if self.index is None or len(self.tuples) == 0:
            return []

        # Step 1: Over-fetch candidates
        fetch_k = min(top_k * fetch_multiplier, len(self.tuples))
        candidates = self.retrieve(query_embedding, top_k=fetch_k)

        # Step 2: Re-rank with weight boost
        for candidate in candidates:
            sim = candidate["similarity_score"]
            w = candidate.get("weight", 0.5)
            candidate["rerank_score"] = sim + (weight_boost * w)

        # Step 3: Take top_k by re-rank score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_candidates = candidates[:top_k]

        # Step 4: Sort by P_j (temporal order) — THIS is what preserves continuity
        top_candidates.sort(key=lambda x: x.get("p_j", 0.5))

        return top_candidates

    # ------------------------------------------------------------------
    #  Format retrieved tuples back into a readable string for the LLM
    # ------------------------------------------------------------------
    @staticmethod
    def format_retrieved_tuples(tuples: List[Dict]) -> str:
        """
        Format retrieved tuples into a clean string for the LLM prompt.
        Sorted by P_j (temporal order) by this point.

        Output format:
            [S_14] (EARLY | W:HIGH) "Patient reports increasing dyspnea"
            [S_80] (MID   | W:HIGH) "SpO2 dropped to 85%, intubated"
            [S_155](LATE  | W:MED)  "Stable on room air at discharge"
        """
        if not tuples:
            return "(No relevant clinical events found)"

        lines = []
        for t in tuples:
            p_j = t.get("p_j", 0.5)
            w = t.get("weight", 0.5)

            # Temporal phase label
            if p_j <= 0.15:
                phase = "ADMISSION"
            elif p_j <= 0.35:
                phase = "EARLY"
            elif p_j <= 0.65:
                phase = "MID"
            elif p_j <= 0.85:
                phase = "LATE"
            else:
                phase = "DISCHARGE"

            # Weight label
            if w >= 0.8:
                w_label = "CRITICAL"
            elif w >= 0.4:
                w_label = "MODERATE"
            else:
                w_label = "ROUTINE"

            sid = t.get("sid", "S_?")
            event = t.get("event", "")
            ts = t.get("timestamp", "")

            lines.append(f'[{sid}] ({phase} | {w_label}) [{ts}] "{event}"')

        return "\n".join(lines)
