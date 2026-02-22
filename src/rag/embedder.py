"""
WTTS Tuple Embedder — Embeds clinical event tuples and CRF queries
into vector space using SentenceTransformers.

Swap the model_name to a clinical model (e.g., MedCPT) when GPU is available.
"""

import re
import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer


class WTTSEmbedder:
    """Embeds WTTS tuples and CRF item queries into dense vectors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Args:
            model_name: SentenceTransformer model ID.
                        CPU default: 'all-MiniLM-L6-v2' (384-dim, fast)
                        GPU clinical: 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
                                      or 'medicalai/ClinicalBERT' 
            device: 'cpu' or 'cuda'
        """
        print(f"Loading embedding model: {model_name} on {device}...")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.embedding_dim}")

    # ------------------------------------------------------------------
    #  Parse WTTS string → structured list of tuple dicts
    # ------------------------------------------------------------------
    def parse_wtts_string(self, wtts_string: str) -> List[Dict]:
        """
        Parses the WTTS pipe-delimited string back into structured dicts.

        Input format:  [S_0] ("2026-01-01", "event text", 0.5, 1.0) | [S_1] (...)
        Output: [
            {"sid": "S_0", "timestamp": "2026-01-01", "event": "event text", "p_j": 0.5, "weight": 1.0},
            ...
        ]
        """
        tuples = []
        # Split by pipe separator
        raw_entries = wtts_string.split(" | ")

        # Pattern to extract: [S_xx] ("timestamp", "event", P_j, W)
        pattern = re.compile(
            r'\[(?P<sid>S_\d+)\]\s*\('
            r'"(?P<timestamp>[^"]*)",\s*'
            r'"(?P<event>[^"]*)",\s*'
            r'(?P<p_j>[\d.]+),\s*'
            r'(?P<weight>[\d.]+)\)'
        )

        for entry in raw_entries:
            entry = entry.strip()
            if not entry:
                continue
            match = pattern.search(entry)
            if match:
                tuples.append({
                    "sid": match.group("sid"),
                    "timestamp": match.group("timestamp"),
                    "event": match.group("event"),
                    "p_j": float(match.group("p_j")),
                    "weight": float(match.group("weight")),
                })

        return tuples

    # ------------------------------------------------------------------
    #  Embed tuple event texts
    # ------------------------------------------------------------------
    def embed_tuples(self, tuples: List[Dict]) -> np.ndarray:
        """
        Embed the event text from each WTTS tuple.

        Args:
            tuples: List of parsed tuple dicts (from parse_wtts_string)

        Returns:
            np.ndarray of shape (n_tuples, embedding_dim)
        """
        if not tuples:
            return np.array([])

        texts = [t["event"] for t in tuples]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
            batch_size=64,
        )
        return np.array(embeddings, dtype=np.float32)

    # ------------------------------------------------------------------
    #  Embed a CRF item query
    # ------------------------------------------------------------------
    def embed_query(self, crf_item: str, valid_options: Optional[List[str]] = None) -> np.ndarray:
        """
        Create an embedding for a CRF item query.
        Combines the item name with its valid options to create a richer query.

        Args:
            crf_item: e.g., "mrc_grade" or "administration of bronchodilators"
            valid_options: e.g., ["y", "n", "unknown"]

        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        # Build a descriptive query string
        query_parts = [crf_item.replace("_", " ")]

        if valid_options:
            # Add option context to help embedding understand what we're looking for
            opts_str = ", ".join(str(o) for o in valid_options[:10])  # limit to avoid huge queries
            query_parts.append(f"options: {opts_str}")

        query_text = " | ".join(query_parts)

        embedding = self.model.encode(
            [query_text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embedding[0], dtype=np.float32)

    # ------------------------------------------------------------------
    #  Batch embed multiple CRF queries at once
    # ------------------------------------------------------------------
    def embed_queries_batch(
        self, 
        crf_items: List[str], 
        valid_options_map: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Embed all CRF items in one batch for efficiency.

        Returns:
            Dict mapping crf_item name → embedding vector
        """
        query_texts = []
        item_names = []

        for item in crf_items:
            item_names.append(item)
            parts = [item.replace("_", " ")]
            opts = valid_options_map.get(item, [])
            if opts:
                opts_str = ", ".join(str(o) for o in opts[:10])
                parts.append(f"options: {opts_str}")
            query_texts.append(" | ".join(parts))

        embeddings = self.model.encode(
            query_texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            batch_size=64,
        )

        return {
            name: np.array(emb, dtype=np.float32)
            for name, emb in zip(item_names, embeddings)
        }
