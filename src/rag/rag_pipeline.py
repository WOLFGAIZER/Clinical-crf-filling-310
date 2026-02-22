"""
RAG CRF Extractor — Orchestrates the full RAG-guided pipeline:
    1. Build WTTS tuples from patient data
    2. Embed all tuples into FAISS index
    3. For each CRF item: retrieve relevant tuples → LLM extraction
    4. Return predictions

This replaces the two-pass (Skeleton → Extraction) approach in main.py.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any

from src.rag.embedder import WTTSEmbedder
from src.rag.retriever import WTTSRetriever


# ---------------------------------------------------------------------------
#  RAG-optimized prompt — shorter, focused on retrieved evidence only
# ---------------------------------------------------------------------------

RAG_EXTRACTION_PROMPT = """\
You are a Clinical Coding Expert.

RETRIEVED CLINICAL EVIDENCE (sorted chronologically, most relevant events for these items):
{retrieved_evidence}

TASK:
For each Clinical Item below, determine the value based ONLY on the evidence above.
1. **Value**: Must come strictly from the "Valid Options".
2. **Evidence**: Cite the [S_xx] ID that supports your choice.
3. If no evidence supports any option, choose "unknown".

ITEMS TO EXTRACT & THEIR OPTIONS:
{chunk_schema_json}

OUTPUT FORMAT (JSON Object):
{{
  "item_name": {{
    "value": "Selected Option",
    "evidence": "S_xx",
    "reasoning": "Brief explanation"
  }},
  ...
}}
"""


class RAGCRFExtractor:
    """
    Orchestrates RAG-guided CRF extraction for clinical notes.
    Replaces the two-pass (Skeleton → Extraction) pipeline with
    per-item retrieval for focused, temporally-ordered evidence.
    """

    def __init__(
        self,
        embedder: WTTSEmbedder,
        generate_fn,
        top_k: int = 15,
        weight_boost: float = 0.3,
        items_per_chunk: int = 5,
    ):
        """
        Args:
            embedder: WTTSEmbedder instance (shared across patients)
            generate_fn: Async function to call LLM (the generate_async from main.py)
            top_k: Number of tuples to retrieve per CRF item group
            weight_boost: Re-ranking boost for critical events
            items_per_chunk: How many CRF items to group per LLM call.
                             Grouped items share the same retrieved evidence pool.
        """
        self.embedder = embedder
        self.generate_fn = generate_fn
        self.top_k = top_k
        self.weight_boost = weight_boost
        self.items_per_chunk = items_per_chunk

    # ------------------------------------------------------------------
    #  Group CRF items by semantic similarity for batched retrieval
    # ------------------------------------------------------------------
    def _group_crf_items(
        self,
        target_items: List[str],
        valid_options: Dict[str, List[str]],
    ) -> List[List[str]]:
        """
        Group CRF items into chunks. Items in the same chunk will share
        a combined retrieval query, so similar items get grouped together.

        For now: simple sequential chunking (items_per_chunk at a time).
        Future: cluster by embedding similarity of item names.
        """
        chunks = []
        for i in range(0, len(target_items), self.items_per_chunk):
            chunk = target_items[i : i + self.items_per_chunk]
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    #  Build combined query for a group of CRF items
    # ------------------------------------------------------------------
    def _build_group_query(
        self,
        items: List[str],
        valid_options: Dict[str, List[str]],
    ):
        """
        Create a combined query embedding for a group of CRF items.
        Averages the individual item query embeddings.
        """
        query_embeddings = self.embedder.embed_queries_batch(items, valid_options)

        # Average the embeddings for a combined query
        import numpy as np
        all_embs = list(query_embeddings.values())
        combined = np.mean(all_embs, axis=0).astype(np.float32)

        # Re-normalize after averaging
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return combined

    # ------------------------------------------------------------------
    #  Main extraction method — full RAG pipeline for one patient
    # ------------------------------------------------------------------
    async def extract_patient(
        self,
        patient_data: Dict,
        builder,  # WTTSBuilder instance
        target_items: List[str],
        valid_options: Dict[str, List[str]],
        semaphore: asyncio.Semaphore,
        model: Any = None,  # Gemini model (passed to generate_fn)
    ) -> Optional[Dict]:
        """
        Full RAG pipeline for a single patient:
            1. Build WTTS string
            2. Parse & embed tuples → FAISS index
            3. For each CRF item group: retrieve → prompt → extract
            4. Return predictions

        Args:
            patient_data: Merged patient dict from DataLoader
            builder: WTTSBuilder instance
            target_items: List of CRF item names to extract
            valid_options: Dict mapping item name → list of valid values
            semaphore: Concurrency limiter for LLM calls
            model: Gemini model instance

        Returns:
            Dict with patient_id, predictions, and debug info
        """
        pid = str(
            patient_data.get('document_id')
            or patient_data.get('patient_id')
            or patient_data.get('hadm_id')
            or 'unknown'
        )

        try:
            # --- Step 1: Build WTTS tuples ---
            wtts_string = builder.build_wtts_string(patient_data)

            if not wtts_string.strip():
                print(f"  [{pid}] No WTTS tuples generated, skipping.")
                return None

            # --- Step 2: Parse and embed tuples ---
            tuples = self.embedder.parse_wtts_string(wtts_string)

            if not tuples:
                print(f"  [{pid}] Failed to parse WTTS tuples, skipping.")
                return None

            tuple_embeddings = self.embedder.embed_tuples(tuples)

            # --- Step 3: Build FAISS index for this patient ---
            retriever = WTTSRetriever(self.embedder)
            retriever.build_index(tuples, tuple_embeddings)

            print(f"  [{pid}] Indexed {len(tuples)} tuples. "
                  f"Retrieving for {len(target_items)} CRF items...")

            # --- Step 4: Group CRF items and extract ---
            item_groups = self._group_crf_items(target_items, valid_options)
            final_predictions = {}

            for group_items in item_groups:
                # Build combined query for this group
                group_query = self._build_group_query(group_items, valid_options)

                # Retrieve relevant tuples
                retrieved = retriever.retrieve_and_rerank(
                    group_query,
                    top_k=self.top_k,
                    weight_boost=self.weight_boost,
                )

                # Format for LLM
                evidence_str = WTTSRetriever.format_retrieved_tuples(retrieved)

                chunk_schema = {
                    item: valid_options.get(item, ["y", "n", "unknown"])
                    for item in group_items
                }

                prompt = RAG_EXTRACTION_PROMPT.format(
                    retrieved_evidence=evidence_str,
                    chunk_schema_json=json.dumps(chunk_schema, indent=2),
                )

                # Call LLM with concurrency control
                async with semaphore:
                    response = await self.generate_fn(prompt, model)

                if isinstance(response, dict):
                    if "error" in response:
                        print(f"  [{pid}] LLM error for items {group_items[:2]}...: "
                              f"{response['error']}")
                    else:
                        final_predictions.update(response)

            # --- Step 5: Return results ---
            return {
                "patient_id": pid,
                "predictions": final_predictions,
                "rag_debug": {
                    "total_tuples": len(tuples),
                    "top_k": self.top_k,
                    "item_groups": len(item_groups),
                },
            }

        except Exception as e:
            print(f"  [{pid}] RAG extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None
