"""
RAG Evaluation Script
=====================
Evaluates retrieval quality against a curated question set.
Requires: built FAISS index (run build_index.py first)

Metrics reported:
  - Recall@K: fraction of expected document IDs retrieved in top-K
  - Citation validity rate: fraction of retrieved chunk_ids that are valid (format check)
  - Empty-context trigger rate: fraction of queries returning 0 results above threshold
  - Top-K configuration and similarity scores

Run:
  python scripts/evaluate_rag.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_PATH = PROJECT_ROOT / "evaluation" / "rag_questions.json"


def run_evaluation():
    print("=" * 65)
    print("RAG Evaluation Report")
    print("=" * 65)

    # Load questions
    if not QUESTIONS_PATH.exists():
        print(f"[ERROR] Evaluation questions not found: {QUESTIONS_PATH}")
        sys.exit(1)

    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"\nEvaluation set: {len(questions)} questions")

    # Load retriever (lazy — will trigger FAISS load)
    from rag.retriever import get_retriever
    import os

    top_k     = int(os.getenv("TOP_K", "5"))
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

    try:
        retriever = get_retriever()
    except FileNotFoundError as e:
        print(f"\n[BLOCKED] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Could not load retriever: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"TOP_K = {top_k}, SIMILARITY_THRESHOLD = {threshold}")
    print()

    # Per-question metrics
    recall_at_k_scores: list[float] = []
    empty_context_count = 0
    total_retrieved_chunks = 0
    valid_chunk_id_count = 0

    for i, q in enumerate(questions, 1):
        question     = q["question"]
        expected_ids = set(q.get("expected_document_ids", []))

        print(f"Q{i}: {question[:70]}...")

        result = retriever.retrieve(question, top_k=top_k, threshold=threshold)
        docs  = result["documents"]
        scores = result["scores"]
        meta   = result["metadata"]

        if not docs:
            empty_context_count += 1
            print(f"  → 0 results above threshold={threshold}")
            recall_at_k_scores.append(0.0)
            continue

        # Retrieved document IDs
        retrieved_doc_ids = {m.get("document_id", "") for m in meta}
        retrieved_chunk_ids = [m.get("chunk_id", "") for m in meta]

        # Recall@K: fraction of expected doc IDs found
        if expected_ids:
            hits    = expected_ids & retrieved_doc_ids
            recall  = len(hits) / len(expected_ids)
        else:
            recall  = 1.0  # No expected IDs — treat as pass

        recall_at_k_scores.append(recall)

        # Chunk ID format validity (basic check: should match pattern)
        valid_ids = sum(
            1 for cid in retrieved_chunk_ids
            if ":" in cid and len(cid.split(":")) == 3
        )
        valid_chunk_id_count += valid_ids
        total_retrieved_chunks += len(retrieved_chunk_ids)

        print(f"  → {len(docs)} chunks retrieved | top score: {scores[0]:.4f}")
        print(f"  → Retrieved doc_ids: {retrieved_doc_ids}")
        if expected_ids:
            print(f"  → Expected doc_ids:  {expected_ids}")
            hit_label = f"HIT ({len(hits)}/{len(expected_ids)})" if hits else f"MISS (0/{len(expected_ids)})"
            print(f"  → Recall@{top_k}: {recall:.2f} — {hit_label}")
        print()

    # Summary
    n = len(questions)
    avg_recall = sum(recall_at_k_scores) / n if n > 0 else 0.0
    empty_rate = empty_context_count / n if n > 0 else 0.0
    chunk_validity = valid_chunk_id_count / total_retrieved_chunks if total_retrieved_chunks > 0 else 0.0

    print("=" * 65)
    print("Summary")
    print("-" * 65)
    print(f"  Total questions        : {n}")
    print(f"  TOP_K                  : {top_k}")
    print(f"  Similarity threshold   : {threshold}")
    print(f"  Avg Recall@{top_k}          : {avg_recall:.3f}")
    print(f"  Empty-context rate     : {empty_rate:.3f} ({empty_context_count}/{n} queries)")
    print(f"  Chunk ID validity rate : {chunk_validity:.3f} ({valid_chunk_id_count}/{total_retrieved_chunks} chunks)")
    print("=" * 65)

    if avg_recall < 0.3:
        print("\n[WARNING] Low average Recall@K — consider adding more sources or adjusting chunking.")
    if empty_rate > 0.5:
        print("[WARNING] Over 50% of queries returned empty context — consider lowering the threshold.")


if __name__ == "__main__":
    run_evaluation()
