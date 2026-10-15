"""Train the learned reranker and benchmark retrieval variants."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402
from oncora.eval.retrieval_eval import run  # noqa: E402
from oncora.rag.index import load_index  # noqa: E402
from oncora.rag.retriever import HybridRetriever  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings.paths.ensure()
    results = run(HybridRetriever(load_index()), settings.paths.reports / "retrieval_eval.json")
    for block in ("pubmed_known_item_test", "clinical_questions_development", "clinical_questions_blind"):
        if block in results:
            print(block)
            for name, m in results[block].items():
                print(f"  {name:22s} R@1 {m['recall_at_1']:.3f}  R@5 {m['recall_at_5']:.3f}  MRR {m['mrr_at_10']:.3f}  nDCG {m['ndcg_at_10']:.3f}")
    print(json.dumps(results["reranker_training"], indent=2))
