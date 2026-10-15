"""Run the complete evaluation suite and regenerate every report and figure.

    python scripts/run_evaluation.py              # everything
    python scripts/run_evaluation.py --skip-retrieval
    python scripts/run_evaluation.py --mode llm   # answers through the local LLM

Reports are written to reports/*.json and figures to reports/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402
from oncora.data.trials import load_statements, load_trials  # noqa: E402
from oncora.eval import answer_eval, nli4ct_eval, retrieval_eval  # noqa: E402
from oncora.rag.index import load_index  # noqa: E402
from oncora.rag.pipeline import AnswerEngine  # noqa: E402
from oncora.rag.retriever import HybridRetriever  # noqa: E402

log = logging.getLogger("evaluate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-retrieval", action="store_true")
    ap.add_argument("--skip-nli", action="store_true")
    ap.add_argument("--mode", default="auto", choices=["auto", "llm", "extractive"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = settings.paths.ensure()

    retriever = HybridRetriever(load_index())
    if not args.skip_retrieval:
        log.info("Retrieval benchmark and reranker training")
        retrieval_eval.run(retriever, paths.reports / "retrieval_eval.json")
    if not args.skip_nli:
        log.info("NLI4CT evidence and claim evaluation")
        nli4ct_eval.run(load_trials(paths.raw), load_statements(paths.raw), settings.random_seed, paths.reports / "nli4ct_eval.json")
    log.info("End to end answer evaluation (%s)", args.mode)
    engine = AnswerEngine(retriever)
    res = answer_eval.run(engine, mode=args.mode, out_path=paths.reports / f"answer_eval_{'llm' if args.mode == 'llm' else 'default'}.json")
    print(json.dumps(res["summary"], indent=2))

    from oncora.eval.figures import render_all

    render_all(paths.reports, paths.figures)
    log.info("Figures written to %s", paths.figures)


if __name__ == "__main__":
    main()
