"""Build the hybrid knowledge index (BM25 plus dense vectors) from raw sources."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.rag.index import build_index  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    idx = build_index()
    print(json.dumps(idx.manifest, indent=2))
