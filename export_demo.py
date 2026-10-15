"""Export the static bundle for the in browser demo engine (frontend/public/demo)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.config import settings  # noqa: E402
from oncora.export.demo_bundle import export  # noqa: E402
from oncora.rag.index import load_index  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = export(load_index(), settings.paths.demo_bundle)
    print(json.dumps(meta["browser_engine"]["evaluation"], indent=1))
    print(json.dumps(meta["sizes_bytes"], indent=1))
