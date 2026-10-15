"""Download every public dataset used by Oncora Nexus.

Usage:
    python scripts/download_data.py            # all sources including PAM50 expression
    python scripts/download_data.py --no-expression
    python scripts/download_data.py --force    # refetch everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oncora.data.acquire import acquire_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refetch sources that already exist")
    parser.add_argument("--no-expression", action="store_true", help="Skip the 690 MB expression stream")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = acquire_all(force=args.force, with_expression=not args.no_expression)
    total = sum(s["bytes"] for s in manifest["sources"]) / 1e6
    print(f"Acquired {len(manifest['sources'])} sources, {total:.1f} MB on disk")


if __name__ == "__main__":
    main()
