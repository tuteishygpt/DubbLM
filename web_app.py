#!/usr/bin/env python3
"""Top-level launcher for the DubbLM web application."""

import sys
from pathlib import Path


SRC_PATH = Path(__file__).resolve().parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dubbing.web.app import main


if __name__ == "__main__":
    main()
