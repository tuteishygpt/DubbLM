#!/usr/bin/env python3
"""Command line launcher for the DubbLM pipeline."""

import os
import sys

# Allow running the CLI directly from a source checkout without installation.
SRC_PATH = os.path.join(os.path.dirname(__file__), "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from dubbing.cli.main import main as _main


def main() -> None:
    """Dispatch to the packaged CLI entry point."""
    _main()


if __name__ == "__main__":
    main()
