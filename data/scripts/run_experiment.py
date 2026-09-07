#!/usr/bin/env python3
"""Command-line entry point for the complete reproducible experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiment import run_experiment


if __name__ == "__main__":
    artifacts = run_experiment(ROOT)
    print(artifacts["performance"].round(4).to_string())
    print(f"\nArtifacts written to {ROOT / 'results'} and {ROOT / 'figures'}")
