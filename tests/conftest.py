"""Pytest shared fixtures for the updated MLOps repo.

These fixtures keep the test suite stable after adding logging and optional
Weights & Biases support.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Ensure the repo root is importable so ``import src.*`` works whether tests are
# executed from the repo root or from inside the tests directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Disable external W&B network activity for all tests.

    The application code treats W&B as optional, so tests should verify the
    pipeline behavior without requiring real credentials or internet access.
    """
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    yield
