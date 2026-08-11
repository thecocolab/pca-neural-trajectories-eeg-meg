"""Headless, data-free test environment."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MNE_DONTWRITE_HOME_CONFIG", "true")
