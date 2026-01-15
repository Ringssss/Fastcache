#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import sys

REPO_ROOT = Path(__file__).resolve().parent
CKPT_DIR = Path(os.environ.get("FASTCACHE_CKPT_DIR", REPO_ROOT / "ckpt")).resolve()
DATASETS_DIR = Path(os.environ.get("FASTCACHE_DATASETS_DIR", REPO_ROOT / "datasets")).resolve()
RESULTS_DIR = Path(os.environ.get("FASTCACHE_RESULTS_DIR", REPO_ROOT / "results")).resolve()


def ensure_sys_paths() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    utils_ccm_dir = REPO_ROOT / "utils_ccm"
    if utils_ccm_dir.exists():
        utils_ccm_str = str(utils_ccm_dir)
        if utils_ccm_str not in sys.path:
            sys.path.insert(0, utils_ccm_str)
