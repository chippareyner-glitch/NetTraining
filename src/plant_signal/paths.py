"""Resolve runtime data paths without embedding a developer's machine paths."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Use an explicit data root, the source checkout, or the current directory."""
    configured = os.environ.get('PLANT_SIGNAL_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    package_dir = Path(__file__).resolve().parent
    if package_dir.parent.name == 'src':
        return package_dir.parents[1]
    return Path.cwd().resolve()


ROOT_DIR = project_root()
