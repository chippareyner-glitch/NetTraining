"""Run tasks from a source checkout without installing into the current environment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from plant_signal.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
