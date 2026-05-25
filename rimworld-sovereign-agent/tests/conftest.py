"""Make the repo root importable so ``import rimworld_agent`` works under pytest without an
install. Mirrors vocab-extend-qlora's test bootstrap.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
