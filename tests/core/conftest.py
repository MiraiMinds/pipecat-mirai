"""Tests of fourier-frequency on its own: nothing here may import Pipecat."""

import sys
from pathlib import Path

# The fake servers (fake_mirai_*.py) live one level up, beside the Pipecat tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
