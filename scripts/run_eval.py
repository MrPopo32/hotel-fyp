"""
Run hotel recommender experiments.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path :
	sys.path.insert(0, str(SRC_DIR))

from expts.run_hotelrec_expt import main


if __name__ == "__main__":
	main()
