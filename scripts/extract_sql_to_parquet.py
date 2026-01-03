"""
Extract SQL tables to parquet files.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if(str(SRC_DIR) not in sys.path):
	sys.path.insert(0, str(SRC_DIR))

from rec.pipelines.extract_sql import main


if __name__ == "__main__":
	main()
