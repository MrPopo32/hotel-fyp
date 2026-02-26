"""
Build the hotel profile dataset used by the experiments.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if(str(SRC_DIR) not in sys.path):
	sys.path.insert(0, str(SRC_DIR))

from rec.dataset import Reader
from rec.util.io import PROCESSED_DIR


def main() :
	reader = Reader()

	dataset = reader.dataset

	outputPath = PROCESSED_DIR / "hotel_profiles.parquet"
	featureOutputPath = PROCESSED_DIR / "hotel_feature_profiles.parquet"
	outputPath.parent.mkdir(parents = True, exist_ok = True )
	profileFrame = dataset.getProfileFrame()
	featureProfileFrame = dataset.getFeatureProfileFrame()

	profileFrame.to_parquet(outputPath, index = False)
	featureProfileFrame.to_parquet(featureOutputPath, index = False)

	print("Saved hotel profiles to " + str(outputPath))
	print("Saved full feature profiles to " + str(featureOutputPath))
	print("Hotels: " + str(len(dataset.getHotelIds())))
	print("Reviews: " + str(len(dataset.reviewFrame)))
	print("Hotel feature rows: " + str(len(featureProfileFrame)))
	print("Distinct features: " + str(featureProfileFrame["feature"].nunique() if len(featureProfileFrame) > 0 else 0))


if __name__ == "__main__":
	main()
