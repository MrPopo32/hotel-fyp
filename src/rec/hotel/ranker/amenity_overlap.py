"""
Rank hotels by shared amenities.
"""

from rec.util.math import getJaccardSimilarity

from .base import Ranker


class AmenityOverlapRanker(Ranker):
	name = "AmenityOverlapRanker"
	isSymmetric = True

	def getRankScore(self, X, Y):
		return getJaccardSimilarity(X.amenities, Y.amenities)
