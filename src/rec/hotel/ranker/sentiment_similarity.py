"""
Rank hotels by fixed sentiment profile similarity.
"""

import math

from rec.dataset import ASPECT_COLUMNS
from rec.util.math import getCosineSimilarity

from .base import Ranker


class SentimentSimilarityRanker(Ranker):
	name = "SentimentSimilarityRanker"
	isSymmetric = True

	def __init__(self, aspectWeights = None):
		self.aspectWeights = aspectWeights

	def getRankScore(self, X, Y):
		if self.aspectWeights is None or len(self.aspectWeights) == 0:
			return getCosineSimilarity(X.sentimentProfile, Y.sentimentProfile)

		dotProduct = 0
		normX = 0
		normY = 0

		for aspect in ASPECT_COLUMNS :
			weight = self.aspectWeights[aspect] if aspect in self.aspectWeights else 0.0
			x = X.sentimentProfile[aspect] if aspect in X.sentimentProfile else 0.0
			y = Y.sentimentProfile[aspect] if aspect in Y.sentimentProfile else 0.0
			dotProduct += weight * x * y
			normX += weight * math.pow(x, 2)
			normY += weight * math.pow(y, 2)

		denom = math.sqrt(normX * normY)
		if denom <= 0:
			return 0.0

		return dotProduct / denom
