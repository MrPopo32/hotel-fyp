"""
Rank hotels by feature mention patterns, ignoring sentiment.
"""

from rec.util.math import getCosineSimilarity
from rec.util.math import getWeightedCosineSimilarity

from .base import Ranker


class FrequencyBasedRanker(Ranker):
	name = "FrequencyBasedRanker"
	isSymmetric = True

	def getRankScore(self, X, Y):
		# User-profile queries use their own feature frequencies as weights.
		if self.usesUserFeatureWeights(X):
			return getWeightedCosineSimilarity(
				X.frequencyFeatureVector,
				Y.frequencyFeatureVector,
				X.frequencyFeatureVector,
			)
		return getCosineSimilarity(X.frequencyFeatureVector, Y.frequencyFeatureVector)

	def usesUserFeatureWeights(self, X):
		return getattr(X, "sourceType", None) == "user_profile" and len(X.frequencyFeatureVector) > 0

	def getSimilarity(self, X, Y):
		# Keep the clamp in case the vector representation changes later.
		score = self.getRankScore(X, Y)
		if score < 0:
			return 0.0
		if score > 1:
			return 1.0
		return score
