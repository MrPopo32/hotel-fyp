"""
Sentiment ranker with similarity plus improvement.
"""

from rec.util.math import getNormalizedCosineSimilarity
from rec.util.math import getWeightedNormalizedCosineSimilarity

from .base import Ranker


class SentimentEnhancedRanker(Ranker):
	name = "SentimentEnhancedRanker"
	isSymmetric = False

	def __init__(self, sentimentWeight = 0.5):
		self.sentimentWeight = sentimentWeight

	def getRankScore(self, X, Y):
		# Blend "feels similar" with "looks better on shared features".
		similarity = self.getSimilarity(X, Y)
		improvement = self.getSentimentImprovement(X, Y)
		return (1.0 - self.sentimentWeight) * similarity + self.sentimentWeight * improvement

	def getSimilarity(self, X, Y):
		# Compare sentiment shape without mixing in frequency here.
		if self.usesUserFeatureWeights(X):
			return getWeightedNormalizedCosineSimilarity(
				X.sentimentFeatureVector,
				Y.sentimentFeatureVector,
				X.frequencyFeatureVector,
			)
		return getNormalizedCosineSimilarity(X.sentimentFeatureVector, Y.sentimentFeatureVector)

	def usesUserFeatureWeights(self, X):
		return getattr(X, "sourceType", None) == "user_profile" and len(X.frequencyFeatureVector) > 0

	def getSentimentImprovement(self, X, Y):
		# Only compare shared reviewed features.
		total = 0.0
		totalWeight = 0.0

		xProfile = X.featureSentimentProfile
		yProfile = Y.featureSentimentProfile
		featureKeys = xProfile.keys()
		if len(yProfile) < len(xProfile):
			featureKeys = yProfile.keys()

		for feature in featureKeys:
			if feature not in xProfile or feature not in yProfile:
				continue

			# Map the sentiment difference into [0, 1], with 0.5 as "same".
			diff = (yProfile[feature] - xProfile[feature]) / 2.0
			score = (diff + 1.0) / 2.0

			if score < 0:
				score = 0.0
			if score > 1:
				score = 1.0

			weight = 1.0
			if self.usesUserFeatureWeights(X):
				weight = X.frequencyFeatureVector[feature] if feature in X.frequencyFeatureVector else 0.0
				if weight <= 0:
					continue

			total += weight * score
			totalWeight += weight

		if totalWeight <= 0:
			# No shared sentiment evidence, so stay neutral.
			return 0.5

		return total / totalWeight
