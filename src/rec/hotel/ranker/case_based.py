"""
Simple case-based ranker using metadata and binary features.
"""

from .base import Ranker


class CaseBasedRanker(Ranker):
	name = "CaseBasedRanker"
	isSymmetric = True

	def prepare(self, dataset):
		self.dataset = dataset

		# Same global yes/no feature list for every hotel.
		self.featureNames = list(dataset.featureNames)

	def getRankScore(self, X, Y):
		# Average the partial similarities that are available.
		total = 0.0
		counter = 0

		starScore = self.getNumericSimilarity(X.star, Y.star)
		if starScore is not None:
			total += starScore
			counter += 1

		priceScore = self.getNumericSimilarity(X.priceLevel, Y.priceLevel)
		if priceScore is not None:
			total += priceScore
			counter += 1

		# Feature presence is just a yes/no match here.
		xFeatures = set(X.binaryFeatureVector.keys())
		yFeatures = set(Y.binaryFeatureVector.keys())
		featureCount = len(self.featureNames)
		if featureCount > 0:
			mismatches = len(xFeatures.symmetric_difference(yFeatures))
			total += featureCount - mismatches
			counter += featureCount

		return total / counter if counter > 0 else 0.0

	def getNumericSimilarity(self, x, y):
		# Missing values are skipped instead of treated as a bad match.
		if x is None or y is None:
			return None
		if x <= 0 or y <= 0:
			return None

		# Scale by the larger value so the score stays in range.
		denom = max(abs(float(x)), abs(float(y)))
		if denom <= 0:
			return None

		score = 1.0 - (abs(float(x) - float(y)) / denom)
		if score < 0:
			return 0.0
		if score > 1:
			return 1.0
		return score
