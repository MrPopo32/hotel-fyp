"""
Popularity and rating baseline ranker.
"""

import math

from .base import Ranker


class ContextualPopularityRanker(Ranker):
	name = "ContextualPopularityRanker"
	isSymmetric = False

	def prepare(self, dataset):
		self.dataset = dataset

		# Max review count is needed for popularity normalization.
		self.maxReviewCount = dataset.getMaxReviewCount()

	def getRankScore(self, X, Y):
		if X.id == Y.id:
			return 0.0

		# Map the 5-star rating into a [0, 1] score.
		ratingScore = Y.averageRating / 5.0
		if ratingScore < 0:
			ratingScore = 0
		if ratingScore > 1:
			ratingScore = 1

		# log1p keeps one huge hotel from dominating too much.
		popularityScore = math.log1p(Y.reviewCount) / math.log1p(self.maxReviewCount)

		# Small boost when star bands are similar.
		starSimilarity = 0.5
		if X.star is not None and Y.star is not None:
			starSimilarity = 1.0 / (1.0 + abs(X.star - Y.star))

		# Older city boost kept as a light extra context signal.
		cityBoost = 0.65
		if X.city == Y.city:
			cityBoost = 1.0

		# Rating matters most, then popularity, then star similarity.
		return cityBoost * (0.5 * ratingScore + 0.35 * popularityScore + 0.15 * starSimilarity)
