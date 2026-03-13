"""
Weighted mix of sentiment, amenity, and popularity rankers.
"""

from .amenity_overlap import AmenityOverlapRanker
from .base import Ranker
from .popularity import ContextualPopularityRanker
from .sentiment_similarity import SentimentSimilarityRanker


class WeightedHybridRanker(Ranker):
	isSymmetric = False

	def __init__(self, sentimentWeight = 0.5, amenityWeight = 0.3, popularityWeight = 0.2, aspectWeights = None):
		self.sentimentWeight = sentimentWeight
		self.amenityWeight = amenityWeight
		self.popularityWeight = popularityWeight
		self.sentimentRanker = SentimentSimilarityRanker(aspectWeights)
		self.amenityRanker = AmenityOverlapRanker()
		self.popularityRanker = ContextualPopularityRanker()
		self.name = "WeightedHybridRanker"

	def prepare(self, dataset):
		self.dataset = dataset
		self.sentimentRanker.prepare(dataset)
		self.amenityRanker.prepare(dataset)
		self.popularityRanker.prepare(dataset)

	def getRankScore(self, X, Y):
		sentimentScore = self.sentimentRanker.getRankScore(X, Y)
		amenityScore = self.amenityRanker.getRankScore(X, Y)
		popularityScore = self.popularityRanker.getRankScore(X, Y)

		return self.sentimentWeight * sentimentScore + self.amenityWeight * amenityScore + self.popularityWeight * popularityScore
