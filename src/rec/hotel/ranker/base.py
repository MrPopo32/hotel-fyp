"""
Scoring interface used by all hotel rankers.
"""


class Ranker:
	# True when score(X, Y) should match score(Y, X).
	isSymmetric = True
	name = "Ranker"

	def prepare(self, dataset):
		self.dataset = dataset

	def getRankScore(self, X, Y):
		raise NotImplementedError("Subclasses must implement getRankScore")

	def getSimilarity(self, X, Y):
		# Diversity reranking needs a bounded similarity value.
		score = self.getRankScore(X, Y)
		if score < 0:
			return 0.0
		if score > 1:
			return 1.0
		return score
