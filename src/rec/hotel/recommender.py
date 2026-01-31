"""
Turn pair scores from a ranker into recommendation lists.
"""


class HotelRecommender:
	def __init__(
		self,
		dataset,
		ranker,
		candidateHotelIds = None,
		sameCityOnly = True,
		useDiversity = False,
		diversityPoolSize = 50,
		diversityTopK = 10,
		diversityLambda = 0.8,
	) :
		self.dataset = dataset
		self.ranker = ranker
		self.candidateHotelIds = set(candidateHotelIds) if candidateHotelIds is not None else None
		self.sameCityOnly = sameCityOnly
		self.useDiversity = useDiversity
		self.diversityPoolSize = diversityPoolSize
		self.diversityTopK = diversityTopK
		self.diversityLambda = diversityLambda
		self.isReady = False

		# Cache hotel-to-hotel recommendations so evaluation does not rerank twice.
		self.cache = {}

	def recommend(self, targetHotelId) :
		if targetHotelId in self.cache:
			return list(self.cache[targetHotelId])

		# Hotel-to-hotel queries use the hotel itself as the profile.
		targetHotel = self.dataset.getHotel(targetHotelId)
		recIds = self.recommendForProfile(targetHotel, set([targetHotelId]))
		self.cache[targetHotelId] = list(recIds)
		return recIds

	def recommendForProfile(self, profile, excludeIds = None) :
		if not self.isReady:
			self.fit()

		# Exclusions cover the target hotel and already-seen hotels.
		excludeIds = set(excludeIds) if excludeIds is not None else set()
		if getattr(profile, "sourceType", None) == "user_profile":
			for hotelId in profile.relatedHotels:
				excludeIds.add(hotelId)

		candidates = []

		# Candidate rules live here so every script uses the same restrictions.
		for hotelId in self.getCandidateHotelIds(profile, excludeIds):
			hotel = self.dataset.getHotel(hotelId)
			score = self.ranker.getRankScore(profile, hotel)
			candidates.append((hotelId, score))

		# Sort by score, then id so ties stay deterministic.
		candidates.sort(key = lambda row: (-row[1], row[0]))
		return self.getRankedCandidateIds(profile, candidates)

	def fit(self) :
		self.ranker.prepare(self.dataset)
		self.isReady = True

	def getCandidateHotelIds(self, profile, excludeIds = None) :
		excludeIds = excludeIds if excludeIds is not None else set()

		# Same-city filtering is the main candidate rule.
		if self.sameCityOnly and self.dataset.hasKnownCity(profile.city):
			candidateIds = self.dataset.getHotelIdsByCity(profile.city)
		else:
			candidateIds = self.dataset.getHotelIds()

		rows = []
		for hotelId in sorted(candidateIds):
			# Optional metadata filters apply after the city rule.
			if self.candidateHotelIds is not None and hotelId not in self.candidateHotelIds:
				continue
			if hotelId not in excludeIds:
				rows.append(hotelId)
		return rows

	def getRankedCandidateIds(self, profile, candidates) :
		if not self.useDiversity:
			recIds = []
			for hotelId, _ in candidates:
				recIds.append(hotelId)
			return recIds

		# Only diversify the top pool so the full ranking stays mostly intact.
		poolSize = min(len(candidates), max(self.diversityTopK, self.diversityPoolSize))
		topPool = candidates[:poolSize]
		remaining = candidates[poolSize:]
		selected = []
		selectedIds = set()

		while len(selected) < min(self.diversityTopK, len(topPool)):
			bestRow = None
			bestObjective = None

			for hotelId, baseScore in topPool:
				if hotelId in selectedIds:
					continue

				hotel = self.dataset.getHotel(hotelId)
				redundancy = self.getRedundancyScore(hotel, selected)

				# Reward rank score, penalize repeats from already selected hotels.
				objective = self.diversityLambda * baseScore - (1.0 - self.diversityLambda) * redundancy

				if bestObjective is None or objective > bestObjective:
					bestObjective = objective
					bestRow = (hotelId, baseScore)

			if bestRow is None:
				break

			selected.append(bestRow)
			selectedIds.add(bestRow[0])

		recIds = []
		for hotelId, _ in selected:
			recIds.append(hotelId)

		for hotelId, _ in topPool:
			if hotelId not in selectedIds:
				recIds.append(hotelId)

		for hotelId, _ in remaining:
			recIds.append(hotelId)

		return recIds

	def getRedundancyScore(self, hotel, selected) :
		if len(selected) == 0:
			return 0.0

		# Closest selected hotel is the redundancy penalty.
		best = 0.0
		for selectedId, _ in selected:
			otherHotel = self.dataset.getHotel(selectedId)
			score = self.ranker.getSimilarity(hotel, otherHotel)
			if score > best:
				best = score
		return best
