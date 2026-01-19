"""
Small similarity helpers used by the rankers.
"""

import math

_PROFILE_NORM_CACHE = {}
_WEIGHTED_PROFILE_NORM_CACHE = {}


def getProfileNorm(profile):
	cacheKey = id(profile)
	cachedNorm = _PROFILE_NORM_CACHE.get(cacheKey)
	if cachedNorm is not None and cachedNorm[0] is profile:
		return cachedNorm[1]

	norm = 0
	for value in profile.values():
		norm += math.pow(value, 2)

	_PROFILE_NORM_CACHE[cacheKey] = (profile, norm)
	return norm


def getCosineSimilarity(profileX, profileY):
	dotProduct = 0
	if len(profileX) <= len(profileY):
		for k, x in profileX.items():
			if k in profileY:
				dotProduct += x * profileY[k]
	else:
		for k, y in profileY.items():
			if k in profileX:
				dotProduct += profileX[k] * y

	normX = getProfileNorm(profileX)
	normY = getProfileNorm(profileY)

	denom = math.sqrt(normX * normY)
	return dotProduct / denom if denom > 0 else 0


def getNormalizedCosineSimilarity(profileX, profileY):
	score = getCosineSimilarity(profileX, profileY)

	# Some rankers expect similarity in [0, 1].
	score = (score + 1.0) / 2.0
	if score < 0:
		return 0.0
	if score > 1:
		return 1.0
	return score


def getWeightedCosineSimilarity(profileX, profileY, featureWeights):
	cacheKey = (id(profileX), id(featureWeights))
	cachedNorm = _WEIGHTED_PROFILE_NORM_CACHE.get(cacheKey)
	if cachedNorm is not None and cachedNorm[0] is profileX and cachedNorm[1] is featureWeights:
		normX = cachedNorm[2]
	else:
		normX = 0
		for k, x in profileX.items():
			weight = featureWeights[k] if k in featureWeights else 0.0
			if weight > 0:
				normX += weight * math.pow(x, 2)
		_WEIGHTED_PROFILE_NORM_CACHE[cacheKey] = (profileX, featureWeights, normX)

	dotProduct = 0
	normY = 0
	for k, y in profileY.items():
		weight = featureWeights[k] if k in featureWeights else 0.0
		if weight <= 0:
			continue

		x = profileX[k] if k in profileX else 0.0
		dotProduct += weight * x * y
		normY += weight * math.pow(y, 2)

	denom = math.sqrt(normX * normY)
	return dotProduct / denom if denom > 0 else 0


def getWeightedNormalizedCosineSimilarity(profileX, profileY, featureWeights):
	score = getWeightedCosineSimilarity(profileX, profileY, featureWeights)
	score = (score + 1.0) / 2.0
	if score < 0:
		return 0.0
	if score > 1:
		return 1.0
	return score


def getJaccardSimilarity(setX, setY):
	if len(setX) == 0 or len(setY) == 0:
		return 0

	# Intersection over union for simple binary overlap.
	union = set()
	for x in setX:
		union.add(x)
	for y in setY:
		union.add(y)

	intersection = set()
	for x in setX:
		if x in setY:
			intersection.add(x)

	return 1.0 * len(intersection) / len(union) if len(union) > 0 else 0
