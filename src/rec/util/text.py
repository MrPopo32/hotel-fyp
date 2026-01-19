"""
Small text cleanup helpers for the recommender pipeline.
"""

import re


AMENITY_FEATURE_ALIASES = {
	"room service": ["room service"],
	"kids_activities": ["kids activities", "kids activity", "children activities", "family friendly"],
	"wifi": ["free high speed internet", "high speed internet", "free internet", "wireless internet", "wifi", "wi fi", "internet"],
	"wheelchair_access": ["wheelchair access", "wheelchair accessible", "accessible"],
	"pets_allowed": ["pets allowed", "pet friendly", "pets"],
	"kitchenette": ["kitchenette", "kitchenette room", "small kitchen"],
	"fitness_center": ["fitness center", "fitness centre", "gym"],
	"shuttle_bus_service": ["shuttle bus service", "shuttle service", "hotel shuttle"],
	"suites": ["suite", "suites"],
	"restaurant": ["restaurant", "restaurants", "dining room"],
	"breakfast": ["free breakfast", "breakfast", "continental breakfast"],
	"swimming_pool": ["swimming pool", "pool", "indoor pool", "outdoor pool"],
	"business_center": ["business center", "business centre", "conference center", "conference centre"],
	"parking": ["free parking", "parking", "car park", "parking lot"],
	"bar_lounge": ["bar lounge", "bar lounge area", "bar", "lounge"],
	"beverage_selection": ["beverage selection", "drink selection", "cocktails", "drinks"],
	"beach": ["beach", "beachfront"],
	"spa": ["spa", "wellness center", "wellness centre"],
	"airport_transportation": ["airport transportation", "airport transfer", "airport shuttle"],
}

AMENITY_NAME_MAP = {
	"room service": "room_service",
	"kids activities": "kids_activities",
	"free high speed internet": "wifi",
	"wheelchair access": "wheelchair_access",
	"pets allowed": "pets_allowed",
	"kitchenette": "kitchenette",
	"fitness center": "fitness_center",
	"shuttle bus service": "shuttle_bus_service",
	"suites": "suites",
	"restaurant": "restaurant",
	"free breakfast": "breakfast",
	"swimming pool": "swimming_pool",
	"business center": "business_center",
	"free parking": "parking",
	"bar/lounge": "bar_lounge",
	"beverage selection": "beverage_selection",
	"beach": "beach",
	"spa": "spa",
	"airport transportation": "airport_transportation",
}


def cleanText(value):
	if value is None:
		return None

	text = str(value).strip()
	if len(text) < 1:
		return None

	# Common null-like strings show up in the raw exports.
	if text.lower() == "null" or text.lower() == "none" or text.lower() == "nan" or text.lower() == "na":
		return None

	return text


def normalizeText(value):
	text = cleanText(value)
	if text is None:
		return ""

	# Basic cleanup is enough for keyword matching here.
	text = text.lower().replace("&amp;", " and ")
	text = re.sub(r"<[^>]+>", " ", text)
	text = re.sub(r"[^a-z0-9]+", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def normalizeWithPadding(value):
	text = normalizeText(value)
	if len(text) < 1:
		return " "

	# Padding avoids partial phrase matches like "wifi" inside another word.
	return " " + text + " "


def toFeatureName(rawAmenity):
	normalized = normalizeText(rawAmenity)
	text = normalized.replace(" ", "_")

	return AMENITY_NAME_MAP.get(normalized, text)


def getAmenityAliases(featureName):
	if featureName in AMENITY_FEATURE_ALIASES:
		return AMENITY_FEATURE_ALIASES[featureName]

	return [featureName.replace("_", " ")]


def containsAmenityMention(normalizedReviewText, featureName):
	for alias in getAmenityAliases(featureName):
		aliasText = normalizeText(alias)

		if len(aliasText) > 0 and (" " + aliasText + " ") in normalizedReviewText:
			return True
	return False


def toAmenitySet(value):
	text = cleanText(value)
	if text is None:
		return set()

	s = set()
	tokens = text.split(",")
	for token in tokens:
		token = normalizeText(token)
		if len(token) > 0:
			s.add(token)
	return s


def toRelatedHotelSet(value):
	text = cleanText(value)
	if text is None:
		return set()

	s = set()
	tokens = text.split(",")
	for token in tokens:
		token = token.strip()
		if len(token) > 0:
			s.add(token)
	return s
