"""
Collapse mined aspect phrases into the feature names used by the model.
"""

from rec.util.text import normalizeText


ASPECT_ALIAS_MAP = {
	"breakfast": [
		"breakfast",
		"free breakfast",
		"continental breakfast",
		"buffet breakfast",
		"morning meal",
		"breakfast quality",
		"breakfast service",
	],
	"cleanliness": [
		"cleanliness",
		"clean",
		"hygiene",
		"spotless",
		"tidiness",
		"room cleanliness",
		"bathroom cleanliness",
		"housekeeping",
	],
	"location": [
		"location",
		"area",
		"neighborhood",
		"neighbourhood",
		"distance to attractions",
		"central",
		"transport",
		"nearby",
	],
	"staff": [
		"staff",
		"service",
		"customer service",
		"hospitality",
		"employees",
		"front desk",
		"helpfulness",
		"friendliness",
		"courtesy",
		"manager",
		"management",
	],
	"room_quality": [
		"room",
		"rooms",
		"room quality",
		"room size",
		"room comfort",
		"comfort",
		"bedroom",
		"bathroom",
		"bed",
		"bed comfort",
		"bedding",
		"bedlinen",
		"bed linen",
		"sheet",
		"bedsheet",
		"towel",
		"bath towel",
		"shower",
		"bath shower",
		"hot shower",
		"water pressure",
		"sleep quality",
	],
	"amenities": [
		"amenities",
		"facility",
		"facilities",
		"hotel facilities",
	],
	"wifi": [
		"wifi",
		"wi fi",
		"wi-fi",
		"internet",
		"internet access",
		"wireless internet",
		"wireless",
		"free wifi",
		"free internet",
	],
	"parking": [
		"parking",
		"free parking",
		"car park",
		"parking lot",
	],
	"noise": [
		"noise",
		"noisy",
		"quiet",
		"quietness",
		"street noise",
		"soundproofing",
		"acoustic",
	],
	"value": [
		"value",
		"price",
		"pricing",
		"price value",
		"value for money",
		"cost",
		"expensive",
		"affordable",
		"refund",
		"refund policy",
		"cancellation",
		"cancellation policy",
		"payment",
		"payment process",
	],
	"air_conditioning": [
		"air conditioning",
		"air conditioner",
		"ac",
		"heating",
	],
	"airport_transportation": [
		"airport transportation",
		"airport transport",
		"airport transfer",
		"airport shuttle",
	],
	"bar_lounge": [
		"bar lounge",
		"bar",
		"lounge",
	],
	"business_center": [
		"business center",
		"business centre",
		"conference",
		"meeting room",
	],
	"check_in": [
		"check in",
		"check out",
		"checkout",
		"reception",
		"reservation",
		"booking",
		"registration",
		"guest registration",
		"arrival",
		"departure",
		"queue",
		"wait time",
	],
	"elevator": [
		"elevator",
		"lift",
	],
	"fitness_center": [
		"fitness center",
		"fitness centre",
		"gym",
	],
	"food": [
		"food",
		"meal",
		"dining",
		"food quality",
	],
	"luggage_storage": [
		"luggage storage",
		"baggage storage",
		"bag storage",
	],
	"restaurant": [
		"restaurant",
		"restaurants",
	],
	"spa": [
		"spa",
		"sauna",
		"massage",
	],
	"swimming_pool": [
		"swimming pool",
		"pool",
	],
	"view": [
		"view",
		"views",
		"harbour view",
		"harbor view",
		"scenery",
	],
	"beach": [
		"beach",
		"beachfront",
	],
	"beverage_selection": [
		"beverage selection",
		"drink selection",
		"cocktail",
		"drinks",
	],
	"kids_activities": [
		"kids activities",
		"kids activity",
		"children activities",
		"family activities",
		"family activity",
		"family friendly",
	],
	"kitchenette": [
		"kitchenette",
		"kitchen",
	],
	"pets_allowed": [
		"pets allowed",
		"pet friendly",
		"pets",
	],
	"room_service": [
		"room service",
		"in room dining",
	],
	"shuttle_bus_service": [
		"shuttle bus service",
		"shuttle service",
		"hotel shuttle",
	],
	"suites": [
		"suite",
		"suites",
	],
	"wheelchair_access": [
		"wheelchair access",
		"wheelchair accessible",
		"accessible room",
	],
	"property_quality": [
		"property quality",
		"condition",
		"hotel condition",
		"room condition",
		"maintenance",
		"hotel maintenance",
		"appearance",
		"decor",
		"design",
		"hotel design",
		"ambiance",
		"ambience",
		"refurbishment",
		"renovation",
		"hotel renovation",
		"modern",
		"newness",
		"furniture",
		"carpet",
		"wallpaper",
		"lighting",
		"layout",
		"clean decor",
	],
	"safety": [
		"safety",
		"security",
		"safe",
	],
	"accessibility": [
		"accessibility",
		"accessible",
		"stairs",
	],
}

LEGACY_ASPECT_FIELD_MAP = {
	"cleanliness": "cleanliness",
	"room_quality": "room_quality",
	"amenities": "amenities",
	"breakfast": "breakfast",
	"wifi": "wifi",
	"noise": "noise",
}

ASPECT_PREFIX_WORDS = {
	# Weak leading words that usually do not change the aspect meaning.
	"hotel",
	"free",
	"good",
	"great",
	"nice",
	"excellent",
	"overall",
}

GENERIC_ASPECT_NAMES = {
	"experience",
	"hotel_experience",
	"impression",
	"quality",
	"recommendation",
	"satisfaction",
	"stay",
	"stay_experience",
}

MAX_FALLBACK_ASPECT_WORDS = 3
CONTAINED_ALIAS_EXCLUDE = {
	# These are useful as exact aliases, but too broad inside longer labels.
	"room",
	"rooms",
	"service",
}

def normalizeAspectName(value):
	return normalizeText(value)


def stripAspectPrefixWords(value):
	tokens = value.split()
	while len(tokens) > 1 and tokens[0] in ASPECT_PREFIX_WORDS:
		tokens = tokens[1:]
	return " ".join(tokens)


def singularizeAspectPhrase(value):
	tokens = value.split()
	if len(tokens) <= 0:
		return value

	lastToken = tokens[-1]

	# Tiny plural cleanup, not full English morphology.
	if len(lastToken) > 3 and lastToken.endswith("ies"):
		tokens[-1] = lastToken[:-3] + "y"
	elif len(lastToken) > 3 and lastToken.endswith("s") and not lastToken.endswith("ss"):
		tokens[-1] = lastToken[:-1]

	return " ".join(tokens)


def getAspectAliasLookup():
	lookup = {}
	for canonicalName in ASPECT_ALIAS_MAP.keys():
		# Canonical names should match themselves too.
		lookup[normalizeAspectName(canonicalName)] = canonicalName
		for alias in ASPECT_ALIAS_MAP[canonicalName]:
			lookup[normalizeAspectName(alias)] = canonicalName
	return lookup


ASPECT_ALIAS_LOOKUP = getAspectAliasLookup()


def getContainedAliasLookup():
	rows = []
	for canonicalName in ASPECT_ALIAS_MAP.keys():
		for alias in ASPECT_ALIAS_MAP[canonicalName]:
			normalizedAlias = normalizeAspectName(alias)
			if len(normalizedAlias) > 0 and normalizedAlias not in CONTAINED_ALIAS_EXCLUDE:
				rows.append((canonicalName, normalizedAlias))
	return sorted(rows, key=lambda row: -len(row[1]))


CONTAINED_ALIAS_LOOKUP = getContainedAliasLookup()


def getContainedAliasMatch(value):
	padded = " " + value + " "
	for canonicalName, normalizedAlias in CONTAINED_ALIAS_LOOKUP:
		if (" " + normalizedAlias + " ") in padded:
			return canonicalName
	return None


def getCanonicalAspectName(rawAspect, useContainedAliases=True):
	normalized = normalizeAspectName(rawAspect)
	if len(normalized) < 1:
		return ""

	if normalized in ASPECT_ALIAS_LOOKUP:
		return ASPECT_ALIAS_LOOKUP[normalized]

	stripped = stripAspectPrefixWords(normalized)
	if len(stripped) > 0 and stripped in ASPECT_ALIAS_LOOKUP:
		return ASPECT_ALIAS_LOOKUP[stripped]

	singular = singularizeAspectPhrase(normalized)
	if len(singular) > 0 and singular in ASPECT_ALIAS_LOOKUP:
		return ASPECT_ALIAS_LOOKUP[singular]

	singularStripped = singularizeAspectPhrase(stripped)
	if len(singularStripped) > 0 and singularStripped in ASPECT_ALIAS_LOOKUP:
		return ASPECT_ALIAS_LOOKUP[singularStripped]

	if useContainedAliases:
		contained = getContainedAliasMatch(singularStripped)
		if contained is not None:
			return contained

	# Keep new aspects as readable underscore names instead of dropping them.
	fallback = singularStripped if len(singularStripped) > 0 else singular
	fallbackName = fallback.replace(" ", "_")
	if fallbackName in GENERIC_ASPECT_NAMES:
		return ""
	if len(fallback.split()) > MAX_FALLBACK_ASPECT_WORDS:
		return ""
	return fallback.replace(" ", "_")


def getLegacyAspectField(canonicalAspectName):
	return LEGACY_ASPECT_FIELD_MAP[canonicalAspectName] if canonicalAspectName in LEGACY_ASPECT_FIELD_MAP else None
