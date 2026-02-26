"""
Data helpers for hotel and user profiles.
"""

import json

import pandas as pd

from rec.util.aspect_aliases import getCanonicalAspectName
from rec.util.text import cleanText
from rec.util.text import containsAmenityMention
from rec.util.text import normalizeText
from rec.util.text import toFeatureName

from .entities import Hotel


ASPECT_COLUMNS = [
    "cleanliness",
    "room_quality",
    "amenities",
    "breakfast",
    "wifi",
    "noise",
]

SENTIMENT_LABEL_SCORES = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
}


def isMissing(value):
    if value is None:
        return True
    try:
        result = pd.isna(value)
        if hasattr(result, "__len__") and not isinstance(result, (str, bytes)):
            return False
        return bool(result)
    except (TypeError, ValueError):
        return False
    return False


def getPriceLevel(priceRange):
    """
    Best-effort parser for messy price range values.
    """
    if isMissing(priceRange):
        return None

    text = cleanText(priceRange)
    if text is None:
        return None

    stripped = text.strip()
    symbolCount = stripped.count("$") + stripped.count("€") + stripped.count("£")
    if symbolCount > 0:
        return min(symbolCount, 4)

    try:
        value = int(float(stripped))
        if value > 0:
            return min(value, 4)
    except ValueError:
        pass

    normalized = normalizeText(stripped)
    if any(token in normalized for token in ["budget", "cheap", "low"]):
        return 1
    if any(token in normalized for token in ["mid", "moderate", "average"]):
        return 2
    if any(token in normalized for token in ["upper", "upscale", "premium"]):
        return 3
    if any(token in normalized for token in ["luxury", "expensive", "high"]):
        return 4

    return None


def normalizeCityName(value):
    text = cleanText(value)
    return normalizeText(text) if text is not None else ""


def citiesMatch(left, right):
    leftText = normalizeCityName(left)
    rightText = normalizeCityName(right)
    if len(leftText) == 0 or len(rightText) == 0:
        return False
    return leftText == rightText


def parseRecommendList(value):
    text = cleanText(value)
    if text is None:
        return set()

    related = set()
    for token in text.replace("|", ",").replace(";", ",").split(","):
        token = token.strip()
        if ":" in token:
            prefix, suffix = token.split(":", 1)
            try:
                float(prefix.strip())
                token = suffix.strip()
            except ValueError:
                pass
        if len(token) > 0:
            related.add(token)
    return related


def sentimentToScore(value):
    """
    Convert sentiment labels or numbers to the [-1, 1] scale.
    """
    if isMissing(value):
        return None

    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = cleanText(value)
        if text is None:
            return None

        normalized = normalizeText(text)
        if normalized in SENTIMENT_LABEL_SCORES:
            return SENTIMENT_LABEL_SCORES[normalized]

        try:
            score = float(text)
        except ValueError:
            return None

    if score > 1:
        score = 1.0
    if score < -1:
        score = -1.0
    return score


def getOverallReviewSentiment(row):
    if "overall_sentiment_score" in row:
        score = sentimentToScore(row.get("overall_sentiment_score"))
        if score is not None:
            return score

    if "overall_sentiment_label" in row:
        score = sentimentToScore(row.get("overall_sentiment_label"))
        if score is not None:
            return score

    if "rating" in row and not isMissing(row.get("rating")):
        try:
            rating = float(row.get("rating"))
            return max(-1.0, min(1.0, (rating - 3.0) / 2.0))
        except (TypeError, ValueError):
            return 0.0

    return 0.0


def parseMinedAspects(value):
    """
    Parse the dynamic aspect JSON from enrichment.
    """
    text = cleanText(value)
    if text is None:
        return []

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []

    if isinstance(payload, dict):
        if "aspects" in payload and isinstance(payload["aspects"], list):
            payload = payload["aspects"]
        else:
            payload = [payload]

    if not isinstance(payload, list):
        return []

    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue

        aspect = item.get("aspect") or item.get("name") or item.get("feature")
        canonical = getCanonicalAspectName(aspect)
        if len(canonical) == 0:
            continue

        score = sentimentToScore(item.get("score"))
        if score is None:
            score = sentimentToScore(item.get("sentiment"))
        if score is None:
            score = sentimentToScore(item.get("label"))
        if score is None:
            score = 0.0

        rows.append((canonical, score))

    return rows


def mergeSentimentSignals(signals):
    if len(signals) == 0:
        return {}

    totals = {}
    counts = {}
    for feature, score in signals:
        if feature not in totals:
            totals[feature] = 0.0
            counts[feature] = 0
        totals[feature] += score
        counts[feature] += 1

    merged = {}
    for feature in totals.keys():
        merged[feature] = totals[feature] / counts[feature] if counts[feature] > 0 else 0.0
    return merged


def getReviewAspectSentimentSignals(row):
    signals = []

    for aspect in ASPECT_COLUMNS:
        if aspect in row:
            score = sentimentToScore(row.get(aspect))
            if score is not None:
                signals.append((aspect, score))

    if "mined_aspects_json" in row:
        signals.extend(parseMinedAspects(row.get("mined_aspects_json")))

    return mergeSentimentSignals(signals)


def getReviewFeatureSignals(row, amenityFeatures=None):
    amenityFeatures = amenityFeatures if amenityFeatures is not None else []
    signals = getReviewAspectSentimentSignals(row)
    overallSentiment = getOverallReviewSentiment(row)

    normalizedReviewText = row.get("normalized_review_content")
    if isMissing(normalizedReviewText):
        normalizedReviewText = " " + normalizeText(row.get("review_content")) + " "

    for amenity in amenityFeatures:
        featureName = toFeatureName(amenity)
        if featureName not in signals and containsAmenityMention(normalizedReviewText, featureName):
            signals[featureName] = overallSentiment

    if "recommend_list" in row:
        for feature in parseRecommendList(row.get("recommend_list")):
            featureName = getCanonicalAspectName(feature)
            if len(featureName) > 0 and featureName not in signals:
                signals[featureName] = overallSentiment

    return signals


def getAspectSentimentProfile(hotelReviews):
    if hotelReviews is None or len(hotelReviews) == 0:
        return {}

    totals = {}
    counts = {}
    for _, row in hotelReviews.iterrows():
        signals = getReviewAspectSentimentSignals(row)
        for aspect in ASPECT_COLUMNS:
            if aspect in signals:
                if aspect not in totals:
                    totals[aspect] = 0.0
                    counts[aspect] = 0
                totals[aspect] += signals[aspect]
                counts[aspect] += 1

    profile = {}
    for aspect in ASPECT_COLUMNS:
        if aspect in totals and counts[aspect] > 0:
            profile[aspect] = totals[aspect] / counts[aspect]
    return profile


def getFeatureProfiles(hotelReviews, amenityFeatures=None):
    if hotelReviews is None or len(hotelReviews) == 0:
        return {}, {}, {}, {}

    amenityFeatures = amenityFeatures if amenityFeatures is not None else []
    sentimentTotals = {}
    mentionCounts = {}
    reviewCount = len(hotelReviews)

    for _, row in hotelReviews.iterrows():
        signals = getReviewFeatureSignals(row, amenityFeatures)
        for feature, score in signals.items():
            if feature not in sentimentTotals:
                sentimentTotals[feature] = 0.0
                mentionCounts[feature] = 0
            sentimentTotals[feature] += score
            mentionCounts[feature] += 1

    frequencyFeatureVector = {}
    sentimentFeatureVector = {}
    featureSentimentProfile = {}

    for feature in sorted(mentionCounts.keys()):
        count = mentionCounts[feature]
        frequencyFeatureVector[feature] = count / reviewCount if reviewCount > 0 else 0.0
        sentiment = sentimentTotals[feature] / count if count > 0 else 0.0
        sentimentFeatureVector[feature] = sentiment
        featureSentimentProfile[feature] = sentiment

    return frequencyFeatureVector, sentimentFeatureVector, featureSentimentProfile, mentionCounts


class Dataset:
    def __init__(self, hotels, hotelFrame=None, reviewFrame=None, memberFrame=None):
        self.hotels = hotels
        self.hotelFrame = hotelFrame
        self.reviewFrame = reviewFrame if reviewFrame is not None else pd.DataFrame()
        self.memberFrame = memberFrame
        self.featureNames = self.getAllFeatureNames()

    def getHotelIds(self):
        return list(self.hotels.keys())

    def hasKnownCity(self, city):
        return len(normalizeCityName(city)) > 0 and normalizeCityName(city) != "unknown"

    def getHotelIdsByCity(self, city):
        ids = []
        for hotelId, hotel in self.hotels.items():
            if citiesMatch(hotel.city, city):
                ids.append(hotelId)
        return ids

    def getHotel(self, hotelId):
        return self.hotels[str(hotelId)]

    def citiesMatch(self, left, right):
        return citiesMatch(left, right)

    def getRelatedHotels(self, hotelId):
        hotel = self.getHotel(hotelId)
        related = set(hotel.relatedHotels)
        if len(related) == 0 and self.reviewFrame is not None and "recommend_list" in self.reviewFrame.columns:
            rows = self.reviewFrame[self.reviewFrame["hotel_id"].astype(str) == str(hotelId)]
            for _, row in rows.iterrows():
                related.update(parseRecommendList(row.get("recommend_list")))
        return related

    def getAllFeatureNames(self):
        featureNames = set()
        for hotel in self.hotels.values():
            featureNames.update(hotel.binaryFeatureVector.keys())
            featureNames.update(hotel.frequencyFeatureVector.keys())
            featureNames.update(hotel.sentimentFeatureVector.keys())
            featureNames.update(hotel.featureSentimentProfile.keys())
        return sorted(featureNames)

    def getAmenityFeatureNames(self):
        featureNames = set()
        for hotel in self.hotels.values():
            featureNames.update(hotel.amenities)
        return sorted(featureNames)

    def getMaxReviewCount(self):
        if len(self.hotels) == 0:
            return 1
        return max(1, max(hotel.reviewCount for hotel in self.hotels.values()))

    def getDefaultTargetHotels(self, limit=5):
        rows = sorted(
            self.hotels.values(),
            key=lambda hotel: (-hotel.reviewCount, -hotel.averageRating, hotel.id),
        )
        return [hotel.id for hotel in rows[:limit]]

    def getEvaluableHotelIds(self):
        ids = []
        for hotelId, hotel in self.hotels.items():
            if hotel.reviewCount > 0:
                ids.append(hotelId)
        ids.sort(key=lambda hotelId: (-self.hotels[hotelId].reviewCount, hotelId))
        return ids

    def getProfileFrame(self):
        rows = []
        for hotel in self.hotels.values():
            rows.append({
                "hotel_id": hotel.id,
                "name": hotel.name,
                "city": hotel.city,
                "star": hotel.star,
                "price_range": hotel.priceRange,
                "price_level": hotel.priceLevel,
                "average_rating": hotel.averageRating,
                "review_count": hotel.reviewCount,
                "amenities_json": json.dumps(sorted(hotel.amenities)),
                "sentiment_profile_json": json.dumps(hotel.sentimentProfile, sort_keys=True),
                "binary_feature_vector_json": json.dumps(hotel.binaryFeatureVector, sort_keys=True),
                "frequency_feature_vector_json": json.dumps(hotel.frequencyFeatureVector, sort_keys=True),
                "sentiment_feature_vector_json": json.dumps(hotel.sentimentFeatureVector, sort_keys=True),
            })
        return pd.DataFrame(rows)

    def getFeatureProfileFrame(self):
        rows = []
        for hotel in self.hotels.values():
            features = set(hotel.frequencyFeatureVector.keys())
            features.update(hotel.sentimentFeatureVector.keys())
            features.update(hotel.featureSentimentProfile.keys())

            for feature in sorted(features):
                rows.append({
                    "hotel_id": hotel.id,
                    "feature": feature,
                    "frequency": hotel.frequencyFeatureVector.get(feature, 0.0),
                    "sentiment": hotel.sentimentFeatureVector.get(feature, 0.0),
                    "profile_sentiment": hotel.featureSentimentProfile.get(feature, 0.0),
                    "is_amenity": feature in hotel.amenities,
                })
        return pd.DataFrame(rows)

    def getSubset(self, hotelIds):
        selected = {}
        for hotelId in hotelIds:
            hotelId = str(hotelId)
            if hotelId in self.hotels:
                selected[hotelId] = self.hotels[hotelId]

        hotelFrame = None
        if self.hotelFrame is not None and "hotel_id" in self.hotelFrame.columns:
            hotelFrame = self.hotelFrame[self.hotelFrame["hotel_id"].astype(str).isin(selected.keys())].copy()

        reviewFrame = self.reviewFrame
        if self.reviewFrame is not None and "hotel_id" in self.reviewFrame.columns:
            reviewFrame = self.reviewFrame[self.reviewFrame["hotel_id"].astype(str).isin(selected.keys())].copy()

        return Dataset(selected, hotelFrame, reviewFrame, self.memberFrame)

    def getExpandedHotelIds(self, targetHotelIds, candidateHotelIds=None):
        candidateSet = set(str(hotelId) for hotelId in candidateHotelIds) if candidateHotelIds is not None else set(self.hotels.keys())
        expanded = set()

        for targetId in targetHotelIds:
            targetId = str(targetId)
            if targetId not in self.hotels:
                continue

            expanded.add(targetId)
            targetHotel = self.hotels[targetId]
            for candidateId in candidateSet:
                if candidateId in self.hotels and citiesMatch(self.hotels[candidateId].city, targetHotel.city):
                    expanded.add(candidateId)
            for relatedId in targetHotel.relatedHotels:
                relatedId = str(relatedId)
                if relatedId in self.hotels:
                    expanded.add(relatedId)

        return sorted(expanded)

    def getFilteredHotelIds(self, city=None, minStar=None, maxStar=None, priceRanges=None, requiredAmenities=None):
        ids = []
        for hotelId, hotel in self.hotels.items():
            if self.matchesFilters(hotel, city, minStar, maxStar, priceRanges, requiredAmenities):
                ids.append(hotelId)
        return sorted(ids)

    def matchesFilters(self, hotel, city=None, minStar=None, maxStar=None, priceRanges=None, requiredAmenities=None):
        if city is not None and not citiesMatch(hotel.city, city):
            return False

        if minStar is not None and (hotel.star is None or hotel.star < minStar):
            return False

        if maxStar is not None and (hotel.star is None or hotel.star > maxStar):
            return False

        if priceRanges is not None and len(priceRanges) > 0:
            allowed = set(normalizeText(priceRange) for priceRange in priceRanges)
            hotelPrice = normalizeText(hotel.priceRange)
            if hotelPrice not in allowed:
                return False

        if requiredAmenities is not None and len(requiredAmenities) > 0:
            hotelAmenities = set(toFeatureName(amenity) for amenity in hotel.amenities)
            for amenity in requiredAmenities:
                if toFeatureName(amenity) not in hotelAmenities and getCanonicalAspectName(amenity) not in hotel.binaryFeatureVector:
                    return False

        return True

    def getMemberReviews(self, memberId):
        if self.reviewFrame is None or "member_id" not in self.reviewFrame.columns:
            return pd.DataFrame()
        return self.reviewFrame[self.reviewFrame["member_id"].astype(str) == str(memberId)].copy()

    def getMembersWithMinReviews(self, minReviews):
        if self.reviewFrame is None or "member_id" not in self.reviewFrame.columns:
            return []

        counts = self.reviewFrame.groupby("member_id").size().sort_values(ascending=False)
        return [str(memberId) for memberId, count in counts.items() if count >= minReviews]

    def buildUserProfile(self, memberId, excludeHotelId=None):
        memberReviews = self.getMemberReviews(memberId)
        if excludeHotelId is not None and "hotel_id" in memberReviews.columns:
            memberReviews = memberReviews[memberReviews["hotel_id"].astype(str) != str(excludeHotelId)]

        if len(memberReviews) == 0:
            return None

        frequencyTotals = {}
        sentimentTotals = {}
        sentimentCounts = {}
        seenHotelIds = set()
        cityCounts = {}

        for _, row in memberReviews.iterrows():
            hotelId = str(row.get("hotel_id"))
            if hotelId in self.hotels:
                seenHotelIds.add(hotelId)
                hotel = self.hotels[hotelId]
                cityCounts[hotel.city] = cityCounts.get(hotel.city, 0) + 1
                ratingWeight = 1.0
                if "rating" in row and not isMissing(row.get("rating")):
                    try:
                        ratingWeight = max(0.2, min(1.0, float(row.get("rating")) / 5.0))
                    except (TypeError, ValueError):
                        ratingWeight = 1.0

                for feature, value in hotel.frequencyFeatureVector.items():
                    frequencyTotals[feature] = frequencyTotals.get(feature, 0.0) + ratingWeight * value

                for feature, value in hotel.sentimentFeatureVector.items():
                    sentimentTotals[feature] = sentimentTotals.get(feature, 0.0) + ratingWeight * value
                    sentimentCounts[feature] = sentimentCounts.get(feature, 0.0) + ratingWeight

        if len(seenHotelIds) == 0:
            return None

        reviewCount = len(seenHotelIds)
        frequencyProfile = {}
        for feature, value in frequencyTotals.items():
            frequencyProfile[feature] = value / reviewCount

        sentimentProfile = {}
        for feature, value in sentimentTotals.items():
            weight = sentimentCounts.get(feature, 0.0)
            if weight > 0:
                sentimentProfile[feature] = value / weight

        binaryFeatures = {}
        for feature in frequencyProfile.keys():
            binaryFeatures[feature] = 1.0

        city = "Unknown"
        if len(cityCounts) > 0:
            city = sorted(cityCounts.items(), key=lambda row: (-row[1], row[0]))[0][0]

        return Hotel(
            str(memberId),
            "Member " + str(memberId),
            city,
            None,
            None,
            0.0,
            len(memberReviews),
            set(),
            seenHotelIds,
            {},
            binaryFeatures,
            frequencyProfile,
            sentimentProfile,
            sentimentProfile,
            None,
            "user_profile",
        )
