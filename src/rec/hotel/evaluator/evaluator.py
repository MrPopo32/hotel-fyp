"""
Lightweight evaluation metrics for hotel recommendations.
"""

import math

from rec.util.math import getJaccardSimilarity
from rec.util.math import getNormalizedCosineSimilarity


class HotelEvaluator:
    def __init__(self, model, dataset, k, targetHotelIds):
        self.model = model
        self.dataset = dataset
        self.k = k
        self.targetHotelIds = list(targetHotelIds)
        self.recommendationCache = {}

    def getTopKRecs(self, hotelId):
        hotelId = str(hotelId)
        if hotelId not in self.recommendationCache:
            self.recommendationCache[hotelId] = self.model.recommend(hotelId)[:self.k]
        return list(self.recommendationCache[hotelId])

    def getHotelNovelty(self, hotelId):
        hotel = self.dataset.getHotel(hotelId)
        maxReviewCount = self.dataset.getMaxReviewCount()
        if maxReviewCount <= 1:
            return 0.0
        popularity = math.log1p(hotel.reviewCount) / math.log1p(maxReviewCount)
        novelty = 1.0 - popularity
        if novelty < 0:
            return 0.0
        if novelty > 1:
            return 1.0
        return novelty

    def getContentDistance(self, profile, hotelId):
        hotel = self.dataset.getHotel(hotelId)
        binarySimilarity = getJaccardSimilarity(
            set(profile.binaryFeatureVector.keys()),
            set(hotel.binaryFeatureVector.keys()),
        )
        sentimentSimilarity = getNormalizedCosineSimilarity(
            profile.sentimentFeatureVector,
            hotel.sentimentFeatureVector,
        )
        similarity = 0.5 * binarySimilarity + 0.5 * sentimentSimilarity
        distance = 1.0 - similarity
        if distance < 0:
            return 0.0
        if distance > 1:
            return 1.0
        return distance

    def getUnexpectedness(self, profile, hotelId):
        return self.getContentDistance(profile, hotelId)

    def getCoverage(self):
        if len(self.targetHotelIds) == 0:
            return 0.0

        covered = 0
        for hotelId in self.targetHotelIds:
            if len(self.getTopKRecs(hotelId)) > 0:
                covered += 1
        return covered / len(self.targetHotelIds)

    def getRecommendationCoverage(self):
        if len(self.dataset.hotels) == 0:
            return 0.0

        uniqueRecs = set()
        for hotelId in self.targetHotelIds:
            uniqueRecs.update(self.getTopKRecs(hotelId))
        return len(uniqueRecs) / len(self.dataset.hotels)

    def getRecommendationRelevance(self):
        total = 0
        relevant = 0

        for hotelId in self.targetHotelIds:
            relatedHotels = self.dataset.getRelatedHotels(hotelId)
            recIds = self.getTopKRecs(hotelId)
            for recId in recIds:
                total += 1
                if recId in relatedHotels:
                    relevant += 1

        return relevant / total if total > 0 else 0.0

    def getCityMatchRate(self):
        total = 0
        matches = 0

        for hotelId in self.targetHotelIds:
            targetHotel = self.dataset.getHotel(hotelId)
            for recId in self.getTopKRecs(hotelId):
                total += 1
                recHotel = self.dataset.getHotel(recId)
                if self.dataset.citiesMatch(targetHotel.city, recHotel.city):
                    matches += 1

        return matches / total if total > 0 else 0.0

    def getRecommendationSentimentAlignment(self):
        total = 0.0
        count = 0

        for hotelId in self.targetHotelIds:
            targetHotel = self.dataset.getHotel(hotelId)
            for recId in self.getTopKRecs(hotelId):
                recHotel = self.dataset.getHotel(recId)
                total += getNormalizedCosineSimilarity(
                    targetHotel.sentimentFeatureVector,
                    recHotel.sentimentFeatureVector,
                )
                count += 1

        return total / count if count > 0 else 0.0

    def getRecommendationRating(self):
        total = 0.0
        count = 0

        for hotelId in self.targetHotelIds:
            for recId in self.getTopKRecs(hotelId):
                total += self.dataset.getHotel(recId).averageRating
                count += 1

        return total / count if count > 0 else 0.0

    def getRecommendationNovelty(self):
        total = 0.0
        count = 0

        for hotelId in self.targetHotelIds:
            for recId in self.getTopKRecs(hotelId):
                total += self.getHotelNovelty(recId)
                count += 1

        return total / count if count > 0 else 0.0

    def getRecommendationSerendipity(self):
        total = 0.0
        count = 0

        for hotelId in self.targetHotelIds:
            targetHotel = self.dataset.getHotel(hotelId)
            relatedHotels = self.dataset.getRelatedHotels(hotelId)
            for recId in self.getTopKRecs(hotelId):
                relevance = 1.0 if recId in relatedHotels else 0.0
                unexpectedness = self.getUnexpectedness(targetHotel, recId)
                novelty = self.getHotelNovelty(recId)
                total += relevance * unexpectedness * novelty
                count += 1

        return total / count if count > 0 else 0.0

    def printRecs(self, hotelId):
        targetHotel = self.dataset.getHotel(hotelId)
        print(str(targetHotel))
        print("Recommendations:")
        for index, recId in enumerate(self.getTopKRecs(hotelId), start=1):
            recHotel = self.dataset.getHotel(recId)
            print(
                "%d. %s | city=%s | rating=%.2f | reviews=%d | id=%s"
                % (index, recHotel.name, recHotel.city, recHotel.averageRating, recHotel.reviewCount, recHotel.id)
            )
        print()

    def printRecommendations(self, hotelId):
        self.printRecs(hotelId)


class PersonalisedHotelEvaluator:
    def __init__(
        self,
        model,
        dataset,
        k,
        minUserReviews=5,
        maxUsers=None,
        maxTargets=None,
        minHeldoutRating=4.0,
        heldoutSentiments=None,
    ):
        self.model = model
        self.dataset = dataset
        self.k = k
        self.minUserReviews = minUserReviews
        self.maxUsers = maxUsers
        self.maxTargets = maxTargets
        self.minHeldoutRating = minHeldoutRating
        self.heldoutSentiments = set(heldoutSentiments) if heldoutSentiments is not None else None
        self.results = self.run()

    def getHotelNovelty(self, hotelId):
        maxReviewCount = self.dataset.getMaxReviewCount()
        if maxReviewCount <= 1:
            return 0.0
        hotel = self.dataset.getHotel(hotelId)
        return 1.0 - (math.log1p(hotel.reviewCount) / math.log1p(maxReviewCount))

    def getContentDistance(self, profile, hotelId):
        hotel = self.dataset.getHotel(hotelId)
        binarySimilarity = getJaccardSimilarity(
            set(profile.binaryFeatureVector.keys()),
            set(hotel.binaryFeatureVector.keys()),
        )
        sentimentSimilarity = getNormalizedCosineSimilarity(
            profile.sentimentFeatureVector,
            hotel.sentimentFeatureVector,
        )
        return 1.0 - (0.5 * binarySimilarity + 0.5 * sentimentSimilarity)

    def getUnexpectedness(self, profile, hotelId):
        distance = self.getContentDistance(profile, hotelId)
        if distance < 0:
            return 0.0
        if distance > 1:
            return 1.0
        return distance

    def isEligibleHeldOutReview(self, row):
        if "hotel_id" not in row:
            return False
        hotelId = str(row.get("hotel_id"))
        if hotelId not in self.dataset.hotels:
            return False

        if "rating" in row:
            try:
                if float(row.get("rating")) < self.minHeldoutRating:
                    return False
            except (TypeError, ValueError):
                return False

        if self.heldoutSentiments is not None:
            label = str(row.get("overall_sentiment_label", "")).lower()
            if label not in self.heldoutSentiments:
                return False

        return True

    def run(self):
        results = []
        memberIds = self.dataset.getMembersWithMinReviews(self.minUserReviews)
        if self.maxUsers is not None:
            memberIds = memberIds[:self.maxUsers]

        for memberId in memberIds:
            memberReviews = self.dataset.getMemberReviews(memberId)
            eligibleRows = []
            for _, row in memberReviews.iterrows():
                if self.isEligibleHeldOutReview(row):
                    eligibleRows.append(row)

            for row in eligibleRows:
                if self.maxTargets is not None and len(results) >= self.maxTargets:
                    return results

                heldoutHotelId = str(row.get("hotel_id"))
                profile = self.dataset.buildUserProfile(memberId, heldoutHotelId)
                if profile is None:
                    continue

                recIds = self.model.recommendForProfile(profile, excludeIds=profile.relatedHotels)[:self.k]
                rank = self.getRank(recIds, heldoutHotelId)
                results.append({
                    "member_id": str(memberId),
                    "heldout_hotel_id": heldoutHotelId,
                    "rank": rank,
                    "hit": 1 if rank is not None else 0,
                    "novelty": self.getHotelNovelty(heldoutHotelId),
                    "serendipity": self.getUnexpectedness(profile, heldoutHotelId) * self.getHotelNovelty(heldoutHotelId),
                })

        return results

    def getRank(self, recIds, hotelId):
        for index, recId in enumerate(recIds, start=1):
            if str(recId) == str(hotelId):
                return index
        return None

    def getTargetCount(self):
        return len(self.results)

    def getHitRate(self):
        if len(self.results) == 0:
            return 0.0
        return sum(row["hit"] for row in self.results) / len(self.results)

    def getMRR(self):
        if len(self.results) == 0:
            return 0.0

        total = 0.0
        for row in self.results:
            if row["rank"] is not None:
                total += 1.0 / row["rank"]
        return total / len(self.results)

    def getNovelty(self):
        if len(self.results) == 0:
            return 0.0
        return sum(row["novelty"] for row in self.results) / len(self.results)

    def getSerendipity(self):
        if len(self.results) == 0:
            return 0.0
        return sum(row["serendipity"] for row in self.results) / len(self.results)


UserProfileEvaluator = PersonalisedHotelEvaluator
