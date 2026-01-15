"""
Read parquet tables and build recommender-friendly Hotel objects.
"""

import json

import pandas as pd
import pyarrow.parquet as pq

from rec.util.io import INTERIM_DIR
from rec.util.io import PROCESSED_DIR
from rec.util.text import cleanText
from rec.util.text import normalizeWithPadding
from rec.util.text import toAmenitySet
from rec.util.text import toFeatureName
from rec.util.text import toRelatedHotelSet

from .dataset import Dataset
from .dataset import getAspectSentimentProfile
from .dataset import getFeatureProfiles
from .dataset import getPriceLevel
from .entities import Hotel


class Reader:
	def __init__(self, hotelFile = None, memberFile = None, reviewFile = None, reviewSentimentFile = None) :
		self.hotelFile = hotelFile if hotelFile is not None else INTERIM_DIR / "hotel.parquet"
		self.memberFile = memberFile if memberFile is not None else INTERIM_DIR / "member.parquet"
		self.reviewFile = reviewFile if reviewFile is not None else INTERIM_DIR / "review.parquet"
		self.reviewSentimentFile = reviewSentimentFile if reviewSentimentFile is not None else PROCESSED_DIR / "review_sentiment.parquet"
		self.hotelProfileFile = PROCESSED_DIR / "hotel_profiles.parquet"
		self.dataset = self.readDataset()

	def getDataset(self) :
		return self.dataset

	def readDataset(self) :
		# Only load columns used later in the pipeline.
		hotelColumns = [
			"hotel_id",
			"name",
			"city",
			"locality",
			"star",
			"price_range",
			"average_rating",
			"review_num",
			"amenities",
			"all_amenities",
			"related_hotels",
		]
		hotelFrame = pd.read_parquet(self.hotelFile, columns = self.getExistingColumns(self.hotelFile, hotelColumns)).copy()

		memberFrame = None
		if(self.memberFile.exists()):
			memberColumns = ["member_id", "name", "city", "review_num"]
			memberFrame = pd.read_parquet(self.memberFile, columns = self.getExistingColumns(self.memberFile, memberColumns)).copy()

		# Prefer enriched reviews when they already exist.
		reviewSourceFile = self.reviewSentimentFile if self.reviewSentimentFile.exists() else self.reviewFile
		reviewColumns = [
			"review_id",
			"member_id",
			"hotel_id",
			"city",
			"rating",
			"review_text",
			"review_content",
			"recommend_list",
			"mined_aspects_json",
			"overall_sentiment_label",
			"overall_sentiment_score",
			"cleanliness",
			"room_quality",
			"amenities",
			"breakfast",
			"wifi",
			"noise",
		]
		reviewFrame = self.readReviewFrame(reviewSourceFile, reviewColumns)

		# Normalize ids once so comparisons stay consistent.
		hotelFrame["hotel_id"] = hotelFrame["hotel_id"].astype(str)
		reviewFrame["hotel_id"] = reviewFrame["hotel_id"].astype(str)
		reviewFrame["review_id"] = reviewFrame["review_id"].astype(str)

		if "member_id" in reviewFrame.columns:
			reviewFrame["member_id"] = reviewFrame["member_id"].astype(str)

		if self.canUseHotelProfileCache(reviewSourceFile):
			hotels = self.readCachedHotels(hotelFrame)
			return Dataset(hotels, hotelFrame, reviewFrame, memberFrame)

		# Group once; hotel construction reuses these groups a lot.
		reviewGroups = reviewFrame.groupby("hotel_id", sort = False)
		hotels = self.readHotels(hotelFrame, reviewGroups)

		return Dataset(hotels, hotelFrame, reviewFrame, memberFrame )

	def canUseHotelProfileCache(self, reviewSourceFile) :
		if not self.hotelProfileFile.exists():
			return False
		if not reviewSourceFile.exists():
			return False
		return self.hotelProfileFile.stat().st_mtime >= reviewSourceFile.stat().st_mtime

	def readCachedHotels(self, hotelFrame) :
		profileFrame = pd.read_parquet(self.hotelProfileFile).copy()
		profileFrame["hotel_id"] = profileFrame["hotel_id"].astype(str)
		profiles = {}
		for _, row in profileFrame.iterrows():
			profiles[str(row["hotel_id"])] = row

		hotels = {}
		for _, row in hotelFrame.iterrows():
			hotelId = str(row["hotel_id"])
			if hotelId not in profiles:
				continue

			profile = profiles[hotelId]
			amenities = set(self.fromJson(profile.get("amenities_json"), []))
			sentimentVector = self.fromJson(profile.get("sentiment_feature_vector_json"), {})

			hotel = Hotel(
				hotelId,
				cleanText(profile.get("name")) or hotelId,
				cleanText(profile.get("city")) or cleanText(row.get("city")) or cleanText(row.get("locality")) or "Unknown",
				self.toFloat(profile.get("star")),
				cleanText(profile.get("price_range")),
				self.toFloat(profile.get("average_rating")) or 0.0,
				self.toInt(profile.get("review_count")),
				amenities,
				toRelatedHotelSet(row.get("related_hotels")),
				self.fromJson(profile.get("sentiment_profile_json"), {}),
				self.fromJson(profile.get("binary_feature_vector_json"), {}),
				self.fromJson(profile.get("frequency_feature_vector_json"), {}),
				sentimentVector,
				sentimentVector,
				self.toFloat(profile.get("price_level")),
				"hotel",
			)
			hotels[hotelId] = hotel

		return hotels

	def fromJson(self, value, defaultValue) :
		if pd.isna(value):
			return defaultValue
		try:
			return json.loads(value)
		except (TypeError, ValueError):
			return defaultValue

	def readHotels(self, hotelFrame, reviewGroups) :
		hotels = {}
		groupKeys = set(reviewGroups.groups.keys())

		for _, row in hotelFrame.iterrows() :
			hotelId = str(row["hotel_id"])
			hotelReviews = reviewGroups.get_group(hotelId) if hotelId in groupKeys else None

			# Metadata rating first, review average as fallback.
			averageRating = 0.0
			metadataRating = self.toFloat(row.get("average_rating"))
			if(pd.notna(metadataRating)):
				averageRating = float(metadataRating)
			elif hotelReviews is not None and "rating" in hotelReviews.columns:
				ratings = pd.to_numeric(hotelReviews["rating"], errors = "coerce")
				if ratings.notna().sum() > 0:
					averageRating = float(ratings.mean())

			# Trust the bigger count between metadata and loaded rows.
			reviewCount = 0
			metadataReviews = self.toInt(row.get("review_num"))
			if metadataReviews > reviewCount:
				reviewCount = metadataReviews
			if(hotelReviews is not None and len(hotelReviews) > reviewCount):
				reviewCount = len(hotelReviews)

			star = None
			starValue = self.toFloat(row.get("star"))
			if pd.notna(starValue) :
				star = float(starValue)

			name = cleanText(row.get("name"))
			if(name is None):
				name = hotelId

			# Different extracts use different city fields.
			city = cleanText(row.get("city"))
			if city is None :
				city = cleanText(row.get("locality"))
			if city is None:
				city = "Unknown"

			priceRange = cleanText(row.get("price_range"))

			# Older extracts used both amenity column names.
			amenityText = cleanText(row.get("amenities"))
			if amenityText is None:
				amenityText = cleanText(row.get("all_amenities"))

			amenities = set()
			for amenity in toAmenitySet(amenityText):
				amenities.add(toFeatureName(amenity))

			# Fixed aspects and mined feature vectors stay separate.
			sentimentProfile = getAspectSentimentProfile(hotelReviews)
			frequencyFeatureVector, sentimentFeatureVector, featureSentimentProfile, mentionCounts = getFeatureProfiles(
				hotelReviews,
				sorted(amenities)
			)

			binaryFeatureVector = {}
			for feature in amenities:
				binaryFeatureVector[feature] = 1.0

			# Review-mined features also count as binary features.
			for feature in mentionCounts.keys():
				binaryFeatureVector[feature] = 1.0

			hotel = Hotel(
				hotelId,
				name,
				city,
				star,
				priceRange,
				averageRating,
				reviewCount,
				amenities,
				toRelatedHotelSet(row.get("related_hotels")),
				sentimentProfile,
				binaryFeatureVector,
				frequencyFeatureVector,
				sentimentFeatureVector,
				featureSentimentProfile,
				getPriceLevel(priceRange),
				"hotel",
			)
			hotels[hotelId] = hotel

		return hotels

	def readReviewFrame(self, reviewSourceFile, reviewColumns) :
		selectedColumns = self.getExistingColumns(reviewSourceFile, reviewColumns)
		reviewFrame = pd.read_parquet(reviewSourceFile, columns = selectedColumns).copy()

		# Normalize the old `review_text` name to `review_content`.
		if "review_text" in reviewFrame.columns and "review_content" not in reviewFrame.columns:
			reviewFrame = reviewFrame.rename(columns = {"review_text": "review_content"})

		# Precompute text cleanup because feature scanning repeats it.
		if "review_content" in reviewFrame.columns and "normalized_review_content" not in reviewFrame.columns:
			reviewFrame["normalized_review_content"] = reviewFrame["review_content"].apply(normalizeWithPadding)

		return reviewFrame

	def getExistingColumns(self, parquetFile, columns) :
		existing = set(pq.ParquetFile(parquetFile).schema.names)

		# Keep requested order; easier to inspect dataframes later.
		selected = []
		for column in columns:
			if column in existing:
				selected.append(column)
		return selected


	def toFloat(self, value) :
		if(pd.isna(value)):
			return None

		try:
			return float(value)
		except:
			return None

	def toInt(self, value) :
		if pd.isna(value):
			return 0

		try:
			return int(float(value))
		except:
			return 0
