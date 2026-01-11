"""
Shared profile object for real hotels and user query profiles.
"""


class Hotel:
	def __init__(
		self,
		id,
		name,
		city,
		star,
		priceRange,
		averageRating,
		reviewCount,
		amenities,
		relatedHotels,
		sentimentProfile,
		binaryFeatureVector = None,
		frequencyFeatureVector = None,
		sentimentFeatureVector = None,
		featureSentimentProfile = None,
		priceLevel = None,
		sourceType = "hotel",
	):
		self.id = id
		self.name = name
		self.city = city
		self.star = star
		self.priceRange = priceRange
		self.averageRating = averageRating
		self.reviewCount = reviewCount
		self.amenities = amenities
		self.relatedHotels = relatedHotels

		self.sentimentProfile = sentimentProfile if sentimentProfile is not None else {}
		self.binaryFeatureVector = binaryFeatureVector if binaryFeatureVector is not None else {}
		self.frequencyFeatureVector = frequencyFeatureVector if frequencyFeatureVector is not None else {}
		self.sentimentFeatureVector = sentimentFeatureVector if sentimentFeatureVector is not None else {}
		self.featureSentimentProfile = featureSentimentProfile if featureSentimentProfile is not None else {}
		self.priceLevel = priceLevel
		self.sourceType = sourceType

	def __str__(self):
		s = ""
		s += self.name + "\n"
		s += "   sourceType: " + str(self.sourceType) + "\n"
		s += "   city: " + str(self.city) + "\n"
		s += "   star: " + str(self.star) + "\n"
		s += "   priceRange: " + str(self.priceRange) + "\n"
		s += "   averageRating: " + str(self.averageRating) + "\n"
		s += "   reviewCount: " + str(self.reviewCount) + "\n"
		s += "   amenities: " + str(self.amenities) + "\n"
		s += "   relatedHotels: " + str(self.relatedHotels) + "\n"
		s += "   binaryFeatures: " + str(len(self.binaryFeatureVector)) + "\n"
		s += "   frequencyFeatures: " + str(len(self.frequencyFeatureVector)) + "\n"
		s += "   sentimentFeatures: " + str(len(self.sentimentFeatureVector)) + "\n"
		s += "   id: " + self.id + "\n"
		return s
