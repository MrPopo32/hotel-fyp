"""
Print example hotel recommendations for quick manual checking.
"""

import argparse

from rec.dataset import Reader
from rec.hotel import HotelRecommender
from rec.hotel.evaluator import HotelEvaluator
from rec.hotel.ranker import CaseBasedRanker
from rec.hotel.ranker import FrequencyBasedRanker
from rec.hotel.ranker import SentimentEnhancedRanker


def getRankers(profileMode, sentimentWeight) :
	if profileMode == "case":
		return [("Case-based Metadata", CaseBasedRanker())]

	if profileMode == "frequency":
		return [("Frequency-based", FrequencyBasedRanker())]

	if profileMode == "sentiment":
		return [("Sentiment-enhanced", SentimentEnhancedRanker(sentimentWeight))]

	return [
		("Case-based Metadata", CaseBasedRanker()),
		("Frequency-based", FrequencyBasedRanker()),
		("Sentiment-enhanced", SentimentEnhancedRanker(sentimentWeight)),
	]


def main() :
	parser = argparse.ArgumentParser()
	parser.add_argument("--k", type = int, default = 10)
	parser.add_argument("--hotel-id", action = "append", default = None)
	parser.add_argument("--city", default = None)
	parser.add_argument("--min-star", type = float, default = None)
	parser.add_argument("--max-star", type = float, default = None)
	parser.add_argument("--price-range", action = "append", default = None)
	parser.add_argument("--amenity", action = "append", default = None)
	parser.add_argument("--profile-mode", choices = ["case", "frequency", "sentiment"], default = None)
	parser.add_argument("--sentiment-weight", type = float, default = 0.5)
	parser.add_argument("--diversity", action = "store_true")
	parser.add_argument("--diversity-pool", type = int, default = 50)
	parser.add_argument("--diversity-lambda", type = float, default = 0.8)
	args = parser.parse_args()

	reader = Reader()
	fullDataset = reader.getDataset()
	activeFilters = args.city is not None or args.min_star is not None or args.max_star is not None or args.price_range is not None or args.amenity is not None

	# Busy hotels make the demo output more useful by default.
	hotelIds = args.hotel_id if args.hotel_id is not None else fullDataset.getDefaultTargetHotels(5)
	filteredHotelIds = fullDataset.getFilteredHotelIds(
		args.city,
		args.min_star,
		args.max_star,
		args.price_range,
		args.amenity
	)

	candidateHotelIds = filteredHotelIds if activeFilters else fullDataset.getHotelIds()

	# Keep the demo subset small but still useful for relevance checks.
	expandedHotelIds = fullDataset.getExpandedHotelIds(hotelIds, candidateHotelIds)

	dataset = fullDataset.getSubset(expandedHotelIds) if len(expandedHotelIds) > 0 else fullDataset

	targetHotelIds = []
	for hotelId in hotelIds :
		if hotelId in dataset.hotels:
			targetHotelIds.append(hotelId)

	rankerRows = getRankers(args.profile_mode, args.sentiment_weight)
	for label, ranker in rankerRows :
		print("\n")
		print("Running with: " + label)
		print("\n")
		if activeFilters:
			print("Filters: city=%s minStar=%s maxStar=%s priceRange=%s amenity=%s" %
					(str(args.city), str(args.min_star), str(args.max_star),
					str(args.price_range), str(args.amenity)))
			print()
		if args.diversity:
			print("Diversity reranking: top-%d from reranked top-%d pool (lambda=%.2f)" %
					(args.k, args.diversity_pool, args.diversity_lambda))
			print()

		model = HotelRecommender(
			dataset = dataset,
			ranker = ranker,
			# Keep display filters active inside the recommender too.
			candidateHotelIds = filteredHotelIds if activeFilters else None,
			useDiversity = args.diversity,
			diversityPoolSize = args.diversity_pool,
			diversityTopK = args.k,
			diversityLambda = args.diversity_lambda
		)
		model.fit()

		eval = HotelEvaluator(model, dataset, args.k, targetHotelIds)
		for hotelId in targetHotelIds:
			eval.printRecs(hotelId)


if __name__ == "__main__":
	main()
