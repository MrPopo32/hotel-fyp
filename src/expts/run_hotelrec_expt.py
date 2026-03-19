"""
Run the main recommender experiments and print metric rows.
"""

import argparse

from rec.dataset import Reader
from rec.hotel import HotelRecommender
from rec.hotel.evaluator import HotelEvaluator
from rec.hotel.evaluator import PersonalisedHotelEvaluator
from rec.hotel.ranker import CaseBasedRanker
from rec.hotel.ranker import FrequencyBasedRanker
from rec.hotel.ranker import SentimentEnhancedRanker


def getRankers(task, profileMode, sentimentWeight) :
	# Explicit ranker choices keep the experiment setup easy to read.
	if profileMode == "case":
		return [("Case-based Metadata", CaseBasedRanker())]

	if profileMode == "frequency":
		return [("Frequency-based", FrequencyBasedRanker())]

	if profileMode == "sentiment":
		return [("Sentiment-enhanced", SentimentEnhancedRanker(sentimentWeight))]

	if task == "personalized":
		return [
			("Frequency-based", FrequencyBasedRanker()),
			("Sentiment-enhanced", SentimentEnhancedRanker(sentimentWeight)),
		]

	return [
		("Case-based Metadata", CaseBasedRanker()),
		("Frequency-based", FrequencyBasedRanker()),
		("Sentiment-enhanced", SentimentEnhancedRanker(sentimentWeight)),
	]


def main() :
	parser = argparse.ArgumentParser()
	parser.add_argument("--task", choices = ["non_personalized", "personalized"], default = "non_personalized")
	parser.add_argument("--k", type = int, default = 10)
	parser.add_argument("--max-hotels", type = int, default = None)
	parser.add_argument("--max-users", type = int, default = None)
	parser.add_argument("--max-targets", type = int, default = None)
	parser.add_argument("--min-user-reviews", type = int, default = 5)
	parser.add_argument("--min-heldout-rating", type = float, default = 4.0)
	parser.add_argument("--heldout-sentiment", action = "append", default = None)
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

	filteredHotelIds = fullDataset.getFilteredHotelIds(
		args.city,
		args.min_star,
		args.max_star,
		args.price_range,
		args.amenity
	)

	rankerRows = getRankers(args.task, args.profile_mode, args.sentiment_weight)

	if args.task == "personalized":
		print("k,Mode,Diversity,Targets,HitRate,MRR,Novelty,Serendipity")
		for label, ranker in rankerRows:
			model = HotelRecommender(
				dataset = fullDataset,
				ranker = ranker,
				# Metadata filters should apply to personalised runs too.
				candidateHotelIds = filteredHotelIds if activeFilters else None,
				useDiversity = args.diversity,
				diversityPoolSize = args.diversity_pool,
				diversityTopK = args.k,
				diversityLambda = args.diversity_lambda
			)
			model.fit()

			eval = PersonalisedHotelEvaluator(
				model,
				fullDataset,
				args.k,
				args.min_user_reviews,
				args.max_users,
				args.max_targets,
				args.min_heldout_rating,
				args.heldout_sentiment
			)
			print("%d,%s,%s,%d,%.4f,%.4f" %
					(args.k, label, str(args.diversity),
					 eval.getTargetCount(),
					 eval.getHitRate(),
					 eval.getMRR()) +
					(",%.4f,%.4f" %
					 (eval.getNovelty(),
					  eval.getSerendipity())))
		return

	targetHotelIds = fullDataset.getEvaluableHotelIds()
	if activeFilters:
		targetHotelIds = [hotelId for hotelId in targetHotelIds if hotelId in filteredHotelIds]
	if args.max_hotels is not None:
		targetHotelIds = targetHotelIds[:args.max_hotels]

	candidateHotelIds = filteredHotelIds if activeFilters else fullDataset.getHotelIds()

	# Smaller eval dataset: targets plus realistic candidates.
	expandedHotelIds = fullDataset.getExpandedHotelIds(targetHotelIds, candidateHotelIds)

	dataset = fullDataset.getSubset(expandedHotelIds) if len(expandedHotelIds) > 0 else fullDataset

	filteredTargetHotelIds = []
	for hotelId in targetHotelIds :
		if(hotelId in dataset.hotels):
			filteredTargetHotelIds.append(hotelId)

	print("k,Mode,Diversity,Relevance,Coverage,RecCoverage,CityMatch,SentimentAlignment,RecRating,Novelty,Serendipity")

	for label, ranker in rankerRows :
		model = HotelRecommender(
			dataset = dataset,
			ranker = ranker,
			# Evaluator should measure the same filtered slice the user requested.
			candidateHotelIds = filteredHotelIds if activeFilters else None,
			useDiversity = args.diversity,
			diversityPoolSize = args.diversity_pool,
			diversityTopK = args.k,
			diversityLambda = args.diversity_lambda
		)
		model.fit()

		eval = HotelEvaluator(model, dataset, args.k, filteredTargetHotelIds)
		print("%d,%s,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f" %
				(args.k, label, str(args.diversity),
				 eval.getRecommendationRelevance(),
				 eval.getCoverage(),
				 eval.getRecommendationCoverage(),
				 eval.getCityMatchRate(),
				 eval.getRecommendationSentimentAlignment(),
				 eval.getRecommendationRating(),
				 eval.getRecommendationNovelty(),
				 eval.getRecommendationSerendipity()))


if __name__ == "__main__":
	main()
