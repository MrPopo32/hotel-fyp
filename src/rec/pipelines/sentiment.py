"""
Simple OpenAI review sentiment enrichment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from rec.util.aspect_aliases import getCanonicalAspectName
from rec.util.io import INTERIM_DIR
from rec.util.io import PROCESSED_DIR
from rec.util.text import cleanText


DEFAULT_INPUT = INTERIM_DIR / "review.parquet"
DEFAULT_OUTPUT = PROCESSED_DIR / "review_sentiment.parquet"
DEFAULT_DIMENSIONS_PATH = PROCESSED_DIR / "aspect_dimensions.json"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_BATCH_SIZE = 25
DEFAULT_MAX_OUTPUT_TOKENS = 1400
DEFAULT_REVIEW_TEXT_MAX_CHARS = 3000

ASPECT_SENTIMENT_FIELDS = [
    "cleanliness",
    "room_quality",
    "amenities",
    "breakfast",
    "wifi",
    "noise",
]

CONTROLLED_ASPECT_DIMENSIONS = [
    "cleanliness",
    "room_quality",
    "staff",
    "location",
    "value",
    "noise",
    "food",
    "amenities",
    "business_center",
    "check_in",
]

SENTIMENT_OUTPUT_FIELDS = [
    "overall_sentiment_label",
    "overall_sentiment_score",
    *ASPECT_SENTIMENT_FIELDS,
    "mined_aspects_json",
    "analysis_model",
    "analysis_request_id",
    "analysis_error",
]

ALLOWED_LABELS = {"positive", "negative", "neutral", "mixed"}
ALLOWED_ASPECT_LABELS = ALLOWED_LABELS | {"not_mentioned"}


def parse_args():
    parser = argparse.ArgumentParser(description="Enrich hotel reviews with sentiment signals.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--review-text-max-chars", type=int, default=DEFAULT_REVIEW_TEXT_MAX_CHARS)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dimensions-path", type=Path, default=DEFAULT_DIMENSIONS_PATH)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if args.start_row < 0:
        raise SystemExit("--start-row cannot be negative.")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit cannot be negative.")
    return args


def load_reviews(path):
    frame = pd.read_parquet(path)
    if "review_text" in frame.columns and "review_content" not in frame.columns:
        frame = frame.rename(columns={"review_text": "review_content"})
    for field in SENTIMENT_OUTPUT_FIELDS:
        if field not in frame.columns:
            frame[field] = None
    return frame


def normalize_label(value, default="neutral"):
    label = str(value or default).lower().strip()
    return label if label in ALLOWED_LABELS else default


def normalize_aspect_label(value):
    label = str(value or "not_mentioned").lower().strip()
    return label if label in ALLOWED_ASPECT_LABELS else "not_mentioned"


def normalize_score(value, label):
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = {
            "positive": 0.7,
            "mixed": 0.0,
            "neutral": 0.0,
            "negative": -0.7,
        }.get(label, 0.0)
    return max(-1.0, min(1.0, score))


def normalize_mined_aspects(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = []
    if not isinstance(value, list):
        value = []

    rows = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        aspect = getCanonicalAspectName(item.get("aspect"))
        if aspect not in CONTROLLED_ASPECT_DIMENSIONS or aspect in seen:
            continue
        sentiment = normalize_label(item.get("sentiment"))
        rows.append({
            "aspect": aspect,
            "sentiment": sentiment,
            "score": normalize_score(item.get("score"), sentiment),
        })
        seen.add(aspect)
    return rows


def normalize_result(raw):
    if not isinstance(raw, dict):
        raw = {}

    label = normalize_label(raw.get("overall_sentiment_label") or raw.get("sentiment"))
    mined_aspects = normalize_mined_aspects(raw.get("mined_aspects"))

    result = {
        "overall_sentiment_label": label,
        "overall_sentiment_score": normalize_score(raw.get("overall_sentiment_score"), label),
        "mined_aspects_json": json.dumps(mined_aspects, sort_keys=True),
        "analysis_error": "",
    }

    fixed_aspects = raw.get("fixed_aspects")
    if not isinstance(fixed_aspects, dict):
        fixed_aspects = {}

    for field in ASPECT_SENTIMENT_FIELDS:
        result[field] = normalize_aspect_label(raw.get(field) or fixed_aspects.get(field))
    return result


def make_review_payload(frame, indices, max_chars):
    payload = []
    for index in indices:
        row = frame.iloc[index]
        text = cleanText(row.get("review_content"))
        if text is None:
            text = ""
        if max_chars is not None and max_chars > 0:
            text = text[:max_chars]
        payload.append({
            "row_index": int(index),
            "rating": safe_json_value(row.get("rating")),
            "city": safe_json_value(row.get("city")),
            "review": text,
        })
    return payload


def safe_json_value(value):
    if pd.isna(value):
        return ""
    return value


def build_messages(review_payload):
    dimensions = ", ".join(CONTROLLED_ASPECT_DIMENSIONS)
    aspects = ", ".join(ASPECT_SENTIMENT_FIELDS)
    system_prompt = (
        "You extract hotel review sentiment. Return JSON only. "
        "The response must be an object with a reviews array. "
        "Each item must contain row_index, overall_sentiment_label, "
        "overall_sentiment_score, these fixed aspect labels: "
        + aspects
        + ", and mined_aspects. "
        "Labels must be positive, negative, neutral, or mixed. "
        "For fixed aspects use not_mentioned when absent. "
        "overall_sentiment_score and mined aspect scores must be between -1 and 1. "
        "mined_aspects must contain only these aspect names: "
        + dimensions
        + "."
    )
    user_prompt = json.dumps({"reviews": review_payload}, ensure_ascii=True)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_openai_batch(client, args, review_payload):
    response = client.chat.completions.create(
        model=args.model,
        messages=build_messages(review_payload),
        response_format={"type": "json_object"},
        max_completion_tokens=args.max_output_tokens,
    )
    content = response.choices[0].message.content
    payload = json.loads(content)
    rows = payload.get("reviews", payload if isinstance(payload, list) else [])
    if not isinstance(rows, list):
        rows = []

    by_index = {}
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("row_index"))
        except (TypeError, ValueError):
            continue
        result = normalize_result(raw)
        result["analysis_model"] = getattr(response, "model", args.model)
        result["analysis_request_id"] = getattr(response, "id", "")
        by_index[index] = result
    return by_index


def write_dimensions(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "count": len(CONTROLLED_ASPECT_DIMENSIONS),
                "dimensions": sorted(CONTROLLED_ASPECT_DIMENSIONS),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def select_indices(frame, args):
    start = min(args.start_row, len(frame))
    indices = list(range(start, len(frame)))
    if args.limit is not None:
        indices = indices[:args.limit]
    return indices


def process_reviews(args):
    frame = load_reviews(args.input)
    indices = select_indices(frame, args)
    print("Input: " + str(args.input))
    print("Output: " + str(args.output))
    print("Rows selected: " + str(len(indices)))
    print("Batch size: " + str(args.batch_size))
    print("Model: " + args.model)

    if args.dry_run:
        print("Dry run only. No API calls were made.")
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)
    processed = 0
    started = time.time()

    for offset in range(0, len(indices), args.batch_size):
        batch_indices = indices[offset:offset + args.batch_size]
        review_payload = make_review_payload(frame, batch_indices, args.review_text_max_chars)
        results = call_openai_batch(client, args, review_payload)

        missing = set(batch_indices) - set(results)
        if missing:
            raise RuntimeError("Missing API results for row indices: " + ", ".join(str(i) for i in sorted(missing)))

        for index in batch_indices:
            for field, value in results[index].items():
                frame.at[index, field] = value
        processed += len(batch_indices)
        print("Processed " + str(processed) + "/" + str(len(indices)) + " reviews", flush=True)

        if args.sleep_seconds > 0 and offset + args.batch_size < len(indices):
            time.sleep(args.sleep_seconds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    write_dimensions(args.dimensions_path)
    elapsed = time.time() - started
    print("Saved enriched reviews to " + str(args.output))
    print("Saved aspect dimensions to " + str(args.dimensions_path))
    print("Elapsed seconds: %.1f" % elapsed)


def main():
    args = parse_args()
    process_reviews(args)


if __name__ == "__main__":
    main()
