import argparse
import csv
import json
import random

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

NUMERIC_LABEL_MAP = {
    0: "sad",
    1: "happy",
    2: "happy",
    3: "angry",
    4: "fear",
    5: "surprise",
    6: "neutral",
    7: "disgust",
}

TEXT_LABEL_MAP = {
    "sad": "sad",
    "sadness": "sad",
    "anger": "angry",
    "angry": "angry",
    "fear": "fear",
    "joy": "happy",
    "happy": "happy",
    "love": "happy",
    "surprise": "surprise",
    "disgust": "disgust",
    "neutral": "neutral",
}

TEXT_FIELDS = ["text", "dialogue", "sentence", "utterance"]
LABEL_FIELDS = ["label", "emotion"]


def normalize_label(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        numeric = int(value)
    except ValueError:
        numeric = None
    if numeric is not None:
        return NUMERIC_LABEL_MAP.get(numeric)
    return TEXT_LABEL_MAP.get(value.lower())


def pick_field(fieldnames, candidates):
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description="Build thought strings from a CSV dataset.")
    parser.add_argument("--input", default="backend/train.csv", help="Path to the CSV dataset.")
    parser.add_argument(
        "--output", default="frontend/public/thoughts.json", help="Where to write the thoughts JSON."
    )
    parser.add_argument("--max-per-label", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    buckets = {emotion: set() for emotion in EMOTIONS}

    with open(args.input, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        text_field = pick_field(reader.fieldnames, TEXT_FIELDS)
        label_field = pick_field(reader.fieldnames, LABEL_FIELDS)
        if not text_field or not label_field:
            raise ValueError(
                f"CSV must contain a text field ({TEXT_FIELDS}) and label field ({LABEL_FIELDS})."
            )

        for row in reader:
            text = (row.get(text_field) or "").strip()
            if not text or len(text) > args.max_length:
                continue
            emotion = normalize_label(row.get(label_field))
            if not emotion or emotion not in buckets:
                continue
            buckets[emotion].add(text)

    output = {}
    for emotion in EMOTIONS:
        items = list(buckets[emotion])
        rng.shuffle(items)
        output[emotion] = items[: args.max_per_label]

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
