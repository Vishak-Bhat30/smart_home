"""
Data Preparation Script for Smart Home Command Finetuning
=========================================================
Loads the CSV dataset, splits into train/val/test, and formats
as chat-template conversations for instruction tuning.
"""

import csv
import json
import os
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a smart home assistant for a 2BHK house. "
    "Given a natural language command, output the corresponding device states as a JSON object. "
    "The house has these rooms: bathroom, bedroom, balcony, dining_room, hall, kitchen, living_room, study_room. "
    "Available devices include: ac, blinds, computer, exhaust, fan, geyser, lights, music_system, tv. "
    "Respond ONLY with a valid JSON object, no explanation."
)

def load_csv(csv_path: str) -> list[dict]:
    """Load the CSV and return list of {input, output} dicts."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row["input"].strip()
            out = row["output"].strip()
            if inp and out:
                # Validate JSON output
                try:
                    json.loads(out)
                    rows.append({"input": inp, "output": out})
                except json.JSONDecodeError:
                    continue
    return rows


def format_as_chat(example: dict) -> dict:
    """Format a single example as a chat conversation."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def split_data(rows: list[dict], train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, seed=42):
    """Split data into train/val/test."""
    random.seed(seed)
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = rows[:n_train]
    val = rows[n_train : n_train + n_val]
    test = rows[n_train + n_val :]
    return train, val, test


def save_jsonl(data: list[dict], path: str):
    """Save list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(data)} examples to {path}")


def main():
    csv_path = "./smart_home_100k_clean.csv"
    output_dir = "./data"

    print("Loading CSV...")
    rows = load_csv(csv_path)
    print(f"  Loaded {len(rows)} valid examples")

    print("Splitting data...")
    train, val, test = split_data(rows)
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("Formatting as chat conversations...")
    train_chat = [format_as_chat(r) for r in train]
    val_chat = [format_as_chat(r) for r in val]
    test_chat = [format_as_chat(r) for r in test]

    # Also save raw test data for evaluation
    test_raw = [{"input": r["input"], "expected_output": r["output"]} for r in test]

    print("Saving JSONL files...")
    save_jsonl(train_chat, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_chat, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_chat, os.path.join(output_dir, "test.jsonl"))
    save_jsonl(test_raw, os.path.join(output_dir, "test_raw.jsonl"))

    # Print a sample
    print("\n--- Sample formatted example ---")
    print(json.dumps(train_chat[0], indent=2))


if __name__ == "__main__":
    main()
