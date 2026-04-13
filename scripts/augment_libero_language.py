import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List


def load_task_map(path: Path) -> Dict[int, str]:
    with open(path, encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    if not isinstance(payload, dict):
        raise ValueError("Input task map must be a JSON object mapping task_index -> instruction")

    parsed = {}
    for key, value in payload.items():
        if isinstance(value, list):
            if not value:
                continue
            text = str(value[0]).strip()
        else:
            text = str(value).strip()
        if not text:
            continue
        parsed[int(key)] = text
    return parsed


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_level_variants(text: str) -> List[str]:
    replacements = {
        " pick up ": [" grab ", " lift "],
        " place ": [" put ", " set "],
        " move ": [" shift ", " relocate "],
        " drawer ": [" cabinet drawer "],
        " tray ": [" bin "],
        " mug ": [" cup "],
        " on ": [" onto "],
        " into ": [" inside "],
    }

    padded = f" {text} "
    variants = []
    for source, targets in replacements.items():
        if source in padded:
            for target in targets:
                candidate = padded.replace(source, target)
                variants.append(clean_text(candidate))
    return variants


def template_variants(text: str) -> List[str]:
    return [
        clean_text(text),
        clean_text(f"Please {text}"),
        clean_text(f"Carefully {text}"),
        clean_text(f"Your task is to {text}"),
        clean_text(f"Try to {text}"),
        clean_text(f"In this episode, {text}"),
    ]


def generate_variants(text: str, target_count: int, seed: int) -> List[str]:
    random.seed(seed)
    pool = []

    for candidate in template_variants(text):
        if candidate and candidate not in pool:
            pool.append(candidate)

    for candidate in word_level_variants(text):
        if candidate and candidate not in pool:
            pool.append(candidate)

    random.shuffle(pool[1:])

    if len(pool) >= target_count:
        return pool[:target_count]

    while len(pool) < target_count:
        pool.append(pool[-1])
    return pool


def main():
    parser = argparse.ArgumentParser(description="Offline language augmentation for LIBERO task maps")
    parser.add_argument("--input", type=Path, required=True, help="Path to canonical task map JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to write augmented task map JSON")
    parser.add_argument("--variants_per_task", type=int, default=8, help="Number of variants per task")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    canonical_map = load_task_map(args.input)
    augmented = {}
    for task_index, instruction in canonical_map.items():
        variants = generate_variants(instruction, target_count=args.variants_per_task, seed=args.seed + task_index)
        augmented[int(task_index)] = variants

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file_handle:
        json.dump(augmented, file_handle, ensure_ascii=False, indent=2)

    print(f"Wrote augmented map for {len(augmented)} tasks -> {args.output}")


if __name__ == "__main__":
    main()
