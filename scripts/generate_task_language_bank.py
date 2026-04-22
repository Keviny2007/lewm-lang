"""Generate a task language bank with OpenRouter.

This precomputes canonical instructions and paraphrases before training so the
HDF5 annotation step can stay deterministic.

Example:
    python scripts/generate_task_language_bank.py \
        --tasks pusht_expert_train tworoom \
        --output annotations/task_language_bank.generated.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path


TASK_SPECS = {
    "pusht_expert_train": {
        "task_name": "PushT",
        "description": (
            "A planar manipulation task where the agent pushes a T-shaped block "
            "to match a target pose in position and orientation."
        ),
        "must_include": [
            "pushing or moving the T-shaped block",
            "the goal is the target pose or configuration",
        ],
    },
    "tworoom": {
        "task_name": "TwoRoom",
        "description": (
            "A navigation task where the agent starts in one room and must reach "
            "the designated goal room."
        ),
        "must_include": [
            "moving from the start room to the goal room",
            "navigation language only, no manipulation wording",
        ],
    },
    "reacher": {
        "task_name": "Reacher",
        "description": (
            "A reaching task where a robot arm or end effector must move to a "
            "target position."
        ),
        "must_include": [
            "moving the end effector or arm to the target",
            "reaching language only, no grasping or pushing wording",
        ],
    },
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_openrouter_key() -> str:
    for key_name in ("OPENROUTER_API_KEY", "openrouter_api_key"):
        value = os.environ.get(key_name, "").strip()
        if value:
            return value
    raise RuntimeError(
        "Missing OpenRouter API key. Set OPENROUTER_API_KEY or openrouter_api_key "
        "in the environment or .env."
    )


def build_prompt(task_key: str, spec: dict, num_variants: int) -> str:
    constraints = "\n".join(f"- {item}" for item in spec["must_include"])
    return f"""Generate language annotations for a robotics dataset.

Task key: {task_key}
Task name: {spec["task_name"]}
Task description: {spec["description"]}

Requirements:
- Return exactly one canonical instruction and {num_variants} paraphrases.
- Keep semantics identical across all variants.
- Use concise, plain English.
- Do not mention colors, objects, rooms, or constraints that are not in the task description.
- Avoid adding hidden assumptions or extra goals.
- Keep each instruction to one sentence.
- Make paraphrases lexically distinct but semantically equivalent.

The output must be valid JSON with this exact schema:
{{
  "canonical": "string",
  "variants": ["string", "..."]
}}

Additional semantic constraints:
{constraints}
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def openrouter_chat(api_key: str, model: str, prompt: str, temperature: float) -> dict:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You produce compact JSON only. Do not wrap the JSON in markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kevinyang/le-wm",
            "X-Title": "le-wm-language-bank-generator",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    return extract_json(content)


def normalize_entry(entry: dict, num_variants: int) -> dict:
    canonical = str(entry["canonical"]).strip()
    variants = [str(v).strip() for v in entry.get("variants", []) if str(v).strip()]
    if canonical and canonical not in variants:
        variants.insert(0, canonical)
    seen = set()
    deduped = []
    for item in variants:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if not canonical:
        raise ValueError("Missing canonical instruction")
    if len(deduped) < num_variants + 1:
        raise ValueError(
            f"Expected at least {num_variants + 1} total strings including canonical, got {len(deduped)}"
        )
    return {"canonical": canonical, "variants": deduped[: num_variants + 1]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a task language bank with OpenRouter")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["pusht_expert_train", "tworoom", "reacher"],
        help="Task keys to generate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("annotations/task_language_bank.generated.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-5.1-chat",
        help="OpenRouter model id",
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=4,
        help="Number of paraphrases to request in addition to the canonical string",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature for generation",
    )
    parser.add_argument(
        "--dotenv",
        type=Path,
        default=Path(".env"),
        help="Optional .env file to load",
    )
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    api_key = get_openrouter_key()

    bank = {}
    for task_key in args.tasks:
        if task_key not in TASK_SPECS:
            raise KeyError(f"Unknown task key: {task_key}")
        spec = TASK_SPECS[task_key]
        prompt = build_prompt(task_key, spec, args.num_variants)
        try:
            raw_entry = openrouter_chat(api_key, args.model, prompt, args.temperature)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            print(f"OpenRouter request failed for {task_key}: {exc.code} {body}", file=sys.stderr)
            return 1
        entry = normalize_entry(raw_entry, args.num_variants)
        bank[task_key] = entry
        print(f"{task_key}: canonical={entry['canonical']!r} variants={len(entry['variants'])}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
