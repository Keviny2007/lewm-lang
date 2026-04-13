"""
Generate varied language annotations for CALVIN tasks using OpenRouter.

Runs locally. Outputs a JSON file mapping task_id -> list of phrasings.
This file is then used by convert_calvin.py during HDF5 conversion.

Usage:
    pip install openai
    OPENROUTER_API_KEY=<your_key> python scripts/generate_calvin_annotations.py \
        --out annotations/calvin_task_annotations.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── CALVIN task IDs ───────────────────────────────────────────────────────────
# All 34 tasks in the CALVIN benchmark
CALVIN_TASKS = [
    "move_slider_left",
    "move_slider_right",
    "turn_on_lightbulb",
    "turn_off_lightbulb",
    "turn_on_led",
    "turn_off_led",
    "push_into_drawer",
    "open_drawer",
    "close_drawer",
    "lift_red_block_table",
    "lift_red_block_slider",
    "lift_red_block_drawer",
    "lift_blue_block_table",
    "lift_blue_block_slider",
    "lift_blue_block_drawer",
    "lift_pink_block_table",
    "lift_pink_block_slider",
    "lift_pink_block_drawer",
    "place_in_slider",
    "place_in_drawer",
    "stack_block",
    "unstack_block",
    "push_red_block_right",
    "push_red_block_left",
    "push_blue_block_right",
    "push_blue_block_left",
    "push_pink_block_right",
    "push_pink_block_left",
    "rotate_red_block_right",
    "rotate_red_block_left",
    "rotate_blue_block_right",
    "rotate_blue_block_left",
    "rotate_pink_block_right",
    "rotate_pink_block_left",
]

ENVIRONMENT_CONTEXT = """
The robot operates on a tabletop with the following objects:
- A red, blue, and pink block (small cubes)
- A drawer (can be opened/closed by pulling/pushing)
- A slider (a horizontal sliding cabinet door, pushed left or right)
- A lightbulb (toggled on/off by pressing a button)
- An LED light (toggled on/off by pressing a button)
The robot arm can grasp blocks, press buttons, and manipulate the drawer and slider.
"""

SYSTEM_PROMPT = f"""You are helping generate training data for a robot manipulation model.
{ENVIRONMENT_CONTEXT.strip()}

Given a robot task ID, generate 10 varied natural language instructions a human might give
to describe that task. Keep instructions short (under 10 words), natural, and diverse in phrasing.
Return a JSON array of strings only, no explanation."""

def task_to_prompt(task_id: str) -> str:
    readable = task_id.replace("_", " ")
    return f'Generate 10 varied natural language instructions for the robot task: "{readable}"'


def generate_phrasings(client: OpenAI, task_id: str, model: str) -> list[str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_to_prompt(task_id)},
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    # strip markdown code fences if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="annotations/calvin_task_annotations.json")
    parser.add_argument("--model", default="openai/gpt-4o-mini",
                        help="OpenRouter model ID")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file if present")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    annotations = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            annotations = json.load(f)
        print(f"Resuming from {len(annotations)} existing tasks")

    for task_id in CALVIN_TASKS:
        if task_id in annotations:
            print(f"  skipping {task_id} (already done)")
            continue

        print(f"  generating: {task_id} ...", end=" ", flush=True)
        try:
            phrasings = generate_phrasings(client, task_id, args.model)
            annotations[task_id] = phrasings
            print(f"{len(phrasings)} phrasings")
        except Exception as e:
            print(f"ERROR: {e}")
            # save progress so far before crashing
            with open(out_path, "w") as f:
                json.dump(annotations, f, indent=2)
            raise

    with open(out_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone. Saved {len(annotations)} tasks to {out_path}")


if __name__ == "__main__":
    main()
