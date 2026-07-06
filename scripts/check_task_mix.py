"""Report the task-type mix of the LLaVA-Instruct-150K subset actually trained on.

Reproduces the trainer's shuffle (seed=42) and classifies each item:
  - conversation : more than one human turn (multi-turn Q&A)
  - detail       : single-turn, description-style prompt (keyword heuristic)
  - reasoning    : single-turn, everything else
NOTE: the detail/reasoning split is a rough keyword heuristic (under-counts
detail); the conversation count is exact (turn-based). Run from the repo root:

    python scripts/check_task_mix.py
"""
import json, random, glob, os
from collections import Counter

DATA_DIR = "data/llava_instruct_150k"
SEED = 42
N_TRAINED = 14400   # ~9%: 450 steps x eff-batch 32 (per --max-steps 450)

DETAIL_KW = ("describe", "detailed", "comprehensive", "elaborate", "thorough",
             "depiction", "rundown", "narrate", "illustrate", "portray",
             "exhaustive", "analysis of the image")


def task_type(row: dict) -> str:
    humans = [c["value"] for c in row.get("conversations", []) if c.get("from") == "human"]
    if len(humans) > 1:
        return "conversation"
    if not humans:
        return "none"
    q = humans[0].lower().replace("<image>", "")
    return "detail" if any(k in q for k in DETAIL_KW) else "reasoning"


def report(name: str, rows: list) -> None:
    c = Counter(task_type(r) for r in rows)
    n = sum(c.values())
    print(name, [(k, v, f"{100*v/n:.0f}%") for k, v in c.most_common()])


def main() -> None:
    path = os.path.join(DATA_DIR, "llava_instruct_150k.json")
    if not os.path.exists(path):
        path = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))[0]
    rows = json.load(open(path))
    print("file:", path, "| total items:", len(rows))
    random.Random(SEED).shuffle(rows)
    report("FULL 150K   ", rows)
    report(f"TRAINED {N_TRAINED}", rows[:N_TRAINED])


if __name__ == "__main__":
    main()
