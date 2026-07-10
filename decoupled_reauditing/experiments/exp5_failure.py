import csv
import json
import re
from collections import Counter
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.utils import parse_final_answer, set_all_seeds
from decoupled_reauditing.verifiers.llm_judge import PROMPT_TEMPLATE

CATEGORIES = ["systematic_misconception", "numeric_slip", "spurious_but_consistent", "other"]


def simple_category(trace):
    if re.search(r"\d+\s*[-+*/]\s*\d+\s*=\s*\d+", trace):
        return "numeric_slip"
    if any(x in trace.lower() for x in ["assume", "approximately", "pattern", "always"]):
        return "systematic_misconception"
    if "answer is" in trace.lower():
        return "spurious_but_consistent"
    return "other"


def load_final_clean():
    for t in reversed(range(config.NUM_GENERATIONS)):
        path = Path("results") / f"_ckpt_exp2_method_gen{t}.jsonl"
        if path.exists():
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("kind") == "clean":
                        rows.append(row)
            return rows
    return []


def main():
    set_all_seeds(config.SEED)
    clean = load_final_clean()
    wrong = [x for x in clean if parse_final_answer(x["trace"]) != x["gold_answer"]]
    counts = Counter(simple_category(x["trace"]) for x in wrong)
    total = sum(counts.values())
    rows = [{"category": c, "share": (counts[c] / total if total else 0.0)} for c in CATEGORIES]
    path = Path("results") / "exp5_failure.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "share"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()

