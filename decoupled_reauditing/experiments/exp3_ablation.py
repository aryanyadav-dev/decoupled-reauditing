import csv
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.experiments.common import run_real_experiment


def main():
    rows = []
    for condition in ("rotation_only", "reaudit_only"):
        exp_name = f"exp3_{condition}"
        sub = run_real_experiment(condition, exp_name, f"{exp_name}.tmp.csv")
        for row in sub:
            rows.append({"condition": condition, **row})
    path = Path("results") / "exp3_ablation.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "gen", "reported", "true", "gap", "contam"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    return rows


if __name__ == "__main__":
    main()

