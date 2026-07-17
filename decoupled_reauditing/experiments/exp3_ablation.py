import csv
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.experiments.common import run_real_experiment


def main():
    # Print active regime for logging/verification
    print("="*80)
    print(f"[exp3_ablation] Active regime: {config.REGIME}")
    print(f"  NUM_PROBLEMS={config.NUM_PROBLEMS}, K_SAMPLES={config.K_SAMPLES}, NUM_GENERATIONS={config.NUM_GENERATIONS}")
    print(f"  EVAL_N={config.EVAL_N}, PROBE_N={config.PROBE_N}, TRAIN_STEPS={config.TRAIN_STEPS}")
    print(f"  SEED={config.SEED}")
    print("="*80)
    
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

