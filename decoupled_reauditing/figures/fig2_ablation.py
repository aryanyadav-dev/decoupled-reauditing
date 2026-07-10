from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ab = pd.read_csv("results/exp3_ablation.csv")
    ov = pd.read_csv("results/exp4_overlap.csv")
    final = ab.sort_values("gen").groupby("condition").tail(1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(final["condition"], final["gap"] * 100)
    axes[0].set_ylabel("final gaming gap (pp)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].plot(ov["overlap"], ov["gap_final"] * 100, marker="o")
    axes[1].set_xlabel("synthetic blind-spot overlap")
    axes[1].set_ylabel("final gaming gap (pp)")
    axes[1].grid(alpha=0.25)
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/fig2_ablation.png", dpi=200)


if __name__ == "__main__":
    main()

