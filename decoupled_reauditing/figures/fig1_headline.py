from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    naive = pd.read_csv("results/exp1_naive.csv")
    method = pd.read_csv("results/exp2_method.csv")
    ymax = max(naive[["reported", "true"]].max().max(), method[["reported", "true"]].max().max(), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, df, title in [(axes[0], naive, "naive"), (axes[1], method, "method")]:
        x = df["gen"]
        rep = df["reported"] * 100
        true = df["true"] * 100
        ax.plot(x, rep, marker="o", label="reported")
        ax.plot(x, true, marker="o", label="true")
        ax.fill_between(x, true, rep, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("generation")
        ax.set_ylim(0, ymax * 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy (%)")
    axes[1].legend()
    Path("figures").mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig("figures/fig1_headline.png", dpi=200)


if __name__ == "__main__":
    main()

