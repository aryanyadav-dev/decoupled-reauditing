import csv
import random
from dataclasses import dataclass
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.utils import set_all_seeds


@dataclass
class SyntheticVerifier:
    omega: set
    name: str

    def accepts(self, problem, trace, context=None):
        pid = context.get("problem_id") if context else problem
        wrong = context.get("wrong", True) if context else True
        return (pid in self.omega) if wrong else True


def make_synthetic_pool(n, omega_size, overlap):
    ids = list(range(n))
    shared_n = int(round(omega_size * overlap))
    private_n = omega_size - shared_n
    shared = set(ids[:shared_n])
    cursor = shared_n
    pool = []
    for i in range(3):
        private = set(ids[cursor: cursor + private_n])
        cursor += private_n
        pool.append(SyntheticVerifier(shared | private, f"synthetic_{i}"))
    return pool


def simulate_gap(overlap):
    n = config.NUM_PROBLEMS
    omega_size = max(1, n // 4)
    pool = make_synthetic_pool(n * 4, omega_size, overlap)
    survivors = set(range(n))
    for t in range(config.NUM_GENERATIONS):
        filt = pool[t % 3]
        audit = pool[(t + 1) % 3]
        accepted_wrong = {pid for pid in survivors if filt.accepts(pid, "", {"problem_id": pid, "wrong": True})}
        survivors = {pid for pid in accepted_wrong if audit.accepts(pid, "", {"problem_id": pid, "wrong": True})}
    reported = 1.0
    true = 1.0 - (len(survivors) / max(n, 1))
    return reported - true


def main():
    set_all_seeds(config.SEED)
    rows = []
    for overlap in (0.0, 0.25, 0.5, 0.75, 1.0):
        rows.append({"overlap": overlap, "gap_final": simulate_gap(overlap)})
    path = Path("results") / "exp4_overlap.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["overlap", "gap_final"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    main()

