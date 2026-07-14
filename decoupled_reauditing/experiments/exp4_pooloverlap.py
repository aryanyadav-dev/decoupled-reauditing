"""Synthetic blind-spot overlap control for Proposition 2.

The universe is the N GSM8K-style problem ids 0..N-1. Each synthetic verifier
wrongly accepts wrong traces on its blind spot; `overlap` controls how much of
each blind spot is shared by all verifiers versus private. Decoupled Re-Auditing
keeps only wrong traces accepted by both filter and auditor, so the final gaming
gap rises as the common blind spot grows.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.utils import set_all_seeds

OVERLAPS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class SyntheticVerifier:
    omega: set
    name: str

    def accepts(self, problem, trace, context=None):
        pid = context.get("problem_id") if context else problem
        wrong = context.get("wrong", True) if context else True
        return (pid in self.omega) if wrong else True


def make_synthetic_pool(n, omega_size, overlap):
    """Create three blind spots inside the same 0..n-1 universe as survivors."""
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}")

    # Each verifier has |Omega_i| = omega_size. The shared block is common to
    # all three; each private block is disjoint. If the requested size cannot
    # fit in the N-id universe at overlap=0, shrink it rather than leaking ids.
    shared_n = int(round(omega_size * overlap))
    private_n = omega_size - shared_n
    required = shared_n + 3 * private_n
    if required > n:
        omega_size = max(1, n // 3)
        shared_n = int(round(omega_size * overlap))
        private_n = omega_size - shared_n
        required = shared_n + 3 * private_n
    assert required <= n, "synthetic blind spots must fit inside the survivor universe"

    ids = list(range(n))
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
    pool = make_synthetic_pool(n, omega_size, overlap)
    survivors = set(range(n))
    for t in range(config.NUM_GENERATIONS):
        filt = pool[t % 3]
        audit = pool[(t + 1) % 3]
        survivors = {
            pid
            for pid in survivors
            if filt.accepts(pid, "", {"problem_id": pid, "wrong": True})
            and audit.accepts(pid, "", {"problem_id": pid, "wrong": True})
        }
    # In this synthetic control, the reported view is conditioned on the wrong
    # traces that survived the in-loop filter/auditor gates. The active filter
    # accepts every such survivor by construction, so reported=1.0 is exact.
    reported = 1.0
    true = 1.0 - (len(survivors) / max(n, 1))
    return reported - true


def assert_monotone(rows, tol=1e-12):
    gaps = [row["gap_final"] for row in rows]
    if any(b + tol < a for a, b in zip(gaps, gaps[1:])):
        raise AssertionError(f"exp4 gaps must be non-decreasing with overlap, got {gaps}")


def main():
    set_all_seeds(config.SEED)
    rows = []
    for overlap in OVERLAPS:
        rows.append({"overlap": overlap, "gap_final": simulate_gap(overlap)})
    assert_monotone(rows)
    path = Path("results") / "exp4_overlap.csv"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["overlap", "gap_final"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"overlap={row['overlap']:.2f}, gap_final={row['gap_final']:.6f}")
    return rows


if __name__ == "__main__":
    main()
