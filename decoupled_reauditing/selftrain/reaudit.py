from typing import List, Tuple


# Round-robin filter with next-in-rotation auditor. This exact pairing is the
# hypothesis of Proposition 1: over M consecutive generations every verifier acts as
# filter (and as auditor) at least once, so any wrong trace outside the common blind
# spot is rejected at least once per cycle. Do not change the pairing without updating
# the paper's theorem.
def rotation_schedule(M, num_gens):
    """Return list of (filt_idx, audit_idx) per generation t=0..num_gens-1."""
    return [(t % M, (t + 1) % M) for t in range(num_gens)]


def rotation_pair(pool: List, t: int) -> Tuple[object, object]:
    v_filt = pool[t % len(pool)]
    v_audit = pool[(t + 1) % len(pool)]
    assert v_audit is not v_filt, "auditor must be disjoint from filter (Proposition 1 hypothesis)"
    return v_filt, v_audit


def accept_set(samples, verifier, contexts):
    accepted = []
    for item in samples:
        ctx = contexts[item["problem_id"]]
        if verifier.accepts(item["problem"], item["trace"], ctx):
            accepted.append(item)
    return accepted


def reaudit_set(accepted, auditor, contexts):
    clean = []
    for item in accepted:
        ctx = contexts[item["problem_id"]]
        if auditor.accepts(item["problem"], item["trace"], ctx):
            clean.append(item)
    return clean


def decoupled_reaudit(samples, pool, t, contexts):
    v_filt, v_audit = rotation_pair(pool, t)
    accepted = accept_set(samples, v_filt, contexts)
    clean = reaudit_set(accepted, v_audit, contexts)
    return v_filt, v_audit, accepted, clean
