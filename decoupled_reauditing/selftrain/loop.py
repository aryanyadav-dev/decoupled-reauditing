import json
import os
import sys
import time
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.metrics import generation_metrics
from decoupled_reauditing.selftrain.finetune import finetune_policy
from decoupled_reauditing.selftrain.reaudit import accept_set, decoupled_reaudit, reaudit_set, rotation_pair
from decoupled_reauditing.selftrain.sampler import greedy_eval_traces, sample_traces


def build_contexts(samples):
    contexts = {}
    for item in samples:
        contexts.setdefault(item["problem_id"], {"samples": item["all_samples"]})
    return contexts


def checkpoint_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_generation(
    policy,
    tokenizer,
    train_data,
    eval_data,
    mode,
    t,
    pool,
    judge,
    exp_name,
    start_time,
    results_dir="results",
):
    elapsed = time.time() - start_time
    if elapsed > config.MAX_WALL_CLOCK_SEC:
        ckpt = Path(results_dir) / f"_ckpt_{exp_name}_gen{t}_budget.jsonl"
        checkpoint_jsonl(ckpt, [{"resume": True, "reason": "wall_clock_budget", "generation": t}])
        print(f"RESUME: wall-clock budget reached before generation {t}; checkpointed {ckpt}")
        sys.exit(0)

    samples = sample_traces(policy, tokenizer, train_data, config.K_SAMPLES)
    contexts = build_contexts(samples)

    if mode == "naive":
        v_filt = pool[0]
        v_audit = None
        accepted = accept_set(samples, v_filt, contexts)
        clean = accepted
    elif mode == "method":
        v_filt, v_audit, accepted, clean = decoupled_reaudit(samples, pool, t, contexts)
    elif mode == "rotation_only":
        v_filt, _ = rotation_pair(pool, t)
        v_audit = None
        accepted = accept_set(samples, v_filt, contexts)
        clean = accepted
    elif mode == "reaudit_only":
        v_filt, v_audit = pool[0], pool[1]
        accepted = accept_set(samples, v_filt, contexts)
        clean = reaudit_set(accepted, v_audit, contexts)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    eval_traces = greedy_eval_traces(policy, tokenizer, eval_data)
    metrics = generation_metrics(eval_data, eval_traces, v_filt, clean, judge)
    metrics.update(
        {
            "gen": t,
            "mode": mode,
            "accepted_n": len(accepted),
            "clean_n": len(clean),
            "empty_clean_set": len(clean) == 0,
            "v_filt": getattr(v_filt, "name", type(v_filt).__name__),
            "v_audit": getattr(v_audit, "name", None) if v_audit else None,
        }
    )
    ckpt_rows = [{"kind": "metric", **metrics}] + [{"kind": "clean", **x} for x in clean]
    checkpoint_jsonl(Path(results_dir) / f"_ckpt_{exp_name}_gen{t}.jsonl", ckpt_rows)
    if clean:
        policy = finetune_policy(policy, tokenizer, clean, str(Path(results_dir) / f"adapter_{exp_name}_gen{t}"))
    else:
        print(f"WARNING: empty clean set at {exp_name} generation {t}; carrying policy forward.")
    return policy, clean, metrics

