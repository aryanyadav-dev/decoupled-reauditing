import csv
import gc
import json
import time
from pathlib import Path

import torch

from decoupled_reauditing import config
from decoupled_reauditing.metrics import make_independent_judge
from decoupled_reauditing.selftrain.loop import run_generation
from decoupled_reauditing.utils import load_gsm8k, load_model, set_all_seeds, check_bnb
from decoupled_reauditing.verifiers import make_real_pool


def load_splits():
    train = load_gsm8k("train")[: config.NUM_PROBLEMS]
    eval_ = load_gsm8k("test")[: config.EVAL_N]
    assert {x["problem"] for x in train}.isdisjoint({x["problem"] for x in eval_})
    return train, eval_


def load_train_eval_probe():
    train = load_gsm8k("train")[: config.NUM_PROBLEMS]
    test = load_gsm8k("test")
    eval_ = test[: config.EVAL_N]
    probe = test[config.EVAL_N: config.EVAL_N + config.PROBE_N]
    train_ids = {x["problem"] for x in train}
    eval_ids = {x["problem"] for x in eval_}
    probe_ids = {x["problem"] for x in probe}
    assert train_ids.isdisjoint(eval_ids)
    assert train_ids.isdisjoint(probe_ids)
    assert eval_ids.isdisjoint(probe_ids)
    return train, eval_, probe


def read_metric_ckpt(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("kind") == "metric":
                return row
    return None


def write_csv(path, fieldnames, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def get_device_placement_strategy(num_gpus):
    """Get device placement strategy for models to avoid OOM.
    
    Only loads policy and llm-judge persistently. Independent judge loaded on-demand.
    
    Returns:
        tuple: (policy_device, llm_judge_device, description)
    """
    if num_gpus >= 2:
        # Primary strategy: GPU 0 = policy only, GPU 1 = llm-judge only
        # Independent judge will be loaded temporarily on GPU 1 when needed for scoring
        return (0, 1, "GPU 0 = policy(7B), GPU 1 = llm-judge(7B), independent-judge loaded on-demand")
    else:
        # Single GPU fallback
        return (0, 0, "Single GPU = policy + llm-judge, independent-judge loaded on-demand")


def get_alternative_device_placement_strategy(num_gpus):
    """Alternative device placement - kept for compatibility but not used with on-demand loading."""
    return get_device_placement_strategy(num_gpus)


def log_gpu_memory():
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(i) / 1024**3  # GB 
            print(f"[GPU-{i}] Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached")
    else:
        print("[GPU] No CUDA devices available")


def free_model_from_gpu(model, model_name="model"):
    """Free a model or tokenizer from GPU memory.
    
    Robustly handles both models (which may have .model attribute) and 
    tokenizers (which don't). Safe to call on any object.
    
    Args:
        model: The model or tokenizer to free
        model_name: Name for logging
    """
    import gc
    
    print(f"[free_model_from_gpu] Freeing {model_name} from GPU...")
    log_gpu_memory()
    
    # Safely delete the .model attribute if it exists
    # Tokenizers don't have .model, so use try/except to avoid AttributeError
    try:
        if getattr(model, 'model', None) is not None:
            del model.model
            model.model = None
    except (AttributeError, TypeError):
        # Object doesn't have .model or it's read-only - skip this cleanup
        pass
    
    del model
    
    # Force garbage collection and GPU cache cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[free_model_from_gpu] {model_name} freed, memory after cleanup:")
    log_gpu_memory()


def create_temporary_judge(device_index=None):
    """Create a Math-Shepherd judge temporarily for scoring, then clean up.
    
    Args:
        device_index: GPU to load judge on, or None for CPU
        
    Returns:
        MathShepherdJudge instance
    """
    import gc
    from decoupled_reauditing.metrics import make_independent_judge
    
    print(f"[create_temporary_judge] Loading Math-Shepherd judge on device {device_index}...")
    log_gpu_memory()
    
    judge = make_independent_judge(device_index=device_index)
    
    print(f"[create_temporary_judge] Judge loaded, memory after loading:")
    log_gpu_memory()
    
    return judge


def cleanup_judge(judge):
    """Clean up judge and free GPU memory."""
    free_model_from_gpu(judge, "Math-Shepherd judge")


def run_real_experiment(mode, exp_name, csv_name):
    set_all_seeds(config.SEED)
    
    # Check bitsandbytes functionality before loading models
    check_bnb()
    
    train, eval_ = load_splits()
    
    # Determine device placement based on available GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    policy_device, llm_judge_device, strategy_desc = get_device_placement_strategy(num_gpus)
    
    # Independent judge device: same as llm_judge_device (will replace it during scoring)
    judge_device = llm_judge_device if torch.cuda.is_available() else None
    
    print(f"[run_real_experiment] GPUs available: {num_gpus}")
    print(f"[run_real_experiment] policy -> cuda:{policy_device} (persistent)")
    print(f"[run_real_experiment] llm-judge -> cuda:{llm_judge_device} (during sampling/filtering)")
    print(f"[run_real_experiment] independent-judge -> cuda:{judge_device} (during scoring, replaces llm-judge)")
    print(f"[run_real_experiment] Strategy: {strategy_desc}")
    
    print(f"[run_real_experiment] Initial memory state:")
    log_gpu_memory()
    
    # Load policy persistently on GPU 0 (never moves - holds LoRA weights being trained)
    policy, tokenizer = load_model(config.MODEL_ID, config.LOAD_IN_4BIT, padding_side="left", device_index=policy_device)
    
    print(f"[run_real_experiment] Memory after loading policy:")
    log_gpu_memory()
    
    rows = []
    start = time.time()
    
    for t in range(config.NUM_GENERATIONS):
        print(f"\n[run_real_experiment] === Generation {t} ===")
        
        ckpt = Path("results") / f"_ckpt_{exp_name}_gen{t}.jsonl"
        if ckpt.exists():
            metric = read_metric_ckpt(ckpt)
            if metric:
                rows.append(metric)
                continue
        
        # Clean up memory before each generation
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"[run_real_experiment] Memory at start of generation {t}:")
        log_gpu_memory()
        
        # PHASE 1: SAMPLING/FILTERING - Load llm-judge on GPU 1
        print(f"\n[run_real_experiment] PHASE 1: SAMPLING/FILTERING")
        llm, llm_tok = load_model(config.LLM_JUDGE_ID, config.LOAD_IN_4BIT, device_index=llm_judge_device)
        pool = make_real_pool(llm, llm_tok)
        
        print(f"[run_real_experiment] Memory after loading llm-judge:")
        log_gpu_memory()
        
        # Run sampling and filtering (no judge needed yet)
        policy, clean, eval_traces, v_filt, contexts = run_sampling_and_filtering(
            policy, tokenizer, train, eval_, mode, t, pool, exp_name, start
        )
        
        # PHASE 2: SCORING
        # Sub-phase 2a: compute reported accuracy (uses v_filt = llm-judge verifier)
        # MUST happen while llm-judge model is still loaded; freeing it first would
        # set its inner .model to None and cause "NoneType is not callable" in generate().
        print(f"\n[run_real_experiment] PHASE 2: SCORING")
        print(f"[run_real_experiment] Sub-phase 2a: computing reported accuracy (llm-judge still live)...")
        from decoupled_reauditing.metrics import compute_reported_acc, compute_contamination, compute_true_acc
        reported = compute_reported_acc(eval_, eval_traces, v_filt, contexts)
        print(f"[run_real_experiment] reported_acc={reported:.4f}")

        # Sub-phase 2b: free llm-judge, then load Math-Shepherd judge for true accuracy
        print(f"[run_real_experiment] Sub-phase 2b: freeing llm-judge, loading Math-Shepherd judge...")
        free_model_from_gpu(llm, "llm-judge model")
        free_model_from_gpu(llm_tok, "llm-judge tokenizer")
        del pool
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[run_real_experiment] Memory after freeing llm-judge (GPU 1 now empty):")
        log_gpu_memory()

        # Load Math-Shepherd judge on the freed GPU slot
        judge = create_temporary_judge(device_index=judge_device)

        true_acc = compute_true_acc(eval_, eval_traces, judge)
        contam = compute_contamination(clean)
        print(f"[run_real_experiment] true_acc={true_acc:.4f}, contamination={contam:.4f}")

        # Free the judge immediately after scoring
        cleanup_judge(judge)

        metrics = {
            "reported": reported,
            "true": true_acc,
            "gap": reported - true_acc,
            "contam": contam,
        }
        
        # Add generation metadata to metrics
        metrics.update({
            "gen": t,
            "mode": mode,
            "accepted_n": len(clean.get("accepted", [])) if isinstance(clean, dict) else len(clean),
            "clean_n": len(clean) if isinstance(clean, list) else clean.get("clean_n", 0),
        })
        
        rows.append(metrics)
        
        print(f"[run_real_experiment] Memory at end of generation {t}:")
        log_gpu_memory()
    
    write_csv(Path("results") / csv_name, ["gen", "reported", "true", "gap", "contam"], rows)
    return rows


def run_sampling_and_filtering(policy, tokenizer, train_data, eval_data, mode, t, pool, exp_name, start_time):
    """Run sampling and filtering phase without judge (judge used later for scoring only).
    
    Returns:
        tuple: (updated_policy, clean_set, eval_traces, v_filt, contexts)
    """
    import sys
    from decoupled_reauditing.selftrain.sampler import greedy_eval_traces, sample_traces
    from decoupled_reauditing.selftrain.reaudit import accept_set, decoupled_reaudit, reaudit_set, rotation_pair
    from decoupled_reauditing.selftrain.finetune import finetune_policy
    
    # Check wall clock budget
    elapsed = time.time() - start_time
    if elapsed > config.MAX_WALL_CLOCK_SEC:
        ckpt = Path("results") / f"_ckpt_{exp_name}_gen{t}_budget.jsonl"
        checkpoint_jsonl(ckpt, [{"resume": True, "reason": "wall_clock_budget", "generation": t}])
        print(f"RESUME: wall-clock budget reached before generation {t}; checkpointed {ckpt}")
        sys.exit(0)
    
    # Sample traces and build contexts
    print(f"\n[Generation {t}] === SAMPLING PHASE ===")
    samples = sample_traces(policy, tokenizer, train_data, config.K_SAMPLES)
    contexts = build_contexts(samples)
    print(f"[Generation {t}] Sampled {len(samples)} traces from {len(train_data)} problems")
    
    # Run filtering based on mode (no judge needed)
    print(f"[Generation {t}] === FILTERING PHASE (mode={mode}) ===")
    if mode == "naive":
        v_filt = pool[0]
        v_audit = None
        print(f"[filter] Scoring {len(samples)} traces with {v_filt.__class__.__name__}")
        accepted = accept_set(samples, v_filt, contexts)
        clean = accepted
        print(f"[filter] Accepted {len(accepted)} traces (no re-audit)")
    elif mode == "method":
        print(f"[filter] Running decoupled re-audit with rotation and re-auditing")
        v_filt, v_audit, accepted, clean = decoupled_reaudit(samples, pool, t, contexts)
        print(f"[filter] Filter accepted {len(accepted)} traces")
        print(f"[re-audit] Re-audited with {v_audit.__class__.__name__}, clean set: {len(clean)} traces")
    elif mode == "rotation_only":
        v_filt, _ = rotation_pair(pool, t)
        v_audit = None
        print(f"[filter] Rotation-only: scoring {len(samples)} traces with {v_filt.__class__.__name__}")
        accepted = accept_set(samples, v_filt, contexts)
        clean = accepted
        print(f"[filter] Accepted {len(accepted)} traces (no re-audit)")
    elif mode == "reaudit_only":
        v_filt, v_audit = pool[0], pool[1]
        print(f"[filter] Re-audit-only: scoring {len(samples)} traces with {v_filt.__class__.__name__}")
        accepted = accept_set(samples, v_filt, contexts)
        print(f"[re-audit] Re-auditing {len(accepted)} traces with {v_audit.__class__.__name__}")
        clean = reaudit_set(accepted, v_audit, contexts)
        print(f"[re-audit] Clean set: {len(clean)} traces")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Generate eval traces
    print(f"[Generation {t}] === EVALUATION GENERATION ===")
    eval_traces = greedy_eval_traces(policy, tokenizer, eval_data)
    print(f"[Generation {t}] Generated {len(eval_traces)} eval traces")
    
    # Checkpoint clean set (without metrics yet - those need judge)
    ckpt_rows = [{"kind": "clean", **x} for x in clean]
    checkpoint_jsonl(Path("results") / f"_ckpt_{exp_name}_gen{t}_partial.jsonl", ckpt_rows)
    
    # Fine-tune if we have clean data
    print(f"[Generation {t}] === FINE-TUNING PHASE ===")
    if clean:
        policy = finetune_policy(policy, tokenizer, clean, str(Path("results") / f"adapter_{exp_name}_gen{t}"))
    else:
        print(f"WARNING: empty clean set at {exp_name} generation {t}; carrying policy forward.")
    
    return policy, clean, eval_traces, v_filt, contexts


def build_contexts(samples):
    """Build contexts dict from samples."""
    contexts = {}
    for item in samples:
        contexts.setdefault(item["problem_id"], {"samples": item["all_samples"]})
    return contexts


def checkpoint_jsonl(path, rows):
    """Write checkpoint JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
