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


def create_temporary_judge(device_index=None):
    """Create a Math-Shepherd judge temporarily for scoring, then clean up.
    
    Args:
        device_index: GPU to load judge on, or None for CPU
        
    Returns:
        MathShepherdJudge instance
    """
    import gc
    from decoupled_reauditing.metrics import make_independent_judge
    
    print(f"[create_temporary_judge] Loading Math-Shepherd judge temporarily...")
    log_gpu_memory()
    
    judge = make_independent_judge(device_index=device_index)
    
    print(f"[create_temporary_judge] Judge loaded, memory after loading:")
    log_gpu_memory()
    
    return judge


def cleanup_judge(judge):
    """Clean up judge and free GPU memory."""
    import gc
    
    print(f"[cleanup_judge] Cleaning up Math-Shepherd judge...")
    
    # Delete the model explicitly
    if hasattr(judge, 'model') and judge.model is not None:
        del judge.model
        judge.model = None
    
    # Delete the judge itself
    del judge
    
    # Force garbage collection and GPU cache cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"[cleanup_judge] Cleanup complete, memory after cleanup:")
    log_gpu_memory()


def run_real_experiment(mode, exp_name, csv_name):
    set_all_seeds(config.SEED)
    
    # Check bitsandbytes functionality before loading models
    check_bnb()
    
    train, eval_ = load_splits()
    
    # Determine device placement based on available GPUs - only policy and llm-judge persistent
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    policy_device, llm_judge_device, strategy_desc = get_device_placement_strategy(num_gpus)
    
    # Independent judge device: prefer GPU 1 if available, else GPU 0, else None (CPU)
    judge_device = 1 if num_gpus >= 2 else 0 if torch.cuda.is_available() else None
    
    print(f"[run_real_experiment] GPUs available: {num_gpus}")
    print(f"[run_real_experiment] policy -> cuda:{policy_device}, LLM-judge -> cuda:{llm_judge_device}")
    print(f"[run_real_experiment] independent-judge -> on-demand cuda:{judge_device}")
    print(f"[run_real_experiment] Strategy: {strategy_desc}")
    
    print(f"[run_real_experiment] Initial memory state:")
    log_gpu_memory()
    
    # Load only policy and llm-judge persistently
    policy, tokenizer = load_model(config.MODEL_ID, config.LOAD_IN_4BIT, padding_side="left", device_index=policy_device)
    llm, llm_tok = load_model(config.LLM_JUDGE_ID, config.LOAD_IN_4BIT, device_index=llm_judge_device)
    pool = make_real_pool(llm, llm_tok)
    
    print(f"[run_real_experiment] Memory after loading persistent models:")
    log_gpu_memory()
    
    # DO NOT create judge here - it will be created on-demand per generation
    
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
        
        # Run generation with temporary judge creation
        policy, clean, metric = run_generation_with_temp_judge(
            policy, tokenizer, train, eval_, mode, t, pool, judge_device, exp_name, start
        )
        
        rows.append(metric)
        
        print(f"[run_real_experiment] Memory at end of generation {t}:")
        log_gpu_memory()
    
    write_csv(Path("results") / csv_name, ["gen", "reported", "true", "gap", "contam"], rows)
    return rows


def run_generation_with_temp_judge(policy, tokenizer, train_data, eval_data, mode, t, pool, judge_device, exp_name, start_time):
    """Run generation with temporary judge loading for metrics computation only."""
    from decoupled_reauditing.selftrain.loop import run_generation
    
    # First run the generation logic without judge (it doesn't need judge until metrics)
    # We need to modify run_generation to accept judge as optional and create it when needed
    
    # For now, create judge temporarily, run generation, then clean up
    judge = create_temporary_judge(device_index=judge_device)
    
    try:
        result = run_generation(policy, tokenizer, train_data, eval_data, mode, t, pool, judge, exp_name, start_time)
        return result
    finally:
        # Always clean up judge after use
        cleanup_judge(judge)
