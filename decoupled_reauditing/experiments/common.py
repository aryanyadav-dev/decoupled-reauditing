import csv
import json
import time
from pathlib import Path

from decoupled_reauditing import config
from decoupled_reauditing.metrics import make_independent_judge
from decoupled_reauditing.selftrain.loop import run_generation
from decoupled_reauditing.utils import load_gsm8k, load_model, set_all_seeds
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


def run_real_experiment(mode, exp_name, csv_name):
    set_all_seeds(config.SEED)
    train, eval_ = load_splits()
    policy, tokenizer = load_model(config.MODEL_ID, config.LOAD_IN_4BIT, padding_side="left")
    llm, llm_tok = load_model(config.LLM_JUDGE_ID, config.LOAD_IN_4BIT)
    pool = make_real_pool(llm, llm_tok)
    judge = make_independent_judge()
    rows = []
    start = time.time()
    for t in range(config.NUM_GENERATIONS):
        ckpt = Path("results") / f"_ckpt_{exp_name}_gen{t}.jsonl"
        if ckpt.exists():
            metric = read_metric_ckpt(ckpt)
            if metric:
                rows.append(metric)
                continue
        policy, clean, metric = run_generation(policy, tokenizer, train, eval_, mode, t, pool, judge, exp_name, start)
        rows.append(metric)
    write_csv(Path("results") / csv_name, ["gen", "reported", "true", "gap", "contam"], rows)
    return rows
