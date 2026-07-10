import argparse
import csv
import importlib
import os
import shutil
import traceback
from pathlib import Path

import torch
from huggingface_hub import model_info

from decoupled_reauditing import config


CSVS = [
    "results/judge_certification.csv",
    "results/exp1_naive.csv",
    "results/exp2_method.csv",
    "results/exp3_ablation.csv",
    "results/exp4_overlap.csv",
    "results/exp5_failure.csv",
]
FIGS = ["figures/fig1_headline.png", "figures/fig2_ablation.png"]


def preflight():
    regime = config.active_regime()
    print("PREFLIGHT")
    print(f"Regime: {regime['REGIME']} sizes={regime}")
    assert regime["REGIME"] in {"SMOKE", "PILOT", "FULL"}
    assert torch.cuda.is_available(), "CUDA GPU required."
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM={props.total_memory / 2**30:.2f} GiB")
    for d in ("results", "figures"):
        Path(d).mkdir(exist_ok=True)
        test = Path(d) / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    for mid in (config.MODEL_ID, config.LLM_JUDGE_ID, config.JUDGE_ID):
        model_info(mid)
        print(f"Resolved model: {mid}")
    agreement = ensure_judge_certification()
    print(f"Judge-vs-gold agreement: {agreement * 100:.1f}%")


def read_judge_certification(path):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("metric") == "judge_gold_agreement":
            return float(row["value"])
    return None


def write_judge_certification(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "judge_gold_agreement", "value": value})


def ensure_judge_certification():
    path = Path("results") / "judge_certification.csv"
    existing = read_judge_certification(path)
    if existing is not None:
        return existing

    from decoupled_reauditing.experiments.common import load_train_eval_probe
    from decoupled_reauditing.metrics import judge_gold_agreement, make_independent_judge
    from decoupled_reauditing.selftrain.sampler import sample_traces
    from decoupled_reauditing.utils import load_model, set_all_seeds

    # Certification probe traces are sampled from the base policy before any
    # fine-tuning. The probe problems are carved from GSM8K test after eval.
    set_all_seeds(config.SEED)
    _, _, probe = load_train_eval_probe()
    policy, tokenizer = load_model(config.MODEL_ID, config.LOAD_IN_4BIT, padding_side="left")
    samples = sample_traces(policy, tokenizer, probe, config.K_SAMPLES)
    probe_set = [(item["problem"], item["trace"], item["gold_answer"]) for item in samples]
    del policy
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    judge = make_independent_judge()
    value = judge_gold_agreement(judge, probe_set)
    write_judge_certification(path, value)
    return value


def apply_regime(smoke_test, pilot_mode):
    config.SMOKE_TEST = smoke_test
    config.PILOT_MODE = pilot_mode
    active = config.active_regime()
    for key, value in active.items():
        setattr(config, key, value)


def env_regime_flags():
    smoke = os.getenv("DRA_SMOKE_TEST", "1") == "1"
    pilot = os.getenv("DRA_PILOT_MODE", "0") == "1"
    return smoke, pilot


def archive_smoke_artifacts():
    dest = Path("results") / "_smoke_pass"
    dest.mkdir(parents=True, exist_ok=True)
    for pattern in ("exp*.csv", "judge_certification.csv", "_ckpt_*.jsonl", "adapter_*"):
        for path in Path("results").glob(pattern):
            target = dest / path.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            path.replace(target)
    figdest = Path("figures") / "_smoke_pass"
    figdest.mkdir(parents=True, exist_ok=True)
    for path in Path("figures").glob("fig*.png"):
        target = figdest / path.name
        if target.exists():
            target.unlink()
        path.replace(target)


def run_modules():
    modules = [
        "decoupled_reauditing.experiments.exp1_naive",
        "decoupled_reauditing.experiments.exp2_method",
        "decoupled_reauditing.experiments.exp3_ablation",
        "decoupled_reauditing.experiments.exp4_pooloverlap",
        "decoupled_reauditing.experiments.exp5_failure",
        "decoupled_reauditing.figures.fig1_headline",
        "decoupled_reauditing.figures.fig2_ablation",
    ]
    for name in modules:
        print(f"RUN {name}")
        importlib.import_module(name).main()


def set_smoke_env():
    apply_regime(True, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    target_smoke, target_pilot = env_regime_flags()
    if not args.skip_smoke:
        try:
            set_smoke_env()
            preflight()
            run_modules()
            if not target_smoke:
                archive_smoke_artifacts()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
    apply_regime(target_smoke, target_pilot)
    preflight()
    run_modules()
    print("FINAL SUMMARY")
    for p in CSVS + FIGS:
        print(f"{p}: {'written' if Path(p).exists() else 'missing'}")
    skipped = sorted(str(p) for p in Path("results").glob("_ckpt_*_budget.jsonl"))
    if skipped:
        print("Resume checkpoints:")
        for p in skipped:
            print(p)


if __name__ == "__main__":
    main()
