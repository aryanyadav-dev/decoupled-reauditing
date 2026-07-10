# Decoupled Re-Auditing

Research code and paper draft for **Decoupled Re-Auditing: Preventing Monitor Gaming in Self-Improving Reasoners**.

The project studies a failure mode in rejection-sampling self-training for reasoning models: an imperfect verifier can become a stationary optimization target, so the policy learns traces the verifier accepts even when they are wrong. The paper calls this **monitor gaming**. The defense is **Decoupled Re-Auditing**: rotate the filtering verifier each generation and re-audit the accepted traces with the next verifier in the rotation before fine-tuning.

## Paper Claim

Naive self-training can make reported verifier accuracy rise while independent process-judge accuracy falls. Decoupled Re-Auditing should keep reported and true accuracy aligned when verifier blind spots are diverse.

The core theorem in `research.tex` depends on the exact round-robin schedule implemented in code:

```text
filter(t) = t % M
audit(t)  = (t + 1) % M
```

Over each cycle, every verifier acts as a filter and as an auditor. A wrong trace that is not in the common blind spot of all verifiers is rejected at least once per cycle, interrupting persistent reinforcement.

## Repository Map

- `research.tex`: AAAI-style paper draft.
- `decoupled_reauditing/config.py`: regimes, model ids, seeds, LoRA settings, generation limits, and probe size.
- `decoupled_reauditing/verifiers/`: symbolic checker, LLM judge verifier, and self-consistency verifier.
- `decoupled_reauditing/selftrain/`: sampling, filtering, re-auditing, LoRA fine-tuning, checkpoints, and schedule helpers.
- `decoupled_reauditing/judge.py`: independent Math-Shepherd PRM judge, used for measurement only.
- `decoupled_reauditing/metrics.py`: reported accuracy, true accuracy, gaming gap, contamination, pool diversity, and judge-vs-gold certification.
- `decoupled_reauditing/experiments/`: five paper experiments.
- `decoupled_reauditing/figures/`: scripts that turn CSVs into paper figures.
- `decoupled_reauditing/tests/`: toy tests for verifier behavior, metrics, judge formatting, and Proposition 1 scheduling.

## Experiments

The five experiment scripts produce the paper tables and figures:

- `exp1_naive.py` -> `results/exp1_naive.csv`
- `exp2_method.py` -> `results/exp2_method.csv`
- `exp3_ablation.py` -> `results/exp3_ablation.csv`
- `exp4_pooloverlap.py` -> `results/exp4_overlap.csv`
- `exp5_failure.py` -> `results/exp5_failure.csv`

The runner also emits:

- `results/judge_certification.csv`: independent judge agreement with GSM8K gold on a held-out probe set.
- `figures/fig1_headline.png`: naive vs method reported/true accuracy.
- `figures/fig2_ablation.png`: ablation and synthetic overlap plots.

## Reproducibility Discipline

The code is designed for single-GPU runs with 4-bit loading. It fixes seed `42`, bounds every generation call, checkpoints every generation to JSONL, and keeps GSM8K train/eval/probe slices disjoint.

The independent Math-Shepherd judge never participates in filtering or re-auditing. It is used only through measurement code for true accuracy, gaming gap, contamination diagnostics, and judge-vs-gold certification.

## Running

Install pinned dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
python -m pytest decoupled_reauditing/tests
```

Run the full smoke-to-configured chain:

```bash
python run_all.py
```

Resume without rerunning smoke:

```bash
python run_all.py --skip-smoke
```

By default `DRA_SMOKE_TEST=1`, so the tiny smoke regime is active. For pilot or full runs:

```bash
export DRA_SMOKE_TEST=0
export DRA_PILOT_MODE=1   # pilot
python run_all.py
```

For full:

```bash
export DRA_SMOKE_TEST=0
export DRA_PILOT_MODE=0
python run_all.py
```

## Paper Draft

`research.tex` contains the current AAAI-style draft with:

- the monitor-gaming setup,
- the Decoupled Re-Auditing algorithm,
- Proposition 1 and the pool-diversity bound,
- GSM8K experimental design,
- result tables and figure placeholders,
- limitations and failure analysis.

