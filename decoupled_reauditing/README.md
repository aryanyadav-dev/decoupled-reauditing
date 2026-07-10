# Decoupled Re-Auditing

Research code for **Decoupled Re-Auditing: Preventing Monitor Gaming in Self-Improving Reasoners**.

## Regimes

Set regimes with environment flags before running:

```bash
export DRA_SMOKE_TEST=1   # smoke, default
export DRA_PILOT_MODE=0
```

Precedence is literal: `SMOKE_TEST`, then `PILOT_MODE`, else full. Full is selected with both flags set to `0`.

## Run

Install pinned dependencies:

```bash
pip install -r decoupled_reauditing/requirements.txt
```

Run one experiment:

```bash
python -m decoupled_reauditing.experiments.exp2_method
```

Run the complete chain:

```bash
python -m decoupled_reauditing.run_all
```

`run_all.py` always runs a smoke pass first, then the configured pilot/full regime. For crash resume:

```bash
python -m decoupled_reauditing.run_all --skip-smoke
```

Each generation writes `results/_ckpt_<experiment>_gen<g>.jsonl` before moving on. Existing generation checkpoints are skipped on restart.

## Outputs

- `results/judge_certification.csv`: `metric,value`; reports one held-out judge-vs-GSM8K-gold agreement number. Probe problems are carved from GSM8K test after the eval slice, never used for training or eval, and traces are sampled from the base policy before any fine-tuning.
- `results/exp1_naive.csv`: `gen,reported,true,gap,contam`; feeds the naive baseline table and Fig. 1.
- `results/exp2_method.csv`: `gen,reported,true,gap,contam`; feeds the method table and Fig. 1.
- `results/exp3_ablation.csv`: `condition,gen,reported,true,gap,contam`; feeds the ablation table and Fig. 2 left.
- `results/exp4_overlap.csv`: `overlap,gap_final`; feeds the synthetic overlap analysis and Fig. 2 right.
- `results/exp5_failure.csv`: `category,share`; feeds the residual-failure characterization table.
- `figures/fig1_headline.png`: reported vs true accuracy for naive and method.
- `figures/fig2_ablation.png`: ablation gap bars and overlap curve.

## Measurement Discipline

Training uses GSM8K train only; evaluation uses GSM8K test only. Gold correctness and contamination are computed from parsed GSM8K final answers, not from any verifier. The independent Math-Shepherd PRM judge is imported only by `metrics.py`-driven measurement code and never participates in filtering.

Experiment 5 is descriptive only. It categorizes final method survivors that are gold-wrong; because `llm_judge` is a pool member, this characterization is not a correctness claim and does not re-verify the method.
