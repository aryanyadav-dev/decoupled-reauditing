from typing import Dict, Iterable, List, Sequence, Set

from .utils import parse_final_answer


def make_independent_judge(*args, **kwargs):
    from .judge import MathShepherdJudge

    return MathShepherdJudge(*args, **kwargs)


def mean_bool(vals: Iterable[bool]) -> float:
    vals = list(vals)
    return sum(bool(v) for v in vals) / len(vals) if vals else 0.0


def judge_gold_agreement(judge, probe_set):
    """Fraction of held-out probe traces where J agrees with GSM8K gold truth.

    The probe set is constructed outside filtering from GSM8K test examples after
    the eval slice, so it is disjoint from both self-training train problems and
    eval problems. Each tuple is (problem, trace, gold_answer); traces are sampled
    from the base policy before any fine-tuning, yielding a natural mix of correct
    and incorrect solutions. This is measurement-only and never touches filtering.
    """
    correct = 0
    for problem, trace, gold in probe_set:
        truth = parse_final_answer(trace) == gold
        verdict = judge.accepts(problem, trace)
        if verdict == truth:
            correct += 1
    return correct / max(1, len(probe_set))


def compute_reported_acc(eval_records, traces, v_filt, contexts=None) -> float:
    contexts = contexts or {}
    vals = []
    for rec, trace in zip(eval_records, traces):
        vals.append(v_filt.accepts(rec["problem"], trace, contexts.get(rec["problem"], {})))
    return mean_bool(vals)


def compute_true_acc(eval_records, traces, judge) -> float:
    return mean_bool(judge.accepts(rec["problem"], trace) for rec, trace in zip(eval_records, traces))


def compute_contamination(clean_set: Sequence[Dict]) -> float:
    if not clean_set:
        return 0.0
    wrong = 0
    for item in clean_set:
        wrong += parse_final_answer(item["trace"]) != item["gold_answer"]
    return wrong / len(clean_set)


def omega_for_verifier(verifier, probe_set, contexts=None) -> Set[int]:
    contexts = contexts or {}
    omega = set()
    for i, item in enumerate(probe_set):
        trace = item["trace"]
        accepted = verifier.accepts(item["problem"], trace, contexts.get(item["problem"], {}))
        wrong = parse_final_answer(trace) != item["gold_answer"]
        if accepted and wrong:
            omega.add(i)
    return omega


def pool_diversity(omegas: Sequence[Set[int]]) -> float:
    if len(omegas) < 2:
        return 1.0
    max_j = 0.0
    for i in range(len(omegas)):
        for j in range(i + 1, len(omegas)):
            union = omegas[i] | omegas[j]
            jac = 1.0 if not union else len(omegas[i] & omegas[j]) / len(union)
            max_j = max(max_j, jac)
    return 1.0 - max_j


def generation_metrics(eval_records, eval_traces, v_filt, clean_set, judge, contexts=None) -> Dict[str, float]:
    reported = compute_reported_acc(eval_records, eval_traces, v_filt, contexts)
    true = compute_true_acc(eval_records, eval_traces, judge)
    contam = compute_contamination(clean_set)
    return {"reported": reported, "true": true, "gap": reported - true, "contam": contam}
