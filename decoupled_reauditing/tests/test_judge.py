import os

import pytest


def test_prm_good_trace_score_greater_than_bad():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from decoupled_reauditing import config
    from decoupled_reauditing.judge import MathShepherdJudge

    try:
        tok = AutoTokenizer.from_pretrained(config.JUDGE_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(config.JUDGE_ID, local_files_only=True)
    except Exception as exc:
        pytest.skip(f"Math-Shepherd model not cached locally: {exc}")
    judge = MathShepherdJudge(model=model, tokenizer=tok)
    problem = "Janet has 16 eggs, eats 3, bakes with 4, and sells the rest for $2 each. Revenue?"
    good = "Step 1: 16 - 3 = 13.\nStep 2: 13 - 4 = 9.\nStep 3: 9 * 2 = 18. The answer is: 18"
    bad = "Step 1: 16 - 3 = 13.\nStep 2: 13 - 4 = 9.\nStep 3: 9 * 2 = 17. The answer is: 17"
    assert judge.score(problem, good) > judge.score(problem, bad)

