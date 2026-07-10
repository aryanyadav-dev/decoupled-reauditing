from decoupled_reauditing.metrics import compute_contamination, generation_metrics, pool_diversity


class V:
    def __init__(self, value):
        self.value = value
    def accepts(self, problem, trace, context=None):
        return self.value


class J:
    def accepts(self, problem, trace):
        return False


def test_gaming_gap_sign_and_reported_uses_current_filter():
    eval_records = [{"problem": "p", "gold_answer": 1}]
    out = generation_metrics(eval_records, ["The answer is: 1"], V(True), [], J())
    assert out["reported"] == 1.0
    assert out["true"] == 0.0
    assert out["gap"] == 1.0


def test_contamination_uses_gold():
    clean = [{"trace": "The answer is: 2", "gold_answer": 1}, {"trace": "The answer is: 1", "gold_answer": 1}]
    assert compute_contamination(clean) == 0.5


def test_pool_diversity_extremes():
    assert pool_diversity([{1, 2}, {3, 4}, {5, 6}]) == 1.0
    assert pool_diversity([{1, 2}, {1, 2}, {1, 2}]) == 0.0

