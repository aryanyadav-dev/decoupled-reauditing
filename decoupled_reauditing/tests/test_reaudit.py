from decoupled_reauditing.selftrain.reaudit import decoupled_reaudit, rotation_pair, rotation_schedule


class V:
    def __init__(self, name, accept):
        self.name = name
        self.accept = accept
    def accepts(self, problem, trace, context=None):
        return self.accept


def test_rotation_indices():
    pool = [V("a", True), V("b", True), V("c", True)]
    assert [x.name for x in rotation_pair(pool, 0)] == ["a", "b"]
    assert [x.name for x in rotation_pair(pool, 1)] == ["b", "c"]
    assert [x.name for x in rotation_pair(pool, 2)] == ["c", "a"]


def test_filter_coverage_per_cycle():
    for M in (2, 3, 4):
        schedule = rotation_schedule(M, M)
        assert {filt_idx for filt_idx, _ in schedule} == set(range(M))


def test_auditor_disjoint():
    for M in (2, 3, 4):
        schedule = rotation_schedule(M, M)
        assert all(filt_idx != audit_idx for filt_idx, audit_idx in schedule)


def test_reaudit_subset():
    samples = [{"problem_id": 0, "problem": "p", "trace": "t"}]
    contexts = {0: {"samples": ["t"]}}
    pool = [V("f", True), V("a", False), V("x", True)]
    _, _, accepted, clean = decoupled_reaudit(samples, pool, 0, contexts)
    assert len(accepted) == 1
    assert clean == []
    assert set(map(id, clean)).issubset(set(map(id, accepted)))
