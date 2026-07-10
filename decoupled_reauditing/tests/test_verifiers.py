import pytest

from decoupled_reauditing.verifiers.self_consistency import SelfConsistencyVerifier
from decoupled_reauditing.verifiers.symbolic import SymbolicChecker


class FakeTok:
    pad_token_id = 0
    def __call__(self, prompt, return_tensors=None):
        return FakeInputs()
    def decode(self, ids, skip_special_tokens=True):
        return self.answer


class FakeInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": [[1, 2]]})
    def to(self, device):
        return self


class FakeModel:
    device = "cpu"
    def generate(self, **kwargs):
        return [[1, 2, 3]]


def test_symbolic_accept_reject():
    v = SymbolicChecker()
    assert v.accepts("p", "2 + 3 = 5. The answer is: 5")
    assert not v.accepts("p", "2 + 3 = 6. The answer is: 6")


def test_self_consistency_accept_reject():
    v = SelfConsistencyVerifier()
    ctx = {"samples": ["The answer is: 4", "The answer is: 4", "The answer is: 5"]}
    assert v.accepts("p", "The answer is: 4", ctx)
    assert not v.accepts("p", "The answer is: 5", ctx)

