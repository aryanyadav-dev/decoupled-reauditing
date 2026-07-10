from collections import Counter
from dataclasses import dataclass

from decoupled_reauditing.utils import parse_final_answer


@dataclass
class SelfConsistencyVerifier:
    name: str = "self_consistency"

    def accepts(self, problem, trace, context=None) -> bool:
        if context is None or "samples" not in context:
            raise ValueError("self_consistency requires context['samples'] with all k traces.")
        answers = [parse_final_answer(s) for s in context["samples"]]
        answers = [a for a in answers if a is not None]
        if not answers:
            return False
        majority = Counter(answers).most_common(1)[0][0]
        return parse_final_answer(trace) == majority

