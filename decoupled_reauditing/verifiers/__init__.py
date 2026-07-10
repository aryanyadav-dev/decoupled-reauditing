from .symbolic import SymbolicChecker
from .llm_judge import LLMJudgeVerifier
from .self_consistency import SelfConsistencyVerifier


def make_real_pool(llm_model=None, llm_tokenizer=None):
    return [
        SymbolicChecker(),
        LLMJudgeVerifier(model=llm_model, tokenizer=llm_tokenizer),
        SelfConsistencyVerifier(),
    ]

