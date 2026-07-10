"""Math-Shepherd PRM measurement wrapper.

This module is measurement-only. It must not be imported by filtering paths.
"""

import inspect
import re
from dataclasses import dataclass
from typing import List

import torch

from . import config
from .utils import load_model

_forbidden = ("verifiers/", "verifiers.", "selftrain/loop.py")
for frame in inspect.stack()[1:]:
    marker = frame.filename.replace("\\", "/")
    mod = frame.frame.f_globals.get("__name__", "")
    if any(x in marker or x in mod for x in _forbidden):
        raise ImportError("judge.py must never be imported by filtering or acceptance paths.")

GOOD_TOKEN = "+"
BAD_TOKEN = "-"
STEP_TAG = "ки"


def split_steps(trace: str) -> List[str]:
    lines = [x.strip() for x in re.split(r"\n+|(?=Step\s+\d+\s*:)", trace) if x.strip()]
    if len(lines) <= 1:
        lines = [x.strip() for x in re.split(r"(?<=[.!?])\s+", trace) if x.strip()]
    return lines or [trace.strip()]


def resolve_prm_tokens(tokenizer):
    candidate_tokens = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]
    step_tag_id = tokenizer.encode(STEP_TAG)[-1]
    if len(candidate_tokens) != 2:
        raise RuntimeError("Could not resolve Math-Shepherd + / - candidate tokens exactly.")
    if step_tag_id is None:
        raise RuntimeError("Could not resolve Math-Shepherd step tag token.")
    if candidate_tokens[0] == candidate_tokens[1]:
        raise RuntimeError("Math-Shepherd positive and negative token ids collapsed.")
    # Model card values for this checkpoint: + -> 648, - -> 387, step tag -> 12902.
    # Raise instead of silently scoring the wrong positions if tokenizer drift occurs.
    if candidate_tokens != [648, 387] or step_tag_id != 12902:
        raise RuntimeError(
            f"Unexpected PRM tokenization: candidates={candidate_tokens}, step_tag_id={step_tag_id}"
        )
    return candidate_tokens, step_tag_id


def format_for_prm(problem: str, trace: str) -> str:
    steps = split_steps(trace)
    tagged = "\n".join(f"{step} {STEP_TAG}" for step in steps)
    return f"{problem} {tagged}"


@dataclass
class MathShepherdJudge:
    model: object = None
    tokenizer: object = None
    threshold: float = config.PRM_THRESHOLD

    def __post_init__(self):
        if self.model is None or self.tokenizer is None:
            self.model, self.tokenizer = load_model(config.JUDGE_ID, config.LOAD_IN_4BIT)
        self.candidate_tokens, self.step_tag_id = resolve_prm_tokens(self.tokenizer)
        self.model.eval()

    def score(self, problem: str, trace: str) -> float:
        text = format_for_prm(problem, trace)
        input_ids = torch.tensor([self.tokenizer.encode(text)], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = scores[input_ids == self.step_tag_id]
        if step_scores.numel() == 0:
            raise RuntimeError("No Math-Shepherd step-token positions found.")
        return float(step_scores.min().detach().cpu())

    def accepts(self, problem: str, trace: str) -> bool:
        return self.score(problem, trace) >= self.threshold

