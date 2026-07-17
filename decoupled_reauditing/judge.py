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
    """Resolve Math-Shepherd PRM token IDs following official recipe.
    
    Math-Shepherd uses '+' for good steps, '-' for bad steps, and 'ки' as step tag.
    Official inference recipe: encode("+ -")[1:] to get candidate tokens.
    
    Args:
        tokenizer: Math-Shepherd tokenizer
        
    Returns:
        tuple: (candidate_tokens, step_tag_id) where candidate_tokens = [good_id, bad_id]
    """
    good_token = "+"
    bad_token = "-"
    step_tag = "ки"
    
    # Method 1: Official Math-Shepherd recipe - encode "+ -" and drop leading token
    # This handles tokenizers that prepend BOS or space tokens
    try:
        candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]
        step_tag_id = tokenizer.encode(step_tag)[-1]
        
        if len(candidate_tokens) == 2 and candidate_tokens[0] != candidate_tokens[1]:
            print(f"[resolve_prm_tokens] Method 1 (encode): good=+={candidate_tokens[0]}, bad=-={candidate_tokens[1]}, step_tag=ки={step_tag_id}")
            
            # Validate against known values for peiyi9979/math-shepherd-mistral-7b-prm
            # Model card values: + -> 648, - -> 387, step tag -> 12902
            if candidate_tokens == [648, 387] and step_tag_id == 12902:
                print(f"[resolve_prm_tokens] ✓ Token IDs match Math-Shepherd expected values")
                return candidate_tokens, step_tag_id
            else:
                print(f"[resolve_prm_tokens] WARNING: Token IDs differ from expected [648, 387, 12902]")
                print(f"[resolve_prm_tokens] This may indicate tokenizer drift or different checkpoint")
                return candidate_tokens, step_tag_id
    except Exception as e:
        print(f"[resolve_prm_tokens] Method 1 failed: {e}")
    
    # Method 2: Fallback - direct token-to-id conversion
    try:
        good_id = tokenizer.convert_tokens_to_ids(good_token)
        bad_id = tokenizer.convert_tokens_to_ids(bad_token)
        step_tag_id = tokenizer.convert_tokens_to_ids(step_tag)
        
        # Verify none are unknown token ID
        unk_id = tokenizer.unk_token_id
        if good_id != unk_id and bad_id != unk_id and step_tag_id != unk_id and good_id != bad_id:
            candidate_tokens = [good_id, bad_id]
            print(f"[resolve_prm_tokens] Method 2 (convert_tokens_to_ids): good=+={good_id}, bad=-={bad_id}, step_tag=ки={step_tag_id}")
            return candidate_tokens, step_tag_id
    except Exception as e:
        print(f"[resolve_prm_tokens] Method 2 failed: {e}")
    
    # Both methods failed - raise error with diagnostic info
    try:
        encoded_combined = tokenizer.encode(f"{good_token} {bad_token}")
        encoded_good = tokenizer.encode(good_token)
        encoded_bad = tokenizer.encode(bad_token)
        encoded_step = tokenizer.encode(step_tag)
    except:
        encoded_combined = encoded_good = encoded_bad = encoded_step = "error"
    
    raise RuntimeError(
        f"Could not resolve Math-Shepherd +/- candidate tokens exactly.\n"
        f"  encode('+ -') = {encoded_combined}\n"
        f"  encode('+') = {encoded_good}\n"
        f"  encode('-') = {encoded_bad}\n"
        f"  encode('ки') = {encoded_step}\n"
        f"Tokenizer may be incompatible with Math-Shepherd PRM checkpoint."
    )


def format_for_prm(problem: str, trace: str) -> str:
    steps = split_steps(trace)
    tagged = "\n".join(f"{step} {STEP_TAG}" for step in steps)
    return f"{problem} {tagged}"


@dataclass
class MathShepherdJudge:
    model: object = None
    tokenizer: object = None
    threshold: float = config.PRM_THRESHOLD
    device_index: int = 1  # Default to cuda:1 for independent judge

    def __post_init__(self):
        if self.model is None or self.tokenizer is None:
            # Fall back to cuda:0 if only one GPU is available
            device_index = self.device_index
            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                print(f"[MathShepherdJudge] Only 1 GPU available, using cuda:0 (requested cuda:{device_index})")
                device_index = 0
            self.model, self.tokenizer = load_model(config.JUDGE_ID, config.LOAD_IN_4BIT, device_index=device_index)
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

