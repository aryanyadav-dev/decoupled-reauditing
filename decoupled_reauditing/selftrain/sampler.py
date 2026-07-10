from typing import Dict, List

import torch
from tqdm import tqdm

from decoupled_reauditing import config
from decoupled_reauditing.utils import build_prompt, parse_final_answer


def sample_traces(policy, tokenizer, data: List[Dict], k: int = config.K_SAMPLES) -> List[Dict]:
    out = []
    for idx, item in enumerate(tqdm(data, desc="sampling")):
        prompt = build_prompt(item["problem"])
        enc = tokenizer([prompt] * k, return_tensors="pt", padding=True).to(policy.device)
        with torch.no_grad():
            gen = policy.generate(
                **enc,
                do_sample=True,
                temperature=config.TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS_POLICY,
                pad_token_id=tokenizer.pad_token_id,
            )
        samples = []
        for row in gen:
            text = tokenizer.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            samples.append(text)
        for trace in samples:
            out.append(
                {
                    "problem_id": idx,
                    "problem": item["problem"],
                    "gold_answer": item["gold_answer"],
                    "trace": trace,
                    "final_answer": parse_final_answer(trace),
                    "all_samples": samples,
                }
            )
    return out


def greedy_eval_traces(policy, tokenizer, data: List[Dict]) -> List[str]:
    traces = []
    for item in tqdm(data, desc="eval-generation"):
        prompt = build_prompt(item["problem"])
        enc = tokenizer(prompt, return_tensors="pt").to(policy.device)
        with torch.no_grad():
            gen = policy.generate(
                **enc,
                do_sample=False,
                max_new_tokens=config.MAX_NEW_TOKENS_POLICY,
                pad_token_id=tokenizer.pad_token_id,
            )
        traces.append(tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return traces

