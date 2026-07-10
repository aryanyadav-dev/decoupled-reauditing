from dataclasses import dataclass

import torch

from decoupled_reauditing import config


PROMPT_TEMPLATE = """[INST] You are a strict math verifier.
Check whether the proposed solution's reasoning steps are correct and internally justified.
Answer with exactly YES or NO.

Problem:
{problem}

Solution:
{trace}
[/INST]"""


@dataclass
class LLMJudgeVerifier:
    model: object = None
    tokenizer: object = None
    name: str = "llm_judge"

    def accepts(self, problem, trace, context=None) -> bool:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLMJudgeVerifier requires a frozen model and tokenizer.")
        prompt = PROMPT_TEMPLATE.format(problem=problem, trace=trace)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=config.MAX_NEW_TOKENS_JUDGE,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        first = decoded.strip().split()[0].upper().strip(".,:;") if decoded.strip() else ""
        if first == "YES":
            return True
        if first == "NO":
            return False
        return decoded.strip().upper().startswith("YES")

