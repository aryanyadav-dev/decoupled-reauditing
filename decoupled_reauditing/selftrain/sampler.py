from typing import Dict, List

import time
import torch
from tqdm import tqdm

from decoupled_reauditing import config
from decoupled_reauditing.utils import build_prompt, parse_final_answer


def sample_traces(policy, tokenizer, data: List[Dict], k: int = config.K_SAMPLES) -> List[Dict]:
    """Sample k reasoning traces per problem using batched generation.
    
    Generates traces in batches of GEN_BATCH_SIZE problems at a time for speed.
    Each problem generates k samples, so actual batch size = GEN_BATCH_SIZE * k.
    Uses left-padding for decoder-only models.
    
    Args:
        policy: Policy model for generation
        tokenizer: Tokenizer for the policy
        data: List of problem dicts with 'problem' and 'gold_answer'
        k: Number of samples per problem (default: config.K_SAMPLES)
    
    Returns:
        List of dicts with problem, trace, final_answer, etc.
    """
    start_time = time.time()
    out = []
    batch_size = config.GEN_BATCH_SIZE
    
    # CRITICAL: Switch policy to inference mode for generation
    # After fine-tuning, the model is left with gradient_checkpointing=ON and use_cache=OFF,
    # which breaks .generate() with KV-cache/attention-mask shape mismatch.
    # Save training state and switch to inference config.
    was_training = policy.training
    if hasattr(policy, 'gradient_checkpointing_disable'):
        policy.gradient_checkpointing_disable()
    if hasattr(policy, 'config'):
        policy.config.use_cache = True
    policy.eval()
    
    print(f"[sample_traces] Switched policy to inference mode (gradient_checkpointing=OFF, use_cache=True, eval)")
    
    # Set left-padding for decoder-only generation (required for batching)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Process data in batches of batch_size problems
    for batch_start in tqdm(range(0, len(data), batch_size), desc="sampling"):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_prompts = []
        batch_metadata = []
        
        # For each problem in this batch, create k copies of the prompt
        for idx_in_full, item in enumerate(batch_data, start=batch_start):
            prompt = build_prompt(item["problem"])
            # Duplicate prompt k times for k samples per problem
            for _ in range(k):
                batch_prompts.append(prompt)
                batch_metadata.append({
                    "problem_id": idx_in_full,
                    "problem": item["problem"],
                    "gold_answer": item["gold_answer"],
                })
        
        # Tokenize entire batch with left-padding
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(policy.device)
        input_len = enc["input_ids"].shape[1]
        
        # Generate for entire batch at once
        with torch.no_grad():
            gen = policy.generate(
                **enc,
                do_sample=True,
                temperature=config.TEMPERATURE,
                max_new_tokens=config.MAX_NEW_TOKENS_POLICY,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode each generated trace
        traces_batch = []
        for row in gen:
            text = tokenizer.decode(row[input_len:], skip_special_tokens=True)
            traces_batch.append(text)
        
        # Group traces by problem (k consecutive traces belong to same problem)
        for prob_idx in range(len(batch_data)):
            start_idx = prob_idx * k
            end_idx = start_idx + k
            samples = traces_batch[start_idx:end_idx]
            
            # Add each trace as a separate dict (preserves original output structure)
            for trace in samples:
                metadata = batch_metadata[start_idx]  # All k samples share same metadata
                out.append({
                    "problem_id": metadata["problem_id"],
                    "problem": metadata["problem"],
                    "gold_answer": metadata["gold_answer"],
                    "trace": trace,
                    "final_answer": parse_final_answer(trace),
                    "all_samples": samples,  # All k samples for this problem
                })
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Restore training config for next fine-tune phase
    # Fine-tuning expects: gradient_checkpointing=ON, use_cache=False, train()
    if hasattr(policy, 'config'):
        policy.config.use_cache = False
    if hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()
    if was_training:
        policy.train()
    
    print(f"[sample_traces] Restored policy to training config (gradient_checkpointing=ON, use_cache=False)")
    
    elapsed = time.time() - start_time
    print(f"[sample_traces] Generated {len(data)} problems × {k} samples = {len(out)} traces in {elapsed:.1f}s")
    print(f"[sample_traces] Throughput: {len(out)/elapsed:.2f} traces/sec")
    
    return out


def greedy_eval_traces(policy, tokenizer, data: List[Dict]) -> List[str]:
    """Generate greedy (deterministic) evaluation traces using batched generation.
    
    Args:
        policy: Policy model for generation
        tokenizer: Tokenizer for the policy
        data: List of problem dicts with 'problem' key
    
    Returns:
        List of generated trace strings (one per problem, in order)
    """
    start_time = time.time()
    traces = []
    batch_size = config.GEN_BATCH_SIZE
    
    # CRITICAL: Switch policy to inference mode for generation
    # Same fix as sample_traces - prevent gradient_checkpointing / use_cache mismatch
    was_training = policy.training
    if hasattr(policy, 'gradient_checkpointing_disable'):
        policy.gradient_checkpointing_disable()
    if hasattr(policy, 'config'):
        policy.config.use_cache = True
    policy.eval()
    
    print(f"[greedy_eval_traces] Switched policy to inference mode (gradient_checkpointing=OFF, use_cache=True, eval)")
    
    # Set left-padding for decoder-only generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Process data in batches
    for batch_start in tqdm(range(0, len(data), batch_size), desc="eval-generation"):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_prompts = [build_prompt(item["problem"]) for item in batch_data]
        
        # Tokenize batch with left-padding
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False).to(policy.device)
        input_len = enc["input_ids"].shape[1]
        
        # Generate for entire batch (greedy/deterministic)
        with torch.no_grad():
            gen = policy.generate(
                **enc,
                do_sample=False,
                max_new_tokens=config.MAX_NEW_TOKENS_POLICY,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode each trace
        for row in gen:
            text = tokenizer.decode(row[input_len:], skip_special_tokens=True)
            traces.append(text)
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    # Restore training config for next fine-tune phase
    if hasattr(policy, 'config'):
        policy.config.use_cache = False
    if hasattr(policy, 'gradient_checkpointing_enable'):
        policy.gradient_checkpointing_enable()
    if was_training:
        policy.train()
    
    print(f"[greedy_eval_traces] Restored policy to training config (gradient_checkpointing=ON, use_cache=False)")
    
    elapsed = time.time() - start_time
    print(f"[greedy_eval_traces] Generated {len(data)} greedy traces in {elapsed:.1f}s")
    print(f"[greedy_eval_traces] Throughput: {len(traces)/elapsed:.2f} traces/sec")
    
    return traces

