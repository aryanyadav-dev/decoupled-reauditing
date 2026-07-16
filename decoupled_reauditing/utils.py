import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset

# Handle transformers 5.0 import changes
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
except ImportError:
    # Fallback for transformers 5.0+ if imports moved
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    try:
        from transformers import set_seed
    except ImportError:
        from transformers.utils import set_seed

from . import config


def set_all_seeds(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def check_bnb():
    """Check if bitsandbytes is working properly with current GPU/CUDA setup.
    
    Raises:
        RuntimeError: If bitsandbytes cannot execute on current hardware
    """
    try:
        import bitsandbytes
        import torch
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available - bitsandbytes requires CUDA")
        
        # Test basic bitsandbytes functionality with a tiny tensor
        device = torch.cuda.current_device()
        test_tensor = torch.randn(8, 8, device=device, dtype=torch.float16)
        
        # Try to quantize a small tensor
        quantized = bitsandbytes.functional.quantize_4bit(test_tensor)
        
        print(f"[check_bnb] ✓ bitsandbytes {bitsandbytes.__version__} working on CUDA device {device}")
        
    except ImportError as e:
        raise RuntimeError(f"bitsandbytes not installed or importable: {e}")
    except Exception as e:
        raise RuntimeError(f"bitsandbytes cannot execute on current GPU/CUDA: {e}")


def load_model(model_id: str, load_in_4bit: bool = True, padding_side: str = "right", device_index: Optional[int] = None):
    """Load a model with optional device placement.
    
    Args:
        model_id: HuggingFace model identifier
        load_in_4bit: Whether to use 4-bit quantization
        padding_side: Tokenizer padding side
        device_index: Target GPU index (0, 1, etc.). If None, uses auto placement.
                      For 4-bit models, this sets device at load time via device_map={"": device_index}.
    
    Returns:
        (model, tokenizer) tuple
    """
    quantization_config = None
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Determine device_map based on device_index and available GPUs
    if device_index is not None and torch.cuda.is_available():
        # Validate device_index against available GPUs
        num_gpus = torch.cuda.device_count()
        if device_index >= num_gpus:
            print(f"[load_model] WARNING: device_index={device_index} >= num_gpus={num_gpus}, falling back to cuda:0")
            device_index = 0
        
        # For 4-bit models, use device_map={"": device_index} to place at load time
        device_map = {"": device_index}
        target_device = f"cuda:{device_index}"
    else:
        # Don't use device_map="auto" if no device_index specified to avoid accelerate requirement
        device_map = None
        target_device = "default"
    
    if load_in_4bit:
        # Ensure dtype is compatible with bitsandbytes and transformers 5.0
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,  # Explicit bfloat16 for Kaggle T4
            bnb_4bit_use_double_quant=True,
        )
    
    # Handle transformers 5.0+ tokenizer API changes
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side=padding_side)
    except TypeError:
        # Fallback if padding_side parameter changed in 5.0
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if hasattr(tokenizer, 'padding_side'):
            tokenizer.padding_side = padding_side
    
    # Handle transformers 5.0+ model loading API changes
    model_kwargs = {
        "torch_dtype": dtype,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Handle pad token setup - compatible with 5.0
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, 'config') and hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.eos_token_id
    elif hasattr(model, 'config') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Log device placement
    actual_device = str(getattr(model, 'device', target_device))
    print(f"[load_model] {model_id} -> {actual_device}")
    
    return model, tokenizer


def parse_final_answer(text: str) -> Optional[int]:
    if text is None:
        return None
    after_marker = re.findall(r"####\s*([-+]?\d[\d,]*)", text)
    if after_marker:
        return int(after_marker[-1].replace(",", ""))
    answer_phrases = re.findall(
        r"(?:answer\s+is|answer:|therefore|so)\s*\$?\s*([-+]?\d[\d,]*)",
        text,
        flags=re.IGNORECASE,
    )
    if answer_phrases:
        return int(answer_phrases[-1].replace(",", ""))
    nums = re.findall(r"[-+]?\d[\d,]*", text)
    if not nums:
        return None
    return int(nums[-1].replace(",", ""))


def load_gsm8k(split: str) -> List[Dict[str, int]]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows = []
    for row in ds:
        gold = parse_final_answer(row["answer"])
        if gold is None:
            raise ValueError(f"Could not parse GSM8K gold answer: {row['answer']}")
        rows.append({"problem": row["question"], "gold_answer": gold})
    return rows


def build_prompt(problem: str) -> str:
    return (
        "[INST] Solve the math problem step by step. End with 'The answer is: <number>'.\n"
        f"{problem} [/INST]"
    )

