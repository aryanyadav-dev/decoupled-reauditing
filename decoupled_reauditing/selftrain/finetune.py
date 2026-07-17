from typing import Dict, List

from datasets import Dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import trl

from decoupled_reauditing import config
from decoupled_reauditing.utils import build_prompt


def format_training_text(item: Dict) -> str:
    """Format a clean_set item into full training text (problem + trace)."""
    return f"{build_prompt(item['problem'])}\n{item['trace']}"


def finetune_policy(policy, tokenizer, clean_set: List[Dict], output_dir: str):
    """Fine-tune policy with LoRA on clean training set.
    
    Args:
        policy: The policy model to fine-tune
        tokenizer: Tokenizer for the model
        clean_set: List of dicts with 'problem' and 'trace' keys
        output_dir: Directory to save adapter weights
    
    Returns:
        Fine-tuned policy model
    """
    print(f"[finetune_policy] trl version: {trl.__version__}")
    
    if not clean_set:
        print("[finetune_policy] Empty clean_set, returning policy unchanged")
        return policy
    
    # Check if model already has LoRA adapter (from previous generation)
    is_peft_model = getattr(policy, "is_peft_model", False)
    
    if is_peft_model:
        print(f"[finetune_policy] Model is already a PeftModel (generation t>0); continuing training on existing adapter")
        # Don't pass peft_config to SFTTrainer - would conflict with existing PeftModel
        lora_config = None
    else:
        print(f"[finetune_policy] Model is not yet a PeftModel (generation 0); SFTTrainer will apply LoRA")
        # Let SFTTrainer apply LoRA via peft_config
        # Prepare model for k-bit training first
        policy = prepare_model_for_kbit_training(policy)
        lora_config = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        print(f"[finetune_policy] LoRA config prepared: r={config.LORA_R}, alpha={config.LORA_ALPHA}")
    
    # Convert clean_set to HuggingFace Dataset with proper format
    # Each item needs a "text" column containing the full training string
    formatted_data = []
    for item in clean_set:
        # Build full training text: problem + trace
        full_text = format_training_text(item)
        formatted_data.append({"text": full_text})
    
    # Create Dataset from list of dicts
    ds = Dataset.from_list(formatted_data)
    
    # Validation: print dataset info before training
    print(f"[finetune_policy] Dataset info:")
    print(f"  - Length: {len(ds)}")
    print(f"  - Column names: {ds.column_names}")
    print(f"  - Features: {ds.features}")
    if len(ds) > 0:
        sample_text = ds[0]["text"]
        print(f"  - Sample text (first 200 chars): {sample_text[:200]}...")
        print(f"  - Sample text length: {len(sample_text)} chars")
    
    # Build SFTConfig with ALL training and dataset parameters
    # trl 1.8.0 API: use max_length (NOT max_seq_length) and dataset_text_field
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=config.TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="steps",
        save_steps=max(config.TRAIN_STEPS, 1),
        report_to=[],
        seed=config.SEED,
        data_seed=config.SEED,
        fp16=True,
        dataset_text_field="text",  # Must match the column name in Dataset
        max_length=1024,
        packing=False,
    )
    
    print(f"[finetune_policy] SFTConfig created: max_steps={config.TRAIN_STEPS}, max_length=1024, dataset_text_field='text'")
    
    # Create trainer with trl 1.8.0 API
    # Multi-generation strategy:
    # - Generation 0: pass peft_config to SFTTrainer, which applies LoRA and trains
    # - Generation t>0: model is already PeftModel, pass peft_config=None, just train existing adapter
    # This avoids "You passed a PeftModel instance together with a peft_config" error
    trainer = SFTTrainer(
        model=policy,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=lora_config,  # None if already PeftModel, LoraConfig if generation 0
    )
    
    print(f"[finetune_policy] Starting training for {config.TRAIN_STEPS} steps...")
    trainer.train()
    print(f"[finetune_policy] Training complete")
    
    return trainer.model

