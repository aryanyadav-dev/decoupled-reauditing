from typing import Dict, List

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Handle trl API changes: trl 0.12+ uses SFTConfig
try:
    from trl import SFTTrainer, SFTConfig
    USE_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    USE_SFT_CONFIG = False
    # Fall back to TrainingArguments for older trl
    try:
        from transformers import TrainingArguments
    except ImportError:
        from transformers.training_args import TrainingArguments

from decoupled_reauditing import config
from decoupled_reauditing.utils import build_prompt


def format_training_text(item: Dict) -> str:
    return f"{build_prompt(item['problem'])}\n{item['trace']}"


def finetune_policy(policy, tokenizer, clean_set: List[Dict], output_dir: str):
    if not clean_set:
        return policy
    
    # Prepare model for k-bit training if not already PEFT model
    if not getattr(policy, "is_peft_model", False):
        policy = prepare_model_for_kbit_training(policy)
        lora = LoraConfig(
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        policy = get_peft_model(policy, lora)
    else:
        # Model already has PEFT, reuse its config
        lora = None
    
    ds = Dataset.from_dict({"text": [format_training_text(x) for x in clean_set]})
    
    if USE_SFT_CONFIG:
        # Modern trl 0.12+ API: use SFTConfig
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=config.TRAIN_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=5,
            save_steps=max(config.TRAIN_STEPS, 1),
            report_to=[],
            seed=config.SEED,
            data_seed=config.SEED,
            fp16=True,
            dataset_text_field="text",
            max_seq_length=1024,
            packing=False,
        )
        
        # Create trainer with modern API
        trainer = SFTTrainer(
            model=policy,
            processing_class=tokenizer,
            train_dataset=ds,
            args=sft_config,
            peft_config=lora if lora is not None else None,
        )
    else:
        # Legacy trl API: use TrainingArguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            max_steps=config.TRAIN_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=5,
            save_steps=max(config.TRAIN_STEPS, 1),
            report_to=[],
            seed=config.SEED,
            data_seed=config.SEED,
            fp16=True,
        )
        
        # Try modern tokenizer parameter name first, fall back to old name
        try:
            trainer = SFTTrainer(
                model=policy,
                processing_class=tokenizer,
                train_dataset=ds,
                dataset_text_field="text",
                max_seq_length=1024,
                args=training_args,
                packing=False,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=policy,
                tokenizer=tokenizer,
                train_dataset=ds,
                dataset_text_field="text",
                max_seq_length=1024,
                args=training_args,
                packing=False,
            )
    
    trainer.train()
    return trainer.model

