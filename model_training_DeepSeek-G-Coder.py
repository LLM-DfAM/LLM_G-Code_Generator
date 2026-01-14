import os
import torch
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    logging,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model

# 0. Configuration and hyperparameters
# ==============================================================================
MODEL_NAME_OR_PATH = "./base_model/DeepSeek-Coder-V2-Lite-Instruct"
DATASET_PATH = "dataset/"
OUTPUT_DIR = "gcode-generator-DeepSeek-G-Coder"
WANDB_PROJECT = "gcode-generator"

# Training parameters
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 5
MAX_SEQ_LENGTH = 20480
LOGGING_STEPS = 10
SAVE_STEPS = 100
WARMUP_STEPS = 30

# 1. Data preprocessing
# ==============================================================================
def tokenize_and_mask_prompt(examples, tokenizer):
    """
    data sample format: "teeth=14... <|endofprompt|> G0 X..."
    DeepSeek ChatML: User: ... Assistant: ...
    """
    input_ids_list = []
    labels_list = []


    SPLIT_TOKEN = "<|endofprompt|>"

    for raw_text in examples["text"]:
        if SPLIT_TOKEN in raw_text:
            parts = raw_text.split(SPLIT_TOKEN)
            user_part = parts[0].strip()
            assistant_part = parts[1].replace("<|endoftext|>", "").strip()
        else:
            user_part = raw_text[:50]
            assistant_part = ""

        # --- Construct Deepseek chat message structure ---
        messages = [
            {"role": "user", "content": user_part},
            {"role": "assistant", "content": assistant_part}
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_generation_prompt=False,
            return_tensors=None
        )

        user_messages = [{"role": "user", "content": user_part}]
        user_only_ids = tokenizer.apply_chat_template(
            user_messages,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            add_generation_prompt=True,
            return_tensors=None
        )

        len_user = len(user_only_ids)

        labels = [-100] * len_user + input_ids[len_user:]

        if len(input_ids) != len(labels):
            min_len = min(len(input_ids), len(labels))
            input_ids = input_ids[:min_len]
            labels = labels[:min_len]

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": [[1] * len(ids) for ids in input_ids_list]
    }


def main():
    logging.set_verbosity_info()
    warnings.filterwarnings("ignore")

    # --- Load Deepseek Native tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME_OR_PATH,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Load dataset ---
    raw_datasets = load_dataset("json", data_dir=DATASET_PATH, cache_dir=f"{OUTPUT_DIR}/.cache")

    tokenized_datasets = raw_datasets.map(
        tokenize_and_mask_prompt,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=raw_datasets["train"].column_names,
        desc="Formatting to DeepSeek Template",
        load_from_cache_file=False
    )

    # --- Load the model configuration ---
    config = AutoConfig.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    config.use_cache = False

    # --- Load LLM ---
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # --- LoRA config ---
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "kv_a_proj_with_mrope", "kv_b_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Training arguments config ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCHS,

        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="wandb",
        run_name=f"{WANDB_PROJECT}-DeepSeek-G-Coder",

        bf16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        group_by_length=True,

        ddp_find_unused_parameters=False,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type="cosine",
        dataloader_num_workers=8,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Start fine-tuning ---
    if list(path for path in os.listdir(training_args.output_dir) if "checkpoint" in path):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # --- Save fine-tuned LLM ---
    final_path = f"{OUTPUT_DIR}/final_model"
    trainer.save_model(final_path)


if __name__ == "__main__":
    main()