import os
import torch
import warnings
import math
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    logging,
    DataCollatorForSeq2Seq
)
from peft import PeftModel

try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2

    USE_LIGER = True
except ImportError:
    USE_LIGER = False

BASE_MODEL_PATH = "./base_model/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "gcode-generator-Qwen2-G-Coder/final_model"
DATASET_PATH = "dataset/"
MAX_SEQ_LENGTH = 20480
EVAL_BATCH_SIZE = 1


def tokenize_and_mask_prompt(examples, tokenizer):
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


def run_evaluation(split_name, trainer, dataset):
    print(f"\n=== Evaluating: {split_name} Set ({len(dataset)} samples) ===")
    try:
        metrics = trainer.evaluate(eval_dataset=dataset)
        eval_loss = metrics["eval_loss"]
        # Prevent exp overflow due to excessive Loss
        perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")

        print(f"--- [{split_name} Evaluation Results] ---")
        print(f"  Sample Count:      {len(dataset)}")
        print(f"  Loss:              {eval_loss:.6f}")
        print(f"  Perplexity:        {perplexity:.6f}")
        print("-----------------------------")
        return eval_loss
    except Exception as e:
        print(f"[{split_name}] Evaluation Failed: {e}")
        return None


def main():
    if USE_LIGER:
        apply_liger_kernel_to_qwen2()

    logging.set_verbosity_info()
    warnings.filterwarnings("ignore")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_datasets = load_dataset("json", data_dir=DATASET_PATH, cache_dir=f"{ADAPTER_PATH}/.cache_eval")

    processed_datasets = raw_datasets.map(
        tokenize_and_mask_prompt,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=["text"],
        desc="Formatting to ChatML",
    )

    config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    model.to("cuda")

    eval_args = TrainingArguments(
        output_dir=f"{ADAPTER_PATH}/eval_logs",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        bf16=True,
        report_to="none",
        prediction_loss_only=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest",
        label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if "validation" in processed_datasets:
        run_evaluation("Validation", trainer, processed_datasets["validation"])

    if "test" in processed_datasets:
        run_evaluation("Test", trainer, processed_datasets["test"])

    print("\nAll evaluations completed.")


if __name__ == "__main__":
    main()