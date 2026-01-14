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
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

BASE_MODEL_PATH = "./base_model/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "gcode-generator-Qwen1-G-Coder/final_model"
DATASET_PATH = "dataset/"
MAX_SEQ_LENGTH = 18000
EVAL_BATCH_SIZE = 1


def get_tokens_to_add():
    standard_g_codes = [
        "G0", "G1", "G2", "G3", "G4", "G10", "G11", "G20", "G21", "G28", "G29", "G90", "G91", "G92",
        " X", " Y", " E", " F", "Z.2"
    ]
    standard_m_codes = [
        "M0", "M1", "M17", "M73", "M82", "M83", "M84", "M104", "M105", "M106", "M107",
        "M109", "M140", "M190", "M204", "M220", "M221", "M400", "M600"
    ]
    bambu_custom_codes = [
        "G29.1", "G29.2", "M412", "M622", "M900", "M1002",
        "; FEATURE:", "; LINE_WIDTH:", "; LAYER:",
        "M624 AQAAAAAAAAA="
    ]
    common_feedrates = [
        " F1800", " F1200", " F42000", " F3600"
    ]
    prompt_keywords_tokens = [
        "module=",
        " teeth_count=",
        " bore_diameter=",
    ]
    long_comment_prefixes = [
        "; start printing object, unique label id: 15",
        "; stop printing object, unique label id: 15"
    ]

    return standard_g_codes + standard_m_codes + bambu_custom_codes + \
        common_feedrates + prompt_keywords_tokens + long_comment_prefixes


def load_base_tokenizer_and_extend(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )

    special_tokens_dict = {
        "additional_special_tokens": ["<|endofprompt|>"]
    }
    tokenizer.add_special_tokens(special_tokens_dict)

    tokens_to_add = get_tokens_to_add()
    existing_vocab = tokenizer.get_vocab()
    new_tokens = [t for t in tokens_to_add if t not in existing_vocab]

    if new_tokens:
        num_added = tokenizer.add_tokens(new_tokens)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_and_mask_prompt(examples, tokenizer):
    eop_token_id = tokenizer.convert_tokens_to_ids("<|endofprompt|>")
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )
    labels = [list(ids) for ids in outputs["input_ids"]]
    for i in range(len(labels)):
        try:
            if eop_token_id in labels[i]:
                eop_index = labels[i].index(eop_token_id)
                for j in range(eop_index + 1):
                    labels[i][j] = -100
        except ValueError:
            pass
    outputs["labels"] = labels
    return outputs


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
    apply_liger_kernel_to_qwen2()

    logging.set_verbosity_info()
    warnings.filterwarnings("ignore")

    tokenizer = load_base_tokenizer_and_extend(BASE_MODEL_PATH)

    raw_datasets = load_dataset("json", data_dir=DATASET_PATH, cache_dir=f"{ADAPTER_PATH}/.cache_eval")

    processed_datasets = raw_datasets.map(
        tokenize_and_mask_prompt,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count() // 2,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    config.max_position_embeddings = MAX_SEQ_LENGTH

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False

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