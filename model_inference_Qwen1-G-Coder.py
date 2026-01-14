import os
import torch
import warnings
import json
import random
import glob
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig
)
from peft import PeftModel


BASE_MODEL_PATH = "./base_model/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "gcode-generator-Qwen1-G-Coder/final_model"
DATASET_DIR = "dataset/test"
GENERATED_GCODE_DIR = "Qwen1-G-Coder_generate_gcode_files"
MAX_SEQ_LENGTH = 18000
PROMPT_END_TOKEN = "<|endofprompt|>"
INFERENCE_BATCH_SIZE = 1


def get_tokens_to_add():
    standard_g_codes = ["G0", "G1", "G2", "G3", "G4", "G10", "G11", "G20", "G21", "G28", "G29", "G90", "G91", "G92",
                        " X", " Y", " E", " F", "Z.2"]
    standard_m_codes = ["M0", "M1", "M17", "M73", "M82", "M83", "M84", "M104", "M105", "M106", "M107", "M109", "M140",
                        "M190", "M204", "M220", "M221", "M400", "M600"]
    bambu_custom_codes = ["G29.1", "G29.2", "M412", "M622", "M900", "M1002", "; FEATURE:", "; LINE_WIDTH:", "; LAYER:",
                          "M624 AQAAAAAAAAA="]
    common_feedrates = [" F1800", " F1200", " F42000", " F3600"]
    prompt_keywords_tokens = ["module=", " teeth_count=", " bore_diameter="]
    long_comment_prefixes = ["; start printing object, unique label id: 15",
                             "; stop printing object, unique label id: 15"]
    return standard_g_codes + standard_m_codes + bambu_custom_codes + common_feedrates + prompt_keywords_tokens + long_comment_prefixes


def load_base_tokenizer_and_extend(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofprompt|>"]})

    tokens_to_add = get_tokens_to_add()
    new_tokens = [t for t in tokens_to_add if t not in tokenizer.get_vocab()]
    if new_tokens: tokenizer.add_tokens(new_tokens)

    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_all_test_samples(num_samples=300):
    target_dir = "dataset/test"
    files = glob.glob(f"{target_dir}/*.jsonl")
    files.sort()

    all_prompts = []

    for file_path in files:
        if len(all_prompts) >= num_samples: break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(all_prompts) >= num_samples: break
                    try:
                        data = json.loads(line)
                        full_text = data["text"]
                        if PROMPT_END_TOKEN in full_text:
                            prompt = full_text.split(PROMPT_END_TOKEN)[0].strip()
                            all_prompts.append(prompt)
                    except:
                        continue
        except:
            continue

    return all_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="Current GPU ID (0, 1, 2)")
    parser.add_argument("--world_size", type=int, default=1, help="Total number of GPUs (3)")
    args = parser.parse_args()

    rank = args.rank
    world_size = args.world_size

    print(f"\n========================================")
    print(f"Start the parallel generation process: GPU {rank} / {world_size}")
    print(f"========================================\n")


    config = AutoConfig.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    config.max_position_embeddings = MAX_SEQ_LENGTH
    if MAX_SEQ_LENGTH > 32768:
        config.rope_scaling = {"type": "dynamic", "factor": 2.0}

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    device = torch.device(f"cuda:0")
    model.to(device)

    tokenizer = load_base_tokenizer_and_extend(BASE_MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()


    all_raw_prompts = get_all_test_samples(num_samples=300)
    full_task_list = all_raw_prompts


    my_tasks = full_task_list[rank::world_size]

    print(f"Total task pool: {len(full_task_list)} 个")
    print(f"GPU {rank} eceives tasks: {len(my_tasks)} (Indices: {list(range(len(full_task_list)))[rank::world_size]})")

    save_dir = Path(GENERATED_GCODE_DIR)
    save_dir.mkdir(exist_ok=True)

    batches = [my_tasks[i:i + INFERENCE_BATCH_SIZE] for i in range(0, len(my_tasks), INFERENCE_BATCH_SIZE)]

    for batch_idx, batch_prompts in enumerate(tqdm(batches, desc=f"GPU {rank} Progress")):

        batch_input_texts = [f"{p} {PROMPT_END_TOKEN}" for p in batch_prompts]
        inputs = tokenizer(batch_input_texts, return_tensors="pt", padding=True).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=MAX_SEQ_LENGTH,
                    do_sample=False,  # Deterministic generation
                    temperature=None,
                    top_p=None,
                    repetition_penalty=1.0,
                    num_beams=1,
                    num_return_sequences=1,
                    early_stopping=True,
                    min_new_tokens=500,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
                )

            input_seq_len = inputs.input_ids.shape[1]

            for i, output_seq in enumerate(outputs):
                param_str = batch_prompts[i]
                generated_ids = output_seq[input_seq_len:]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

                stop_marker = "; stop printing object, unique label id: 15"
                if stop_marker in output_text:
                    output_text = output_text.split(stop_marker)[0] + stop_marker + "\n"

                safe_params = param_str.replace("=", "-").replace(" ", "_")
                filename = f"GPU{rank}-Qwen-{safe_params}.gcode"

                with open(save_dir / filename, "w", encoding="utf-8") as f:
                    f.write(output_text)

        except Exception as e:
            print(f"GPU {rank} Error: {e}")
            torch.cuda.empty_cache()

    print(f"GPU {rank} has completed all tasks!")


if __name__ == "__main__":
    main()