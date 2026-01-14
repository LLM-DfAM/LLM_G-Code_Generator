import os
import random
import re
import torch
import warnings
import json
import glob
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
ADAPTER_PATH = "gcode-generator-Qwen2-G-Coder/final_model"
DATASET_DIR = "dataset/test"
GENERATED_GCODE_DIR = "Qwen1-G-Coder_generate_gcode_files"
MAX_SEQ_LENGTH = 20480
INFERENCE_BATCH_SIZE = 1


def load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def get_all_test_samples(num_samples=300):
    files = glob.glob(f"{DATASET_DIR}/*.jsonl")
    files.sort()

    all_prompts = []
    SPLIT_TOKEN = "<|endofprompt|>"

    for file_path in files:
        if len(all_prompts) >= num_samples: break
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(all_prompts) >= num_samples: break
                    try:
                        data = json.loads(line)
                        full_text = data["text"]
                        if SPLIT_TOKEN in full_text:
                            prompt = full_text.split(SPLIT_TOKEN)[0].strip()
                            all_prompts.append(prompt)
                        else:
                            all_prompts.append(full_text[:50])
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
    config.use_cache = True

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )

    device = torch.device(f"cuda:0")
    model.to(device)

    tokenizer = load_tokenizer(BASE_MODEL_PATH)

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    all_raw_prompts = get_all_test_samples(num_samples=300)
    my_tasks = all_raw_prompts[rank::world_size]

    save_dir = Path(GENERATED_GCODE_DIR)
    save_dir.mkdir(exist_ok=True)

    batches = [my_tasks[i:i + INFERENCE_BATCH_SIZE] for i in range(0, len(my_tasks), INFERENCE_BATCH_SIZE)]

    for batch_prompts in tqdm(batches, desc=f"GPU {rank} Generating"):

        batch_inputs_ids = []

        for p in batch_prompts:
            messages = [
                {"role": "user", "content": p}
            ]
            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(device)
            batch_inputs_ids.append(input_ids)

        input_tensor = batch_inputs_ids[0]

        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_tensor,
                    max_length=MAX_SEQ_LENGTH,
                    do_sample=False,  # Deterministic generation
                    num_beams=1,
                    repetition_penalty=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            input_len = input_tensor.shape[1]
            generated_ids = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            output_text = output_text.strip()

            stop_marker = "; stop printing object, unique label id: 15"
            if stop_marker in output_text:
                output_text = output_text.split(stop_marker)[0] + stop_marker + "\n"

            param_str = batch_prompts[0]
            target_keys = ["module", "teeth_count", "bore_diameter"]
            name_parts = []
            for key in target_keys:
                match = re.search(rf"{key}\s*=\s*([\d\.]+)", param_str)
                if match:
                    value = match.group(1)
                    name_parts.append(f"{key}-{value}")
            if name_parts:
                safe_params = "_".join(name_parts)
            else:
                safe_params = "params_extract_failed"
            filename = f"GPU{rank}_Qwen_{safe_params}.gcode"

            with open(save_dir / filename, "w", encoding="utf-8") as f:
                f.write(output_text)

        except Exception as e:
            print(f"GPU {rank} Error: {e}")
            torch.cuda.empty_cache()

    print(f"GPU {rank} has completed all tasks!")


if __name__ == "__main__":
    main()