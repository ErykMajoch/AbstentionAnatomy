import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import gc
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import setup
from src.utils import save_jsonl

config = setup(None)
out_dir = Path("datasets/processed")

# =============================
# = A. Generate all responses =
# =============================

responses_path = out_dir / "responses.jsonl"

if responses_path.exists():
    print(f"Responses already exist at '{responses_path}', skipping generation")
    print("Delete the file to re-generate\n")
else:
    prompts = []
    with open(out_dir / "all_prompts.jsonl", encoding="utf-8") as file:
        for line in file:
            prompts.append(json.loads(line))

    print(f"Generating responses for {len(prompts)} prompts")

    model_name = config["model"]["name"]
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=config["_dtype"],
        device_map="cuda",
    )

    tokeniser.padding_side = "left"
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token

    alloc = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {alloc:.2f}GB / {total:.1f}GB")

    batch_size = 16
    max_seq_len = config["model"]["max_seq_len"]
    max_new_tokens = config["model"]["generation_max_tokens"]

    items = []
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_items = prompts[batch_start : batch_start + batch_size]

        chat_prompts = []
        for item in batch_items:
            messages = [{"role": "user", "content": item["prompt"]}]
            chat_prompts.append(
                tokeniser.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        encoded = tokeniser(
            chat_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = encoded.input_ids.shape[1]
        for j, item in enumerate(batch_items):
            generated_tokens = output_ids[j, input_len:]
            response = tokeniser.decode(generated_tokens, skip_special_tokens=True)
            items.append({**item, "response": response})

        if (batch_start // batch_size + 1) % 10 == 0:
            save_jsonl(items, out_dir / "responses_partial.jsonl")

    save_jsonl(items, responses_path)
    print(f"Saved {len(items)} responses to '{responses_path}'")

    del model, tokeniser
    torch.cuda.empty_cache()
    gc.collect()
    print("Model unloaded\n")
