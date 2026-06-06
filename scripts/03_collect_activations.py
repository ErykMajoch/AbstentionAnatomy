import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import torch
import numpy as np
from pathlib import Path
from src.config import setup
from src.model import load_model, load_sae, print_memory_report
from src.activations import collect_activations

config = setup()

prompts = []
labels = []
classes = []
for split in ["train", "val"]:
    with open(f"datasets/processed/{split}.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            prompts.append(item["prompt"])
            labels.append(item["behaviour_label"])
            classes.append(item["class"])

print(f"Loaded {len(prompts)} prompts for activation collection")

model = load_model(config)
print_memory_report()

save_root = Path(config["reproducibility"]["activation_dir"])
target_layers = config["sae"]["layers"]
token_positions = config["activations"]["token_positions"]

for layer in target_layers:
    print(f"Collecting layer {layer}")
    sae = load_sae(config, layer)
    print_memory_report()

    collect_activations(
        model=model,
        sae=sae,
        prompts=prompts,
        layer=layer,
        token_positions=token_positions,
        batch_size=config["activations"]["batch_size"],
        max_seq_len=config["model"]["max_seq_len"],
        save_path=save_root / f"layer_{layer}",
    )

    del sae
    torch.cuda.empty_cache()
    print_memory_report()

np.save(
    save_root / "labels.npy", np.array([1 if l == "abstain" else 0 for l in labels])
)
np.save(save_root / "classes.npy", np.array(classes))

metadata = [
    {"prompt": p, "label": l, "class": c} for p, l, c in zip(prompts, labels, classes)
]
with open(save_root / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)

print(f"Activation collection complete and saved to '{save_root}'")
