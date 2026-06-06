import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
from pathlib import Path
import torch
from src.config import setup
from src.model import load_model, load_sae, print_memory_report
from src.neuronpedia import fetch_neuronpedia_explanations

config = setup()
model = load_model(config)
sae = load_sae(config, layer=config["sae"]["primary_layer"])
print_memory_report()

# =======================================
# = A. Test a factual completion prompt =
# =======================================

prompt = "The Eiffel Tower is in the city of"
tokens = model.to_tokens(prompt)
print(f"\nPrompt: '{prompt}'")
print(f"Tokens shape: {tokens.shape} (batch,seq)")

logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

top_token = logits[0, -1].argmax()
print(f"Top predicted token: '{model.tokenizer.decode(top_token)}'")

# ====================================================
# = B. Extract SAE feature activations at last token =
# ====================================================

layer = config["sae"]["primary_layer"]
hook_name = model.get_sae_hook_name(sae, internal="hook_sae_acts_post")
sae_acts = cache[hook_name]  # shape: [1, seq_len, n_features]
print(f"\nSAE activations shape: {sae_acts.shape}")

last_token_acts = sae_acts[0, -1, :]  # shape: [n_features]

n_active = (last_token_acts > 0).sum().item()
print(f"Active features at last token: {n_active} / {last_token_acts.shape[0]}")
print(f"Sparsity (L0): {n_active}")

top_k = 20
top_values, top_indices = last_token_acts.topk(top_k)
print(
    f"\nTop-{top_k} features at last token ('{model.tokenizer.decode(tokens[0, -1])}'):"
)
print(f"{'Rank':<6} {'Feature ID':<12} {'Activation':<12}")
print("-" * 30)
for rank, (index, value) in enumerate(zip(top_indices, top_values)):
    print(f"{rank + 1:<6} {index.item():<12} {value.item():<12.4f}")

# =================================
# = C. Neuronpedia feature lookup =
# =================================

model_id = config["model"]["name"].split("/")[-1]
sae_id = f"{layer}-gemmascope-2-res-16k"
feature_ids = [idx.item() for idx in top_indices]
activations = [val.item() for val in top_values]

print(f"\nFetching Neuronpedia data for {len(feature_ids)} features")
Path("results/00").mkdir(parents=True, exist_ok=True)
explanations = fetch_neuronpedia_explanations(
    model_id,
    sae_id,
    feature_ids,
    activations,
    save_path="results/00/top_features.csv",
)

# =============================
# = D. Reconstruction Quality =
# =============================

sae_input_hook = model.get_sae_hook_name(sae, internal="hook_sae_input")
sae_output_hook = model.get_sae_hook_name(sae, internal="hook_sae_output")

# Cast to f32 to avoid fp16 overflow!
resid_original = cache[sae_input_hook].float()  # shape: [1, seq_len, d_model]
resid_reconstructed = cache[sae_output_hook].float()  # shape: [1, seq_len, d_model]

error = resid_original - resid_reconstructed
mse_per_pos = (error**2).mean(dim=-1)  # shape: [1, seq_len]
variance_per_pos = (
    (resid_original - resid_original.mean(dim=-1, keepdim=True)) ** 2
).mean(dim=-1)

r_squared = 1 - mse_per_pos / variance_per_pos
print("\nReconstruction quality (per token position):")
for position in range(tokens.shape[1]):
    token_string = model.tokenizer.decode(tokens[0, position])
    print(
        f"Position {position} ('{token_string}'): R^2 = {r_squared[0, position].item():.4f}, MSE = {mse_per_pos[0, position].item():.4f}"
    )

overall_mse = mse_per_pos.mean().item()
overall_r2 = r_squared.mean().item()
print(f"\nOverall R^2 = {overall_r2:.4f}, MSE = {overall_mse:.4f}")

Path("results/00").mkdir(parents=True, exist_ok=True)
metrics = {
    "prompt": prompt,
    "layer": layer,
    "n_active_features": n_active,
    "overall_r_squared": overall_r2,
    "overall_mse": overall_mse,
    "top_prediction": model.tokenizer.decode(top_token),
}

with open("results/00/metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

print("\nMetrics saved to 'results/00/metrics.json'\n")

# =================================================
# = E. Batch verification across multiple prompts =
# =================================================

test_prompts = [
    "The Eiffel Tower is in the city of",
    "The capital of Japan is",
    "Water freezes at",
    "The largest planet in our solar system is",
    "Python is a programming",
    "Shakespeare wrote Romeo and",
    "The speed of light is approximately",
    "Photosynthesis converts sunlight into",
    "The Great Wall of China is located in",
    "The chemical symbol for gold is",
]

results = []
for prompt in test_prompts:
    tokens = model.to_tokens(prompt)
    logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

    sae_acts = cache[hook_name][0, -1, :]
    top_value, top_index = sae_acts.topk(5)

    top_prediction = model.tokenizer.decode(logits[0, -1].argmax())
    n_active = (sae_acts > 0).sum().item()

    results.append(
        {
            "prompt": prompt,
            "top_prediction": top_prediction,
            "L0": n_active,
            "top_5_features": [index.item() for index in top_index],
            "top_5_activations": [value.item() for value in top_value],
        }
    )

    print(f"'{prompt}' -> '{top_prediction}' (L0 = {n_active})")

with open("results/00/batch_metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nBatch metrics saved to 'results/00/batch_metrics.json'\n")

# ===========================
# = F. FP16 Stability Check =
# ===========================

print("\nFP16 Stability Check")

stability_prompts = [
    "The",
    "A very long sentence with many words to test whether the model produces NaN values "
    * 3,
    "1 + 1 =",
]

instability = False
for prompt in stability_prompts:
    tokens = model.to_tokens(prompt)
    if tokens.shape[1] > config["model"]["max_seq_len"]:
        tokens = tokens[:, : config["model"]["max_seq_len"]]

    logits, cache = model.run_with_cache_with_saes(tokens, saes=[sae])
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    sae_acts = cache[hook_name]
    sae_nan = torch.isnan(sae_acts).any().item()

    print(
        f"Prompt length={tokens.shape[1]}: logit_nan={has_nan}, logit_inf={has_inf}, sae_nan={sae_nan}"
    )
    if has_nan or has_inf or sae_nan:
        print("Numerical instability detected!")
        instability = True

if not instability:
    print("Numerical stability passed")
