import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import torch
from pathlib import Path
from tqdm import tqdm
from src.config import setup
from src.utils import save_jsonl, load_jsonl
from src.metrics import load_labelled_responses

config = setup()
layer = config["sae"]["primary_layer"]

with open("results/tables/discovery_results.json") as f:
    discovery = json.load(f)

candidate_features = discovery[str(layer)]["consensus_2of3"]

test_prompts = load_jsonl("datasets/processed/test.jsonl")
abstain_prompts = [p for p in test_prompts if p["intended_label"] == "abstain"]

n_eval = config["steering"]["n_generations_per_coeff"]
abstain_subset = abstain_prompts[:n_eval]
prompts = [p["prompt"] for p in abstain_subset]

response_dir = Path("results/ablation/responses")
labelled_dir = Path("results/ablation/labelled")
out_dir = Path(config["reproducibility"]["table_dir"])
response_path = response_dir / "ablation.jsonl"

# ==================================
# = A. Generate baseline responses =
# ==================================

needs_generation = not response_path.exists() or response_path.stat().st_size == 0

if needs_generation:
    from src.model import load_model, load_sae
    from src.steering import ablate_and_generate

    model = load_model(config)
    sae = load_sae(config, layer)

    response_dir.mkdir(parents=True, exist_ok=True)
    all_responses = []

    print(f"\nGenerating baseline responses for {len(prompts)} abstain prompts")
    for prompt in tqdm(prompts, desc="Baseline"):
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            output = model.generate(
                tokens,
                max_new_tokens=config["model"]["generation_max_tokens"],
                temperature=0.0,
                top_k=1,
            )
        generated_tokens = output[0, tokens.shape[1] :]
        response = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        all_responses.append(
            {
                "prompt": prompt,
                "feature_id": None,
                "ablated_features": [],
                "layer": layer,
                "condition": "baseline",
                "response": response,
            }
        )

    # =================================
    # = B. Generate ablated responses =
    # =================================

    for feature_id in candidate_features:
        print(f"\nAblating feature {feature_id}")
        for prompt in tqdm(prompts, desc=f"Feature {feature_id}", leave=False):
            result = ablate_and_generate(
                model,
                sae,
                prompt,
                [feature_id],
                layer,
                max_new_tokens=config["model"]["generation_max_tokens"],
            )
            all_responses.append(
                {
                    "prompt": result["prompt"],
                    "feature_id": feature_id,
                    "ablated_features": result["ablated_features"],
                    "layer": layer,
                    "condition": "ablated",
                    "response": result["response"],
                }
            )

    save_jsonl(all_responses, response_path)
    print(f"\nSaved {len(all_responses)} responses to '{response_path}'")
else:
    print(f"Responses already exist at '{response_path}', skipping generation")

# =================================
# = C. Analyse labelled responses =
# =================================

labelled_path = labelled_dir / "ablation.jsonl"

if not labelled_path.exists():
    print("Judge labels not found! Classify responses and then re-run this script!")
    sys.exit(0)

print("\nAnalysing labelled responses")

labelled = load_labelled_responses(labelled_path)

baselines = {}
ablated_by_feature = {}

for item in labelled:
    if item["condition"] == "baseline":
        baselines[item["prompt"]] = item["behaviour_label"]
    else:
        fid = item["feature_id"]
        if fid not in ablated_by_feature:
            ablated_by_feature[fid] = {}
        ablated_by_feature[fid][item["prompt"]] = item["behaviour_label"]

results = {}
for feature_id in candidate_features:
    fid_labels = ablated_by_feature.get(feature_id, {})
    n_total = 0
    n_flipped = 0
    flips = {"abstain_to_answer": 0, "answer_to_abstain": 0}

    for prompt, ablated_label in fid_labels.items():
        baseline_label = baselines.get(prompt)
        if baseline_label is None:
            continue
        n_total += 1
        if baseline_label != ablated_label:
            n_flipped += 1
            if baseline_label == "abstain" and ablated_label == "answer":
                flips["abstain_to_answer"] += 1
            else:
                flips["answer_to_abstain"] += 1

    flip_rate = n_flipped / n_total if n_total > 0 else 0
    baseline_abstain = sum(1 for p in fid_labels if baselines.get(p) == "abstain")
    ablated_abstain = sum(1 for l in fid_labels.values() if l == "abstain")

    results[str(feature_id)] = {
        "flip_rate": flip_rate,
        "n_flipped": n_flipped,
        "n_total": n_total,
        "flips": flips,
        "baseline_abstain_rate": baseline_abstain / n_total if n_total > 0 else 0,
        "ablated_abstain_rate": ablated_abstain / n_total if n_total > 0 else 0,
    }

    print(f"\nFeature {feature_id}:")
    print(f" Flip rate: {flip_rate:.2%} ({n_flipped}/{n_total})")
    print(
        f" Baseline abstain rate: {results[str(feature_id)]['baseline_abstain_rate']:.2%}"
    )
    print(
        f" Ablated abstain rate: {results[str(feature_id)]['ablated_abstain_rate']:.2%}"
    )
    print(
        f" Flips: {flips['abstain_to_answer']} abstain -> answer, {flips['answer_to_abstain']} answer -> abstain"
    )

out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to '{out_dir / 'ablation_results.json'}'")
