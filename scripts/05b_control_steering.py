import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from src.config import setup
from src.utils import save_jsonl, load_jsonl
from src.metrics import load_labelled_responses, bootstrap_ci

config = setup()
layer = config["sae"]["primary_layer"]
coefficients = config["steering"]["coefficients"]
n_eval = config["steering"]["n_generations_per_coeff"]

REFERENCE_FEATURE = 622
COMPARISON_FEATURES = [622, 763, 340]
UNRELATED_FEATURES = [1, 21, 90]
N_RANDOM_SEEDS = 10
RANDOM_SEED_BASE = 42

test_prompts = load_jsonl("datasets/processed/test.jsonl")
answer_prompts = [p["prompt"] for p in test_prompts if p["intended_label"] == "answer"]
answer_subset = answer_prompts[:n_eval]

response_dir = Path("results/controls/responses")
labelled_dir = Path("results/controls/labelled")
out_dir = Path(config["reproducibility"]["table_dir"])
response_path = response_dir / "control_steering.jsonl"

# ==========================================
# = A. Generate control steering responses =
# ==========================================

if response_path.exists() and response_path.stat().st_size > 0:
    print(f"Responses already exist at '{response_path}', skipping generation")
else:
    from src.model import load_model, load_sae
    from src.steering import (
        get_steering_direction,
        steer_with_direction,
        batch_steer_experiment,
    )

    model = load_model(config)
    sae = load_sae(config, layer)

    response_dir.mkdir(parents=True, exist_ok=True)
    all_responses = []

    # Control 1: Random directions with matched norm

    ref_direction = get_steering_direction(sae, REFERENCE_FEATURE)
    ref_norm = ref_direction.norm().item()
    print(f"\nReference feature {REFERENCE_FEATURE} decoder norm: {ref_norm:.4f}")

    for seed_offset in range(N_RANDOM_SEEDS):
        seed = RANDOM_SEED_BASE + seed_offset
        label = f"random_seed_{seed}"
        print(f"\n=== Control 1: {label} ===")

        gen = torch.Generator(device=ref_direction.device)
        gen.manual_seed(seed)
        random_dir = torch.randn(
            ref_direction.shape,
            generator=gen,
            device=ref_direction.device,
            dtype=ref_direction.dtype,
        )
        random_dir = random_dir / random_dir.norm() * ref_norm

        for coeff in coefficients:
            print(f"  coeff {coeff:+.0f}")
            for prompt in tqdm(answer_subset, desc=f"{label} c={coeff}", leave=False):
                response = steer_with_direction(
                    model,
                    random_dir,
                    prompt,
                    coeff,
                    layer,
                    max_new_tokens=config["model"]["generation_max_tokens"],
                )
                all_responses.append(
                    {
                        "prompt": prompt,
                        "feature_id": REFERENCE_FEATURE,
                        "coeff": coeff,
                        "layer": layer,
                        "condition": "random_direction",
                        "random_seed": seed,
                        "control_label": label,
                        "response": response,
                    }
                )

    # Control 2: Unrelated features

    for feature_id in UNRELATED_FEATURES:
        label = f"unrelated_{feature_id}"
        print(f"\n=== Control 2: {label} ===")

        responses = batch_steer_experiment(
            model,
            sae,
            answer_subset,
            feature_id,
            coefficients,
            layer,
            max_new_tokens=config["model"]["generation_max_tokens"],
        )
        for r in responses:
            all_responses.append(
                {
                    "prompt": r["prompt"],
                    "feature_id": r["feature_id"],
                    "coeff": r["coeff"],
                    "layer": r["layer"],
                    "condition": "unrelated_feature",
                    "random_seed": None,
                    "control_label": label,
                    "response": r["response"],
                }
            )

    save_jsonl(all_responses, response_path)
    print(f"\nSaved {len(all_responses)} control responses to '{response_path}'")

# =========================================
# = C. Analyse labelled control responses =
# =========================================

labelled_path = labelled_dir / "control_steering.jsonl"

if not labelled_path.exists():
    print("\nJudge labels not found! Run judge classification then re-run this script.")
    print(
        f"  python tools/judge_classify.py --response-dir {response_dir} --output-dir {labelled_dir}"
    )
    sys.exit(0)

print("\nAnalysing labelled control responses")

labelled = load_labelled_responses(labelled_path)

groups = defaultdict(list)
for item in labelled:
    key = (item["condition"], item["control_label"], item["coeff"])
    groups[key].append(item["behaviour_label"])

control_rates = {}
for (condition, label, coeff), labels in groups.items():
    n_abstain = sum(1 for l in labels if l == "abstain")
    n = len(labels)
    if condition not in control_rates:
        control_rates[condition] = {}
    if label not in control_rates[condition]:
        control_rates[condition][label] = {}
    control_rates[condition][label][str(coeff)] = {
        "abstain_rate": n_abstain / n,
        "n": n,
        "abstain": n_abstain,
        "answer": n - n_abstain,
    }

with open(out_dir / "steering_answer_to_abstain.json") as f:
    real_results = json.load(f)

comparison = {}
for fid in COMPARISON_FEATURES:
    fid_key = str(fid)
    if fid_key in real_results:
        comparison[fid_key] = real_results[fid_key]

print("\nControl 1: Random Directions")
for seed_offset in range(N_RANDOM_SEEDS):
    label = f"random_seed_{RANDOM_SEED_BASE + seed_offset}"
    if label in control_rates.get("random_direction", {}):
        rates = control_rates["random_direction"][label]
        max_rate = max(v["abstain_rate"] for v in rates.values())
        print(f"  {label}: max abstention rate = {max_rate:.2%}")

print("\nControl 2: Unrelated Features")
for fid in UNRELATED_FEATURES:
    label = f"unrelated_{fid}"
    if label in control_rates.get("unrelated_feature", {}):
        rates = control_rates["unrelated_feature"][label]
        max_rate = max(v["abstain_rate"] for v in rates.values())
        print(f"  Feature {fid}: max abstention rate = {max_rate:.2%}")

print("\nComparison: Real Abstention Features")
for fid in COMPARISON_FEATURES:
    fid_key = str(fid)
    if fid_key in comparison:
        max_rate = max(v["abstain_rate"] for v in comparison[fid_key].values())
        print(f"  Feature {fid}: max abstention rate = {max_rate:.2%}")

summary = {}

random_max_rates = []
random_rates = control_rates.get("random_direction", {})
for label, rates in random_rates.items():
    max_rate = max(v["abstain_rate"] for v in rates.values())
    random_max_rates.append(max_rate)

if random_max_rates:
    arr = np.array(random_max_rates)
    lo, mean, hi = bootstrap_ci(arr)
    summary["random_mean_max_abstain_rate"] = float(mean)
    summary["random_std_max_abstain_rate"] = float(arr.std())
    summary["random_ci_95"] = [lo, hi]
    summary["random_n_seeds"] = len(random_max_rates)

unrelated_max = {}
unrelated_rates = control_rates.get("unrelated_feature", {})
for label, rates in unrelated_rates.items():
    max_rate = max(v["abstain_rate"] for v in rates.values())
    fid = label.replace("unrelated_", "")
    unrelated_max[fid] = max_rate
summary["unrelated_max_abstain_rates"] = unrelated_max

real_max = {}
for fid in COMPARISON_FEATURES:
    fid_key = str(fid)
    if fid_key in comparison:
        real_max[fid_key] = max(v["abstain_rate"] for v in comparison[fid_key].values())
summary["real_max_abstain_rates"] = real_max

all_control_maxes = random_max_rates + list(unrelated_max.values())
all_real_maxes = list(real_max.values())
summary["specificity_confirmed"] = bool(
    all_real_maxes
    and all_control_maxes
    and min(all_real_maxes) > max(all_control_maxes)
)

per_coeff_null = {}
for coeff in coefficients:
    coeff_key = str(coeff)
    rates_at_coeff = []
    for label, rates in random_rates.items():
        if coeff_key in rates:
            rates_at_coeff.append(rates[coeff_key]["abstain_rate"])
    if rates_at_coeff:
        arr = np.array(rates_at_coeff)
        lo, mean, hi = bootstrap_ci(arr)
        per_coeff_null[coeff_key] = {
            "mean": float(mean),
            "std": float(arr.std()),
            "ci_95": [lo, hi],
            "values": [float(v) for v in rates_at_coeff],
        }

results = {
    "random_direction": control_rates.get("random_direction", {}),
    "unrelated_feature": control_rates.get("unrelated_feature", {}),
    "comparison_features": comparison,
    "per_coefficient_null": per_coeff_null,
    "summary": summary,
}

out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "control_steering_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Summary")
if random_max_rates:
    print(
        f"Random directions (n={len(random_max_rates)}): "
        f"max abstention = {summary['random_mean_max_abstain_rate']:.2%} "
        f"± {summary['random_std_max_abstain_rate']:.2%}"
    )
for fid, rate in unrelated_max.items():
    print(f"Unrelated feature {fid}: max abstention = {rate:.2%}")
for fid, rate in real_max.items():
    print(f"Real feature {fid}: max abstention = {rate:.2%}")
print(f"Specificity confirmed: {summary['specificity_confirmed']}")
print(f"\nResults saved to '{out_dir / 'control_steering_results.json'}'")
