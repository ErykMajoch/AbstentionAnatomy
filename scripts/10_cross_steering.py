import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from src.config import setup
from src.utils import save_jsonl, load_jsonl
from src.metrics import load_labelled_responses

config = setup()
layer = config["sae"]["primary_layer"]

CROSS_COEFFICIENTS = [0, 500, 1000]
N_PROMPTS_PER_CELL = 30

with open("results/tables/multiclass_results.json") as f:
    multiclass = json.load(f)

class_top_features = {}
for class_name, info in multiclass["class_features"].items():
    class_top_features[class_name] = info["top_features_mad"][0]

print("Per-class top features:")
for cn, fid in class_top_features.items():
    print(f" {cn}: Feature {fid}")

test_prompts = load_jsonl("datasets/processed/test.jsonl")

class_answer_prompts = defaultdict(list)
for p in test_prompts:
    if p["intended_label"] == "answer":
        class_answer_prompts[p["class"]].append(p["prompt"])

print("\nAnswer prompts per class in test set:")
for cn, prompts in class_answer_prompts.items():
    print(f" {cn}: {len(prompts)} (using {min(len(prompts), N_PROMPTS_PER_CELL)})")

response_dir = Path("results/cross_steering/responses")
labelled_dir = Path("results/cross_steering/labelled")
out_dir = Path(config["reproducibility"]["table_dir"])
fig_dir = Path(config["reproducibility"]["figure_dir"])
response_path = response_dir / "cross_steering.jsonl"

# =======================================
# = A. Generate cross-steered responses =
# =======================================

if response_path.exists() and response_path.stat().st_size > 0:
    print(f"\nResponses already exist at '{response_path}', skipping generation")
else:
    from src.model import load_model, load_sae
    from src.steering import steer_and_generate

    model = load_model(config)
    sae = load_sae(config, layer)

    response_dir.mkdir(parents=True, exist_ok=True)
    all_responses = []

    for source_class, source_feature in class_top_features.items():
        print(f"\nSource: {source_class} (Feature {source_feature})")

        for target_class, target_prompts in class_answer_prompts.items():
            subset = target_prompts[:N_PROMPTS_PER_CELL]
            print(f" -> Target: {target_class} ({len(subset)} prompts)")

            for coeff in CROSS_COEFFICIENTS:
                for prompt in tqdm(
                    subset,
                    desc=f"{source_class} -> {target_class} c={coeff}",
                    leave=False,
                ):
                    result = steer_and_generate(
                        model,
                        sae,
                        prompt,
                        source_feature,
                        coeff,
                        layer,
                        max_new_tokens=config["model"]["generation_max_tokens"],
                    )
                    all_responses.append(
                        {
                            "prompt": result["prompt"],
                            "feature_id": result["feature_id"],
                            "coeff": result["coeff"],
                            "layer": result["layer"],
                            "source_class": source_class,
                            "target_class": target_class,
                            "response": result["response"],
                        }
                    )

    save_jsonl(all_responses, response_path)
    print(f"\nSaved {len(all_responses)} responses to '{response_path}'")

# =================================
# = B. Analyse labelled responses =
# =================================

labelled_path = labelled_dir / "cross_steering.jsonl"

if not labelled_path.exists():
    print("\nJudge labels not found! Classify responses then re-run this script.")
    print(
        f"  python tools/judge_classify.py --response-dir {response_dir} --output-dir {labelled_dir}"
    )
    sys.exit(0)

print("\nAnalysing labelled cross-steering responses")

labelled = load_labelled_responses(labelled_path)

groups = defaultdict(list)
for item in labelled:
    key = (item["source_class"], item["target_class"], item["coeff"])
    groups[key].append(item["behaviour_label"])

class_names = multiclass["class_names"]
n_classes = len(class_names)

rates = {}
for (source, target, coeff), labels in groups.items():
    n_abstain = sum(1 for l in labels if l == "abstain")
    n = len(labels)
    if source not in rates:
        rates[source] = {}
    if target not in rates[source]:
        rates[source][target] = {}
    rates[source][target][str(coeff)] = {
        "abstain_rate": n_abstain / n,
        "n": n,
        "abstain": n_abstain,
        "answer": n - n_abstain,
    }

strong_coeff = str(max(CROSS_COEFFICIENTS))
baseline_coeff = "0"

print(f"\nCross-steering matrix (coeff={strong_coeff})")
print(f"{'Source feature →':<25}", end="")
for cn in class_names:
    print(f"{cn:>18}", end="")
print()
print(f"{'Target prompts ↓':<25}", end="")
for cn in class_names:
    fid = class_top_features.get(cn, "?")
    print(f"{'F' + str(fid):>18}", end="")
print()
print("-" * (25 + 18 * n_classes))

delta_matrix = np.zeros((n_classes, n_classes))
steered_matrix = np.zeros((n_classes, n_classes))
baseline_matrix = np.zeros((n_classes, n_classes))

for i, target in enumerate(class_names):
    print(f"{target:<25}", end="")
    for j, source in enumerate(class_names):
        baseline_rate = (
            rates.get(source, {})
            .get(target, {})
            .get(baseline_coeff, {})
            .get("abstain_rate", float("nan"))
        )
        steered_rate = (
            rates.get(source, {})
            .get(target, {})
            .get(strong_coeff, {})
            .get("abstain_rate", float("nan"))
        )
        delta = steered_rate - baseline_rate

        baseline_matrix[i, j] = baseline_rate
        steered_matrix[i, j] = steered_rate
        delta_matrix[i, j] = delta

        print(f"{steered_rate:>13.0%} ({delta:+.0%})", end="")
    print()

diag_mean = np.nanmean(np.diag(delta_matrix))
off_diag = delta_matrix[~np.eye(n_classes, dtype=bool)]
off_diag_mean = np.nanmean(off_diag)
print(f"\nDiagonal mean delta:     {diag_mean:+.2%}")
print(f"Off-diagonal mean delta: {off_diag_mean:+.2%}")
print(f"Specificity ratio:       {diag_mean / (off_diag_mean + 1e-8):.2f}x")

# Delta matrix heatmap

fig_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    delta_matrix,
    xticklabels=[f"{cn}\n(F{class_top_features[cn]})" for cn in class_names],
    yticklabels=class_names,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    vmin=0,
    ax=axes[0],
)
axes[0].set_xlabel("Source class (feature steered)")
axes[0].set_ylabel("Target class (prompts)")
axes[0].set_title(f"Cross-steering delta (coeff={strong_coeff})")

sns.heatmap(
    steered_matrix,
    xticklabels=[f"{cn}\n(F{class_top_features[cn]})" for cn in class_names],
    yticklabels=class_names,
    cmap="YlOrRd",
    annot=True,
    fmt=".2f",
    vmin=0,
    vmax=1,
    ax=axes[1],
)
axes[1].set_xlabel("Source class (feature steered)")
axes[1].set_ylabel("Target class (prompts)")
axes[1].set_title(f"Cross-steering abstention rate (coeff={strong_coeff})")

plt.tight_layout()
plt.savefig(fig_dir / "cross_steering_matrix.png", dpi=300, bbox_inches="tight")
plt.savefig(fig_dir / "cross_steering_matrix.pdf", bbox_inches="tight")
plt.close()
print(f"\nHeatmap saved to '{fig_dir / 'cross_steering_matrix.png'}'")


out_dir.mkdir(parents=True, exist_ok=True)

results = {
    "layer": layer,
    "coefficients": CROSS_COEFFICIENTS,
    "n_prompts_per_cell": N_PROMPTS_PER_CELL,
    "class_top_features": class_top_features,
    "class_names": class_names,
    "rates": rates,
    "delta_matrix": delta_matrix.tolist(),
    "steered_matrix": steered_matrix.tolist(),
    "baseline_matrix": baseline_matrix.tolist(),
    "summary": {
        "strong_coefficient": int(strong_coeff),
        "diagonal_mean_delta": float(diag_mean),
        "off_diagonal_mean_delta": float(off_diag_mean),
        "specificity_ratio": float(diag_mean / (off_diag_mean + 1e-8)),
    },
}

with open(out_dir / "cross_steering_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to '{out_dir / 'cross_steering_results.json'}'")
