import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from src.config import setup
from src.utils import save_jsonl, load_jsonl

config = setup()
layer = config["sae"]["primary_layer"]

EVAL_FEATURES = [622, 763, 340]
EVAL_COEFFICIENTS = [0, 150, 300, 500, 800, 1000]
N_SAMPLES = config["steering"]["capability_eval_samples"]

response_dir = Path("results/capability/responses")
out_dir = Path(config["reproducibility"]["table_dir"])
response_path = response_dir / "capability_eval.jsonl"

# ==========================================
# = A. Generate steered TriviaQA responses =
# ==========================================

if response_path.exists() and response_path.stat().st_size > 0:
    print(f"Responses already exist at '{response_path}', skipping generation")
else:
    from src.model import load_model, load_sae
    from src.steering import steer_and_generate

    model = load_model(config)
    sae = load_sae(config, layer)

    print(f"\nLoading TriviaQA validation set ({N_SAMPLES} samples)")
    dataset = load_dataset("mandarjoshi/trivia_qa", "unfiltered", split="validation")
    eval_items = list(dataset.select(range(N_SAMPLES)))

    response_dir.mkdir(parents=True, exist_ok=True)
    all_responses = []

    for feature_id in EVAL_FEATURES:
        for coeff in EVAL_COEFFICIENTS:
            print(f"\nFeature {feature_id}, coeff {coeff:+d}")
            for item in tqdm(eval_items, desc=f"f={feature_id} c={coeff}", leave=False):
                question = item["question"]
                aliases = item["answer"]["aliases"]

                result = steer_and_generate(
                    model,
                    sae,
                    question,
                    feature_id,
                    coeff,
                    layer,
                    max_new_tokens=32,
                )
                all_responses.append(
                    {
                        "question": question,
                        "answer_aliases": aliases,
                        "feature_id": feature_id,
                        "coeff": coeff,
                        "layer": layer,
                        "response": result["response"],
                    }
                )

    save_jsonl(all_responses, response_path)
    print(f"\nSaved {len(all_responses)} responses to '{response_path}'")

# ========================
# = B. Evaluate accuracy =
# ========================

print("\nEvaluating TriviaQA accuracy")

responses = load_jsonl(response_path)

groups = defaultdict(list)
for item in responses:
    groups[(item["feature_id"], item["coeff"])].append(item)

results = {}
for (feature_id, coeff), items in sorted(groups.items()):
    correct = 0
    for item in items:
        response_lower = item["response"].lower()
        hit = any(alias.lower() in response_lower for alias in item["answer_aliases"])
        if hit:
            correct += 1
    total = len(items)
    accuracy = correct / total if total > 0 else 0

    fid_key = str(feature_id)
    if fid_key not in results:
        results[fid_key] = {}
    results[fid_key][str(coeff)] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
    }

abstention_rates = {}
steering_path = out_dir / "steering_answer_to_abstain.json"
if steering_path.exists():
    with open(steering_path) as f:
        steering_data = json.load(f)
    for fid in EVAL_FEATURES:
        fid_key = str(fid)
        if fid_key in steering_data:
            abstention_rates[fid_key] = {
                k: v["abstain_rate"] for k, v in steering_data[fid_key].items()
            }

print(f"\n{'Feature':<10} {'Coeff':<8} {'Accuracy':<12} {'Abstain rate':<14}")
print("-" * 44)
for fid in EVAL_FEATURES:
    fid_key = str(fid)
    if fid_key not in results:
        continue
    for coeff in EVAL_COEFFICIENTS:
        coeff_key = str(coeff)
        if coeff_key not in results[fid_key]:
            continue
        acc = results[fid_key][coeff_key]["accuracy"]
        abstain = abstention_rates.get(fid_key, {}).get(coeff_key, None)
        abstain_str = f"{abstain:.2%}" if abstain is not None else "N/A"
        print(f"{fid_key:<10} {coeff:+8d} {acc:<12.3f} {abstain_str:<14}")
    print()

baseline_accuracies = {}
for fid in EVAL_FEATURES:
    fid_key = str(fid)
    if fid_key in results and "0" in results[fid_key]:
        baseline_accuracies[fid_key] = results[fid_key]["0"]["accuracy"]

best_tradeoff = None
for fid in EVAL_FEATURES:
    fid_key = str(fid)
    baseline = baseline_accuracies.get(fid_key)
    if baseline is None:
        continue
    for coeff in EVAL_COEFFICIENTS:
        if coeff == 0:
            continue
        coeff_key = str(coeff)
        entry = results.get(fid_key, {}).get(coeff_key)
        if entry is None:
            continue
        acc = entry["accuracy"]
        drop = baseline - acc
        within_5pct = drop <= 0.05
        abstain_rate = abstention_rates.get(fid_key, {}).get(coeff_key)
        baseline_abstain = abstention_rates.get(fid_key, {}).get("0", 0.08)
        shift = (abstain_rate - baseline_abstain) if abstain_rate is not None else None

        if shift is not None and within_5pct:
            if best_tradeoff is None or shift > best_tradeoff["abstention_shift"]:
                best_tradeoff = {
                    "feature_id": int(fid),
                    "coeff": coeff,
                    "accuracy": acc,
                    "accuracy_drop": drop,
                    "abstention_shift": shift,
                    "within_5pct": True,
                }

summary = {"baseline_accuracies": baseline_accuracies}
if best_tradeoff:
    summary["best_tradeoff"] = best_tradeoff
    print(
        f"Best tradeoff: Feature {best_tradeoff['feature_id']} at "
        f"coeff={best_tradeoff['coeff']}: "
        f"accuracy={best_tradeoff['accuracy']:.3f} "
        f"(drop={best_tradeoff['accuracy_drop']:.3f}), "
        f"abstention shift={best_tradeoff['abstention_shift']:.2%}"
    )
else:
    print(
        "No coefficient found that preserves accuracy within 5% of baseline "
        "while shifting abstention."
    )

results["summary"] = summary

out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "capability_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to '{out_dir / 'capability_eval_results.json'}'")
