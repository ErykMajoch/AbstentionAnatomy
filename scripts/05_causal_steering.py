import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
from pathlib import Path
from src.config import setup
from src.utils import save_jsonl, load_jsonl
from src.metrics import load_labelled_responses, compute_abstention_rates

config = setup()
layer = config["sae"]["primary_layer"]

with open("results/tables/discovery_results.json") as f:
    discovery = json.load(f)

candidate_features = discovery[str(layer)]["consensus_2of3"]
coefficients = config["steering"]["coefficients"]

test_prompts = load_jsonl("datasets/processed/test.jsonl")

answer_prompts = [p["prompt"] for p in test_prompts if p["intended_label"] == "answer"]
abstain_prompts = [
    p["prompt"] for p in test_prompts if p["intended_label"] == "abstain"
]

n_eval = config["steering"]["n_generations_per_coeff"]
answer_subset = answer_prompts[:n_eval]
abstain_subset = abstain_prompts[:n_eval]

response_dir = Path("results/steering/responses")
labelled_dir = Path("results/steering/labelled")
out_dir = Path(config["reproducibility"]["table_dir"])

# ==============================================
# = A. Steer answer prompts towards abstention =
# ==============================================

a2a_path = response_dir / "answer_to_abstain.jsonl"

if a2a_path.exists() and a2a_path.stat().st_size > 0:
    print(f"Responses already exist at '{a2a_path}', skipping generation")
else:
    from src.model import load_model, load_sae
    from src.steering import batch_steer_experiment

    if "model" not in dir():
        model = load_model(config)
        sae = load_sae(config, layer)

    response_dir.mkdir(parents=True, exist_ok=True)

    print("\nSteering answer prompts toward abstention")
    all_responses = []
    for feature_id in candidate_features:
        print(f"\nFeature {feature_id}:")
        responses = batch_steer_experiment(
            model,
            sae,
            answer_subset,
            feature_id,
            coefficients,
            layer,
        )
        all_responses.extend(responses)

    save_jsonl(all_responses, a2a_path)
    print(f"\nSaved {len(all_responses)} responses to '{a2a_path}'")

# ==============================================
# = B. Steer abstain prompts towards answering =
# ==============================================

a2ans_path = response_dir / "abstain_to_answer.jsonl"

if a2ans_path.exists() and a2ans_path.stat().st_size > 0:
    print(f"Responses already exist at '{a2ans_path}', skipping generation")
else:
    from src.model import load_model, load_sae
    from src.steering import batch_steer_experiment

    if "model" not in dir():
        model = load_model(config)
        sae = load_sae(config, layer)

    response_dir.mkdir(parents=True, exist_ok=True)

    neg_coefficients = [-c for c in coefficients if c > 0]

    print("\nSteering abstain prompts toward answering")
    all_responses = []
    for feature_id in candidate_features:
        print(f"\nFeature {feature_id} (negative coefficients):")
        responses = batch_steer_experiment(
            model,
            sae,
            abstain_subset,
            feature_id,
            neg_coefficients,
            layer,
        )
        all_responses.extend(responses)

    save_jsonl(all_responses, a2ans_path)
    print(f"\nSaved {len(all_responses)} responses to '{a2ans_path}'")

# ========================================
# = C. Analyse labelled responses        =
# ========================================

a2a_labelled = labelled_dir / "answer_to_abstain.jsonl"
a2ans_labelled = labelled_dir / "abstain_to_answer.jsonl"

if not a2a_labelled.exists() or not a2ans_labelled.exists():
    print("Judge labels not found! Classify responses and then re-run this script!")
    sys.exit(0)

print("\nAnalysing judge-labelled responses")

for direction, path in [
    ("answer_to_abstain", a2a_labelled),
    ("abstain_to_answer", a2ans_labelled),
]:
    labelled = load_labelled_responses(path)
    rates = compute_abstention_rates(labelled)

    print(f"\n--- {direction} ---")
    current_feature = None
    for (feature_id, coeff), stats in sorted(rates.items()):
        if feature_id != current_feature:
            current_feature = feature_id
            print(f"\nFeature {feature_id}:")
        print(
            f" coeff {coeff:+8.0f}: abstention rate {stats['abstain_rate']:.2%} ({stats['abstain']}/{stats['n']})"
        )

    results = {}
    for (feature_id, coeff), stats in rates.items():
        feature_key = str(feature_id)
        if feature_key not in results:
            results[feature_key] = {}
        results[feature_key][str(coeff)] = stats

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"steering_{direction}.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\nResults saved to '{out_dir}'")
