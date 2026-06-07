import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import setup
from src.model import load_model, load_sae, print_memory_report
from src.discovery import mean_activation_difference, sparse_linear_probe, activation_frequency, baseline_raw_residual_probe, baseline_logprob_entropy

config = setup()
activation_dir = Path(config["reproducibility"]["activation_dir"])
labels = np.load(activation_dir / "labels.npy")

# ========================
# = A. Feature discovery =
# ========================

results_all_layers = {}

for layer in config["sae"]["layers"]:
    print(f"== Layer {layer} ==")

    acts = np.load(activation_dir / f"layer_{layer}" / f"layer_{layer}_last.npy")
    print(f"Activations shape: {acts.shape}")

    # Mean activation difference
    print("\nMean Activation Difference (Cohen's d)")
    mad_results = mean_activation_difference(acts, labels)
    top_mad = np.argsort(np.abs(mad_results["cohen_d"]))[::-1][:20]
    print(f"Top 20 features by |Cohen's d|:")
    for rank, index in enumerate(top_mad):
        d = mad_results["cohen_d"][index]
        p = mad_results["p_values"][index]
        print(f"  {rank+1}. Feature {index}: d={d:.3f}, p={p:.2e}")

    # Sparse linear probe
    print("\nSparse Linear Probe (L1 logistic)")
    probe_results = sparse_linear_probe(
        acts, labels,
        C=config["discovery"]["probe_regularisation"],
        cv_folds=config["discovery"]["probe_cv_folds"],
    )
    print(f"CV accuracy: {probe_results['cv_accuracy']:.3f} ± {probe_results['cv_std']:.3f}")
    print(f"Non-zero features: {probe_results['n_nonzero']}")
    top_probe = np.argsort(np.abs(probe_results["weights"]))[::-1][:20]
    print(f"Top 20 features by |weight|:")
    for rank, index in enumerate(top_probe):
        w = probe_results["weights"][index]
        print(f"  {rank+1}. Feature {index}: weight={w:.4f}")

    # Activation frequency
    print("\nActivation Frequency")
    freq_results = activation_frequency(acts, labels)
    top_freq = np.argsort(freq_results["f1"])[::-1][:20]
    print(f"Top 20 features by F1 (as abstention detector):")
    for rank, index in enumerate(top_freq):
        f1 = freq_results["f1"][index]
        prec = freq_results["precision"][index]
        rec = freq_results["recall"][index]
        print(f"  {rank+1}. Feature {index}: F1={f1:.3f}, P={prec:.3f}, R={rec:.3f}")

    # Consensus features
    top_k = config["discovery"]["top_k_features"]
    mad_set = set(top_mad[:top_k])
    probe_set = set(top_probe[:top_k])
    freq_set = set(top_freq[:top_k])

    consensus_2of3 = (mad_set & probe_set) | (mad_set & freq_set) | (probe_set & freq_set)
    consensus_3of3 = mad_set & probe_set & freq_set

    print(f"Consensus")
    print(f"Top-{top_k} overlap (2-of-3 methods): {len(consensus_2of3)} features: {sorted(consensus_2of3)}")
    print(f"Top-{top_k} overlap (3-of-3 methods): {len(consensus_3of3)} features: {sorted(consensus_3of3)}")

    results_all_layers[layer] = {
        "mad_top20": [int(x) for x in top_mad],
        "probe_top20": [int(x) for x in top_probe],
        "freq_top20": [int(x) for x in top_freq],
        "consensus_2of3": sorted([int(x) for x in consensus_2of3]),
        "consensus_3of3": sorted([int(x) for x in consensus_3of3]),
        "probe_cv_accuracy": float(probe_results["cv_accuracy"]),
        "probe_cv_std": float(probe_results["cv_std"]),
        "probe_n_nonzero": int(probe_results["n_nonzero"]),
    }

out_dir = Path(config["reproducibility"]["table_dir"])
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "discovery_results.json", "w") as f:
    json.dump(results_all_layers, f, indent=2)

print(f"\nResults saved to '{out_dir / 'discovery_results.json'}'")

# ===============
# = B. Baseline =
# ===============

with open(activation_dir / "metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
prompts = [item["prompt"] for item in metadata]

model = load_model(config)
print_memory_report()

baseline_results = {}

# Raw residual probe (per layer)
for layer in config["sae"]["layers"]:
    print(f"\nRaw residual probe: layer {layer}")
    sae = load_sae(config, layer)
    raw_result = baseline_raw_residual_probe(
        model, sae, prompts, labels,
        max_seq_len=config["model"]["max_seq_len"],
        cv_folds=config["discovery"]["probe_cv_folds"],
    )
    print(f"CV accuracy: {raw_result['cv_accuracy']:.3f} ± {raw_result['cv_std']:.3f}")
    print(f"Residual dim: {raw_result['residual_dim']}")
    baseline_results[f"raw_residual_layer_{layer}"] = {
        "cv_accuracy": float(raw_result["cv_accuracy"]),
        "cv_std": float(raw_result["cv_std"]),
        "residual_dim": int(raw_result["residual_dim"]),
    }
    del sae
    torch.cuda.empty_cache()
    print_memory_report()

# Logprob entropy baseline
print("\nLogprob entropy baseline")
logprob_result = baseline_logprob_entropy(
    model, prompts, labels,
    max_seq_len=config["model"]["max_seq_len"],
    cv_folds=config["discovery"]["probe_cv_folds"],
)
print(f"CV accuracy: {logprob_result['cv_accuracy']:.3f} ± {logprob_result['cv_std']:.3f}")
baseline_results["logprob_entropy"] = {
    "cv_accuracy": float(logprob_result["cv_accuracy"]),
    "cv_std": float(logprob_result["cv_std"]),
    "feature_names": logprob_result["feature_names"],
}

# Comparison table
print("Comparison: SAE probe vs baselines")
print(f"{'Method':<35} {'CV Accuracy':>12}")
print(f"{'-'*47}")
for layer in config["sae"]["layers"]:
    sae_acc = results_all_layers[layer]["probe_cv_accuracy"]
    raw_acc = baseline_results[f"raw_residual_layer_{layer}"]["cv_accuracy"]
    print(f"  Layer {layer} SAE probe (L1):         {sae_acc:>8.3f}")
    print(f"  Layer {layer} raw residual (L2):       {raw_acc:>8.3f}")
print(f"  Logprob entropy:                   {baseline_results['logprob_entropy']['cv_accuracy']:>8.3f}")

with open(out_dir / "baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(baseline_results, f, indent=2)

print(f"\nBaseline results saved to '{out_dir / 'baseline_results.json'}'")

# ============================
# = C. Comparison table CSV  =
# ============================

rows = []
for layer in config["sae"]["layers"]:
    rows.append({
        "layer": layer,
        "method": "SAE probe (L1)",
        "cv_accuracy": results_all_layers[layer]["probe_cv_accuracy"],
        "cv_std": results_all_layers[layer]["probe_cv_std"],
        "notes": f"Interpretable, {results_all_layers[layer]['probe_n_nonzero']} nonzero features",
    })
    raw_key = f"raw_residual_layer_{layer}"
    rows.append({
        "layer": layer,
        "method": "Raw residual (L2)",
        "cv_accuracy": baseline_results[raw_key]["cv_accuracy"],
        "cv_std": baseline_results[raw_key]["cv_std"],
        "notes": f"Upper bound, {baseline_results[raw_key]['residual_dim']}-dim",
    })
rows.append({
    "layer": "",
    "method": "Logprob entropy",
    "cv_accuracy": baseline_results["logprob_entropy"]["cv_accuracy"],
    "cv_std": baseline_results["logprob_entropy"]["cv_std"],
    "notes": "3 features (entropy, top logprob, top-5 mass)",
})

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(out_dir / "discovery_comparison_table.csv", index=False)
print(f"Comparison table saved to '{out_dir / 'discovery_comparison_table.csv'}'")

# =========================
# = D. Neuronpedia lookup =
# =========================

neuronpedia_rows = []
for layer in config["sae"]["layers"]:
    acts = np.load(activation_dir / f"layer_{layer}" / f"layer_{layer}_last.npy")
    mad_results = mean_activation_difference(acts, labels)
    cohen_d = mad_results["cohen_d"]

    for feature_id in results_all_layers[layer]["consensus_2of3"]:
        neuronpedia_rows.append({
            "layer": layer,
            "feature_id": feature_id,
            "cohen_d": round(float(cohen_d[feature_id]), 4),
            "neuronpedia_url": f"https://www.neuronpedia.org/gemma-3-1b-it/{layer}-gemmascope-2-res-16k/{feature_id}",
            "interpretation": "",
        })

interpretations_df = pd.DataFrame(neuronpedia_rows)
interpretations_df.to_csv(out_dir / "candidate_features_interpretations.csv", index=False)
print(f"Candidate features CSV saved to '{out_dir / 'candidate_features_interpretations.csv'}'")
print(f"{len(neuronpedia_rows)} consensus features across {len(config['sae']['layers'])} layers")
