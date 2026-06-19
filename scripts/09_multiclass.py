import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.config import setup
from src.discovery import mean_activation_difference, sparse_linear_probe

config = setup()

activation_dir = Path(config["reproducibility"]["activation_dir"])
labels = np.load(activation_dir / "labels.npy")
classes = np.load(activation_dir / "classes.npy", allow_pickle=True)
layer = config["sae"]["primary_layer"]
acts = np.load(activation_dir / f"layer_{layer}" / f"layer_{layer}_last.npy")
top_k = config["discovery"]["top_k_features"]

fig_dir = Path(config["reproducibility"]["figure_dir"])
out_dir = Path(config["reproducibility"]["table_dir"])
fig_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Activations shape: {acts.shape}")
print(f"Labels: {labels.sum()} abstain, {(1 - labels).sum()} answer")
print(f"Classes: {dict(zip(*np.unique(classes, return_counts=True)))}")

# ==================================
# = A. Per-class feature discovery =
# ==================================

class_features = {}
class_cohen_d = {}

for class_name in config["dataset"]["classes"]:
    print(f"\n== Class: {class_name} ==")

    class_mask = classes == class_name
    class_acts = acts[class_mask]
    class_labels = labels[class_mask]

    n_abstain = int(class_labels.sum())
    n_answer = int((1 - class_labels).sum())

    if n_abstain < 10 or n_answer < 10:
        print(f"  Skipping: too few samples (abstain={n_abstain}, answer={n_answer})")
        continue

    print(f"  Samples: {n_abstain} abstain, {n_answer} answer")

    # Cohen's d per feature
    mad = mean_activation_difference(class_acts, class_labels)
    top_features_mad = np.argsort(np.abs(mad["cohen_d"]))[::-1][:top_k]

    print(f" Top {top_k} features by |Cohen's d|:")
    for rank, index in enumerate(top_features_mad[:10]):
        d = mad["cohen_d"][index]
        p = mad["p_values"][index]
        print(f"    {rank + 1}. Feature {index}: d={d:.3f}, p={p:.2e}")

    # L1 logistic probe
    probe = sparse_linear_probe(
        class_acts,
        class_labels,
        C=config["discovery"]["probe_regularisation"],
        cv_folds=config["discovery"]["probe_cv_folds"],
    )
    top_features_probe = np.argsort(np.abs(probe["weights"]))[::-1][:top_k]

    print(f" Probe CV accuracy: {probe['cv_accuracy']:.3f} ± {probe['cv_std']:.3f}")
    print(f" Probe nonzero features: {probe['n_nonzero']}")

    class_features[class_name] = {
        "top_features_mad": [int(x) for x in top_features_mad],
        "top_features_probe": [int(x) for x in top_features_probe],
        "probe_accuracy": float(probe["cv_accuracy"]),
        "probe_std": float(probe["cv_std"]),
        "probe_n_nonzero": int(probe["n_nonzero"]),
        "n_abstain": n_abstain,
        "n_answer": n_answer,
    }
    class_cohen_d[class_name] = mad["cohen_d"]

# =======================================
# = B. Overlap analysis (Jaccard index) =
# =======================================

class_names = list(class_features.keys())
n_classes = len(class_names)

print(f"\n== Feature overlap (Jaccard index, top-{top_k} by |Cohen's d|) ==")

jaccard_matrix = np.zeros((n_classes, n_classes))
for i, c1 in enumerate(class_names):
    for j, c2 in enumerate(class_names):
        set1 = set(class_features[c1]["top_features_mad"][:top_k])
        set2 = set(class_features[c2]["top_features_mad"][:top_k])
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        jaccard_matrix[i, j] = intersection / union if union > 0 else 0
        if i < j:
            shared = sorted(set1 & set2)
            print(
                f"  {c1} vs {c2}: Jaccard = {jaccard_matrix[i, j]:.3f}, shared = {shared}"
            )

# ==============================
# = C. Feature x class heatmap =
# ==============================

all_top_features = set()
for cn in class_names:
    all_top_features.update(class_features[cn]["top_features_mad"][:top_k])
all_top_features = sorted(all_top_features)

print(f"\n{len(all_top_features)} unique features across {n_classes} classes")

heatmap_data = np.zeros((len(all_top_features), n_classes))
for j, cn in enumerate(class_names):
    for i, fid in enumerate(all_top_features):
        heatmap_data[i, j] = class_cohen_d[cn][fid]

fig, ax = plt.subplots(figsize=(10, max(6, len(all_top_features) * 0.4)))
sns.heatmap(
    heatmap_data,
    xticklabels=class_names,
    yticklabels=[f"F{fid}" for fid in all_top_features],
    cmap="RdBu_r",
    center=0,
    annot=True,
    fmt=".2f",
    ax=ax,
)
ax.set_xlabel("Abstention class")
ax.set_ylabel("SAE Feature")
ax.set_title(f"Feature x Class Cohen's d (Layer {layer})")
plt.tight_layout()

plt.savefig(fig_dir / "feature_class_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig(fig_dir / "feature_class_heatmap.pdf", bbox_inches="tight")
plt.close()
print(f"Heatmap saved to '{fig_dir / 'feature_class_heatmap.png'}'")

# ===========================================
# = D. Cosine similarity between class-mean =
# =    directions in SAE feature space      =
# ===========================================

print(f"\nCosine similarity between class-mean directions")

class_mean_directions = {}
for cn in class_names:
    class_mask = classes == cn
    abstain_mask = class_mask & (labels == 1)
    answer_mask = class_mask & (labels == 0)
    direction = acts[abstain_mask].astype(np.float32).mean(axis=0) - acts[
        answer_mask
    ].astype(np.float32).mean(axis=0)
    norm = np.linalg.norm(direction)
    class_mean_directions[cn] = direction / (norm + 1e-8)
    print(f" {cn}: direction norm = {norm:.4f}")

cosine_matrix = np.zeros((n_classes, n_classes))
for i, c1 in enumerate(class_names):
    for j, c2 in enumerate(class_names):
        cosine_matrix[i, j] = float(
            np.dot(class_mean_directions[c1], class_mean_directions[c2])
        )
        if i < j:
            print(f"  {c1} vs {c2}: cosine similarity = {cosine_matrix[i, j]:.3f}")

results = {
    "layer": layer,
    "top_k": top_k,
    "class_names": class_names,
    "class_features": class_features,
    "jaccard_matrix": jaccard_matrix.tolist(),
    "cosine_matrix": cosine_matrix.tolist(),
    "all_top_features": all_top_features,
}

with open(out_dir / "multiclass_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to '{out_dir / 'multiclass_results.json'}'")
