import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import setup

config = setup()

# Load steering results from Step D.2
fig_dir = Path(config["reproducibility"]["figure_dir"])
fig_dir.mkdir(parents=True, exist_ok=True)

with open("results/tables/discovery_results.json") as f:
    discovery = json.load(f)

layer = config["sae"]["primary_layer"]
candidate_features = discovery[str(layer)]["consensus_2of3"]

with open("results/tables/steering_answer_to_abstain.json") as f:
    answer_to_abstain = json.load(f)
with open("results/tables/steering_abstain_to_answer.json") as f:
    abstain_to_answer = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for feature_id in candidate_features:
    feat_key = str(feature_id)

    if feat_key in answer_to_abstain:
        data = answer_to_abstain[feat_key]
        coeffs = sorted([float(k) for k in data.keys()])
        rates = [data[str(int(c))]["abstain_rate"] for c in coeffs]
        axes[0].plot(coeffs, rates, marker="o", label=f"Feature {feature_id}")

    if feat_key in abstain_to_answer:
        data2 = abstain_to_answer[feat_key]
        coeffs2 = sorted([float(k) for k in data2.keys()])
        rates2 = [data2[str(int(c))]["abstain_rate"] for c in coeffs2]
        axes[1].plot(coeffs2, rates2, marker="s", label=f"Feature {feature_id}")

axes[0].set_xlabel("Steering coefficient")
axes[0].set_ylabel("Abstention rate")
axes[0].set_title("Answer prompts steered toward abstention")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Steering coefficient (negative = suppress)")
axes[1].set_ylabel("Abstention rate")
axes[1].set_title("Abstain prompts steered toward answering")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_dir / "dose_response_curves.png", dpi=300, bbox_inches="tight")
plt.savefig(fig_dir / "dose_response_curves.pdf", bbox_inches="tight")
print(f"Saved dose-response plots to {fig_dir}")
