"""Evaluation metrics for abstention detection and capability preservation."""

from collections import defaultdict
import numpy as np
from src.utils import load_jsonl


def load_labelled_responses(path) -> list[dict]:
    return load_jsonl(path)


def compute_abstention_rates(
    labelled: list[dict], group_keys: tuple = ("feature_id", "coeff")
) -> dict:
    groups = defaultdict(list)
    for item in labelled:
        key = tuple(item[k] for k in group_keys)
        groups[key].append(item["behaviour_label"])

    rates = {}
    for key, labels in groups.items():
        n_abstain = sum(1 for l in labels if l == "abstain")
        rates[key] = {
            "abstain_rate": n_abstain / len(labels),
            "n": len(labels),
            "abstain": n_abstain,
            "answer": len(labels) - n_abstain,
        }
    return rates


def bootstrap_ci(
    data: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        stats.append(stat_fn(sample))
    stats = np.array(stats)
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(stats, 100 * alpha)),
        float(stat_fn(data)),
        float(np.percentile(stats, 100 * (1 - alpha))),
    )
