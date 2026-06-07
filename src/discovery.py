import torch
import numpy as np
from tqdm import tqdm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# Compute mean difference (Cohen's d per feature)
def mean_activation_difference(activations: np.ndarray, labels: np.ndarray) -> dict:
    activations = activations.astype(np.float32)

    abstain_mask = labels == 1
    answer_mask = labels == 0

    mean_abstain = activations[abstain_mask].mean(axis=0)
    mean_answer = activations[answer_mask].mean(axis=0)

    std_abstain = activations[abstain_mask].std(axis=0) + 1e-8
    std_answer = activations[answer_mask].std(axis=0) + 1e-8

    n_abstain = abstain_mask.sum()
    n_answer = answer_mask.sum()

    pooled_std = np.sqrt(
        ((n_abstain - 1) * std_abstain**2 + (n_answer - 1) * std_answer**2)
        / (n_abstain + n_answer - 2)
    )
    pooled_std = np.maximum(pooled_std, 1e-8)

    cohen_d = (mean_abstain - mean_answer) / pooled_std

    # Welch's t-test -> Two sample t-test per feature (for p-values, unequal variance)
    t_stats, p_values = stats.ttest_ind(
        activations[abstain_mask],
        activations[answer_mask],
        equal_var=False,
        axis=0,
    )

    return {
        "cohen_d": cohen_d,
        "mean_abstain": mean_abstain,
        "mean_answer": mean_answer,
        "t_stats": t_stats,
        "p_values": p_values,
    }


# L1 regularised logistic regression probe
def sparse_linear_probe(
    activations: np.ndarray, labels: np.ndarray, C: float = 1.0, cv_folds: int = 5
) -> dict:
    activations = activations.astype(np.float32)
    clf = LogisticRegression(
        C=C, solver="liblinear", l1_ratio=1, max_iter=5000, random_state=42
    )

    cv_scores = cross_val_score(
        clf, activations, labels, cv=cv_folds, scoring="accuracy"
    )

    clf.fit(activations, labels)
    weights = clf.coef_[0]  # shape: [n_features]
    nonzero_mask = np.abs(weights) > 1e-6
    nonzero_features = np.where(nonzero_mask)[0]

    return {
        "weights": weights,
        "nonzero_features": nonzero_features,
        "n_nonzero": len(nonzero_features),
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores,
    }


# Per feature precision and recall as a binary abstention detector
def activation_frequency(activations: np.ndarray, labels: np.ndarray) -> dict:
    activations = activations.astype(np.float32)
    binary_fires = (activations > 0).astype(np.float32)

    abstain_mask = labels == 1
    answer_mask = labels == 0

    abstain_fire_rate = binary_fires[abstain_mask].mean(axis=0)
    answer_fire_rate = binary_fires[answer_mask].mean(axis=0)

    n_features = activations.shape[1]
    precision = np.zeros(n_features)
    recall = np.zeros(n_features)
    f1 = np.zeros(n_features)

    for i in range(n_features):
        predictions = binary_fires[:, i]
        if predictions.sum() == 0:
            continue
        tp = (predictions[abstain_mask] == 1).sum()
        fp = (predictions[answer_mask] == 1).sum()
        fn = (predictions[abstain_mask] == 0).sum()

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = (
            2 * precision[i] * recall[i] / (precision[i] + recall[i])
            if (precision[i] + recall[i]) > 0
            else 0
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "abstain_fire_rate": abstain_fire_rate,
        "answer_fire_rate": answer_fire_rate,
        "selectivity": abstain_fire_rate - answer_fire_rate,
    }


# Baseline linear probe on raw residual stream
def baseline_raw_residual_probe(
    model,
    sae,
    prompts: list[str],
    labels: np.ndarray,
    max_seq_len: int = 128,
    cv_folds: int = 5,
) -> dict:
    sae_input_hook = model.get_sae_hook_name(sae, internal="hook_sae_input")

    all_residuals = []
    for prompt in tqdm(prompts, desc="Collecting raw residuals"):
        tokens = model.to_tokens(prompt)

        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        with torch.no_grad():
            _, cache = model.run_with_cache_with_saes(tokens, saes=[sae])

        residual = cache[sae_input_hook][0, -1, :].cpu().numpy()
        all_residuals.append(residual)

        del cache
        torch.cuda.empty_cache()

    residuals = np.stack(all_residuals).astype(np.float32)

    clf = LogisticRegression(
        C=1.0, solver="liblinear", l1_ratio=0, max_iter=5000, random_state=42
    )
    cv_scores = cross_val_score(clf, residuals, labels, cv=cv_folds, scoring="accuracy")

    return {
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores,
        "residual_dim": residuals.shape[1],
    }


# Baseline sequence entropy / log-probability as abstention predictor
def baseline_logprob_entropy(
    model,
    prompts: list[str],
    labels: np.ndarray,
    max_seq_len: int = 128,
    cv_folds: int = 5,
) -> dict:
    features = []
    for prompt in tqdm(prompts, desc="Computing logprop features"):
        tokens = model.to_tokens(prompt)

        if tokens.shape[1] > max_seq_len:
            tokens = tokens[:, :max_seq_len]

        with torch.no_grad():
            logits = model(tokens)

        probs = torch.softmax(logits[0, -1, :].float(), dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        top_logprob = torch.log(probs.max()).item()
        top5_prob_mass = probs.topk(5).values.sum().item()

        features.append([entropy, top_logprob, top5_prob_mass])

    features = np.array(features)

    clf = LogisticRegression(
        C=1.0, solver="liblinear", l1_ratio=0, max_iter=5000, random_state=42
    )
    cv_scores = cross_val_score(clf, features, labels, cv=cv_folds, scoring="accuracy")

    return {
        "cv_accuracy": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "feature_names": ["entropy", "top_logprob", "top5_prob_mass"],
    }
