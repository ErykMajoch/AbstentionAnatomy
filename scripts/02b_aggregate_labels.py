import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import json
import random
import numpy as np
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config import setup
from src.judge import majority_vote, cohens_kappa, fleiss_kappa
from src.utils import save_jsonl, load_jsonl

config = setup()
random.seed(config["seed"])

processed_dir = Path("datasets/processed")
validation_dir = Path("datasets/human_validation")
tables_dir = Path(config["reproducibility"]["table_dir"])

validation_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)

JUDGES = {
    "deepseek": processed_dir / "labelled_prompts_deepseek.jsonl",
    "nemotron": processed_dir / "labelled_prompts_nemotron.jsonl",
    "qwen": processed_dir / "labelled_prompts_qwen.jsonl",
}

# ===========================================
# = A. Load judge outputs and majority vote =
# ===========================================

print("Computing judge majority vote")
judge_data = {}
for name, path in JUDGES.items():
    if not path.exists():
        print(f"ERROR: '{path}' not found")
        sys.exit(1)
    judge_data[name] = load_jsonl(path)
    print(f"  {name}: {len(judge_data[name])} items")

judge_names = list(judge_data.keys())
n_items = len(judge_data[judge_names[0]])
for name in judge_names[1:]:
    assert len(judge_data[name]) == n_items, (
        f"Judge {name} has {len(judge_data[name])} items, expected {n_items}"
    )

labelled = []
for i in range(n_items):
    base = judge_data[judge_names[0]][i]
    labels = {name: judge_data[name][i]["behaviour_label"] for name in judge_names}
    vote = majority_vote(list(labels.values()))
    agreement = sum(1 for l in labels.values() if l == vote)

    labelled.append(
        {
            "prompt": base["prompt"],
            "intended_label": base["intended_label"],
            "class": base["class"],
            "subtype": base["subtype"],
            "source": base["source"],
            "pair_id": base["pair_id"],
            "response": base["response"],
            "behaviour_label": vote,
            "judge_labels": labels,
            "judge_agreement": agreement,
            "keyword_label": base["keyword_label"],
            "keyword_hits": base["keyword_hits"],
        }
    )

save_jsonl(labelled, processed_dir / "labelled_prompts.jsonl")

n_unanimous = sum(1 for item in labelled if item["judge_agreement"] == 3)
n_split = n_items - n_unanimous
print(f"Total: {n_items}")
print(f"Unanimous (3/3): {n_unanimous} ({100 * n_unanimous / n_items:.1f}%)")
print(f"Split (2/3): {n_split} ({100 * n_split / n_items:.1f}%)")
print(f"Saved to '{processed_dir / 'labelled_prompts.jsonl'}'")


# ========================
# = B. Agreement metrics =
# ========================

print("\nComputing inter-judge agreement")
all_labels = {
    name: [judge_data[name][i]["behaviour_label"] for i in range(n_items)]
    for name in judge_names
}
label_categories = ["abstain", "answer"]
cat_idx = {c: i for i, c in enumerate(label_categories)}

ratings_matrix = np.zeros((n_items, len(label_categories)), dtype=int)
for i in range(n_items):
    for name in judge_names:
        cat = all_labels[name][i]
        ratings_matrix[i, cat_idx[cat]] += 1

overall_fleiss = fleiss_kappa(ratings_matrix)

pairs = [
    (a, b) for idx_a, a in enumerate(judge_names) for b in judge_names[idx_a + 1 :]
]
pairwise = {}
for a, b in pairs:
    k = cohens_kappa(all_labels[a], all_labels[b])
    pairwise[f"{a}_{b}"] = round(k, 4)

overall_agreement = n_unanimous / n_items

classes = config["dataset"]["classes"]
per_class = {}
for cls in classes:
    cls_indices = [i for i in range(n_items) if labelled[i]["class"] == cls]
    n_cls = len(cls_indices)
    if n_cls == 0:
        continue

    cls_ratings = ratings_matrix[cls_indices]
    cls_fleiss = fleiss_kappa(cls_ratings)

    cls_pairwise = {}
    for a, b in pairs:
        a_labels = [all_labels[a][i] for i in cls_indices]
        b_labels = [all_labels[b][i] for i in cls_indices]
        cls_pairwise[f"{a}_{b}"] = round(cohens_kappa(a_labels, b_labels), 4)

    cls_unanimous = sum(1 for i in cls_indices if labelled[i]["judge_agreement"] == 3)
    per_class[cls] = {
        "n": n_cls,
        "fleiss_kappa": round(cls_fleiss, 4),
        "pairwise_cohens_kappa": cls_pairwise,
        "unanimous_pct": round(100 * cls_unanimous / n_cls, 1),
    }

agreement_report = {
    "n_total": n_items,
    "n_unanimous": n_unanimous,
    "n_split": n_split,
    "fleiss_kappa": round(overall_fleiss, 4),
    "pairwise_cohens_kappa": pairwise,
    "overall_agreement_pct": round(100 * overall_agreement, 1),
    "per_class": per_class,
    "judges": judge_names,
}

with open(tables_dir / "judge_agreement.json", "w", encoding="utf-8") as f:
    json.dump(agreement_report, f, indent=2, ensure_ascii=False)

print(f"Fleiss' kappa (3 raters): {overall_fleiss:.4f}")
print(f"Overall unanimous: {100 * overall_agreement:.1f}%")
print(f"\nPairwise Cohen's kappa:")
for pair_name, kappa in pairwise.items():
    print(f" {pair_name}: {kappa:.4f}")
print(f"\nPer class:")
for cls, metrics in per_class.items():
    print(
        f" {cls}: Fleiss={metrics['fleiss_kappa']:.4f}, unanimous={metrics['unanimous_pct']:.1f}% (n={metrics['n']})"
    )
print(f"Saved to '{tables_dir / 'judge_agreement.json'}'")


# =========================================
# = C. Human validation sample generation =
# =========================================

print("\nHuman validation samples")
n_per_class = config["judge"]["human_validation_n"] // len(classes)

by_class = {cls: [] for cls in classes}
for i, item in enumerate(labelled):
    by_class[item["class"]].append(i)

sampled_indices = []
for cls in classes:
    pool = by_class[cls]
    n_sample = min(n_per_class, len(pool))
    sampled_indices.extend(random.sample(pool, n_sample))

sampled_indices.sort()

samples = []
answer_key = []
for rank, idx in enumerate(sampled_indices):
    item = labelled[idx]
    samples.append(
        {
            "index": rank,
            "prompt": item["prompt"],
            "response": item["response"],
            "class": item["class"],
        }
    )
    answer_key.append(
        {
            "index": rank,
            "behaviour_label": item["behaviour_label"],
            "judge_labels": item["judge_labels"],
            "judge_agreement": item["judge_agreement"],
            "intended_label": item["intended_label"],
        }
    )

save_jsonl(samples, validation_dir / "samples.jsonl")
with open(validation_dir / "answer_key.json", "w", encoding="utf-8") as f:
    json.dump(answer_key, f, indent=2, ensure_ascii=False)

print(f"Sampled {len(samples)} items ({n_per_class} per class)")
for cls in classes:
    n = sum(1 for s in samples if s["class"] == cls)
    print(f"{cls}: {n}")
print(f"Saved to '{validation_dir / 'samples.jsonl'}'")
print(f"Answer key: '{validation_dir / 'answer_key.json'}'")


# ==================================
# = D. Human validation evaluation =
# ==================================

print("\nEvaluating human evaluation")

annotations_path = validation_dir / "annotations.jsonl"

if not annotations_path.exists():
    print(
        f"'{annotations_path}' not found! Annotate the samples in '{validation_dir / 'samples.jsonl'}' and create '{annotations_path}' with one JSON object per line:"
    )
    print(f'{{"index": 0, "human_label": "abstain"}}')
    print(f'{{"index": 1, "human_label": "answer"}}')
    print(f"Then re-run this script!")
else:
    annotations = {
        item["index"]: item["human_label"] for item in load_jsonl(annotations_path)
    }
    key_by_index = {item["index"]: item for item in answer_key}

    matched_indices = sorted(set(annotations.keys()) & set(key_by_index.keys()))
    print(f"Matched {len(matched_indices)} annotations")

    human_labels = [annotations[i] for i in matched_indices]
    panel_labels = [key_by_index[i]["behaviour_label"] for i in matched_indices]

    human_vs_panel = cohens_kappa(human_labels, panel_labels)

    human_vs_judge = {}
    for name in judge_names:
        judge_labels_for_sample = [
            key_by_index[i]["judge_labels"][name] for i in matched_indices
        ]
        human_vs_judge[name] = round(
            cohens_kappa(human_labels, judge_labels_for_sample), 4
        )

    agreement_count = sum(1 for h, p in zip(human_labels, panel_labels) if h == p)
    agreement_pct = 100 * agreement_count / len(matched_indices)

    confusion = {
        "abstain_abstain": 0,
        "abstain_answer": 0,
        "answer_abstain": 0,
        "answer_answer": 0,
    }
    for h, p in zip(human_labels, panel_labels):
        confusion[f"{h}_{p}"] += 1

    sample_classes = [samples[i]["class"] for i in matched_indices]
    per_class_kappa = {}
    for cls in classes:
        cls_mask = [i for i, c in enumerate(sample_classes) if c == cls]
        if len(cls_mask) < 2:
            continue
        cls_human = [human_labels[i] for i in cls_mask]
        cls_panel = [panel_labels[i] for i in cls_mask]
        per_class_kappa[cls] = round(cohens_kappa(cls_human, cls_panel), 4)

    threshold = config["judge"]["human_kappa_threshold"]
    passed = bool(human_vs_panel >= threshold)

    validation_report = {
        "n_annotated": len(matched_indices),
        "human_vs_panel_kappa": round(human_vs_panel, 4),
        "human_vs_panel_agreement_pct": round(agreement_pct, 1),
        "human_vs_individual_judge_kappa": human_vs_judge,
        "confusion_matrix": confusion,
        "per_class_kappa": per_class_kappa,
        "threshold": threshold,
        "passed": passed,
    }

    with open(tables_dir / "human_validation.json", "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)

    status = "PASS" if passed else "FAIL"
    print(
        f"Human vs panel Cohen's kappa: {human_vs_panel:.4f} [{status}] (threshold: {threshold})"
    )
    print(
        f"Raw agreement: {agreement_pct:.1f}% ({agreement_count}/{len(matched_indices)})"
    )
    print(f"Human vs individual judges:")
    for name, k in human_vs_judge.items():
        print(f" {name}: {k:.4f}")
    print(f"\nConfusion matrix (human x panel):")
    print(f"               panel=abstain  panel=answer")
    print(
        f"  human=abstain  {confusion['abstain_abstain']:>8}       {confusion['abstain_answer']:>8}"
    )
    print(
        f"  human=answer   {confusion['answer_abstain']:>8}       {confusion['answer_answer']:>8}"
    )
    if per_class_kappa:
        print(f"\nPer-class kappa:")
        for cls, k in per_class_kappa.items():
            print(f" {cls}: {k:.4f}")
    print(f"Saved to '{tables_dir / 'human_validation.json'}'")


# ===============================
# = E. Train / val / test split =
# ===============================

print("\nCreating training, validation and test splits")

pair_items = {}
for item in labelled:
    pid = item["pair_id"]
    pair_items.setdefault(pid, []).append(item)

pair_ids = list(pair_items.keys())
pair_classes = []
for pid in pair_ids:
    items = pair_items[pid]
    pair_classes.append(items[0]["class"])

test_ratio = config["dataset"]["test_ratio"]
val_ratio = config["dataset"]["val_ratio"]

train_val_ids, test_ids = train_test_split(
    pair_ids,
    test_size=test_ratio,
    stratify=pair_classes,
    random_state=config["seed"],
)

train_val_classes = [pair_classes[pair_ids.index(pid)] for pid in train_val_ids]
relative_val = val_ratio / (1.0 - test_ratio)
train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=relative_val,
    stratify=train_val_classes,
    random_state=config["seed"],
)

split_map = {}
for pid in train_ids:
    split_map[pid] = "train"
for pid in val_ids:
    split_map[pid] = "val"
for pid in test_ids:
    split_map[pid] = "test"

splits = {"train": [], "val": [], "test": []}
for item in labelled:
    split_name = split_map[item["pair_id"]]
    splits[split_name].append(item)

for split_name, items in splits.items():
    save_jsonl(items, processed_dir / f"{split_name}.jsonl")

print(f"Split sizes:")
for split_name in ["train", "val", "test"]:
    items = splits[split_name]
    class_counts = Counter(item["class"] for item in items)
    label_counts = Counter(item["behaviour_label"] for item in items)
    print(
        f"{split_name}: {len(items)} prompts | {dict(label_counts)} | {dict(class_counts)}"
    )

pair_ids_per_split = {
    s: set(item["pair_id"] for item in items) for s, items in splits.items()
}
leakage = (
    len(pair_ids_per_split["train"] & pair_ids_per_split["test"])
    + len(pair_ids_per_split["train"] & pair_ids_per_split["val"])
    + len(pair_ids_per_split["val"] & pair_ids_per_split["test"])
)
if leakage:
    print(f"WARNING: {leakage} pair IDs leaked across splits!")
else:
    print(f"Pair leakage check: PASS")

print(f"Saved to '{processed_dir / '{train,val,test}.jsonl'}'")
