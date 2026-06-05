import numpy as np
from collections import Counter


def majority_vote(labels):
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def cohens_kappa(rater_a, rater_b):
    assert len(rater_a) == len(rater_b)
    n = len(rater_a)
    categories = sorted(set(rater_a) | set(rater_b))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    confusion = np.zeros((k, k), dtype=int)
    for a, b in zip(rater_a, rater_b):
        confusion[cat_idx[a], cat_idx[b]] += 1

    po = np.trace(confusion) / n
    row_sums = confusion.sum(axis=1)
    col_sums = confusion.sum(axis=0)
    pe = (row_sums * col_sums).sum() / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings_matrix):
    n_items, _ = ratings_matrix.shape
    n_raters = ratings_matrix.sum(axis=1)[0]

    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)
    P_i = (ratings_matrix**2).sum(axis=1) - n_raters
    P_i = P_i / (n_raters * (n_raters - 1))

    P_bar = P_i.mean()
    P_e = (p_j**2).sum()

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)
