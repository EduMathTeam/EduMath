import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.main import CourseRecommender


def precision_at_k(recommended, relevant, k=10):
    top = set(recommended[:k])
    return len(top & set(relevant)) / k if top else 0.0


def ndcg_at_k(recommended, relevant_grades, k=10):
    dcg = 0.0
    for i, cid in enumerate(recommended[:k]):
        g = relevant_grades.get(cid, 0.0)
        if g > 0:
            dcg += (2 ** g - 1) / np.log2(i + 2)
    ideal = sorted(relevant_grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def make_synthetic_interactions(rec, n_users=50, per_user=8, seed=42):
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for _, row in rec.courses.iterrows():
        cat = row["category"]
        if cat and cat != "Unknown":
            by_cat[cat].append(int(row["index"]))

    cats = [c for c, ids in by_cat.items() if len(ids) >= 20]
    interactions = {}
    for u in range(1, n_users + 1):
        primary = rng.choice(cats)
        secondary = rng.choice([c for c in cats if c != primary])
        pool = by_cat[primary] + by_cat[secondary]
        chosen = rng.sample(pool, min(per_user, len(pool)))
        interactions[u] = [(cid, 1.0) for cid in chosen]
    return interactions


def split_train_test(interactions, test_ratio=0.2, seed=42):
    rng = random.Random(seed + 1)
    train, test = {}, {}
    for u, hist in interactions.items():
        hist = list(hist)
        rng.shuffle(hist)
        n_test = max(1, int(round(len(hist) * test_ratio)))
        test[u] = hist[:n_test]
        train[u] = hist[n_test:]
    return train, test


def main():
    data_dir = Path(os.getenv("OPENEDU_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    rec = CourseRecommender(data_dir=data_dir, interactions={})

    interactions = make_synthetic_interactions(rec, n_users=50, per_user=8)
    train, test = split_train_test(interactions)

    precisions, ndcgs = [], []
    for u, test_hist in test.items():
        if not test_hist:
            continue
        train_hist = train[u]
        recs = rec.recommend(user_id=u, top_k=10, history=train_hist)
        rec_ids = [r["course_id"] for r in recs]

        relevant_ids = {cid for cid, _ in test_hist}
        grades = {cid: float(w) for cid, w in test_hist}

        precisions.append(precision_at_k(rec_ids, relevant_ids, k=10))
        ndcgs.append(ndcg_at_k(rec_ids, grades, k=10))

    print(f"Users evaluated: {len(precisions)}")
    print(f"Precision@10: {np.mean(precisions):.4f}")
    print(f"NDCG@10:      {np.mean(ndcgs):.4f}")

    # Для контроля: сравним с «случайным» baseline
    print("\n(Sanity) Полностью «слепые» рекомендации для тех же пользователей:")
    blind_prec, blind_ndcg = [], []
    for u, test_hist in test.items():
        if not test_hist:
            continue
        recs = rec.recommend(user_id=u, top_k=10, history=[])  # без истории
        rec_ids = [r["course_id"] for r in recs]
        relevant_ids = {cid for cid, _ in test_hist}
        grades = {cid: float(w) for cid, w in test_hist}
        blind_prec.append(precision_at_k(rec_ids, relevant_ids, k=10))
        blind_ndcg.append(ndcg_at_k(rec_ids, grades, k=10))
    print(f"Precision@10 (cold): {np.mean(blind_prec):.4f}")
    print(f"NDCG@10 (cold):      {np.mean(blind_ndcg):.4f}")


if __name__ == "__main__":
    main()
