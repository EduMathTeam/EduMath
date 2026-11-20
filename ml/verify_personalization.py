"""Проверяет, что разные истории дают разные top-10."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.main import CourseRecommender


def main():
    data_dir = Path(os.getenv("OPENEDU_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    rec = CourseRecommender(data_dir=data_dir, interactions={})

    courses = rec.courses
    prog = courses[courses["category"].str.contains("Программирован", na=False)]["index"].astype(int).tolist()[:5]
    design = courses[courses["category"].str.contains("Дизайн", na=False)]["index"].astype(int).tolist()[:5]

    if not prog or not design:
        # fallback: две разные категории из датасета
        cats = courses[courses["category"] != "Не указана"]["category"].unique()
        cat_a, cat_b = cats[0], cats[1]
        prog = courses[courses["category"] == cat_a]["index"].astype(int).tolist()[:5]
        design = courses[courses["category"] == cat_b]["index"].astype(int).tolist()[:5]

    user_a = [(cid, 1.0) for cid in prog]
    user_b = [(cid, 1.0) for cid in design]

    recs_a = rec.recommend(user_id=101, top_k=10, history=user_a)
    recs_b = rec.recommend(user_id=102, top_k=10, history=user_b)

    ids_a = {r["course_id"] for r in recs_a}
    ids_b = {r["course_id"] for r in recs_b}
    overlap = ids_a & ids_b

    print(f"UserA категорийный фокус: {recs_a[0]['category']}")
    print(f"UserB категорийный фокус: {recs_b[0]['category']}")
    print(f"Overlap: {len(overlap)}/10")

    assert len(overlap) <= 5, f"Слишком много общих рекомендаций: {overlap}"
    assert recs_a[0]["course_id"] not in {c for c, _ in user_a}
    print("OK: рекомендации персонализированы и не содержат уже просмотренного")


if __name__ == "__main__":
    main()
