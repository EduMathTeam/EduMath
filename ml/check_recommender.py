"""Проверка рекомендателя на полном датасете + тайминги."""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.main import CourseRecommender


def main():
    data_dir = Path(os.getenv("OPENEDU_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
    print(f"DATA_DIR = {data_dir} (exists={data_dir.is_dir()})")

    t0 = time.time()
    rec = CourseRecommender(data_dir=data_dir, interactions={})
    print(f"Load: {time.time() - t0:.2f}s, courses={len(rec.courses)}")

    # 1) cold start
    t0 = time.time()
    cold = rec.recommend(user_id=999999, top_k=10)
    print(f"Cold start: {len(cold)} recs, {time.time() - t0:.3f}s")

    # 2) несколько разных «профилей»
    sample_ids = rec.courses["index"].astype(int).tolist()[:20]
    recs1 = rec.recommend(user_id=1, top_k=10,
                          history=[(cid, 1.0) for cid in sample_ids[:5]])
    recs2 = rec.recommend(user_id=2, top_k=10,
                          history=[(cid, 1.0) for cid in sample_ids[10:15]])

    print(f"User1: {[r['course_id'] for r in recs1]}")
    print(f"User2: {[r['course_id'] for r in recs2]}")

    overlap = set(r["course_id"] for r in recs1) & set(r["course_id"] for r in recs2)
    print(f"Overlap user1/user2: {len(overlap)}/10")

    # 3) sanity: рекомендации не содержат уже просмотренного
    assert all(r["course_id"] not in set(sample_ids[:5]) for r in recs1)
    print("OK: seen-курсы исключены")


if __name__ == "__main__":
    main()
