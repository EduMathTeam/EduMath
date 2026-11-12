import ast
import os
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(os.getenv("OPENEDU_DATA_DIR", "/data/raw"))

COURSES_CANDIDATES = (
    "openedu_courses_full_info.csv",
    "openedu_courses_base_info.csv",
)
GROUPS_CANDIDATES = ("openedu_groups_map.csv",)


def _resolve(data_dir: Path, candidates) -> Path:
    for name in candidates:
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Не нашёл ни один из файлов {candidates} в {data_dir}. "
        f"Содержимое: {sorted(os.listdir(data_dir)) if data_dir.is_dir() else 'нет такой папки'}"
    )


class CourseRecommender:
    """
    Гибридный рекомендатель для openedu.ru:
    - content-based (TF-IDF по title/about/competence/…)
    - popularity prior (длина описания + наличие сертификата)
    - диверсификация: не более 3 курсов одного направления в топ-10

    По умолчанию читает CSV из /data/raw.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = DATA_DIR,
        courses_file: Union[str, None] = None,
        groups_file: Union[str, None] = None,
        interactions: Union[dict, pd.DataFrame, None] = None,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"data_dir не существует: {self.data_dir}")

        courses_path = (
            self.data_dir / courses_file
            if courses_file else _resolve(self.data_dir, COURSES_CANDIDATES)
        )
        groups_path = (
            self.data_dir / groups_file
            if groups_file else _resolve(self.data_dir, GROUPS_CANDIDATES)
        )
        print(f"[recommender] курсы: {courses_path}")
        print(f"[recommender] группы: {groups_path}")

        self.groups = pd.read_csv(groups_path, encoding="utf-8-sig")
        self.courses = self._load_courses(courses_path)
        self._build_index()
        self.interactions = interactions if interactions is not None else {}

    # ---------- загрузка ----------
    def _load_courses(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")

        # в base_info нет части колонок (about, certificate, directions в нужном виде)
        # поэтому делаем все обращения к колонкам безопасными
        if "index" not in df.columns:
            raise ValueError(f"В {path} нет колонки 'index'")

        gmap = {
            int(r["index"]): (str(r["code"]), str(r["title"]))
            for _, r in self.groups.iterrows()
        }

        def pick_category(directions_str):
            if pd.isna(directions_str):
                return "Unknown", "Не указана"
            try:
                ids = ast.literal_eval(directions_str)
            except (ValueError, SyntaxError):
                return "Unknown", "Не указана"
            codes = [gmap[i] for i in ids if i in gmap]
            if not codes:
                return "Unknown", "Не указана"
            (code, title), _ = Counter(codes).most_common(1)[0]
            return code, title

        if "directions" in df.columns:
            cats = df["directions"].apply(pick_category)
            df["category_code"] = cats.str[0]
            df["category"] = cats.str[1]
        else:
            df["category_code"] = "Unknown"
            df["category"] = "Не указана"

        def s(col):
            return df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df))

        df["_text"] = (
            s("title") + " " + s("about") + " " + s("competence") + " " +
            s("result_abilities") + " " + s("result_skills") + " " +
            s("intro") + " " + s("university") + " " + s("category")
        )
        return df

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(self.courses["_text"])

        self.course_ids = self.courses["index"].astype(int).values
        self.id_to_pos = {cid: i for i, cid in enumerate(self.course_ids)}

        lengths = self.courses["_text"].str.len().values.astype(float)
        lengths = lengths / (lengths.max() + 1e-9)
        cert = self.courses["certificate"].fillna(0).values.astype(float) \
            if "certificate" in self.courses.columns else np.zeros(len(self.courses))
        self.popularity = 0.7 * lengths + 0.3 * cert

    # ---------- сигналы пользователя ----------
    def _history(self, user_id: int):
        if isinstance(self.interactions, pd.DataFrame):
            sub = self.interactions[self.interactions["user_id"] == user_id]
            return list(zip(sub["course_id"].astype(int), sub["weight"].astype(float)))
        return self.interactions.get(user_id, [])

    def _user_profile(self, user_id: int):
        rows = [(self.id_to_pos[cid], w) for cid, w in self._history(user_id)
                if cid in self.id_to_pos]
        if not rows:
            return None
        pos, w = zip(*rows)
        w = np.asarray(w, dtype=float)
        return np.asarray(self.matrix[list(pos)].multiply(w[:, None]).mean(axis=0))

    # ---------- диверсификация ----------
    def _diversify(self, order, top_k, max_per_cat=3):
        out, per_cat = [], Counter()
        for pos in order:
            cat = self.courses.iloc[pos]["category"]
            if per_cat[cat] >= max_per_cat:
                continue
            out.append(pos)
            per_cat[cat] += 1
            if len(out) >= top_k:
                break
        return out

    # ---------- публичный API ----------
    def recommend(self, user_id: int, top_k: int = 10):
        user_vec = self._user_profile(user_id)
        seen = {cid for cid, _ in self._history(user_id)}

        if user_vec is None:
            scores = self.popularity.copy()                 # cold start
        else:
            content = cosine_similarity(user_vec, self.matrix).ravel()
            scores = 0.85 * content + 0.15 * self.popularity

        for cid in seen:
            if cid in self.id_to_pos:
                scores[self.id_to_pos[cid]] = -np.inf

        order = np.argsort(-scores)
        top_pos = self._diversify(order, top_k=top_k)

        return [{
            "course_id": int(self.courses.iloc[pos]["index"]),
            "title": self.courses.iloc[pos]["title"],
            "category": self.courses.iloc[pos]["category"],
            "score": round(float(scores[pos]), 4),
        } for pos in top_pos]


# ---- Синглтон + удобная функция ----
_recommender: CourseRecommender | None = None

def recommend_courses(user_id: int, top_k: int = 10, data_dir: Union[str, Path] = DATA_DIR):
    global _recommender
    if _recommender is None:
        _recommender = CourseRecommender(
            data_dir=data_dir,
            interactions=_load_interactions_from_db(),  # заглушка, см. ниже
        )
    return _recommender.recommend(user_id, top_k=top_k)


def _load_interactions_from_db():
    """
    Заглушка. В проде:
      return pd.read_sql("SELECT user_id, course_id, weight FROM interactions", conn)
    """
    return {
        1: [(1, 1.0), (18, 1.0), (150, 0.8), (716, 1.0), (76, 0.5)],
        2: [(35, 1.0), (251, 1.0), (319, 1.0), (296, 0.7)],
    }


if __name__ == "__main__":
    for item in recommend_courses(user_id=1, top_k=10):
        print(item)
