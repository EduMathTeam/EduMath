import ast
import os
from collections import Counter
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
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
        f"Не нашёл ни один из {candidates} в {data_dir}. "
        f"Содержимое: {sorted(os.listdir(data_dir)) if data_dir.is_dir() else 'нет папки'}"
    )


def _as_binary(series: pd.Series) -> np.ndarray:
    """Приводит certificate к 0/1, не падая на строках 'Да'/'Нет'."""
    def norm(v):
        if pd.isna(v):
            return 0.0
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            return 1.0 if v > 0 else 0.0
        s = str(v).strip().lower()
        return 1.0 if s in {"1", "true", "да", "yes", "y", "есть"} else 0.0
    return series.map(norm).values.astype(float)


def _minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class SVDCollaborative:
    """Простейшая коллаборативная фильтрация через усечённое SVD."""

    def __init__(self, n_components: int = 20):
        self.n_components = n_components
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def is_fitted(self) -> bool:
        return self.user_factors is not None

    def fit(self, interactions: dict[int, list[tuple[int, float]]]) -> "SVDCollaborative":
        users = sorted(interactions.keys())
        items = sorted({cid for hist in interactions.values() for cid, _ in hist})
        if not users or not items:
            return self

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {c: i for i, c in enumerate(items)}
        self.idx_to_item = {i: c for c, i in self.item_to_idx.items()}

        R = np.zeros((len(users), len(items)), dtype=float)
        for u, hist in interactions.items():
            for cid, w in hist:
                R[self.user_to_idx[u], self.item_to_idx[cid]] = float(w)

        k = max(1, min(self.n_components, min(R.shape) - 1))
        svd = TruncatedSVD(n_components=k, random_state=42)
        self.user_factors = svd.fit_transform(R)          # users × k
        self.item_factors = svd.components_.T             # items × k
        return self

    def score(self, user_id: int) -> dict[int, float]:
        if not self.is_fitted() or user_id not in self.user_to_idx:
            return {}
        u = self.user_to_idx[user_id]
        s = self.user_factors[u] @ self.item_factors.T
        return {self.idx_to_item[i]: float(s[i]) for i in range(len(self.idx_to_item))}


class CourseRecommender:
    """Гибридный рекомендатель: content-based + popularity + опционально SVD."""

    def __init__(
        self,
        data_dir: Union[str, Path] = DATA_DIR,
        courses_file: Union[str, None] = None,
        groups_file: Union[str, None] = None,
        interactions: Union[dict, pd.DataFrame, None] = None,
        cf_weight: float = 0.0,          # 0 — только контент; 0.3-0.5 — гибрид
        cf_components: int = 20,
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
        self.cf_weight = float(cf_weight)
        self.cf_components = int(cf_components)
        self._cf: SVDCollaborative | None = None

        if self.cf_weight > 0 and self.interactions:
            self.fit_cf()

    # ---------- загрузка ----------
    def _load_courses(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, encoding="utf-8-sig")
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
            codes = []
            for i in ids:
                try:
                    key = int(i)
                except (TypeError, ValueError):
                    continue
                if key in gmap:
                    codes.append(gmap[key])
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
        cert = _as_binary(self.courses["certificate"]) \
            if "certificate" in self.courses.columns else np.zeros(len(self.courses))
        self.popularity = 0.7 * lengths + 0.3 * cert

    # ---------- CF ----------
    def fit_cf(self, n_components: int | None = None) -> SVDCollaborative | None:
        """Переобучает SVD. Возвращает объект CF или None, если данных нет."""
        if not self.interactions:
            return None
        self._cf = SVDCollaborative(n_components=n_components or self.cf_components)
        self._cf.fit(self.interactions)
        return self._cf

    # ---------- сигналы пользователя ----------
    def _history(self, user_id: int, history=None):
        if history is not None:
            return list(history)
        if isinstance(self.interactions, pd.DataFrame):
            sub = self.interactions[self.interactions["user_id"] == user_id]
            return list(zip(sub["course_id"].astype(int), sub["weight"].astype(float)))
        return list(self.interactions.get(user_id, []))

    def _user_profile(self, history):
        rows = [(self.id_to_pos[cid], w) for cid, w in history if cid in self.id_to_pos]
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
        # fallback: если категорий слишком мало — добираем без капа
        if len(out) < top_k:
            chosen = set(out)
            for pos in order:
                if pos in chosen:
                    continue
                out.append(pos)
                if len(out) >= top_k:
                    break
        return out

    # ---------- публичный API ----------
    def recommend(self, user_id: int, top_k: int = 10, history=None):
        """
        history — опциональный явный список [(course_id, weight)].
        Если не передан, берётся из self.interactions.
        """
        history = self._history(user_id, history)
        user_vec = self._user_profile(history)
        seen = {cid for cid, _ in history}

        if user_vec is None:
            content_scores = self.popularity.copy()
        else:
            content = cosine_similarity(user_vec, self.matrix).ravel()
            content_scores = 0.85 * content + 0.15 * self.popularity

        if self._cf is not None and self._cf.is_fitted() and self.cf_weight > 0:
            cf_dict = self._cf.score(user_id)
            cf_scores = np.zeros(len(self.courses))
            for cid, s in cf_dict.items():
                pos = self.id_to_pos.get(cid)
                if pos is not None:
                    cf_scores[pos] = s
            scores = (1 - self.cf_weight) * _minmax(content_scores) \
                     + self.cf_weight * _minmax(cf_scores)
        else:
            scores = content_scores

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
