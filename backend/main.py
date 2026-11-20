
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(__file__).resolve().parent / "db.sqlite3"
DATA_DIR = PROJECT_ROOT / "data" / "raw"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.main import CourseRecommender


app = FastAPI(title="EduMath API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


class Interaction(BaseModel):
    user_id: int = Field(gt=0)
    course_id: int = Field(gt=0)
    weight: float = Field(gt=0, le=1)


# Загружаем модель один раз
@lru_cache(maxsize=1)
def get_recommender():
    return CourseRecommender(
        data_dir=DATA_DIR,
        interactions={}
    )


# Защита общей модели при одновременных запросах
recommendation_lock = Lock()


@app.get("/courses")
def get_courses(search: str | None = None, category: str | None = None):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM courses ORDER BY course_id"
        ).fetchall()

    result = [dict(row) for row in rows]

    if search:
        result = [
            course for course in result
            if search.lower() in course["title"].lower()
        ]

    if category:
        result = [
            course for course in result
            if category.lower() in (course["category"] or "").lower()
        ]

    return result


@app.get("/recommendations")
def get_recommendations(user_id: int):
    # Получаем историю пользователя из SQLite
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT course_id, weight FROM interactions WHERE user_id = ?",
            (user_id,)
        ).fetchall()

    history = [
        (int(course_id), float(weight))
        for course_id, weight in rows
    ]

    # Передаём историю ML-модели
    with recommendation_lock:
        recommender = get_recommender()
        recommender.interactions = {user_id: history}

        ml_results = recommender.recommend(
            user_id=user_id,
            top_k=10
        )

    if not ml_results:
        return []

    course_ids = [item["course_id"] for item in ml_results]
    placeholders = ", ".join("?" for _ in course_ids)

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT course_id, title, category FROM courses "
            f"WHERE course_id IN ({placeholders})",
            course_ids
        ).fetchall()

    courses_by_id = {
        row["course_id"]: dict(row)
        for row in rows
    }

    recommendations = []

    for item in ml_results:
        course = courses_by_id.get(item["course_id"])

        if course is not None:
            recommendations.append({
                "course_id": course["course_id"],
                "title": course["title"],
                "category": course["category"],
                "score": item["score"]
            })

    return recommendations


@app.post("/interactions")
def save_interaction(interaction: Interaction):
    with sqlite3.connect(DB_PATH) as conn:
        course = conn.execute(
            "SELECT course_id FROM courses WHERE course_id = ?",
            (interaction.course_id,)
        ).fetchone()

        if course is None:
            raise HTTPException(
                status_code=404,
                detail="Курс не найден"
            )

        conn.execute("""
            INSERT INTO interactions (user_id, course_id, weight)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, course_id)
            DO UPDATE SET weight = excluded.weight
        """, (
            interaction.user_id,
            interaction.course_id,
            interaction.weight
        ))

        conn.commit()

    return {
        "status": "saved",
        "user_id": interaction.user_id,
        "course_id": interaction.course_id,
        "weight": interaction.weight
    }
