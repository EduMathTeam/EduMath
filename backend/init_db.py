
import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR.parent / "data" / "processed" / "courses.csv"
DB_PATH = BASE_DIR / "db.sqlite3"


def init_db():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as file:
        courses = list(csv.DictReader(file))

    conn = sqlite3.connect(DB_PATH)

    try:
        # Таблица настоящих курсов
        conn.execute("""
            CREATE TABLE IF NOT EXISTS courses (
                course_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                category TEXT,
                description TEXT,
                university TEXT,
                language TEXT,
                directions TEXT
            )
        """)

        conn.execute("DELETE FROM courses")

        for course in courses:
            conn.execute("""
                INSERT INTO courses (
                    course_id, title, category, description,
                    university, language, directions
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                int(course["course_id"]),
                course["title"],
                course["category"] or None,
                course["description"],
                course["university"],
                course["language"],
                course["directions"] or None
            ))

        # Таблица взаимодействий пользователей с курсами
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                user_id INTEGER NOT NULL,
                course_id INTEGER NOT NULL,
                weight REAL NOT NULL,
                PRIMARY KEY (user_id, course_id)
            )
        """)

        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM courses"
        ).fetchone()[0]

        print(f"Загружено курсов: {count}")
        print("Таблица interactions готова")

    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
