
from fastapi import FastAPI

app = FastAPI(title="EduMath API")

# Пока используем тестовые курсы
courses = [
    {"course_id": 1, "title": "Основы Python", "category": "Программирование"},
    {"course_id": 2, "title": "Основы Java", "category": "Программирование"},
    {"course_id": 3, "title": "Машинное обучение", "category": "Искусственный интеллект"},
    {"course_id": 4, "title": "Введение в SQL", "category": "Базы данных"},
    {"course_id": 5, "title": "Веб-разработка", "category": "Программирование"},
    {"course_id": 6, "title": "Нейронные сети", "category": "Искусственный интеллект"},
    {"course_id": 7, "title": "Основы HTML и CSS", "category": "Веб-разработка"},
    {"course_id": 8, "title": "Анализ данных", "category": "Data Science"},
    {"course_id": 9, "title": "Алгоритмы и структуры данных", "category": "Программирование"},
    {"course_id": 10, "title": "Основы математики", "category": "Математика"}
]


@app.get("/courses")
def get_courses():
    return courses


@app.get("/recommendations")
def get_recommendations(user_id: int):
    # Временные рекомендации, позже подключим ML-модель
    recommendations = []

    for i, course in enumerate(courses):
        recommendations.append({
            **course,
            "score": round(1.0 - i * 0.05, 2)
        })

    return recommendations
