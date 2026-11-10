
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="EduMath API")

# Разрешаем локальному Frontend обращаться к Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_methods=["GET"],
    allow_headers=["*"]
)


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
def get_courses(search: str | None = None, category: str | None = None):
    result = courses

    # Поиск по названию курса
    if search:
        result = [
            course for course in result
            if search.lower() in course["title"].lower()
        ]

    # Фильтрация по категории
    if category:
        result = [
            course for course in result
            if category.lower() in course["category"].lower()
        ]

    return result



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
