
import sqlite3

import pytest
from fastapi.testclient import TestClient

from backend import main, init_db


client = TestClient(main.app)


# Для каждого теста создаём отдельную временную базу
@pytest.fixture(autouse=True)
def test_database(tmp_path, monkeypatch):
    test_db = tmp_path / "test.sqlite3"

    monkeypatch.setattr(init_db, "DB_PATH", test_db)
    monkeypatch.setattr(main, "DB_PATH", test_db)

    init_db.init_db()


# 1. Проверяем получение всех курсов
def test_get_courses():
    response = client.get("/courses")

    assert response.status_code == 200
    assert len(response.json()) == 1189
    assert "course_id" in response.json()[0]
    assert "title" in response.json()[0]
    assert "description" in response.json()[0]


# 2. Проверяем поиск по названию
def test_search_courses():
    courses = client.get("/courses").json()
    title = courses[0]["title"]

    response = client.get("/courses", params={"search": title})

    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert all(
        title.lower() in course["title"].lower()
        for course in response.json()
    )


# 3. Проверяем фильтрацию по категории
def test_filter_courses():
    courses = client.get("/courses").json()
    category = next(
        course["category"]
        for course in courses
        if course["category"]
    )

    response = client.get("/courses", params={"category": category})

    assert response.status_code == 200
    assert len(response.json()) >= 1
    assert all(
        category.lower() in course["category"].lower()
        for course in response.json()
    )


# 4. Проверяем поиск и категорию одновременно
def test_search_and_filter():
    courses = client.get("/courses").json()
    course = next(
        course for course in courses
        if course["category"]
    )

    response = client.get("/courses", params={
        "search": course["title"],
        "category": course["category"]
    })

    assert response.status_code == 200
    assert course["course_id"] in [
        item["course_id"] for item in response.json()
    ]


# 5. Проверяем поиск несуществующего курса
def test_no_results():
    response = client.get(
        "/courses",
        params={"search": "НесуществующийКурс123456789"}
    )

    assert response.status_code == 200
    assert response.json() == []


# 6. Проверяем рекомендации ML
def test_recommendations():
    response = client.get(
        "/recommendations",
        params={"user_id": 1}
    )

    assert response.status_code == 200
    assert len(response.json()) == 10

    for course in response.json():
        assert "course_id" in course
        assert "title" in course
        assert "category" in course
        assert "score" in course


# 7. Проверяем сохранение взаимодействия
def test_save_interaction():
    response = client.post("/interactions", json={
        "user_id": 1,
        "course_id": 899,
        "weight": 1.0
    })

    assert response.status_code == 200
    assert response.json()["status"] == "saved"

    with sqlite3.connect(main.DB_PATH) as conn:
        saved = conn.execute(
            "SELECT weight FROM interactions "
            "WHERE user_id = ? AND course_id = ?",
            (1, 899)
        ).fetchone()

    assert saved is not None
    assert saved[0] == 1.0


# 8. Проверяем обновление взаимодействия
def test_update_interaction():
    client.post("/interactions", json={
        "user_id": 1,
        "course_id": 899,
        "weight": 1.0
    })

    response = client.post("/interactions", json={
        "user_id": 1,
        "course_id": 899,
        "weight": 0.5
    })

    assert response.status_code == 200

    with sqlite3.connect(main.DB_PATH) as conn:
        rows = conn.execute(
            "SELECT weight FROM interactions "
            "WHERE user_id = ? AND course_id = ?",
            (1, 899)
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == 0.5


# 9. Проверяем сохранение несуществующего курса
def test_invalid_course():
    response = client.post("/interactions", json={
        "user_id": 1,
        "course_id": 999999,
        "weight": 1.0
    })

    assert response.status_code == 404


# 10. Проверяем неправильный вес взаимодействия
def test_invalid_weight():
    response = client.post("/interactions", json={
        "user_id": 1,
        "course_id": 899,
        "weight": 2.0
    })

    assert response.status_code == 422


# 11. Проверяем влияние истории на рекомендации
def test_personal_recommendations():
    user_id = 999

    initial = client.get(
        "/recommendations",
        params={"user_id": user_id}
    )

    assert initial.status_code == 200
    assert len(initial.json()) == 10

    course_id = initial.json()[0]["course_id"]

    saved = client.post("/interactions", json={
        "user_id": user_id,
        "course_id": course_id,
        "weight": 1.0
    })

    assert saved.status_code == 200

    updated = client.get(
        "/recommendations",
        params={"user_id": user_id}
    )

    assert updated.status_code == 200
    assert len(updated.json()) == 10
    assert course_id not in [
        course["course_id"] for course in updated.json()
    ]
