
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# Проверяем получение всех курсов
def test_get_courses():
    response = client.get("/courses")

    assert response.status_code == 200
    assert len(response.json()) == 10


# Проверяем поиск по названию
def test_search_courses():
    response = client.get("/courses?search=Python")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["title"] == "Основы Python"


# Проверяем фильтрацию по категории
def test_filter_courses():
    response = client.get("/courses?category=Программирование")

    assert response.status_code == 200
    assert len(response.json()) == 4


# Проверяем поиск и категорию одновременно
def test_search_and_filter():
    response = client.get(
        "/courses?search=Python&category=Программирование"
    )

    assert response.status_code == 200
    assert len(response.json()) == 1


# Проверяем поиск несуществующего курса
def test_no_results():
    response = client.get("/courses?search=Несуществующий")

    assert response.status_code == 200
    assert response.json() == []


# Проверяем рекомендации
def test_recommendations():
    response = client.get("/recommendations?user_id=1")

    assert response.status_code == 200
    assert len(response.json()) == 10
    assert "score" in response.json()[0]
