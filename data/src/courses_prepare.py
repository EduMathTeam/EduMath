from pathlib import Path
import pandas as pd
from collections import Counter

# Пути к файлам
DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Загружаем исходные таблицы
courses = pd.read_csv(RAW_DIR / "openedu_courses_full_info.csv")
groups = pd.read_csv(RAW_DIR / "openedu_groups_map.csv")

# Проверяем загрузку
print(f"Загружено курсов: {len(courses)}")
print(f"Загружено направлений: {len(groups)}")
print("Столбцы курсов:", courses.columns.tolist())


# Создаём словарь: ID направления -> его название
direction_map = dict(
    zip(groups["index"], groups["title"])
)

# Проверяем несколько значений
print("\nПример направлений:")
for direction_id, title in list(direction_map.items())[:5]:
    print(f"{direction_id}: {title}")


# Преобразуем ID направлений в их названия
def get_direction_titles(directions):
    if pd.isna(directions):
        return []

    direction_ids = eval(directions)

    return [
        direction_map[direction_id]
        for direction_id in direction_ids
        if direction_id in direction_map
    ]

# Добавляем названия направлений в таблицу
courses["directions"] = courses["directions"].apply(get_direction_titles)

# Проверяем результат
print("\nПримеры обработанных направлений:")
print(courses[["title", "directions"]].head())


# Подсчитываем, как часто встречается каждое направление
direction_counts = Counter(
    direction
    for directions in courses["directions"]
    for direction in directions
)

# Проверяем 10 самых популярных направлений
print("\nТоп-10 направлений:")
for direction, count in direction_counts.most_common(10):
    print(f"{direction}: {count}")


# Выбираем основную категорию для каждого курса
def get_main_category(directions):
    if not directions:
        return None

    return max(
        directions,
        key=lambda direction: direction_counts[direction]
    )


courses["category"] = courses["directions"].apply(get_main_category)

# Проверяем результат
print("\nПримеры основных категорий:")
print(courses[["title", "category"]].head(10))


# Формируем итоговую таблицу
processed_courses = courses[
    [
        "index",
        "title",
        "category",
        "about",
        "university",
        "language",
        "directions"
    ]
].copy()

# Переименовываем столбцы
processed_courses = processed_courses.rename(
    columns={
        "index": "course_id",
        "about": "description"
    }
)

# Преобразуем списки направлений в строки через ;
processed_courses["directions"] = processed_courses["directions"].apply(
    lambda directions: "; ".join(directions)
    if directions else ""
)

print(processed_courses.head())
print("\nРазмер таблицы:", processed_courses.shape)
print("\nТип directions:", type(processed_courses.loc[0, "directions"]))


# Создаём папку processed, если её ещё нет
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Сохраняем подготовленный датасет
output_path = PROCESSED_DIR / "courses.csv"
processed_courses.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nДатасет сохранён: {output_path}")