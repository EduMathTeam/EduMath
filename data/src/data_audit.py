
import ast
from collections import Counter
from pathlib import Path

import pandas as pd


# ============================================================
# 1. Загрузка данных
# ============================================================

DATA_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = DATA_DIR / "raw"

base_info = pd.read_csv(RAW_DIR / "openedu_courses_base_info.csv")
full_info = pd.read_csv(RAW_DIR / "openedu_courses_full_info.csv")
groups_map = pd.read_csv(RAW_DIR / "openedu_groups_map.csv")

datasets = {
    "Base info": base_info,
    "Full info": full_info,
    "Groups map": groups_map,
}


# ============================================================
# 2. Вспомогательные функции
# ============================================================

def print_section(title):
    """Выводит заголовок отдельного раздела."""
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


def parse_directions(value):
    """Преобразует строку со списком ID направлений в Python-список."""
    if pd.isna(value):
        return []

    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def get_direction_names(direction_ids, lookup):
    """Заменяет ID направлений их названиями."""
    return [
        lookup[direction_id]
        for direction_id in direction_ids
        if direction_id in lookup
    ]


# ============================================================
# 3. Первичная информация о таблицах
# ============================================================

def analyze_dataset_structure():
    for name, df in datasets.items():
        print_section(f"Dataset: {name}")

        print(f"Shape: {df.shape}")
        print("\nColumns:")
        print(df.columns.tolist())

        print("\nData types:")
        print(df.dtypes)


def analyze_dataset_quality():
    for name, df in datasets.items():
        print_section(f"Data quality: {name}")

        # Исключаем созданные нами списковые столбцы
        original_columns = df.drop(
            columns=["directions_list", "direction_names"],
            errors="ignore",
        )

        print("Missing values:")
        print(df.isna().sum().to_string())

        print("\nDuplicate rows:")
        print(original_columns.duplicated().sum())

        print("\nUnique IDs:")
        print(df["index"].nunique())


# ============================================================
# 4. Обработка и проверка направлений
# ============================================================

full_info["directions_list"] = full_info["directions"].apply(
    parse_directions
)

direction_lookup = groups_map.set_index("index")["title"].to_dict()

full_info["direction_names"] = full_info["directions_list"].apply(
    lambda ids: get_direction_names(ids, direction_lookup)
)


def analyze_directions():
    print_section("Educational directions")

    print("Examples:")
    print(
        full_info[
            ["directions", "directions_list"]
        ].head(10).to_string(index=False)
    )

    course_direction_ids = {
        direction
        for directions in full_info["directions_list"]
        for direction in directions
    }

    group_ids = set(groups_map["index"])
    missing_ids = course_direction_ids - group_ids

    print("\nDirection ID check:")
    print(f"Unique direction IDs in courses: {len(course_direction_ids)}")
    print(f"Available direction IDs: {len(group_ids)}")
    print(f"Missing direction IDs: {missing_ids}")

    print(
        "\nCourses without directions:",
        full_info["directions_list"].apply(len).eq(0).sum(),
    )

    print(
        "Average directions per course:",
        round(full_info["directions_list"].apply(len).mean(), 2),
    )

    print("\nExamples with direction names:")
    print(
        full_info[
            ["title", "directions_list", "direction_names"]
        ].head(10).to_string(index=False)
    )

    print(
        "\nUnique group codes:",
        groups_map["code"].nunique(),
    )

    direction_counts = Counter(
        direction
        for directions in full_info["direction_names"]
        for direction in directions
    )

    print("\nTop 20 educational directions:")

    for direction, count in direction_counts.most_common(20):
        print(f"{direction}: {count}")

    print(f"\nTotal unique directions: {len(direction_counts)}")


# ============================================================
# 5. Сравнение исходных таблиц
# ============================================================

def compare_course_tables():
    base_ids = set(base_info["index"])
    full_ids = set(full_info["index"])

    print_section("Course ID comparison")

    print("IDs only in base:", base_ids - full_ids)
    print("IDs only in full:", full_ids - base_ids)

    common_columns = [
        "title",
        "university",
        "url",
        "start_date",
        "end_date",
    ]

    base_compare = (
        base_info.set_index("index")[common_columns].sort_index()
    )

    full_compare = (
        full_info.set_index("index")[common_columns].sort_index()
    )

    print_section("Common columns comparison")

    for column in common_columns:
        differences = base_compare[column] != full_compare[column]

        print(f"{column}: {differences.sum()} differences")

        if differences.any():
            print("Examples:")
            print(base_compare.loc[differences, column].head())
            print(full_compare.loc[differences, column].head())


# ============================================================
# 6. Числовые характеристики курсов
# ============================================================

def analyze_numeric_columns():
    numeric_columns = [
        "weeks_min",
        "weeks_max",
        "hours_min",
        "hours_max",
        "credit_units",
    ]

    print_section("Numeric columns statistics")
    print(full_info[numeric_columns].describe())

    print_section("Invalid numeric values")

    print(
        "Negative hours:",
        full_info[["hours_min", "hours_max"]]
        .lt(0)
        .any(axis=1)
        .sum(),
    )

    print(
        "Negative weeks:",
        full_info[["weeks_min", "weeks_max"]]
        .lt(0)
        .any(axis=1)
        .sum(),
    )

    print(
        "Max hours less than min hours:",
        (full_info["hours_max"] < full_info["hours_min"]).sum(),
    )

    print(
        "Max weeks less than min weeks:",
        (full_info["weeks_max"] < full_info["weeks_min"]).sum(),
    )


# ============================================================
# 7. Проверка дат
# ============================================================

def analyze_dates():
    date_columns = ["start_date", "end_date"]

    for column in date_columns:
        full_info[column] = pd.to_datetime(
            full_info[column],
            errors="coerce",
        )

    print_section("Date columns statistics")

    for column in date_columns:
        print(f"\n{column}:")
        print("Minimum:", full_info[column].min())
        print("Maximum:", full_info[column].max())
        print("Missing or invalid:", full_info[column].isna().sum())

    invalid_dates = (
        full_info["end_date"] < full_info["start_date"]
    ).sum()

    print("\nEnd date earlier than start date:", invalid_dates)


# ============================================================
# 8. Категориальные характеристики
# ============================================================

def analyze_categories():
    print_section("Courses by language")
    print(full_info["language"].value_counts(dropna=False))

    print_section("Courses by university")
    print(full_info["university"].value_counts().head(15))

    print(
        "\nUnique universities:",
        full_info["university"].nunique(),
    )

    print_section("Course formats")
    print(full_info["course_format"].value_counts(dropna=False).to_string())

    print_section("Certificate availability")
    print(full_info["certificate"].value_counts(dropna=False).to_string())


# ============================================================
# 9. Анализ текстовых характеристик
# ============================================================

def analyze_text_columns():
    text_columns = [
        "about",
        "syllabus",
        "results",
        "intro",
    ]

    print_section("Text columns analysis")

    for column in text_columns:
        text_data = (
            full_info[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        text_data = text_data[text_data != ""]

        text_lengths = text_data.str.len()
        word_counts = text_data.str.split().str.len()

        print(f"\nColumn: {column}")
        print(f"Filled values: {len(text_data)}")
        print(f"Missing values: {full_info[column].isna().sum()}")
        print(f"Average characters: {text_lengths.mean():.2f}")
        print(f"Average words: {word_counts.mean():.2f}")

        if not text_lengths.empty:
            print(f"Minimum characters: {text_lengths.min()}")
            print(f"Maximum characters: {text_lengths.max()}")


def analyze_short_texts():
    text_columns = [
        "about",
        "syllabus",
        "results",
        "intro",
    ]

    print_section("Short text analysis")

    for column in text_columns:
        text_data = (
            full_info[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        empty_count = text_data.eq("").sum()

        short_mask = text_data.ne("") & text_data.str.len().le(10)
        short_count = short_mask.sum()

        print(f"\nColumn: {column}")
        print(f"Empty texts: {empty_count}")
        print(f"Texts up to 10 characters: {short_count}")

        if short_count > 0:
            short_examples = full_info.loc[
                short_mask,
                ["title", column],
            ]

            print(short_examples.to_string(index=False))


def analyze_duplicate_texts():
    text_columns = [
        "about",
        "syllabus",
        "results",
        "intro",
    ]

    print_section("Duplicate text values")

    for column in text_columns:
        text_data = (
            full_info[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        valid_texts = text_data[text_data != ""]
        duplicates = valid_texts.duplicated().sum()

        print(f"{column}: {duplicates} duplicate values")


# ============================================================
# 10. Запуск аудита
# ============================================================

def main():
    analyze_dataset_structure()
    analyze_dataset_quality()
    analyze_directions()
    compare_course_tables()
    analyze_numeric_columns()
    analyze_dates()
    analyze_categories()
    analyze_text_columns()
    analyze_short_texts()
    analyze_duplicate_texts()


if __name__ == "__main__":
    main()
