"""
Константы и конфигурация проекта CL Cup IT 2026
"""

# Пути к данным
PATH_A = "data/raw/cup_it_example_src_A.csv"
PATH_B = "data/raw/cup_it_example_src_B.csv"

# CRS (Coordinate Reference Systems)
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_METRIC = 32636

# Физические ограничения для значений
PHYSICAL_BOUNDS = {
    "height": (0, 462),
    "stairs": (1, 87),
    "avg_floor_height": (2.5, 8),
    "gkh_floor_count_min": (1, 87),
    "gkh_floor_count_max": (1, 87),
    "area_sq_m": (16, 20000),
}

# Городские caps
CITY_CAPS = {
    "spb": {
        "height": 200,
        "stairs": 40,
    }
}

# Параметры очистки
IQR_K = 1.5
LISA_P_THRESHOLD = 0.05
LISA_THRESHOLD_M = 200

# Параметры сопоставления
SPATIAL_MATCHING_IOU_THR = 0.3
SPATIAL_MATCHING_OVERLAP_THR = 0.8

# Параметры восстановления высоты
FLOOR_HEIGHT_MIN = 2.8
FLOOR_HEIGHT_MAX = 3.0
CONFIDENCE_REVIEW_THR = 0.65

# Параметры плотности застройки
DENSITY_RADIUS_M = 250
DENSITY_NEIGHBORS_QUANTILE = 0.90
DENSITY_COVERAGE_QUANTILE = 0.90

# Параметры ML-модели
CATBOOST_ITERATIONS = 500
CATBOOST_DEPTH = 6
CATBOOST_LEARNING_RATE = 0.1

# Вывод
OUTPUT_PROCESSED = "data/processed/"
OUTPUT_FRONTEND_GEOJSON = "data/processed/frontend.geojson"
OUTPUT_FINAL_CSV = "data/processed/final_results.csv"
OUTPUT_COMPONENTS_CSV = "data/processed/matched_buildings.csv"
