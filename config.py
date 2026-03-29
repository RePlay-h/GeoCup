from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PATH_A = RAW_DIR / "cup_it_example_src_A.csv"
PATH_B = RAW_DIR / "cup_it_example_src_B.csv"

CRS_GEOGRAPHIC = "EPSG:4326"
CRS_METRIC = 32636  # UTM zone for Saint Petersburg

CITY_CODE = "spb"

LISA_DISTANCE_THRESHOLD_M = 200
LISA_P_THRESHOLD = 0.05

MATCH_IOU_THRESHOLD = 0.30
MATCH_OVERLAP_THRESHOLD = 0.80

HEIGHT_REVIEW_CONFIDENCE_THRESHOLD = 0.65

DENSE_RADIUS_M = 250
DENSE_NEIGHBORS_Q = 0.90
DENSE_COVERAGE_Q = 0.90

OUTPUT_FILES = {
    "a_prepared": PROCESSED_DIR / "A_prepared.csv",
    "b_prepared": PROCESSED_DIR / "B_prepared.csv",
    "strong_pairs": PROCESSED_DIR / "strong_pairs.csv",
    "matched_buildings": PROCESSED_DIR / "matched_buildings.csv",
    "final_results": PROCESSED_DIR / "final_results.csv",
    "frontend_geojson": PROCESSED_DIR / "frontend.geojson",
    "geometry_diag_a_before": PROCESSED_DIR / "geometry_diag_a_before.csv",
    "geometry_diag_a_after": PROCESSED_DIR / "geometry_diag_a_after.csv",
    "geometry_diag_b_before": PROCESSED_DIR / "geometry_diag_b_before.csv",
    "geometry_diag_b_after": PROCESSED_DIR / "geometry_diag_b_after.csv",
    "ml_cv_summary": PROCESSED_DIR / "ml_cv_summary.csv",
}

FRONTEND_COLUMNS = [
    "component_id",
    "match_type",
    "height_final_full",
    "confidence_score",
    "eco_risk_score",
    "eco_risk_level",
    "dense_area",
    "dense_score",
    "geometry",
]
