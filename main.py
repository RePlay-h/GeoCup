import argparse

from config import (
    CITY_CODE,
    CRS_GEOGRAPHIC,
    CRS_METRIC,
    DENSE_COVERAGE_Q,
    DENSE_NEIGHBORS_Q,
    DENSE_RADIUS_M,
    FRONTEND_COLUMNS,
    HEIGHT_REVIEW_CONFIDENCE_THRESHOLD,
    LISA_DISTANCE_THRESHOLD_M,
    LISA_P_THRESHOLD,
    MATCH_IOU_THRESHOLD,
    MATCH_OVERLAP_THRESHOLD,
    OUTPUT_FILES,
    PATH_A,
    PATH_B,
    PROCESSED_DIR,
)
from src.data_loading import load_sources
from src.feature_cleaning import clean_sources, feature_cleaning_summary
from src.geometry_qa import run_topology_qa
from src.height_recovery import run_height_recovery
from src.risk_scoring import run_risk_scoring
from src.spatial_matching import run_spatial_matching
from src.utils import ensure_dir, save_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cup IT pipeline refactored from notebook to files.")
    parser.add_argument("--path-a", default=str(PATH_A), help="Path to source A csv")
    parser.add_argument("--path-b", default=str(PATH_B), help="Path to source B csv")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR), help="Output directory")
    parser.add_argument("--disable-ml", action="store_true", help="Disable ML fallback for missing heights")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processed_dir = ensure_dir(args.processed_dir)

    print("1. Loading raw data...")
    gdf_a, gdf_b = load_sources(args.path_a, args.path_b, crs_geographic=CRS_GEOGRAPHIC)

    print("2. Geometry QA...")
    gdf_a_fixed, diag_a_before, diag_a_after, _ = run_topology_qa(gdf_a, "A", metric_crs=CRS_METRIC)
    gdf_b_fixed, diag_b_before, diag_b_after, _ = run_topology_qa(gdf_b, "B", metric_crs=CRS_METRIC)

    save_csv(diag_a_before, processed_dir / OUTPUT_FILES["geometry_diag_a_before"].name)
    save_csv(diag_a_after, processed_dir / OUTPUT_FILES["geometry_diag_a_after"].name)
    save_csv(diag_b_before, processed_dir / OUTPUT_FILES["geometry_diag_b_before"].name)
    save_csv(diag_b_after, processed_dir / OUTPUT_FILES["geometry_diag_b_after"].name)

    print("3. Feature cleaning...")
    a_clean, b_clean = clean_sources(
        gdf_a_fixed,
        gdf_b_fixed,
        metric_crs=CRS_METRIC,
        city=CITY_CODE,
        lisa_distance_threshold_m=LISA_DISTANCE_THRESHOLD_M,
        lisa_p_threshold=LISA_P_THRESHOLD,
    )
    print(feature_cleaning_summary(a_clean, b_clean))

    save_csv(a_clean, processed_dir / OUTPUT_FILES["a_prepared"].name)
    save_csv(b_clean, processed_dir / OUTPUT_FILES["b_prepared"].name)

    print("4. Spatial matching...")
    strong_pairs_df, components_df, components_gdf, a_clean, b_clean = run_spatial_matching(
        a_clean,
        b_clean,
        metric_crs=CRS_METRIC,
        iou_thr=MATCH_IOU_THRESHOLD,
        overlap_thr=MATCH_OVERLAP_THRESHOLD,
    )
    save_csv(strong_pairs_df, processed_dir / OUTPUT_FILES["strong_pairs"].name)
    save_csv(components_df, processed_dir / OUTPUT_FILES["matched_buildings"].name)

    print("5. Height recovery...")
    matched_buildings, cv_summary = run_height_recovery(
        components_gdf,
        a_clean,
        b_clean,
        geographic_crs=CRS_GEOGRAPHIC,
        metric_crs=CRS_METRIC,
        conf_review_thr=HEIGHT_REVIEW_CONFIDENCE_THRESHOLD,
        enable_ml=not args.disable_ml,
    )

    if not cv_summary.empty:
        cv_summary.to_csv(processed_dir / OUTPUT_FILES["ml_cv_summary"].name, index=False)

    print("6. Risk scoring...")
    final_gdf = run_risk_scoring(
        matched_buildings,
        geographic_crs=CRS_GEOGRAPHIC,
        metric_crs=CRS_METRIC,
        radius_m=DENSE_RADIUS_M,
        neighbors_q=DENSE_NEIGHBORS_Q,
        coverage_q=DENSE_COVERAGE_Q,
    )

    save_csv(final_gdf, processed_dir / OUTPUT_FILES["final_results"].name)

    frontend_cols = [col for col in FRONTEND_COLUMNS if col in final_gdf.columns]
    final_gdf[frontend_cols].to_file(
        processed_dir / OUTPUT_FILES["frontend_geojson"].name,
        driver="GeoJSON",
    )

    print("Done.")
    print(f"Outputs saved to: {processed_dir}")


if __name__ == "__main__":
    main()