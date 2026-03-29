import sys
import os

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.metrics import mean_squared_error

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    OUTPUT_PROCESSED,
    CATBOOST_ITERATIONS,
    CATBOOST_DEPTH,
    CATBOOST_LEARNING_RATE,
)

from data_loading import load_both
from geometry_qa import run_topology_qa
from feature_cleaning import (
    add_physical_flags,
    add_iqr_flag,
    prepare_height_for_lisa,
    add_height_zscore_for_lisa,
    add_distance_band_weights,
    add_lisa_flags,
    add_review_groups,
    add_height_clean_final,
    add_height_drop_reason,
    apply_domain_cap,
    add_clean_stairs,
    add_clean_avg_floor_height,
    clean_gkh_floors,
    clean_area_sq_m,
)
from spatial_matching import (
    build_candidate_pairs,
    compute_pair_metrics,
    filter_strong_pairs,
    build_match_components,
    build_components_df,
    add_component_metrics,
    norm_text,
    extract_house_number,
    validate_component_addresses_flexible,
    add_component_review_flags,
    build_component_geometry,
)
from height_recovery import (
    select_final_height_for_row,
    add_height_pipeline_flags,
    aggregate_component_features,
    add_baseline_component_features,
    clean_for_catboost,
    prepare_train_test_data,
)
from risk_scoring import (
    add_dense_development_features,
    add_eco_risk_score,
)
from utils import (
    feature_cleaning_summary,
    height_cleaning_summary,
    save_results,
)


def main():
    print("=" * 60)
    print("CL Cup IT 2026: Building Height Recovery Pipeline")
    print("=" * 60)

    # Создаем директорию вывода
    os.makedirs(OUTPUT_PROCESSED, exist_ok=True)

    print("\n1. Loading data...")
    gdf_a, gdf_b = load_both()

    print("\n2. Geometry QA...")
    gdf_a_fixed, _, _, _ = run_topology_qa(gdf_a, "A")
    gdf_b_fixed, _, _, _ = run_topology_qa(gdf_b, "B")


    print("\n3. Adding physical bounds flags...")
    A = add_physical_flags(gdf_a_fixed, ["area_sq_m", "gkh_floor_count_min", "gkh_floor_count_max"])
    B = add_physical_flags(gdf_b_fixed, ["height", "stairs", "avg_floor_height"])

    print("\n4. Adding IQR flags...")
    B = add_iqr_flag(B, "height")
    B = add_iqr_flag(B, "stairs")
    B = add_iqr_flag(B, "avg_floor_height")

    print("\n5. LISA spatial analysis...")
    B = prepare_height_for_lisa(B)
    B = add_height_zscore_for_lisa(B)
    B, B_m, W = add_distance_band_weights(B)
    B = add_lisa_flags(B)
    B = add_review_groups(B)

    print("LISA statistics:")
    print(f"  Islands: {B['is_island'].sum()}")
    print(f"  LISA flags: {B['lisa_flag'].sum()}")
    print(B["review_group"].value_counts(dropna=False))

    print("\n6. Final height cleaning for B...")
    B = apply_domain_cap(B, city="spb")
    B = add_height_clean_final(B)
    B = add_height_drop_reason(B)
    print(height_cleaning_summary(B))

    B = add_clean_stairs(B)
    B = add_clean_avg_floor_height(B)

    print("\n7. Feature cleaning for A...")
    A = clean_gkh_floors(A)
    A = clean_area_sq_m(A, small_area_threshold=20)

    print(feature_cleaning_summary(A, B))

    # ========== STAGE 8: Пространственное сопоставление ==========
    print("\n8. Spatial matching...")
    candidates_df = build_candidate_pairs(A, B)
    print(f"  Candidate pairs: {len(candidates_df)}")

    A_m = A.to_crs(32636).copy()
    B_m = B.to_crs(32636).copy()
    A_m["area"] = A_m.geometry.area
    B_m["area"] = B_m.geometry.area

    pairs_df = compute_pair_metrics(A_m, B_m, candidates_df)
    print(f"  Pairs with overlap: {len(pairs_df)}")

    strong_pairs_df = filter_strong_pairs(pairs_df)
    print(f"  Strong pairs: {len(strong_pairs_df)}")

    G, components = build_match_components(A, B, strong_pairs_df)
    components_df = build_components_df(components)
    components_df = add_component_metrics(components_df, strong_pairs_df)

    print(f"  Components: {len(components_df)}")
    print("  Match types:")
    print(components_df["match_type"].value_counts())

    # ========== STAGE 9: Адресное сопоставление ==========
    print("\n9. Address matching...")
    A["address_a_norm"] = A["gkh_address"].apply(norm_text)
    A["house_a"] = A["gkh_address"].apply(extract_house_number)
    B["street_b"] = B["name_street"].apply(norm_text)
    B["house_b"] = B["number"].apply(norm_text)

    components_df = validate_component_addresses_flexible(components_df, A, B)
    components_df = add_component_review_flags(components_df)

    print(f"  Needs review: {components_df['needs_review'].sum()}/{len(components_df)}")

    # ========== STAGE 10: Геометрия компонент ==========
    print("\n10. Building component geometry...")
    components_df = build_component_geometry(components_df, A, B)
    components_gdf = gpd.GeoDataFrame(components_df, geometry="geometry", crs=A.crs)

    # ========== STAGE 11: Агрегирование признаков ==========
    print("\n11. Aggregating features for height recovery...")
    matched_buildings = aggregate_component_features(components_df, A, B)

    # ========== STAGE 12: Выбор высоты по алгоритму ==========
    print("\n12. Initial height selection...")
    height_result = matched_buildings.apply(select_final_height_for_row, axis=1)
    matched_buildings = pd.concat([matched_buildings, height_result], axis=1)

    print("  Height sources:")
    print(matched_buildings["height_source"].value_counts(dropna=False))

    matched_buildings = add_height_pipeline_flags(matched_buildings)
    print(f"  Needs ML: {matched_buildings['needs_ml'].sum()}")
    print(f"  Ready for use: {matched_buildings['ready_for_use'].sum()}")

    # ========== STAGE 13: ML-модель для заполнения пропусков (CatBoost) ==========
    print("\n13. Training CatBoost model for height prediction...")
    
    feature_cols = [
        "stairs_clean_component",
        "avg_floor_height_component",
        "a_floor_mid_component",
        "a_floor_min_component",
        "a_floor_max_component",
        "n_a",
        "n_b",
        "max_iou",
        "mean_iou",
        "max_overlap_a",
        "max_overlap_b",
        "match_type",
        "district_comp",
        "locality_comp",
        "purpose_comp",
        "type_comp",
    ]

    cat_features = [
        "match_type",
        "district_comp",
        "locality_comp",
        "purpose_comp",
        "type_comp",
    ]

    matched_buildings = add_baseline_component_features(matched_buildings, A, B)

    from catboost import CatBoostRegressor

    X_train, X_val, y_train, y_val, X_pred, train_idx, pred_idx = prepare_train_test_data(
            matched_buildings,
            feature_cols,
            cat_features,
            test_size=0.2
    )

    if len(X_train) > 10 and len(X_pred) > 0:
        print(f"  Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        print(f"  Predicting for {len(X_pred)} samples")

        model = CatBoostRegressor(
                iterations=CATBOOST_ITERATIONS,
                depth=CATBOOST_DEPTH,
                learning_rate=CATBOOST_LEARNING_RATE,
                loss_function="RMSE",
                verbose=False
        )

        model.fit(X_train, y_train, cat_features=cat_features)

        val_preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        print(f"  Validation RMSE: {rmse:.2f} m")

        preds = model.predict(X_pred)
        matched_buildings.loc[pred_idx, "height_final"] = preds
        matched_buildings["height_final_full"] = matched_buildings["height_final"]
    else:
        matched_buildings["height_final_full"] = matched_buildings["height_final"]
        print("  Insufficient data for ML training")


    print("\n14. Computing building density...")
    df = matched_buildings.drop(['height_final'] if 'height_final' in matched_buildings.columns else [], axis=1)
    df = add_dense_development_features(df)

    print("\n15. Computing Eco Risk Score...")
    eco_df = add_eco_risk_score(df)

    print("  Eco Risk Levels:")
    print(eco_df["eco_risk_level"].value_counts())

    print("\n16. Saving results...")
    save_results(
        eco_df,
        A,
        B,
        matched_buildings,
        strong_pairs_df,
        output_dir=OUTPUT_PROCESSED
    )

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Results saved to {OUTPUT_PROCESSED}")
    print("=" * 60)

    return eco_df, matched_buildings, A, B


if __name__ == "__main__":
    eco_df, matched_buildings, A, B = main()
