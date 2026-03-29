"""
Вспомогательные функции: визуализация, статистика, I/O
"""
import pandas as pd
import geopandas as gpd



def feature_cleaning_summary(A, B):
    """Печатает сводку по очистке признаков."""
    summary = {
        "A_rows": len(A),
        "B_rows": len(B),
        "A_area_notna_before": int(A["area_sq_m"].notna().sum()),
        "A_area_notna_after": int(A["area_sq_m_clean"].notna().sum()),
        "A_area_small_flag": int(A["area_sq_m_small_flag"].sum()),
        "A_gkh_floor_mid_notna": int(A["gkh_floor_mid"].notna().sum()),
        "B_height_notna_before": int(B["height"].notna().sum()),
        "B_height_notna_after": int(B["height_clean_final"].notna().sum()),
        "B_stairs_notna_before": int(B["stairs"].notna().sum()),
        "B_stairs_notna_after": int(B["stairs_clean"].notna().sum()),
        "B_avg_floor_height_notna_before": int(B["avg_floor_height"].notna().sum()),
        "B_avg_floor_height_notna_after": int(B["avg_floor_height_clean"].notna().sum()),
    }
    return pd.Series(summary)


def height_cleaning_summary(df):
    """Печатает сводку по очистке высоты."""
    summary = {
        "rows_total": len(df),
        "height_notna_before": int(df["height"].notna().sum()),
        "height_phys_bad": int(df["height_phys_bad"].sum()),
        "height_iqr_bad": int(df["height_iqr_bad"].sum()),
        "is_island": int(df["is_island"].sum()),
        "lisa_flag": int(df["lisa_flag"].sum()),
        "iqr_and_lisa": int((df["height_iqr_bad"] & df["lisa_flag"]).sum()),
        "height_notna_after": int(df["height_clean_final"].notna().sum()),
        "height_dropped_total": int(df["height"].notna().sum() - df["height_clean_final"].notna().sum()),
    }
    return pd.Series(summary)


def save_results(
    eco_df: gpd.GeoDataFrame,
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame,
    matched_buildings: pd.DataFrame,
    strong_pairs: pd.DataFrame,
    output_dir: str = "data/processed/"
):
    """Сохраняет результаты в файлы."""
    # Промежуточные данные A и B
    A.to_csv(f'{output_dir}A_prepared.csv', index=False)
    B.to_csv(f'{output_dir}B_prepared.csv', index=False)

    # Результаты сопоставления
    matched_buildings.to_csv(f"{output_dir}matched_buildings.csv", index=False)
    strong_pairs.to_csv(f"{output_dir}strong_pairs.csv", index=False)

    # Итоговые результаты
    cols_for_frontend = [
        "match_type",
        "height_final_full",
        "confidence_score",
        "eco_risk_score",
        "eco_risk_level",
        "dense_area",
        "dense_score",
        "geometry",
    ]

    eco_df_export = eco_df[cols_for_frontend].to_crs(epsg=4326)
    eco_df_export.to_file(f"{output_dir}frontend.geojson", driver="GeoJSON")
    eco_df.to_csv(f'{output_dir}final_results.csv', index=False)

    print(f"Results saved to {output_dir}")
