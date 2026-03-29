"""
Вспомогательные функции: визуализация, статистика, I/O
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np


def plot_review_map(gdf, title="Height review map"):
    """Строит карту итоговых review-групп после очистки height."""
    fig, ax = plt.subplots(figsize=(15, 15))

    clean = gdf[gdf["review_group"] == "clean"]
    iqr_only = gdf[gdf["review_group"] == "iqr_only"]
    lisa_only = gdf[gdf["review_group"] == "lisa_only"]
    iqr_and_lisa = gdf[gdf["review_group"] == "iqr_and_lisa"]
    physical_bad = gdf[gdf["review_group"] == "physical_bad"]

    if len(clean) > 0:
        clean.plot(ax=ax, color="lightgrey", linewidth=0.1, alpha=0.6, label="clean")
    if len(iqr_only) > 0:
        iqr_only.plot(ax=ax, color="royalblue", linewidth=0.5, alpha=0.9, label="iqr_only")
    if len(lisa_only) > 0:
        lisa_only.plot(ax=ax, color="orange", linewidth=0.5, alpha=0.9, label="lisa_only")
    if len(iqr_and_lisa) > 0:
        iqr_and_lisa.plot(ax=ax, color="red", linewidth=0.8, alpha=1.0, label="iqr_and_lisa")
    if len(physical_bad) > 0:
        physical_bad.plot(ax=ax, color="black", linewidth=0.8, alpha=1.0, label="physical_bad")

    ax.set_title(title)
    ax.legend()
    ax.set_axis_off()
    plt.show()


def plot_needs_review_map(gdf):
    """Строит карту объектов, требующих проверки."""
    fig, ax = plt.subplots(figsize=(12, 12))

    clean = gdf[~gdf["needs_review"]]
    review = gdf[gdf["needs_review"]]

    if len(clean) > 0:
        clean.plot(ax=ax, color="blue", linewidth=0.1, alpha=0.5, label="clean")

    if len(review) > 0:
        review.plot(ax=ax, color="red", linewidth=0.6, alpha=1.0, label="needs_review")

    ax.set_title("Matching quality: needs_review")
    ax.legend()
    ax.set_axis_off()
    plt.show()


def plot_match_type_map(gdf):
    """Строит карту типов сопоставления."""
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = {
        "1:1": "lightgrey",
        "1:N": "blue",
        "N:1": "green",
        "N:N": "purple",
        "A_only": "orange",
        "B_only": "black",
    }

    for mt, color in colors.items():
        sub = gdf[gdf["match_type"] == mt]
        if len(sub) > 0:
            sub.plot(ax=ax, color=color, linewidth=0.4, alpha=0.8, label=mt)

    ax.set_title("Match types")
    ax.legend()
    ax.set_axis_off()
    plt.show()


def plot_dense_area_map(gdf):
    """Строит карту плотности застройки."""
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = {
        False: "lightgrey",
        True: "red",
    }

    for dense_flag, color in colors.items():
        sub = gdf[gdf["dense_area"] == dense_flag]
        if len(sub) > 0:
            sub.plot(
                ax=ax,
                color=color,
                linewidth=0.4,
                alpha=0.8,
                label=f"dense_area={dense_flag}"
            )

    ax.set_title("Dense area")
    ax.legend()
    ax.set_axis_off()
    plt.show()


def plot_dense_score_map(gdf, figsize=(12, 12)):
    """Строит карту dense score."""
    plot_gdf = gdf.copy()
    fig, ax = plt.subplots(figsize=figsize)

    plot_gdf.plot(
        ax=ax,
        column="dense_score",
        cmap="Reds",
        legend=True,
        linewidth=0.3,
        alpha=0.8,
    )

    ax.set_title("Dense development score")
    ax.set_axis_off()
    plt.show()


def plot_eco_risk_score_map(gdf, figsize=(12, 12), sample=None):
    """Строит карту Eco Risk Score."""
    plot_gdf = gdf.copy()

    plot_gdf = plot_gdf[
        plot_gdf.geometry.notna() &
        (~plot_gdf.geometry.is_empty) &
        plot_gdf["eco_risk_score"].notna()
    ].copy()

    if len(plot_gdf) == 0:
        print("Нечего рисовать: после фильтрации нет валидных geometry или eco_risk_score.")
        return

    if sample is not None and sample < len(plot_gdf):
        plot_gdf = plot_gdf.sample(sample, random_state=42)

    fig, ax = plt.subplots(figsize=figsize)

    plot_gdf.plot(
        ax=ax,
        column="eco_risk_score",
        cmap="YlOrRd",
        legend=True,
        linewidth=0.2,
        alpha=0.9,
        vmin=0,
        vmax=1,
        legend_kwds={"label": "Eco Risk Score", "shrink": 0.7}
    )

    ax.set_title("Eco Risk Score Map")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_eco_risk_level_map(gdf):
    """Строит карту уровней Eco Risk."""
    fig, ax = plt.subplots(figsize=(12, 12))

    colors = {
        "low": "green",
        "medium": "orange",
        "high": "red",
    }

    for level, color in colors.items():
        sub = gdf[gdf["eco_risk_level"] == level]
        if len(sub) > 0:
            sub.plot(
                ax=ax,
                color=color,
                linewidth=0.4,
                alpha=0.8,
                label=level
            )

    ax.set_title("Eco Risk Level")
    ax.legend()
    ax.set_axis_off()
    plt.show()


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
