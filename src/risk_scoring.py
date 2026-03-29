"""
Расчет плотности застройки и экологического риска (Eco Risk Score)
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from config import (
    CRS_METRIC,
    DENSITY_RADIUS_M,
    DENSITY_NEIGHBORS_QUANTILE,
    DENSITY_COVERAGE_QUANTILE,
)


def add_dense_development_features(
    df: pd.DataFrame,
    radius_m: float = DENSITY_RADIUS_M,
    neighbors_q: float = DENSITY_NEIGHBORS_QUANTILE,
    coverage_q: float = DENSITY_COVERAGE_QUANTILE,
) -> gpd.GeoDataFrame:
    """Считает локальную плотность застройки вокруг каждого объекта."""
    gdf = gpd.GeoDataFrame(df.copy(), geometry="geometry", crs="EPSG:4326")
    gdf_m = gdf.to_crs(CRS_METRIC).copy()

    gdf_m["centroid"] = gdf_m.geometry.centroid
    coords = np.column_stack([
        gdf_m["centroid"].x.values,
        gdf_m["centroid"].y.values
    ])

    gdf_m["footprint_area_m2"] = gdf_m.geometry.area

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius_m)

    neighbor_count = []
    neighbor_area_sum = []

    for i, idxs in enumerate(neighbor_lists):
        idxs_wo_self = [j for j in idxs if j != i]
        neighbor_count.append(len(idxs_wo_self))

        if len(idxs_wo_self) > 0:
            area_sum = gdf_m.iloc[idxs_wo_self]["footprint_area_m2"].sum()
        else:
            area_sum = 0.0

        neighbor_area_sum.append(area_sum)

    gdf_m["neighbors_250m"] = neighbor_count
    gdf_m["neighbor_footprint_sum_250m"] = neighbor_area_sum

    circle_area = np.pi * (radius_m ** 2)
    gdf_m["coverage_ratio_250m"] = gdf_m["neighbor_footprint_sum_250m"] / circle_area

    neigh_thr = gdf_m["neighbors_250m"].quantile(neighbors_q)
    cov_thr = gdf_m["coverage_ratio_250m"].quantile(coverage_q)

    gdf_m["dense_by_neighbors"] = gdf_m["neighbors_250m"] >= neigh_thr
    gdf_m["dense_by_coverage"] = gdf_m["coverage_ratio_250m"] >= cov_thr

    gdf_m["dense_area"] = (
        gdf_m["dense_by_neighbors"] |
        gdf_m["dense_by_coverage"]
    )

    gdf_m["dense_score"] = (
        gdf_m["neighbors_250m"].rank(pct=True) * 0.5 +
        gdf_m["coverage_ratio_250m"].rank(pct=True) * 0.5
    )

    print(f"radius_m = {radius_m}")
    print(f"neighbors threshold = {neigh_thr:.2f}")
    print(f"coverage threshold = {cov_thr:.4f}")
    print("dense_area share =", gdf_m["dense_area"].mean())

    return gdf_m


def add_eco_risk_score(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Вычисляет итоговый Eco Risk Score на основе плотности, высоты и сложности."""
    gdf = df.copy()

    # 1. Площадь здания
    area = gdf.geometry.area
    area[gdf.geometry.isna()] = np.nan
    area[gdf.geometry.is_empty] = np.nan
    area[area <= 0] = np.nan

    area_median = area.median()
    if pd.isna(area_median):
        area_median = 1.0

    gdf["footprint_area_m2"] = area.fillna(area_median).values

    # 2. Нормированные компоненты
    gdf["dense_score_norm"] = gdf["dense_score"].fillna(0).clip(0, 1)

    h = gdf["height_final_full"].copy()
    h = h.fillna(h.median() if h.notna().any() else 0)
    gdf["height_score"] = h.rank(pct=True)

    a = gdf["footprint_area_m2"].copy()
    a = a.fillna(a.median() if a.notna().any() else 0)
    gdf["footprint_score"] = a.rank(pct=True)

    # 3. Сложность городской ткани
    match_risk_map = {
        "1:1": 0.2,
        "1:N": 0.7,
        "N:1": 0.7,
        "N:N": 1.0,
        "A_only": 0.5,
        "B_only": 0.5,
    }
    gdf["match_risk_score"] = gdf["match_type"].map(match_risk_map).fillna(0.5)

    complexity_raw = gdf["n_a"].fillna(0) + gdf["n_b"].fillna(0)
    gdf["component_complexity_score"] = complexity_raw.rank(pct=True)

    gdf["urban_complexity_score"] = (
        0.7 * gdf["match_risk_score"] +
        0.3 * gdf["component_complexity_score"]
    )

    # 4. Итоговый Eco Risk Score
    gdf["eco_risk_score"] = (
        0.45 * gdf["dense_score_norm"] +
        0.25 * gdf["height_score"] +
        0.15 * gdf["footprint_score"] +
        0.15 * gdf["urban_complexity_score"]
    ).clip(0, 1)

    gdf["eco_risk_score"] = gdf["eco_risk_score"].fillna(gdf["eco_risk_score"].median())

    # 5. Категории
    gdf["eco_risk_level"] = pd.cut(
        gdf["eco_risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"]
    )

    print("geometry type sample:", type(gdf.geometry.dropna().iloc[0]) if gdf.geometry.notna().any() else None)
    print("crs:", gdf.crs)
    print("footprint_area_m2 min:", gdf["footprint_area_m2"].min())
    print("footprint_area_m2 max:", gdf["footprint_area_m2"].max())
    print("footprint_area_m2 nulls:", gdf["footprint_area_m2"].isna().sum())
    print("eco_risk_score nulls:", gdf["eco_risk_score"].isna().sum())

    return gdf
