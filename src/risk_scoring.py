import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def add_dense_development_features(
    gdf_metric: gpd.GeoDataFrame,
    radius_m: float = 250,
    neighbors_q: float = 0.90,
    coverage_q: float = 0.90,
) -> gpd.GeoDataFrame:
    gdf_metric = gdf_metric.copy()

    centroids = gdf_metric.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    gdf_metric["footprint_area_m2"] = gdf_metric.geometry.area

    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, r=radius_m)

    neighbor_count = []
    neighbor_area_sum = []

    for i, idxs in enumerate(neighbor_lists):
        idxs_wo_self = [j for j in idxs if j != i]
        neighbor_count.append(len(idxs_wo_self))

        if idxs_wo_self:
            area_sum = gdf_metric.iloc[idxs_wo_self]["footprint_area_m2"].sum()
        else:
            area_sum = 0.0
        neighbor_area_sum.append(area_sum)

    gdf_metric["neighbors_250m"] = neighbor_count
    gdf_metric["neighbor_footprint_sum_250m"] = neighbor_area_sum

    circle_area = np.pi * (radius_m ** 2)
    gdf_metric["coverage_ratio_250m"] = gdf_metric["neighbor_footprint_sum_250m"] / circle_area

    neigh_thr = gdf_metric["neighbors_250m"].quantile(neighbors_q)
    cov_thr = gdf_metric["coverage_ratio_250m"].quantile(coverage_q)

    gdf_metric["dense_by_neighbors"] = gdf_metric["neighbors_250m"] >= neigh_thr
    gdf_metric["dense_by_coverage"] = gdf_metric["coverage_ratio_250m"] >= cov_thr
    gdf_metric["dense_area"] = gdf_metric["dense_by_neighbors"] | gdf_metric["dense_by_coverage"]
    gdf_metric["dense_score"] = (
        gdf_metric["neighbors_250m"].rank(pct=True) * 0.5
        + gdf_metric["coverage_ratio_250m"].rank(pct=True) * 0.5
    )
    return gdf_metric


def add_eco_risk_score(gdf_metric: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf_metric = gdf_metric.copy()

    area = gdf_metric.geometry.area
    area[gdf_metric.geometry.isna()] = np.nan
    area[gdf_metric.geometry.is_empty] = np.nan
    area[area <= 0] = np.nan

    area_median = area.median()
    if pd.isna(area_median):
        area_median = 1.0

    gdf_metric["footprint_area_m2"] = area.fillna(area_median).values
    gdf_metric["dense_score_norm"] = gdf_metric["dense_score"].fillna(0).clip(0, 1)

    height = gdf_metric["height_final_full"].copy()
    height = height.fillna(height.median() if height.notna().any() else 0)
    gdf_metric["height_score"] = height.rank(pct=True)

    footprint = gdf_metric["footprint_area_m2"].copy()
    footprint = footprint.fillna(footprint.median() if footprint.notna().any() else 0)
    gdf_metric["footprint_score"] = footprint.rank(pct=True)

    match_risk_map = {
        "1:1": 0.2,
        "1:N": 0.7,
        "N:1": 0.7,
        "N:N": 1.0,
        "A_only": 0.5,
        "B_only": 0.5,
    }
    gdf_metric["match_risk_score"] = gdf_metric["match_type"].map(match_risk_map).fillna(0.5)

    complexity_raw = gdf_metric["n_a"].fillna(0) + gdf_metric["n_b"].fillna(0)
    gdf_metric["component_complexity_score"] = complexity_raw.rank(pct=True)

    gdf_metric["urban_complexity_score"] = (
        0.7 * gdf_metric["match_risk_score"]
        + 0.3 * gdf_metric["component_complexity_score"]
    )

    gdf_metric["eco_risk_score"] = (
        0.45 * gdf_metric["dense_score_norm"]
        + 0.25 * gdf_metric["height_score"]
        + 0.15 * gdf_metric["footprint_score"]
        + 0.15 * gdf_metric["urban_complexity_score"]
    ).clip(0, 1)

    gdf_metric["eco_risk_score"] = gdf_metric["eco_risk_score"].fillna(gdf_metric["eco_risk_score"].median())

    gdf_metric["eco_risk_level"] = pd.cut(
        gdf_metric["eco_risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["low", "medium", "high"],
    )
    return gdf_metric


def run_risk_scoring(
    df: pd.DataFrame,
    geographic_crs: str = "EPSG:4326",
    metric_crs: int | str = 32636,
    radius_m: float = 250,
    neighbors_q: float = 0.90,
    coverage_q: float = 0.90,
) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(df.copy(), geometry="geometry", crs=geographic_crs)
    gdf_metric = gdf.to_crs(metric_crs).copy()

    gdf_metric = add_dense_development_features(
        gdf_metric,
        radius_m=radius_m,
        neighbors_q=neighbors_q,
        coverage_q=coverage_q,
    )
    gdf_metric = add_eco_risk_score(gdf_metric)
    return gdf_metric.to_crs(geographic_crs)
