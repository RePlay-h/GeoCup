from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from src.utils import first_notna, mean_notna, safe_mean, weighted_mean


MODEL_BASE_FEATURES = [
    "n_a",
    "n_b",
    "max_iou",
    "match_type",
    "max_overlap_a",
    "max_overlap_b",
    "centroid_x",
    "centroid_y",
    "area",
    "perimeter",
    "compactness",
    "bbox_width",
    "bbox_height",
    "elongation",
]

MODEL_CAT_FEATURES = ["match_type"]


def aggregate_component_features(
    components_df: pd.DataFrame,
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
    metric_crs: int | str = 32636,
) -> pd.DataFrame:
    components_df = components_df.copy()

    b_height_list = []
    b_stairs_list = []
    b_afh_list = []
    b_area_list = []

    a_floor_min_list = []
    a_floor_max_list = []
    a_floor_mid_list = []

    for _, row in components_df.iterrows():
        a_idxs = row["a_idxs"]
        b_idxs = row["b_idxs"]

        if b_idxs:
            sub_b = b_gdf.loc[b_idxs].copy()
            heights = sub_b["height_clean_final"].dropna() if "height_clean_final" in sub_b.columns else pd.Series(dtype=float)
            stairs = sub_b["stairs_clean"].dropna() if "stairs_clean" in sub_b.columns else pd.Series(dtype=float)
            afhs = sub_b["avg_floor_height_clean"].dropna() if "avg_floor_height_clean" in sub_b.columns else pd.Series(dtype=float)

            sub_b_metric = sub_b.to_crs(metric_crs).copy()
            sub_b["area_b_geom"] = sub_b_metric.geometry.area

            b_height_list.append(list(heights.values))
            b_stairs_list.append(list(stairs.values))
            b_afh_list.append(list(afhs.values))
            b_area_list.append(list(sub_b["area_b_geom"].values))
        else:
            b_height_list.append([])
            b_stairs_list.append([])
            b_afh_list.append([])
            b_area_list.append([])

        if a_idxs:
            sub_a = a_gdf.loc[a_idxs].copy()
            floor_min = sub_a["gkh_floor_count_min_clean"].dropna() if "gkh_floor_count_min_clean" in sub_a.columns else pd.Series(dtype=float)
            floor_max = sub_a["gkh_floor_count_max_clean"].dropna() if "gkh_floor_count_max_clean" in sub_a.columns else pd.Series(dtype=float)
            floor_mid = sub_a["gkh_floor_mid"].dropna() if "gkh_floor_mid" in sub_a.columns else pd.Series(dtype=float)

            a_floor_min_list.append(list(floor_min.values))
            a_floor_max_list.append(list(floor_max.values))
            a_floor_mid_list.append(list(floor_mid.values))
        else:
            a_floor_min_list.append([])
            a_floor_max_list.append([])
            a_floor_mid_list.append([])

    components_df["b_height_values"] = b_height_list
    components_df["b_stairs_values"] = b_stairs_list
    components_df["b_afh_values"] = b_afh_list
    components_df["b_area_values"] = b_area_list
    components_df["a_floor_min_values"] = a_floor_min_list
    components_df["a_floor_max_values"] = a_floor_max_list
    components_df["a_floor_mid_values"] = a_floor_mid_list
    return components_df


def get_a_height_range(row: pd.Series, floor_h_min: float = 2.8, floor_h_max: float = 3.0) -> tuple[float, float]:
    floor_mins = [x for x in row["a_floor_min_values"] if pd.notna(x)]
    floor_maxs = [x for x in row["a_floor_max_values"] if pd.notna(x)]

    if len(floor_mins) == 0 or len(floor_maxs) == 0:
        return np.nan, np.nan

    floor_min = min(floor_mins)
    floor_max = max(floor_maxs)
    return floor_min * floor_h_min, floor_max * floor_h_max


def check_b_internal_consistency(
    height_value: float,
    stairs_values: list[float],
    afh_values: list[float],
    tol: float = 3.0,
) -> Optional[bool]:
    if pd.isna(height_value):
        return None

    if len(stairs_values) == 0 or len(afh_values) == 0:
        return None

    stairs_mean = safe_mean(stairs_values)
    afh_mean = safe_mean(afh_values)

    if pd.isna(stairs_mean) or pd.isna(afh_mean):
        return None

    expected_height = stairs_mean * afh_mean
    return abs(height_value - expected_height) <= tol


def select_final_height_for_row(row: pd.Series) -> pd.Series:
    b_heights = [x for x in row["b_height_values"] if pd.notna(x)]
    b_stairs = [x for x in row["b_stairs_values"] if pd.notna(x)]
    b_afhs = [x for x in row["b_afh_values"] if pd.notna(x)]
    b_areas = row["b_area_values"]

    a_h_min, a_h_max = get_a_height_range(row)

    if len(b_heights) == 1:
        height_b = float(b_heights[0])
        b_internal_ok = check_b_internal_consistency(height_b, b_stairs, b_afhs)

        if pd.notna(a_h_min) and pd.notna(a_h_max):
            a_agree = a_h_min <= height_b <= a_h_max
        else:
            a_agree = None

        if b_internal_ok is True and a_agree is True:
            return pd.Series(
                {
                    "height_final": height_b,
                    "height_source": "B_direct_high",
                    "confidence_score": 0.95,
                    "b_internal_ok": True,
                    "a_agree_with_b": True,
                }
            )

        if b_internal_ok is True and a_agree is None:
            return pd.Series(
                {
                    "height_final": height_b,
                    "height_source": "B_direct_high",
                    "confidence_score": 0.90,
                    "b_internal_ok": True,
                    "a_agree_with_b": None,
                }
            )

        if a_agree is False:
            return pd.Series(
                {
                    "height_final": height_b,
                    "height_source": "B_conflict_with_A",
                    "confidence_score": 0.60,
                    "b_internal_ok": b_internal_ok,
                    "a_agree_with_b": False,
                }
            )

        return pd.Series(
            {
                "height_final": height_b,
                "height_source": "B_direct_medium",
                "confidence_score": 0.80,
                "b_internal_ok": b_internal_ok,
                "a_agree_with_b": a_agree,
            }
        )

    if len(b_heights) > 1:
        height_b = weighted_mean(b_heights, b_areas)

        if pd.notna(a_h_min) and pd.notna(a_h_max):
            a_agree = a_h_min <= height_b <= a_h_max
        else:
            a_agree = None

        if a_agree is False:
            return pd.Series(
                {
                    "height_final": height_b,
                    "height_source": "B_weighted_conflict_with_A",
                    "confidence_score": 0.55,
                    "b_internal_ok": None,
                    "a_agree_with_b": False,
                }
            )

        return pd.Series(
            {
                "height_final": height_b,
                "height_source": "B_weighted",
                "confidence_score": 0.75,
                "b_internal_ok": None,
                "a_agree_with_b": a_agree,
            }
        )

    if pd.notna(a_h_min) and pd.notna(a_h_max):
        height_a = (a_h_min + a_h_max) / 2
        return pd.Series(
            {
                "height_final": height_a,
                "height_source": "A_range_only",
                "confidence_score": 0.45,
                "b_internal_ok": None,
                "a_agree_with_b": None,
            }
        )

    return pd.Series(
        {
            "height_final": np.nan,
            "height_source": "unknown",
            "confidence_score": 0.00,
            "b_internal_ok": None,
            "a_agree_with_b": None,
        }
    )


def add_height_pipeline_flags(df: pd.DataFrame, conf_review_thr: float = 0.65) -> pd.DataFrame:
    df = df.copy()

    df["needs_ml"] = df["height_source"].eq("unknown") | df["height_final"].isna()

    conflict_sources = [
        "B_conflict_with_A",
        "B_weighted_conflict_with_A",
    ]

    df["needs_visual_review"] = (
        df["needs_review"].fillna(False)
        | df["height_source"].isin(conflict_sources)
        | (df["confidence_score"] < conf_review_thr)
    )

    df["ready_for_use"] = (
        df["height_final"].notna()
        & (~df["needs_ml"])
        & (~df["needs_visual_review"])
    )
    return df


def add_baseline_component_features(
    components_df: pd.DataFrame,
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    components_df = components_df.copy()

    stairs_comp = []
    afh_comp = []

    a_floor_min_comp = []
    a_floor_max_comp = []
    a_floor_mid_comp = []

    district_comp = []
    locality_comp = []
    purpose_comp = []
    type_comp = []

    for _, row in components_df.iterrows():
        a_idxs = row["a_idxs"]
        b_idxs = row["b_idxs"]

        if b_idxs:
            sub_b = b_gdf.loc[b_idxs]
            stairs_comp.append(mean_notna(sub_b["stairs_clean"]) if "stairs_clean" in sub_b.columns else np.nan)
            afh_comp.append(mean_notna(sub_b["avg_floor_height_clean"]) if "avg_floor_height_clean" in sub_b.columns else np.nan)
            district_comp.append(first_notna(sub_b["district"]) if "district" in sub_b.columns else np.nan)
            locality_comp.append(first_notna(sub_b["locality"]) if "locality" in sub_b.columns else np.nan)
            purpose_comp.append(first_notna(sub_b["purpose_of_building"]) if "purpose_of_building" in sub_b.columns else np.nan)
            type_comp.append(first_notna(sub_b["type"]) if "type" in sub_b.columns else np.nan)
        else:
            stairs_comp.append(np.nan)
            afh_comp.append(np.nan)
            district_comp.append(np.nan)
            locality_comp.append(np.nan)
            purpose_comp.append(np.nan)
            type_comp.append(np.nan)

        if a_idxs:
            sub_a = a_gdf.loc[a_idxs]
            a_floor_min_comp.append(mean_notna(sub_a["gkh_floor_count_min_clean"]) if "gkh_floor_count_min_clean" in sub_a.columns else np.nan)
            a_floor_max_comp.append(mean_notna(sub_a["gkh_floor_count_max_clean"]) if "gkh_floor_count_max_clean" in sub_a.columns else np.nan)
            a_floor_mid_comp.append(mean_notna(sub_a["gkh_floor_mid"]) if "gkh_floor_mid" in sub_a.columns else np.nan)
        else:
            a_floor_min_comp.append(np.nan)
            a_floor_max_comp.append(np.nan)
            a_floor_mid_comp.append(np.nan)

    components_df["stairs_clean_component"] = stairs_comp
    components_df["avg_floor_height_component"] = afh_comp
    components_df["a_floor_min_component"] = a_floor_min_comp
    components_df["a_floor_max_component"] = a_floor_max_comp
    components_df["a_floor_mid_component"] = a_floor_mid_comp
    components_df["district_comp"] = district_comp
    components_df["locality_comp"] = locality_comp
    components_df["purpose_comp"] = purpose_comp
    components_df["type_comp"] = type_comp
    return components_df


def clean_for_model(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    keep_cols = list(dict.fromkeys(feature_cols + ["height_final"]))
    keep_cols = [col for col in keep_cols if col in df.columns]
    model_df = df[keep_cols].copy()

    numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col == "height_final":
            model_df.loc[model_df[col] <= 0, col] = np.nan
            continue
        model_df.loc[model_df[col] < 0, col] = np.nan

    return model_df


def add_features(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    df = df.copy()

    df["area"] = df.geometry.area
    df["perimeter"] = df.geometry.length
    df["compactness"] = 4 * np.pi * df["area"] / (df["perimeter"] ** 2 + 1e-6)

    bounds = df.geometry.bounds
    df["bbox_width"] = bounds["maxx"] - bounds["minx"]
    df["bbox_height"] = bounds["maxy"] - bounds["miny"]
    df["elongation"] = df["bbox_width"] / (df["bbox_height"] + 1e-6)

    centroids = df.geometry.centroid
    df["centroid_x"] = centroids.x
    df["centroid_y"] = centroids.y
    return df


def add_spatial_groups(df: gpd.GeoDataFrame, cell_size: float = 2000) -> gpd.GeoDataFrame:
    df = df.copy()

    centroids = df.geometry.centroid
    df["centroid_x"] = centroids.x
    df["centroid_y"] = centroids.y

    x_bin = (df["centroid_x"] // cell_size).astype(int)
    y_bin = (df["centroid_y"] // cell_size).astype(int)
    df["spatial_group"] = x_bin.astype(str) + "_" + y_bin.astype(str)
    return df


def add_knn_features(
    train_df: pd.DataFrame,
    target_col: str = "height_final",
    k: int = 5,
    valid_df: Optional[pd.DataFrame] = None,
):
    train_df = train_df.copy()

    if len(train_df) == 0:
        if valid_df is None:
            return train_df
        return train_df, valid_df.copy()

    effective_k = max(1, min(k, len(train_df)))
    train_coords = np.vstack([train_df["centroid_x"], train_df["centroid_y"]]).T
    nbrs = NearestNeighbors(n_neighbors=effective_k).fit(train_coords)

    d_train, idx_train = nbrs.kneighbors(train_coords)
    train_df["knn_height_mean"] = [train_df.iloc[idx][target_col].mean() for idx in idx_train]
    train_df["knn_dist_mean"] = d_train.mean(axis=1)

    if valid_df is None:
        return train_df

    valid_df = valid_df.copy()
    valid_coords = np.vstack([valid_df["centroid_x"], valid_df["centroid_y"]]).T
    d_valid, idx_valid = nbrs.kneighbors(valid_coords)
    valid_df["knn_height_mean"] = [train_df.iloc[idx][target_col].mean() for idx in idx_valid]
    valid_df["knn_dist_mean"] = d_valid.mean(axis=1)

    return train_df, valid_df


def _build_candidate_models():
    models = {}

    try:  # pragma: no cover
        from catboost import CatBoostRegressor

        models["catboost"] = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            loss_function="RMSE",
            verbose=False,
        )
    except ImportError:
        pass

    try:  # pragma: no cover
        from lightgbm import LGBMRegressor

        models["lightgbm"] = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
        )
    except ImportError:
        pass

    if not models:
        from sklearn.ensemble import RandomForestRegressor

        models["random_forest"] = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )

    return models


def run_spatial_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str = "height_final",
    n_splits: int = 5,
):
    df = df.copy()

    groups = df["spatial_group"].fillna("missing")
    unique_groups = groups.nunique()

    if len(df) < 10 or unique_groups < 2:
        return pd.DataFrame(), pd.DataFrame(), _build_candidate_models()

    effective_splits = min(n_splits, unique_groups)
    if effective_splits < 2:
        return pd.DataFrame(), pd.DataFrame(), _build_candidate_models()

    gkf = GroupKFold(n_splits=effective_splits)
    models = _build_candidate_models()
    all_results = []

    for model_name, base_model in models.items():
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(df[feature_cols], df[target_col], groups=groups),
            start=1,
        ):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            train_df, val_df = add_knn_features(train_df, target_col=target_col, k=5, valid_df=val_df)

            fold_features = feature_cols.copy()
            for extra_col in ["knn_height_mean", "knn_dist_mean"]:
                if extra_col not in fold_features:
                    fold_features.append(extra_col)

            x_train = train_df[fold_features].copy()
            y_train = train_df[target_col].copy()
            x_val = val_df[fold_features].copy()
            y_val = val_df[target_col].copy()

            fold_cat = [col for col in cat_features if col in fold_features]

            model = base_model.__class__(**base_model.get_params())

            if model_name == "catboost":
                for col in fold_cat:
                    x_train[col] = x_train[col].fillna("missing").astype(str)
                    x_val[col] = x_val[col].fillna("missing").astype(str)
                model.fit(x_train, y_train, cat_features=fold_cat)
            else:
                for col in fold_cat:
                    x_train[col] = x_train[col].fillna("missing").astype("category")
                    x_val[col] = x_val[col].fillna("missing").astype("category")
                model.fit(x_train, y_train)

            pred = model.predict(x_val)

            fold_metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "MAE": mean_absolute_error(y_val, pred),
                    "RMSE": np.sqrt(mean_squared_error(y_val, pred)),
                    "R2": r2_score(y_val, pred),
                }
            )

        all_results.extend(fold_metrics)

    results_df = pd.DataFrame(all_results)
    summary_df = (
        results_df.groupby("model")[["MAE", "RMSE", "R2"]]
        .mean()
        .reset_index()
        .sort_values("RMSE")
    )
    return results_df, summary_df, models


def fit_best_model_and_predict(
    train_df: gpd.GeoDataFrame,
    pred_df: gpd.GeoDataFrame,
    feature_cols: list[str],
    cat_features: list[str],
    target_col: str = "height_final",
):
    if len(train_df) == 0 or len(pred_df) == 0:
        return pred_df.copy(), pd.DataFrame(), None

    cv_results, cv_summary, models = run_spatial_cv(
        train_df,
        feature_cols=feature_cols,
        cat_features=cat_features,
        target_col=target_col,
        n_splits=5,
    )

    if cv_summary.empty:
        model_name, model = next(iter(models.items()))
    else:
        model_name = cv_summary.iloc[0]["model"]
        model = models[model_name]

    train_df, pred_df = add_knn_features(train_df, target_col=target_col, k=5, valid_df=pred_df)

    full_features = feature_cols.copy()
    for extra_col in ["knn_height_mean", "knn_dist_mean"]:
        if extra_col not in full_features:
            full_features.append(extra_col)

    x_train = train_df[full_features].copy()
    y_train = train_df[target_col].copy()
    x_pred = pred_df[full_features].copy()

    effective_cat = [col for col in cat_features if col in full_features]

    if model_name == "catboost":
        for col in effective_cat:
            x_train[col] = x_train[col].fillna("missing").astype(str)
            x_pred[col] = x_pred[col].fillna("missing").astype(str)
        model.fit(x_train, y_train, cat_features=effective_cat)
    else:
        for col in effective_cat:
            x_train[col] = x_train[col].fillna("missing").astype("category")
            x_pred[col] = x_pred[col].fillna("missing").astype("category")
        model.fit(x_train, y_train)

    pred_df = pred_df.copy()
    pred_df["height_ml"] = model.predict(x_pred)
    pred_df["ml_model_name"] = model_name
    return pred_df, cv_summary, model_name


def run_height_recovery(
    components_df: pd.DataFrame,
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
    geographic_crs: str = "EPSG:4326",
    metric_crs: int | str = 32636,
    conf_review_thr: float = 0.65,
    enable_ml: bool = True,
):
    matched_buildings = aggregate_component_features(
        components_df,
        a_gdf=a_gdf,
        b_gdf=b_gdf,
        metric_crs=metric_crs,
    )

    height_result = matched_buildings.apply(select_final_height_for_row, axis=1)
    matched_buildings = pd.concat([matched_buildings, height_result], axis=1)
    matched_buildings = add_height_pipeline_flags(matched_buildings, conf_review_thr=conf_review_thr)
    matched_buildings = add_baseline_component_features(matched_buildings, a_gdf, b_gdf)

    matched_buildings["height_final_full"] = matched_buildings["height_final"]
    matched_buildings["height_ml"] = np.nan
    matched_buildings["ml_model_name"] = None

    if not enable_ml:
        return matched_buildings, pd.DataFrame()

    feature_cols = MODEL_BASE_FEATURES + ["geometry"]
    model_df = clean_for_model(matched_buildings, feature_cols)

    train_df = model_df[model_df["height_final"].notna()].copy()
    pred_idx = matched_buildings.index[matched_buildings["needs_ml"]].tolist()
    pred_df = model_df.loc[model_df.index.intersection(pred_idx)].copy()

    if len(train_df) < 10 or len(pred_df) == 0:
        return matched_buildings, pd.DataFrame()

    train_gdf = gpd.GeoDataFrame(train_df, geometry="geometry", crs=geographic_crs).to_crs(metric_crs)
    pred_gdf = gpd.GeoDataFrame(pred_df, geometry="geometry", crs=geographic_crs).to_crs(metric_crs)

    train_gdf = add_features(train_gdf)
    pred_gdf = add_features(pred_gdf)

    train_gdf = add_spatial_groups(train_gdf)
    pred_gdf = add_spatial_groups(pred_gdf)

    pred_gdf, cv_summary, model_name = fit_best_model_and_predict(
        train_gdf,
        pred_gdf,
        feature_cols=MODEL_BASE_FEATURES,
        cat_features=MODEL_CAT_FEATURES,
        target_col="height_final",
    )

    if "height_ml" not in pred_gdf.columns:
        return matched_buildings, cv_summary

    matched_buildings.loc[pred_gdf.index, "height_ml"] = pred_gdf["height_ml"]
    matched_buildings.loc[pred_gdf.index, "ml_model_name"] = model_name
    fill_mask = matched_buildings["height_final_full"].isna() & matched_buildings["height_ml"].notna()
    matched_buildings.loc[fill_mask, "height_final_full"] = matched_buildings.loc[fill_mask, "height_ml"]
    return matched_buildings, cv_summary
