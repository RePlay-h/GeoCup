"""
Восстановление высоты: алгоритм и ML-модель
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split

from config import (
    FLOOR_HEIGHT_MIN,
    FLOOR_HEIGHT_MAX,
    CONFIDENCE_REVIEW_THR,
)


def safe_mean(values):
    """Безопасно вычисляет среднее (пропускает NaN)."""
    values = [x for x in values if pd.notna(x)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values))


def weighted_mean(values, weights):
    """Вычисляет взвешенное среднее (пропускает NaN)."""
    pairs = [(v, w) for v, w in zip(values, weights) if pd.notna(v) and pd.notna(w) and w > 0]
    if len(pairs) == 0:
        return np.nan
    vals = np.array([v for v, _ in pairs], dtype=float)
    wts = np.array([w for _, w in pairs], dtype=float)
    return float(np.average(vals, weights=wts))


def get_a_height_range(row, floor_h_min=FLOOR_HEIGHT_MIN, floor_h_max=FLOOR_HEIGHT_MAX):
    """Вычисляет диапазон высоты на основе этажности из A."""
    floor_mins = [x for x in row["a_floor_min_values"] if pd.notna(x)]
    floor_maxs = [x for x in row["a_floor_max_values"] if pd.notna(x)]

    if len(floor_mins) == 0 or len(floor_maxs) == 0:
        return (np.nan, np.nan)

    floor_min = min(floor_mins)
    floor_max = max(floor_maxs)

    height_min_a = floor_min * floor_h_min
    height_max_a = floor_max * floor_h_max

    return (height_min_a, height_max_a)


def check_b_internal_consistency(height_value, stairs_values, afh_values, tol=3.0):
    """Проверяет внутреннюю согласованность высоты из B."""
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


def select_final_height_for_row(row):
    """Выбирает итоговую высоту для одной записи по сложному алгоритму."""
    b_heights = [x for x in row["b_height_values"] if pd.notna(x)]
    b_stairs = [x for x in row["b_stairs_values"] if pd.notna(x)]
    b_afhs = [x for x in row["b_afh_values"] if pd.notna(x)]
    b_areas = row["b_area_values"]

    a_h_min, a_h_max = get_a_height_range(row)

    # CASE 1: один B height
    if len(b_heights) == 1:
        height_b = float(b_heights[0])
        b_internal_ok = check_b_internal_consistency(height_b, b_stairs, b_afhs)

        if pd.notna(a_h_min) and pd.notna(a_h_max):
            a_agree = (a_h_min <= height_b <= a_h_max)
        else:
            a_agree = None

        if b_internal_ok is True and a_agree is True:
            return pd.Series({
                "height_final": height_b,
                "height_source": "B_direct_high",
                "confidence_score": 0.95,
                "b_internal_ok": True,
                "a_agree_with_b": True,
            })

        if b_internal_ok is True and a_agree is None:
            return pd.Series({
                "height_final": height_b,
                "height_source": "B_direct_high",
                "confidence_score": 0.90,
                "b_internal_ok": True,
                "a_agree_with_b": None,
            })

        if a_agree is False:
            return pd.Series({
                "height_final": height_b,
                "height_source": "B_conflict_with_A",
                "confidence_score": 0.60,
                "b_internal_ok": b_internal_ok,
                "a_agree_with_b": False,
            })

        return pd.Series({
            "height_final": height_b,
            "height_source": "B_direct_medium",
            "confidence_score": 0.80,
            "b_internal_ok": b_internal_ok,
            "a_agree_with_b": a_agree,
        })

    # CASE 2: несколько B height
    if len(b_heights) > 1:
        height_b = weighted_mean(b_heights, b_areas)

        if pd.notna(a_h_min) and pd.notna(a_h_max):
            a_agree = (a_h_min <= height_b <= a_h_max)
        else:
            a_agree = None

        if a_agree is False:
            return pd.Series({
                "height_final": height_b,
                "height_source": "B_weighted_conflict_with_A",
                "confidence_score": 0.55,
                "b_internal_ok": None,
                "a_agree_with_b": False,
            })

        return pd.Series({
            "height_final": height_b,
            "height_source": "B_weighted",
            "confidence_score": 0.75,
            "b_internal_ok": None,
            "a_agree_with_b": a_agree,
        })

    # CASE 3: только A
    if pd.notna(a_h_min) and pd.notna(a_h_max):
        height_a = (a_h_min + a_h_max) / 2

        return pd.Series({
            "height_final": height_a,
            "height_source": "A_range_only",
            "confidence_score": 0.45,
            "b_internal_ok": None,
            "a_agree_with_b": None,
        })

    # CASE 4: unknown
    return pd.Series({
        "height_final": np.nan,
        "height_source": "unknown",
        "confidence_score": 0.00,
        "b_internal_ok": None,
        "a_agree_with_b": None,
    })


def add_height_pipeline_flags(
    df: pd.DataFrame,
    conf_review_thr: float = CONFIDENCE_REVIEW_THR
) -> pd.DataFrame:
    """Добавляет флаги для фильтрации объектов по стадиям пайплайна."""
    df = df.copy()

    df["needs_ml"] = (
        df["height_source"].eq("unknown") |
        df["height_final"].isna()
    )

    conflict_sources = [
        "B_conflict_with_A",
        "B_weighted_conflict_with_A",
    ]

    df["needs_visual_review"] = (
        df["needs_review"].fillna(False) |
        df["height_source"].isin(conflict_sources) |
        (df["confidence_score"] < conf_review_thr)
    )

    df["ready_for_use"] = (
        df["height_final"].notna() &
        (~df["needs_ml"]) &
        (~df["needs_visual_review"])
    )

    return df


def aggregate_component_features(
    components_df: pd.DataFrame,
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Агрегирует признаки из компонент для использования в модели."""
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

        if len(b_idxs) > 0:
            sub_b = B.loc[b_idxs].copy()
            heights = sub_b["height_clean_final"].dropna()
            stairs = sub_b["stairs_clean"].dropna()
            afhs = sub_b["avg_floor_height_clean"].dropna()

            if "geometry" in sub_b.columns:
                sub_b_m = sub_b.to_crs(32636).copy()
                sub_b["area_b_geom"] = sub_b_m.geometry.area
            else:
                sub_b["area_b_geom"] = np.nan

            b_height_list.append(list(heights.values))
            b_stairs_list.append(list(stairs.values))
            b_afh_list.append(list(afhs.values))
            b_area_list.append(list(sub_b["area_b_geom"].values))
        else:
            b_height_list.append([])
            b_stairs_list.append([])
            b_afh_list.append([])
            b_area_list.append([])

        if len(a_idxs) > 0:
            sub_a = A.loc[a_idxs].copy()
            floor_min = sub_a["gkh_floor_count_min_clean"].dropna()
            floor_max = sub_a["gkh_floor_count_max_clean"].dropna()
            floor_mid = sub_a["gkh_floor_mid"].dropna()

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


def add_baseline_component_features(
    components_df: pd.DataFrame,
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Добавляет агрегированные признаки для модели."""
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

    def first_notna(series):
        s = series.dropna()
        if len(s) == 0:
            return np.nan
        return s.iloc[0]

    def mean_notna(series):
        s = series.dropna()
        if len(s) == 0:
            return np.nan
        return float(s.mean())

    for _, row in components_df.iterrows():
        a_idxs = row["a_idxs"]
        b_idxs = row["b_idxs"]

        if len(b_idxs) > 0:
            sub_b = B.loc[b_idxs]
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

        if len(a_idxs) > 0:
            sub_a = A.loc[a_idxs]
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


def clean_for_catboost(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Подготавливает данные для CatBoost."""
    df = df.copy()

    keep_cols = feature_cols + ["height_final"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    model_df = df[keep_cols].copy()

    numeric_cols = model_df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col == "height_final":
            model_df.loc[model_df[col] <= 0, col] = np.nan
            continue
        model_df.loc[model_df[col] < 0, col] = np.nan

    if "stairs_clean_component" in model_df.columns:
        model_df.loc[model_df["stairs_clean_component"] > 100, "stairs_clean_component"] = np.nan

    if "avg_floor_height_component" in model_df.columns:
        model_df.loc[
            (model_df["avg_floor_height_component"] < 1.5) |
            (model_df["avg_floor_height_component"] > 20),
            "avg_floor_height_component"
        ] = np.nan

    if "a_floor_mid_component" in model_df.columns:
        model_df.loc[
            (model_df["a_floor_mid_component"] < 1) |
            (model_df["a_floor_mid_component"] > 100),
            "a_floor_mid_component"
        ] = np.nan

    return model_df


def prepare_train_test_data(
    matched_buildings: pd.DataFrame,
    feature_cols: list,
    cat_features: list,
    test_size: float = 0.2
) -> tuple:
    """Подготавливает данные для обучения и предсказания."""
    model_df = clean_for_catboost(matched_buildings, feature_cols)

    train_df = model_df[model_df["height_final"].notna()].copy()
    pred_idx = matched_buildings.index[matched_buildings["needs_ml"]].tolist()
    pred_df = model_df.loc[model_df.index.intersection(pred_idx)].copy()

    X = train_df[feature_cols].copy()
    y = train_df["height_final"].copy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
    X_pred = pred_df[feature_cols].copy() if len(pred_df) > 0 else pd.DataFrame(columns=feature_cols)

    for col in cat_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            if len(X_pred) > 0:
                X_pred[col] = X_pred[col].astype(str)

    return X_train, X_val, y_train, y_val, X_pred, train_df.index, pred_df.index if len(pred_df) > 0 else pd.Index([])
