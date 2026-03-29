"""
Очистка признаков в источниках A и B
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import libpysal
import esda

from config import (
    PHYSICAL_BOUNDS,
    IQR_K,
    LISA_P_THRESHOLD,
    LISA_THRESHOLD_M,
    CITY_CAPS,
    CRS_METRIC,
)


def add_physical_flags(df, cols):
    """Ставит флаги физически невозможных значений, но не удаляет строки."""
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        lo, hi = PHYSICAL_BOUNDS[col]
        df[f"{col}_is_nan"] = df[col].isna()
        df[f"{col}_phys_bad"] = (
            df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        )

    if {"gkh_floor_count_min", "gkh_floor_count_max"}.issubset(df.columns):
        df["gkh_min_gt_max"] = (
            df["gkh_floor_count_min"].notna()
            & df["gkh_floor_count_max"].notna()
            & (df["gkh_floor_count_min"] > df["gkh_floor_count_max"])
        )

    return df


def add_iqr_flag(df, col, k=IQR_K):
    """Ставит флаг статистического выброса по правилу Tukey fence."""
    df = df.copy()

    if col not in df.columns:
        return df

    s = df[col].dropna()
    if len(s) == 0:
        df[f"{col}_iqr_bad"] = False
        df[f"{col}_q1"] = np.nan
        df[f"{col}_q3"] = np.nan
        df[f"{col}_iqr"] = np.nan
        df[f"{col}_lower_fence"] = np.nan
        df[f"{col}_upper_fence"] = np.nan
        return df

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    df[f"{col}_q1"] = q1
    df[f"{col}_q3"] = q3
    df[f"{col}_iqr"] = iqr
    df[f"{col}_lower_fence"] = lower_fence
    df[f"{col}_upper_fence"] = upper_fence
    df[f"{col}_iqr_bad"] = (
        df[col].notna() & ((df[col] < lower_fence) | (df[col] > upper_fence))
    )

    return df


def prepare_height_for_lisa(df):
    """Создаёт рабочую версию высоты, очищенную от грубых выбросов перед LISA."""
    df = df.copy()
    df["height_clean_for_lisa"] = df["height"]

    if "height_phys_bad" in df.columns:
        df.loc[df["height_phys_bad"], "height_clean_for_lisa"] = np.nan
    if "height_iqr_bad" in df.columns:
        df.loc[df["height_iqr_bad"], "height_clean_for_lisa"] = np.nan

    return df


def add_height_zscore_for_lisa(df):
    """Стандартизирует высоту после предварительной очистки."""
    df = df.copy()
    clean = df["height_clean_for_lisa"].dropna()

    mean_h = clean.mean()
    std_h = clean.std()

    df["height_mean_for_lisa"] = mean_h
    df["height_std_for_lisa"] = std_h

    if pd.isna(std_h) or std_h == 0:
        df["height_z"] = np.nan
    else:
        df["height_z"] = (df["height"] - mean_h) / std_h

    return df


def add_distance_band_weights(gdf, threshold=LISA_THRESHOLD_M):
    """Строит матрицу соседства по радиусу в метрах."""
    gdf = gdf.copy()
    gdf_m = gdf.to_crs(CRS_METRIC).copy()
    W = libpysal.weights.DistanceBand.from_dataframe(
        gdf_m,
        threshold=threshold,
        binary=True,
        silence_warnings=True,
    )

    gdf["is_island"] = gdf.index.isin(W.islands)
    return gdf, gdf_m, W


def add_lisa_flags(gdf, threshold=LISA_THRESHOLD_M, p_threshold=LISA_P_THRESHOLD):
    """Считает локальный Moran's I и флагирует значимые HL-объекты."""
    gdf = gdf.copy()

    gdf["lisa_I"] = np.nan
    gdf["lisa_p"] = np.nan
    gdf["lisa_quad"] = np.nan
    gdf["lisa_flag"] = False

    mask = (~gdf["is_island"]) & (gdf["height_z"].notna())
    gdf_lisa = gdf.loc[mask].copy()

    if len(gdf_lisa) == 0:
        return gdf

    gdf_lisa_m = gdf_lisa.to_crs(CRS_METRIC).copy()
    W_lisa = libpysal.weights.DistanceBand.from_dataframe(
        gdf_lisa_m,
        threshold=threshold,
        binary=True,
        silence_warnings=True,
    )

    lisa = esda.Moran_Local(
        gdf_lisa["height_z"].values,
        W_lisa,
        permutations=999,
    )

    gdf.loc[gdf_lisa.index, "lisa_I"] = lisa.Is
    gdf.loc[gdf_lisa.index, "lisa_p"] = lisa.p_sim
    gdf.loc[gdf_lisa.index, "lisa_quad"] = lisa.q

    gdf.loc[gdf_lisa.index, "lisa_flag"] = (
        (gdf.loc[gdf_lisa.index, "lisa_p"] < p_threshold)
        & (gdf.loc[gdf_lisa.index, "lisa_quad"] == 4)
    )

    return gdf


def add_review_groups(df):
    """Сводит все флаги в читаемые группы для анализа и карт."""
    df = df.copy()
    df["needs_review"] = df["height_phys_bad"] | df["height_iqr_bad"] | df["lisa_flag"]
    df["review_group"] = "clean"

    df.loc[df["height_iqr_bad"] & ~df["lisa_flag"], "review_group"] = "iqr_only"
    df.loc[~df["height_iqr_bad"] & df["lisa_flag"], "review_group"] = "lisa_only"
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "review_group"] = "iqr_and_lisa"
    df.loc[df["height_phys_bad"], "review_group"] = "physical_bad"

    return df


def add_height_clean_final(df):
    """Формирует финальную очищенную высоту без удаления самих объектов."""
    df = df.copy()
    df["height_clean_final"] = df["height"]

    df.loc[df["height_phys_bad"], "height_clean_final"] = np.nan
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "height_clean_final"] = np.nan
    df.loc[df["height_clean_final"] > 200, "height_clean_final"] = np.nan

    return df


def add_height_drop_reason(df):
    """Добавляет текстовую причину зануления значения height."""
    df = df.copy()
    df["height_drop_reason"] = None

    df.loc[df["height_phys_bad"], "height_drop_reason"] = "physical_bounds"
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "height_drop_reason"] = "iqr_and_lisa"
    df.loc[df["height"] > 200, "height_drop_reason"] = "city_cap"

    return df


def add_clean_stairs(df):
    """Очищает stairs по физическим границам и IQR-флагам."""
    df = df.copy()
    df["stairs_clean"] = df["stairs"]
    df["stairs_drop_reason"] = None

    df.loc[df["stairs_phys_bad"], "stairs_clean"] = np.nan
    df.loc[df["stairs_phys_bad"], "stairs_drop_reason"] = "physical_bounds"

    df.loc[df["stairs_iqr_bad"], "stairs_clean"] = np.nan
    df.loc[df["stairs_iqr_bad"], "stairs_drop_reason"] = "iqr"

    return df


def add_clean_avg_floor_height(df):
    """Очищает avg_floor_height по физическим границам и IQR-флагам."""
    df = df.copy()
    df["avg_floor_height_clean"] = df["avg_floor_height"]
    df["avg_floor_height_drop_reason"] = None

    df.loc[df["avg_floor_height_phys_bad"], "avg_floor_height_clean"] = np.nan
    df.loc[df["avg_floor_height_phys_bad"], "avg_floor_height_drop_reason"] = "physical_bounds"

    df.loc[df["avg_floor_height_iqr_bad"], "avg_floor_height_clean"] = np.nan
    df.loc[df["avg_floor_height_iqr_bad"], "avg_floor_height_drop_reason"] = "iqr"

    return df


def clean_gkh_floors(df):
    """Очищает min/max этажности и строит midpoint-признак."""
    df = df.copy()

    df["gkh_floor_count_min_clean"] = df["gkh_floor_count_min"]
    df["gkh_floor_count_max_clean"] = df["gkh_floor_count_max"]

    if "gkh_floor_count_min_phys_bad" in df.columns:
        df.loc[df["gkh_floor_count_min_phys_bad"], "gkh_floor_count_min_clean"] = np.nan
    if "gkh_floor_count_max_phys_bad" in df.columns:
        df.loc[df["gkh_floor_count_max_phys_bad"], "gkh_floor_count_max_clean"] = np.nan

    mask_swap = (
        df["gkh_floor_count_min_clean"].notna()
        & df["gkh_floor_count_max_clean"].notna()
        & (df["gkh_floor_count_min_clean"] > df["gkh_floor_count_max_clean"])
    )

    tmp = df.loc[mask_swap, "gkh_floor_count_min_clean"].copy()
    df.loc[mask_swap, "gkh_floor_count_min_clean"] = df.loc[mask_swap, "gkh_floor_count_max_clean"]
    df.loc[mask_swap, "gkh_floor_count_max_clean"] = tmp

    df["gkh_floor_mid"] = np.nan
    mask_mid = df["gkh_floor_count_min_clean"].notna() & df["gkh_floor_count_max_clean"].notna()
    df.loc[mask_mid, "gkh_floor_mid"] = (
        df.loc[mask_mid, "gkh_floor_count_min_clean"] + df.loc[mask_mid, "gkh_floor_count_max_clean"]
    ) / 2

    return df


def clean_area_sq_m(df, small_area_threshold=20):
    """Очищает площадь и ставит флаг для очень маленьких объектов."""
    df = df.copy()
    df["area_sq_m_clean"] = df["area_sq_m"]

    if "area_sq_m_phys_bad" in df.columns:
        df.loc[df["area_sq_m_phys_bad"], "area_sq_m_clean"] = np.nan

    df["area_sq_m_small_flag"] = (
        df["area_sq_m_clean"].notna() & (df["area_sq_m_clean"] < small_area_threshold)
    )

    return df


def apply_domain_cap(df, city="spb"):
    """Применяет городской cap к высоте."""
    df = df.copy()

    height_cap = CITY_CAPS[city]["height"]

    if "height_clean_final" not in df.columns:
        df["height_clean_final"] = df["height"]

    if "height_drop_reason" not in df.columns:
        df["height_drop_reason"] = None

    df["height_domain_cap_bad"] = (
        df["height_clean_final"].notna() &
        (df["height_clean_final"] > height_cap)
    )

    df.loc[df["height_domain_cap_bad"], "height_clean_final"] = np.nan
    df.loc[df["height_domain_cap_bad"], "height_drop_reason"] = "domain_cap"

    return df
