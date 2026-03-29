import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

try:
    import esda
    import libpysal
except ImportError:  # pragma: no cover
    esda = None
    libpysal = None


PHYSICAL_BOUNDS = {
    "height": (2, 462),
    "stairs": (1, 87),
    "avg_floor_height": (2.5, 8),
    "gkh_floor_count_min": (1, 100),
    "gkh_floor_count_max": (1, 100),
    "area_sq_m": (16, 182000),
}

CITY_CAPS = {
    "spb": {
        "height": 200,
        "stairs": 40,
    }
}


def add_physical_flags(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        if col not in df.columns or col not in PHYSICAL_BOUNDS:
            continue

        lower, upper = PHYSICAL_BOUNDS[col]
        df[f"{col}_is_nan"] = df[col].isna()
        df[f"{col}_phys_bad"] = df[col].notna() & ((df[col] < lower) | (df[col] > upper))

    if {"gkh_floor_count_min", "gkh_floor_count_max"}.issubset(df.columns):
        df["gkh_min_gt_max"] = (
            df["gkh_floor_count_min"].notna()
            & df["gkh_floor_count_max"].notna()
            & (df["gkh_floor_count_min"] > df["gkh_floor_count_max"])
        )

    return df


def add_iqr_flag(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.DataFrame:
    df = df.copy()

    if col not in df.columns:
        return df

    series = df[col].dropna()
    if len(series) == 0:
        df[f"{col}_iqr_bad"] = False
        df[f"{col}_q1"] = np.nan
        df[f"{col}_q3"] = np.nan
        df[f"{col}_iqr"] = np.nan
        df[f"{col}_lower_fence"] = np.nan
        df[f"{col}_upper_fence"] = np.nan
        return df

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr

    df[f"{col}_q1"] = q1
    df[f"{col}_q3"] = q3
    df[f"{col}_iqr"] = iqr
    df[f"{col}_lower_fence"] = lower_fence
    df[f"{col}_upper_fence"] = upper_fence
    df[f"{col}_iqr_bad"] = df[col].notna() & ((df[col] < lower_fence) | (df[col] > upper_fence))
    return df


def prepare_height_for_lisa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["height_clean_for_lisa"] = df["height"]

    if "height_phys_bad" in df.columns:
        df.loc[df["height_phys_bad"], "height_clean_for_lisa"] = np.nan
    if "height_iqr_bad" in df.columns:
        df.loc[df["height_iqr_bad"], "height_clean_for_lisa"] = np.nan

    return df


def add_height_zscore_for_lisa(df: pd.DataFrame) -> pd.DataFrame:
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


def add_distance_band_weights(gdf: gpd.GeoDataFrame, threshold: float = 200, metric_crs: int | str = 32636):
    gdf = gdf.copy()

    if libpysal is None:
        warnings.warn("libpysal is not installed. LISA-related features will be skipped.")
        gdf["is_island"] = False
        return gdf, gdf.to_crs(metric_crs), None

    gdf_m = gdf.to_crs(metric_crs).copy()
    weights = libpysal.weights.DistanceBand.from_dataframe(
        gdf_m,
        threshold=threshold,
        binary=True,
        silence_warnings=True,
    )

    gdf["is_island"] = gdf.index.isin(weights.islands)
    return gdf, gdf_m, weights


def add_lisa_flags(
    gdf: gpd.GeoDataFrame,
    threshold: float = 200,
    p_threshold: float = 0.05,
    metric_crs: int | str = 32636,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf["lisa_I"] = np.nan
    gdf["lisa_p"] = np.nan
    gdf["lisa_quad"] = np.nan
    gdf["lisa_flag"] = False

    if esda is None or libpysal is None:
        return gdf

    mask = (~gdf["is_island"]) & (gdf["height_z"].notna())
    gdf_lisa = gdf.loc[mask].copy()

    if len(gdf_lisa) == 0:
        return gdf

    gdf_lisa_m = gdf_lisa.to_crs(metric_crs).copy()
    weights_lisa = libpysal.weights.DistanceBand.from_dataframe(
        gdf_lisa_m,
        threshold=threshold,
        binary=True,
        silence_warnings=True,
    )

    lisa = esda.Moran_Local(
        gdf_lisa["height_z"].values,
        weights_lisa,
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


def add_review_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["needs_review"] = df["height_phys_bad"] | df["height_iqr_bad"] | df["lisa_flag"]
    df["review_group"] = "clean"

    df.loc[df["height_iqr_bad"] & ~df["lisa_flag"], "review_group"] = "iqr_only"
    df.loc[~df["height_iqr_bad"] & df["lisa_flag"], "review_group"] = "lisa_only"
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "review_group"] = "iqr_and_lisa"
    df.loc[df["height_phys_bad"], "review_group"] = "physical_bad"
    return df


def add_height_clean_final(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["height_clean_final"] = df["height"]

    df.loc[df["height_phys_bad"], "height_clean_final"] = np.nan
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "height_clean_final"] = np.nan
    return df


def add_height_drop_reason(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["height_drop_reason"] = None

    df.loc[df["height_phys_bad"], "height_drop_reason"] = "physical_bounds"
    df.loc[df["height_iqr_bad"] & df["lisa_flag"], "height_drop_reason"] = "iqr_and_lisa"
    return df


def apply_domain_cap(df: pd.DataFrame, city: str = "spb") -> pd.DataFrame:
    df = df.copy()

    if city not in CITY_CAPS:
        return df

    height_cap = CITY_CAPS[city]["height"]

    if "height_clean_final" not in df.columns:
        df["height_clean_final"] = df["height"]

    if "height_drop_reason" not in df.columns:
        df["height_drop_reason"] = None

    df["height_domain_cap_bad"] = (
        df["height_clean_final"].notna()
        & (df["height_clean_final"] > height_cap)
    )
    df.loc[df["height_domain_cap_bad"], "height_clean_final"] = np.nan
    df.loc[df["height_domain_cap_bad"], "height_drop_reason"] = "domain_cap"
    return df


def add_clean_stairs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stairs_clean"] = df["stairs"]
    df["stairs_drop_reason"] = None

    if "stairs_phys_bad" in df.columns:
        df.loc[df["stairs_phys_bad"], "stairs_clean"] = np.nan
        df.loc[df["stairs_phys_bad"], "stairs_drop_reason"] = "physical_bounds"

    if "stairs_iqr_bad" in df.columns:
        df.loc[df["stairs_iqr_bad"], "stairs_clean"] = np.nan
        df.loc[df["stairs_iqr_bad"], "stairs_drop_reason"] = "iqr"

    return df


def add_clean_avg_floor_height(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_floor_height_clean"] = df["avg_floor_height"]
    df["avg_floor_height_drop_reason"] = None

    if "avg_floor_height_phys_bad" in df.columns:
        df.loc[df["avg_floor_height_phys_bad"], "avg_floor_height_clean"] = np.nan
        df.loc[df["avg_floor_height_phys_bad"], "avg_floor_height_drop_reason"] = "physical_bounds"

    if "avg_floor_height_iqr_bad" in df.columns:
        df.loc[df["avg_floor_height_iqr_bad"], "avg_floor_height_clean"] = np.nan
        df.loc[df["avg_floor_height_iqr_bad"], "avg_floor_height_drop_reason"] = "iqr"

    return df


def clean_gkh_floors(df: pd.DataFrame) -> pd.DataFrame:
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


def clean_area_sq_m(df: pd.DataFrame, small_area_threshold: float = 20) -> pd.DataFrame:
    df = df.copy()
    df["area_sq_m_clean"] = df["area_sq_m"]

    if "area_sq_m_phys_bad" in df.columns:
        df.loc[df["area_sq_m_phys_bad"], "area_sq_m_clean"] = np.nan

    df["area_sq_m_small_flag"] = df["area_sq_m_clean"].notna() & (df["area_sq_m_clean"] < small_area_threshold)
    return df


def height_cleaning_summary(df: pd.DataFrame) -> pd.Series:
    summary = {
        "rows_total": len(df),
        "height_notna_before": int(df["height"].notna().sum()),
        "height_phys_bad": int(df.get("height_phys_bad", pd.Series(dtype=bool)).sum()),
        "height_iqr_bad": int(df.get("height_iqr_bad", pd.Series(dtype=bool)).sum()),
        "is_island": int(df.get("is_island", pd.Series(dtype=bool)).sum()),
        "lisa_flag": int(df.get("lisa_flag", pd.Series(dtype=bool)).sum()),
        "height_notna_after": int(df["height_clean_final"].notna().sum()),
        "height_dropped_total": int(df["height"].notna().sum() - df["height_clean_final"].notna().sum()),
    }
    return pd.Series(summary)


def feature_cleaning_summary(a_df: pd.DataFrame, b_df: pd.DataFrame) -> pd.Series:
    summary = {
        "A_rows": len(a_df),
        "B_rows": len(b_df),
        "A_area_notna_before": int(a_df["area_sq_m"].notna().sum()) if "area_sq_m" in a_df.columns else 0,
        "A_area_notna_after": int(a_df["area_sq_m_clean"].notna().sum()) if "area_sq_m_clean" in a_df.columns else 0,
        "A_area_small_flag": int(a_df["area_sq_m_small_flag"].sum()) if "area_sq_m_small_flag" in a_df.columns else 0,
        "A_gkh_floor_mid_notna": int(a_df["gkh_floor_mid"].notna().sum()) if "gkh_floor_mid" in a_df.columns else 0,
        "B_height_notna_before": int(b_df["height"].notna().sum()) if "height" in b_df.columns else 0,
        "B_height_notna_after": int(b_df["height_clean_final"].notna().sum()) if "height_clean_final" in b_df.columns else 0,
        "B_stairs_notna_before": int(b_df["stairs"].notna().sum()) if "stairs" in b_df.columns else 0,
        "B_stairs_notna_after": int(b_df["stairs_clean"].notna().sum()) if "stairs_clean" in b_df.columns else 0,
        "B_avg_floor_height_notna_before": int(b_df["avg_floor_height"].notna().sum()) if "avg_floor_height" in b_df.columns else 0,
        "B_avg_floor_height_notna_after": int(b_df["avg_floor_height_clean"].notna().sum()) if "avg_floor_height_clean" in b_df.columns else 0,
    }
    return pd.Series(summary)


def clean_sources(
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
    metric_crs: int | str = 32636,
    city: str = "spb",
    lisa_distance_threshold_m: float = 200,
    lisa_p_threshold: float = 0.05,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    a_df = add_physical_flags(
        a_gdf,
        ["area_sq_m", "gkh_floor_count_min", "gkh_floor_count_max"],
    )
    b_df = add_physical_flags(
        b_gdf,
        ["height", "stairs", "avg_floor_height"],
    )

    for col in ["height", "stairs", "avg_floor_height"]:
        b_df = add_iqr_flag(b_df, col)

    b_df["height_needs_review"] = b_df["height_phys_bad"] | b_df["height_iqr_bad"]

    b_df = prepare_height_for_lisa(b_df)
    b_df = add_height_zscore_for_lisa(b_df)
    b_df, _, _ = add_distance_band_weights(
        b_df,
        threshold=lisa_distance_threshold_m,
        metric_crs=metric_crs,
    )
    b_df = add_lisa_flags(
        b_df,
        threshold=lisa_distance_threshold_m,
        p_threshold=lisa_p_threshold,
        metric_crs=metric_crs,
    )
    b_df = add_review_groups(b_df)
    b_df = add_height_clean_final(b_df)
    b_df = add_height_drop_reason(b_df)
    b_df = apply_domain_cap(b_df, city=city)
    b_df = add_clean_stairs(b_df)
    b_df = add_clean_avg_floor_height(b_df)

    a_df = clean_gkh_floors(a_df)
    a_df = clean_area_sq_m(a_df, small_area_threshold=20)

    return a_df, b_df
