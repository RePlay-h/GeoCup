from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry.base import BaseGeometry


def ensure_geometry_objects(series: pd.Series) -> pd.Series:
    """Converts WKT strings to shapely objects only when conversion is needed."""
    first_valid = next((value for value in series if value is not None and not pd.isna(value)), None)

    if first_valid is None:
        return series
    if isinstance(first_valid, BaseGeometry):
        return series
    if isinstance(first_valid, str):
        return series.apply(lambda value: wkt.loads(value) if pd.notna(value) else None)

    raise TypeError(f"Unsupported geometry type: {type(first_valid)}")


def drop_source_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    df = df.drop_duplicates()

    if "geometry" in df.columns:
        geom_col = "geometry"
    elif "wkt" in df.columns:
        geom_col = "wkt"
    else:
        geom_col = None

    if "id" in df.columns and geom_col is not None:
        df = df.drop_duplicates(subset=["id", geom_col])

    after = len(df)
    print(f"{name}: removed {before - after} duplicates")
    return df


def load_a(path: str | Path, crs_geographic: str = "EPSG:4326") -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = drop_source_duplicates(df, "A")

    if "geometry" not in df.columns:
        raise KeyError("Source A must contain `geometry` column.")

    df["geometry"] = ensure_geometry_objects(df["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs_geographic)


def load_b(path: str | Path, crs_geographic: str = "EPSG:4326") -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = drop_source_duplicates(df, "B")

    if "geometry" not in df.columns:
        if "wkt" not in df.columns:
            raise KeyError("Source B must contain `geometry` or `wkt` column.")
        df["geometry"] = df["wkt"]

    df["geometry"] = ensure_geometry_objects(df["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs=crs_geographic)


def load_sources(
    path_a: str | Path,
    path_b: str | Path,
    crs_geographic: str = "EPSG:4326",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return load_a(path_a, crs_geographic=crs_geographic), load_b(path_b, crs_geographic=crs_geographic)
