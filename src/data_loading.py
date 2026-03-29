"""
Загрузка и инициализация данных
"""
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry.base import BaseGeometry

from config import CRS_GEOGRAPHIC, PATH_A, PATH_B


def ensure_geometry_objects(series: pd.Series) -> pd.Series:
    """Преобразует колонку геометрии в объекты Shapely только если это действительно нужно."""
    first_valid = next((x for x in series if x is not None), None)

    if first_valid is None:
        return series
    if isinstance(first_valid, BaseGeometry):
        return series
    if isinstance(first_valid, str):
        return series.apply(wkt.loads)

    raise TypeError(f"Unsupported geometry type: {type(first_valid)}")


def drop_source_duplicates(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """удалить дубликаты из источника"""
    df = df.copy()
    before = len(df)

    # 1. Убираем полные дубликаты строк
    df = df.drop_duplicates()

    # 2. Если есть id и geometry/wkt — убираем дубли объектов
    # с одинаковым идентификатором и одинаковой геометрией
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


def load_a(path: str = PATH_A) -> gpd.GeoDataFrame:
    """Чтение источника A: геометрия ожидается в колонке `geometry`."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = drop_source_duplicates(df, "A")

    df["geometry"] = ensure_geometry_objects(df["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_GEOGRAPHIC)


def load_b(path: str = PATH_B) -> gpd.GeoDataFrame:
    """Чтение источника B: геометрия может лежать в колонке `wkt`."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = drop_source_duplicates(df, "B")

    if "geometry" not in df.columns:
        df["geometry"] = df["wkt"]

    df["geometry"] = ensure_geometry_objects(df["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs=CRS_GEOGRAPHIC)


def load_both() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Загружает оба источника и возвращает кортеж (gdf_a, gdf_b)."""
    gdf_a = load_a()
    gdf_b = load_b()
    
    print("A:", gdf_a.shape)
    print("B:", gdf_b.shape)
    
    return gdf_a, gdf_b
