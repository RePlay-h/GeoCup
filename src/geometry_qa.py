"""
Геометрический QA: проверка и исправление геометрии
"""
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import is_valid, is_valid_reason, make_valid
from shapely.geometry import GeometryCollection, Polygon, MultiPolygon
from shapely.geometry.polygon import orient

from config import CRS_METRIC


def polygon_has_wrong_orientation(geom) -> bool:
    """Soft QA-check: exterior должен быть CCW, holes — CW."""
    if geom is None or geom.is_empty:
        return False

    def check_polygon(poly: Polygon) -> bool:
        exterior_ok = poly.exterior.is_ccw
        holes_ok = all(not ring.is_ccw for ring in poly.interiors)
        return not (exterior_ok and holes_ok)

    if geom.geom_type == "Polygon":
        return check_polygon(geom)
    if geom.geom_type == "MultiPolygon":
        return any(check_polygon(p) for p in geom.geoms)
    return False


def normalize_orientation(geom):
    """Приводит полигоны к одной конвенции: exterior CCW, holes CW."""
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type == "Polygon":
        return orient(geom, sign=1.0)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([orient(p, sign=1.0) for p in geom.geoms])

    return geom


def geometry_diagnostics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Сводная диагностика геометрии по каждому объекту."""
    geom = gdf.geometry
    report = pd.DataFrame(index=gdf.index)

    report["is_missing"] = geom.isna()
    report["is_empty"] = geom.apply(lambda x: getattr(x, "is_empty", False))
    report["is_valid"] = geom.apply(lambda x: False if x is None else is_valid(x))
    report["valid_reason"] = geom.apply(lambda x: None if x is None else is_valid_reason(x))
    report["geom_type"] = geom.apply(lambda x: None if x is None else x.geom_type)

    # Площадь считаем только в метрической CRS
    gdf_m = gdf.to_crs(CRS_METRIC)
    report["area_m2"] = gdf_m.area
    report["zero_area"] = (~report["is_empty"]) & (report["area_m2"] <= 1e-9)

    report["wrong_ring_orientation"] = geom.apply(
        lambda x: False if x is None else polygon_has_wrong_orientation(x)
    )

    report["needs_attention"] = (
        report["is_missing"]
        | report["is_empty"]
        | (~report["is_valid"])
        | report["zero_area"]
    )

    return report


def extract_polygonal(geom):
    """После make_valid оставляем только полигональные части."""
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if len(polys) == 0:
            return None
        if len(polys) == 1:
            return polys[0]

        flat = []
        for g in polys:
            if g.geom_type == "Polygon":
                flat.append(g)
            else:
                flat.extend(list(g.geoms))
        return MultiPolygon(flat)

    return None


def repair_geometry(geom):
    """Точечно исправляет только невалидные геометрии."""
    if geom is None or geom.is_empty:
        return geom

    fixed = geom
    if not is_valid(geom):
        fixed = make_valid(geom, method="structure")

    fixed = extract_polygonal(fixed)
    fixed = normalize_orientation(fixed)
    return fixed


def repair_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Создаёт исправленную версию GeoDataFrame, не теряя оригинальную геометрию."""
    out = gdf.copy()
    out["geometry_original"] = out.geometry
    out["geometry"] = out.geometry.apply(repair_geometry)
    return out.set_geometry("geometry", crs=gdf.crs)


def coverage_checks(gdf: gpd.GeoDataFrame, gap_width: float = 0.0):
    """Проверка coverage-проблем: gaps и edge-matching между соседними полигонами."""
    gs = gdf.geometry
    is_cov_valid = gs.is_valid_coverage(gap_width=gap_width)
    invalid_edges = gs.invalid_coverage_edges(gap_width=gap_width)
    return is_cov_valid, invalid_edges


def run_topology_qa(gdf: gpd.GeoDataFrame, name: str, metric_crs: int = CRS_METRIC):
    """Печатает краткую QA-сводку до и после возможного ремонта геометрии."""
    print(f"\n=== {name}: BEFORE REPAIR ===")
    diag_before = geometry_diagnostics(gdf)
    print("rows:", len(gdf))
    print("invalid:", int((~diag_before["is_valid"]).sum()))
    print("empty:", int(diag_before["is_empty"].sum()))
    print("zero_area:", int(diag_before["zero_area"].sum()))
    print("wrong_orientation:", int(diag_before["wrong_ring_orientation"].sum()))
    print(diag_before["valid_reason"].value_counts(dropna=False).head(10))

    fixed = repair_gdf(gdf)

    print(f"\n=== {name}: AFTER REPAIR ===")
    diag_after = geometry_diagnostics(fixed)
    print("invalid:", int((~diag_after["is_valid"]).sum()))
    print("empty:", int(diag_after["is_empty"].sum()))
    print("zero_area:", int(diag_after["zero_area"].sum()))
    print("wrong_orientation:", int(diag_after["wrong_ring_orientation"].sum()))

    fixed_m = fixed.to_crs(metric_crs)
    try:
        cov_ok, bad_edges = coverage_checks(fixed_m, gap_width=0.2)
        bad_count = int((~bad_edges.is_empty).sum())
        print("coverage_valid:", cov_ok)
        print("coverage_problem_polygons:", bad_count)
    except Exception as e:
        print("coverage check skipped:", e)
        bad_edges = None

    return fixed, diag_before, diag_after, bad_edges
