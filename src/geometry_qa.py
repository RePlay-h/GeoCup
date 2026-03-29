import geopandas as gpd
import pandas as pd
from shapely import is_valid, is_valid_reason, make_valid
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient


def polygon_has_wrong_orientation(geom) -> bool:
    """Soft QA: exterior ring must be CCW, holes must be CW."""
    if geom is None or geom.is_empty:
        return False

    def check_polygon(poly: Polygon) -> bool:
        exterior_ok = poly.exterior.is_ccw
        holes_ok = all(not ring.is_ccw for ring in poly.interiors)
        return not (exterior_ok and holes_ok)

    if geom.geom_type == "Polygon":
        return check_polygon(geom)
    if geom.geom_type == "MultiPolygon":
        return any(check_polygon(poly) for poly in geom.geoms)
    return False


def normalize_orientation(geom):
    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == "Polygon":
        return orient(geom, sign=1.0)
    if geom.geom_type == "MultiPolygon":
        return MultiPolygon([orient(poly, sign=1.0) for poly in geom.geoms])
    return geom


def geometry_diagnostics(gdf: gpd.GeoDataFrame, metric_crs: int | str) -> pd.DataFrame:
    geom = gdf.geometry
    report = pd.DataFrame(index=gdf.index)

    report["is_missing"] = geom.isna()
    report["is_empty"] = geom.apply(lambda value: getattr(value, "is_empty", False))
    report["is_valid"] = geom.apply(lambda value: False if value is None else is_valid(value))
    report["valid_reason"] = geom.apply(lambda value: None if value is None else is_valid_reason(value))
    report["geom_type"] = geom.apply(lambda value: None if value is None else value.geom_type)

    gdf_m = gdf.to_crs(metric_crs)
    report["area_m2"] = gdf_m.area
    report["zero_area"] = (~report["is_empty"]) & (report["area_m2"] <= 1e-9)
    report["wrong_ring_orientation"] = geom.apply(
        lambda value: False if value is None else polygon_has_wrong_orientation(value)
    )

    report["needs_attention"] = (
        report["is_missing"]
        | report["is_empty"]
        | (~report["is_valid"])
        | report["zero_area"]
    )
    return report


def extract_polygonal(geom):
    if geom is None or geom.is_empty:
        return geom

    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom

    if geom.geom_type == "GeometryCollection":
        polygons = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polygons:
            return None
        if len(polygons) == 1:
            return polygons[0]

        flat = []
        for geometry in polygons:
            if geometry.geom_type == "Polygon":
                flat.append(geometry)
            else:
                flat.extend(list(geometry.geoms))
        return MultiPolygon(flat)

    return None


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return geom

    fixed = geom
    if not is_valid(geom):
        fixed = make_valid(geom)

    fixed = extract_polygonal(fixed)
    fixed = normalize_orientation(fixed)
    return fixed


def repair_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["geometry_original"] = out.geometry
    out["geometry"] = out.geometry.apply(repair_geometry)
    return out.set_geometry("geometry", crs=gdf.crs)


def coverage_checks(gdf: gpd.GeoDataFrame, gap_width: float = 0.0):
    gs = gdf.geometry
    if not hasattr(gs, "is_valid_coverage") or not hasattr(gs, "invalid_coverage_edges"):
        return None, None
    return gs.is_valid_coverage(gap_width=gap_width), gs.invalid_coverage_edges(gap_width=gap_width)


def run_topology_qa(
    gdf: gpd.GeoDataFrame,
    name: str,
    metric_crs: int | str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, object]:
    print(f"\n=== {name}: BEFORE REPAIR ===")
    diag_before = geometry_diagnostics(gdf, metric_crs=metric_crs)
    print("rows:", len(gdf))
    print("invalid:", int((~diag_before["is_valid"]).sum()))
    print("empty:", int(diag_before["is_empty"].sum()))
    print("zero_area:", int(diag_before["zero_area"].sum()))
    print("wrong_orientation:", int(diag_before["wrong_ring_orientation"].sum()))

    fixed = repair_gdf(gdf)

    print(f"\n=== {name}: AFTER REPAIR ===")
    diag_after = geometry_diagnostics(fixed, metric_crs=metric_crs)
    print("invalid:", int((~diag_after["is_valid"]).sum()))
    print("empty:", int(diag_after["is_empty"].sum()))
    print("zero_area:", int(diag_after["zero_area"].sum()))
    print("wrong_orientation:", int(diag_after["wrong_ring_orientation"].sum()))

    bad_edges = None
    try:
        fixed_m = fixed.to_crs(metric_crs)
        cov_ok, bad_edges = coverage_checks(fixed_m, gap_width=0.2)
        if cov_ok is not None:
            bad_count = int((~bad_edges.is_empty).sum()) if bad_edges is not None else 0
            print("coverage_valid:", cov_ok)
            print("coverage_problem_polygons:", bad_count)
    except Exception as exc:
        print("coverage check skipped:", exc)

    return fixed, diag_before, diag_after, bad_edges
