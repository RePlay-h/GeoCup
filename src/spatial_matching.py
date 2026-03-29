import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.utils import extract_house_number, norm_text


def build_candidate_pairs(a_gdf: gpd.GeoDataFrame, b_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    geoms_b = list(b_gdf.geometry.values)
    idxs_b = b_gdf.index.to_numpy()
    tree = STRtree(geoms_b)

    geometry_id_to_idx = {id(geom): idx for geom, idx in zip(geoms_b, idxs_b)}
    candidates: list[tuple[int, int]] = []

    for idx_a, geom_a in a_gdf.geometry.items():
        if geom_a is None or geom_a.is_empty:
            continue

        hits = tree.query(geom_a)
        for hit in hits:
            if isinstance(hit, (int, np.integer)):
                idx_b = idxs_b[int(hit)]
            else:  # compatibility with older shapely versions
                idx_b = geometry_id_to_idx.get(id(hit))
                if idx_b is None:
                    continue
            candidates.append((idx_a, idx_b))

    return pd.DataFrame(candidates, columns=["idx_a", "idx_b"]).drop_duplicates()


def compute_pair_metrics(
    a_metric: gpd.GeoDataFrame,
    b_metric: gpd.GeoDataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in candidates_df.iterrows():
        idx_a = row["idx_a"]
        idx_b = row["idx_b"]

        geom_a = a_metric.loc[idx_a, "geometry"]
        geom_b = b_metric.loc[idx_b, "geometry"]

        if geom_a is None or geom_b is None or geom_a.is_empty or geom_b.is_empty:
            continue

        intersection = geom_a.intersection(geom_b)
        inter_area = intersection.area

        if inter_area <= 0:
            continue

        area_a = a_metric.loc[idx_a, "area_a"]
        area_b = b_metric.loc[idx_b, "area_b"]
        union_area = area_a + area_b - inter_area

        rows.append(
            {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "area_a": area_a,
                "area_b": area_b,
                "inter_area": inter_area,
                "union_area": union_area,
                "iou": inter_area / union_area if union_area > 0 else np.nan,
                "overlap_a": inter_area / area_a if area_a > 0 else np.nan,
                "overlap_b": inter_area / area_b if area_b > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def filter_strong_pairs(
    pairs_df: pd.DataFrame,
    iou_thr: float = 0.3,
    overlap_thr: float = 0.8,
) -> pd.DataFrame:
    strong_pairs_df = pairs_df[
        (pairs_df["iou"] >= iou_thr)
        | (pairs_df["overlap_a"] >= overlap_thr)
        | (pairs_df["overlap_b"] >= overlap_thr)
    ].copy()

    strong_pairs_df["strong_by_iou"] = strong_pairs_df["iou"] >= iou_thr
    strong_pairs_df["strong_by_a"] = strong_pairs_df["overlap_a"] >= overlap_thr
    strong_pairs_df["strong_by_b"] = strong_pairs_df["overlap_b"] >= overlap_thr
    return strong_pairs_df


def build_match_components(
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
    strong_pairs_df: pd.DataFrame,
) -> tuple[nx.Graph, list[set[str]]]:
    graph = nx.Graph()

    for idx_a in a_gdf.index:
        graph.add_node(f"A_{idx_a}", source="A", raw_idx=idx_a)

    for idx_b in b_gdf.index:
        graph.add_node(f"B_{idx_b}", source="B", raw_idx=idx_b)

    for _, row in strong_pairs_df.iterrows():
        node_a = f"A_{row['idx_a']}"
        node_b = f"B_{row['idx_b']}"
        graph.add_edge(
            node_a,
            node_b,
            iou=row["iou"],
            overlap_a=row["overlap_a"],
            overlap_b=row["overlap_b"],
        )

    return graph, list(nx.connected_components(graph))


def get_match_type(n_a: int, n_b: int) -> str:
    if n_a > 0 and n_b == 0:
        return "A_only"
    if n_a == 0 and n_b > 0:
        return "B_only"
    if n_a == 1 and n_b == 1:
        return "1:1"
    if n_a == 1 and n_b > 1:
        return "1:N"
    if n_a > 1 and n_b == 1:
        return "N:1"
    if n_a > 1 and n_b > 1:
        return "N:N"
    return "unknown"


def build_components_df(components: list[set[str]]) -> pd.DataFrame:
    rows = []

    for component_id, component in enumerate(components):
        a_nodes = sorted(node for node in component if node.startswith("A_"))
        b_nodes = sorted(node for node in component if node.startswith("B_"))

        a_idxs = [int(node.split("_", 1)[1]) for node in a_nodes]
        b_idxs = [int(node.split("_", 1)[1]) for node in b_nodes]

        rows.append(
            {
                "component_id": component_id,
                "a_idxs": a_idxs,
                "b_idxs": b_idxs,
                "n_a": len(a_idxs),
                "n_b": len(b_idxs),
                "match_type": get_match_type(len(a_idxs), len(b_idxs)),
            }
        )

    return pd.DataFrame(rows)


def add_component_metrics(components_df: pd.DataFrame, strong_pairs_df: pd.DataFrame) -> pd.DataFrame:
    components_df = components_df.copy()

    max_iou_list = []
    mean_iou_list = []
    max_overlap_a_list = []
    max_overlap_b_list = []

    for _, component in components_df.iterrows():
        a_idxs = component["a_idxs"]
        b_idxs = component["b_idxs"]

        if not a_idxs or not b_idxs:
            max_iou_list.append(np.nan)
            mean_iou_list.append(np.nan)
            max_overlap_a_list.append(np.nan)
            max_overlap_b_list.append(np.nan)
            continue

        subset = strong_pairs_df[
            strong_pairs_df["idx_a"].isin(a_idxs)
            & strong_pairs_df["idx_b"].isin(b_idxs)
        ]

        if len(subset) == 0:
            max_iou_list.append(np.nan)
            mean_iou_list.append(np.nan)
            max_overlap_a_list.append(np.nan)
            max_overlap_b_list.append(np.nan)
            continue

        max_iou_list.append(subset["iou"].max())
        mean_iou_list.append(subset["iou"].mean())
        max_overlap_a_list.append(subset["overlap_a"].max())
        max_overlap_b_list.append(subset["overlap_b"].max())

    components_df["max_iou"] = max_iou_list
    components_df["mean_iou"] = mean_iou_list
    components_df["max_overlap_a"] = max_overlap_a_list
    components_df["max_overlap_b"] = max_overlap_b_list
    return components_df


def add_address_features(a_gdf: gpd.GeoDataFrame, b_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    a_gdf = a_gdf.copy()
    b_gdf = b_gdf.copy()

    a_address_col = "gkh_address" if "gkh_address" in a_gdf.columns else "address"
    if a_address_col in a_gdf.columns:
        a_gdf["address_a_norm"] = a_gdf[a_address_col].apply(norm_text)
        a_gdf["house_a"] = a_gdf[a_address_col].apply(extract_house_number)
    else:
        a_gdf["address_a_norm"] = ""
        a_gdf["house_a"] = ""

    if "name_street" in b_gdf.columns:
        b_gdf["street_b"] = b_gdf["name_street"].apply(norm_text)
    else:
        b_gdf["street_b"] = ""

    if "number" in b_gdf.columns:
        b_gdf["house_b"] = b_gdf["number"].apply(norm_text)
    else:
        b_gdf["house_b"] = ""

    return a_gdf, b_gdf


def validate_component_addresses_flexible(
    components_df: pd.DataFrame,
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    components_df = components_df.copy()

    strong_list = []
    weak_list = []
    conflict_list = []

    for _, component in components_df.iterrows():
        a_idxs = component["a_idxs"]
        b_idxs = component["b_idxs"]

        a_nums = set(a_gdf.loc[a_idxs, "house_a"].dropna()) if a_idxs else set()
        b_nums = set(b_gdf.loc[b_idxs, "house_b"].dropna()) if b_idxs else set()

        if len(a_nums) == 0 or len(b_nums) == 0:
            strong = False
            weak = False
            conflict = False
        else:
            intersection = a_nums.intersection(b_nums)
            weak = len(intersection) > 0
            strong = weak
            conflict = len(intersection) == 0

        strong_list.append(strong)
        weak_list.append(weak)
        conflict_list.append(conflict)

    components_df["address_match_strong"] = strong_list
    components_df["address_match_weak"] = weak_list
    components_df["address_conflict_flag"] = conflict_list
    return components_df


def add_component_review_flags(components_df: pd.DataFrame) -> pd.DataFrame:
    components_df = components_df.copy()
    components_df["needs_review"] = False

    components_df.loc[components_df["address_conflict_flag"], "needs_review"] = True

    weak_mask = components_df["max_iou"].notna() & (components_df["max_iou"] < 0.4)
    components_df.loc[weak_mask, "needs_review"] = True

    complex_mask = components_df["match_type"].isin(["1:N", "N:1", "N:N"])
    components_df.loc[complex_mask, "needs_review"] = True

    return components_df


def build_component_geometry(
    components_df: pd.DataFrame,
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    geometries = []

    for _, row in components_df.iterrows():
        a_idxs = row["a_idxs"]
        b_idxs = row["b_idxs"]

        geometry = None
        if b_idxs:
            geometry = unary_union(b_gdf.loc[b_idxs].geometry)
        elif a_idxs:
            geometry = unary_union(a_gdf.loc[a_idxs].geometry)

        geometries.append(geometry)

    output = components_df.copy()
    output["geometry"] = geometries
    return gpd.GeoDataFrame(output, geometry="geometry", crs=a_gdf.crs)


def run_spatial_matching(
    a_gdf: gpd.GeoDataFrame,
    b_gdf: gpd.GeoDataFrame,
    metric_crs: int | str = 32636,
    iou_thr: float = 0.3,
    overlap_thr: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    a_gdf, b_gdf = add_address_features(a_gdf, b_gdf)

    candidates_df = build_candidate_pairs(a_gdf, b_gdf)

    a_metric = a_gdf.to_crs(metric_crs).copy()
    b_metric = b_gdf.to_crs(metric_crs).copy()
    a_metric["area_a"] = a_metric.geometry.area
    b_metric["area_b"] = b_metric.geometry.area

    pairs_df = compute_pair_metrics(a_metric, b_metric, candidates_df)
    strong_pairs_df = filter_strong_pairs(pairs_df, iou_thr=iou_thr, overlap_thr=overlap_thr)

    _, components = build_match_components(a_gdf, b_gdf, strong_pairs_df)
    components_df = build_components_df(components)
    components_df = add_component_metrics(components_df, strong_pairs_df)
    components_df = validate_component_addresses_flexible(components_df, a_gdf, b_gdf)
    components_df = add_component_review_flags(components_df)

    components_gdf = build_component_geometry(components_df, a_gdf, b_gdf)
    return strong_pairs_df, components_df, components_gdf, a_gdf, b_gdf
