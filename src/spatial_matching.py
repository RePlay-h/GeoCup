"""
Пространственное сопоставление зданий из разных источников
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import re
from shapely.strtree import STRtree
from shapely.ops import unary_union

from config import SPATIAL_MATCHING_IOU_THR, SPATIAL_MATCHING_OVERLAP_THR, CRS_METRIC


def build_candidate_pairs(A: gpd.GeoDataFrame, B: gpd.GeoDataFrame) -> pd.DataFrame:
    """Быстро ищем пары A-B, которые потенциально пересекаются."""
    geoms_b = B.geometry.values
    idxs_b = B.index.to_numpy()

    tree = STRtree(geoms_b)

    candidates = []

    for idx_a, geom_a in A.geometry.items():
        if geom_a is None or geom_a.is_empty:
            continue

        hit_pos = tree.query(geom_a)

        for pos in hit_pos:
            idx_b = idxs_b[pos]
            candidates.append((idx_a, idx_b))

    candidates_df = pd.DataFrame(candidates, columns=["idx_a", "idx_b"])
    return candidates_df


def compute_pair_metrics(
    A_m: gpd.GeoDataFrame,
    B_m: gpd.GeoDataFrame,
    candidates_df: pd.DataFrame
) -> pd.DataFrame:
    """Считает IoU и overlap для каждой пары."""
    rows = []

    for _, row in candidates_df.iterrows():
        idx_a = row["idx_a"]
        idx_b = row["idx_b"]

        geom_a = A_m.loc[idx_a, "geometry"]
        geom_b = B_m.loc[idx_b, "geometry"]

        if geom_a is None or geom_b is None:
            continue
        if geom_a.is_empty or geom_b.is_empty:
            continue

        inter = geom_a.intersection(geom_b)
        inter_area = inter.area

        if inter_area <= 0:
            continue

        area_a = A_m.loc[idx_a, "area"]
        area_b = B_m.loc[idx_b, "area"]

        union_area = area_a + area_b - inter_area

        iou = inter_area / union_area if union_area > 0 else np.nan
        overlap_a = inter_area / area_a if area_a > 0 else np.nan
        overlap_b = inter_area / area_b if area_b > 0 else np.nan

        rows.append({
            "idx_a": idx_a,
            "idx_b": idx_b,
            "area_a": area_a,
            "area_b": area_b,
            "inter_area": inter_area,
            "union_area": union_area,
            "iou": iou,
            "overlap_a": overlap_a,
            "overlap_b": overlap_b,
        })

    pairs_df = pd.DataFrame(rows)
    return pairs_df


def filter_strong_pairs(
    pairs_df: pd.DataFrame,
    iou_thr: float = SPATIAL_MATCHING_IOU_THR,
    overlap_thr: float = SPATIAL_MATCHING_OVERLAP_THR
) -> pd.DataFrame:
    """Фильтрует сильные пары по IoU или overlap."""
    strong_pairs_df = pairs_df[
        (pairs_df["iou"] >= iou_thr) |
        (pairs_df["overlap_a"] >= overlap_thr) |
        (pairs_df["overlap_b"] >= overlap_thr)
    ].copy()

    strong_pairs_df["strong_by_iou"] = strong_pairs_df["iou"] >= iou_thr
    strong_pairs_df["strong_by_a"] = strong_pairs_df["overlap_a"] >= overlap_thr
    strong_pairs_df["strong_by_b"] = strong_pairs_df["overlap_b"] >= overlap_thr

    return strong_pairs_df


def build_match_components(
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame,
    strong_pairs_df: pd.DataFrame
) -> tuple[nx.Graph, list]:
    """Строит граф связности и выделяет компоненты."""
    G = nx.Graph()

    for idx_a in A.index:
        G.add_node(f"A_{idx_a}", source="A", raw_idx=idx_a)

    for idx_b in B.index:
        G.add_node(f"B_{idx_b}", source="B", raw_idx=idx_b)

    for _, row in strong_pairs_df.iterrows():
        node_a = f"A_{row['idx_a']}"
        node_b = f"B_{row['idx_b']}"

        G.add_edge(
            node_a,
            node_b,
            iou=row["iou"],
            overlap_a=row["overlap_a"],
            overlap_b=row["overlap_b"]
        )

    components = list(nx.connected_components(G))
    return G, components


def get_match_type(n_a: int, n_b: int) -> str:
    """Определяет тип сопоставления."""
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


def build_components_df(components: list) -> pd.DataFrame:
    """Строит DataFrame из компонент графа."""
    rows = []

    for comp_id, comp in enumerate(components):
        a_nodes = sorted([x for x in comp if x.startswith("A_")])
        b_nodes = sorted([x for x in comp if x.startswith("B_")])

        a_idxs = [int(x.split("_", 1)[1]) for x in a_nodes]
        b_idxs = [int(x.split("_", 1)[1]) for x in b_nodes]

        n_a = len(a_idxs)
        n_b = len(b_idxs)

        rows.append({
            "component_id": comp_id,
            "a_idxs": a_idxs,
            "b_idxs": b_idxs,
            "n_a": n_a,
            "n_b": n_b,
            "match_type": get_match_type(n_a, n_b),
        })

    components_df = pd.DataFrame(rows)
    return components_df


def add_component_metrics(
    components_df: pd.DataFrame,
    strong_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """Добавляет метрики qualit я к компонентам."""
    components_df = components_df.copy()

    max_iou_list = []
    mean_iou_list = []
    max_overlap_a_list = []
    max_overlap_b_list = []

    for _, comp_row in components_df.iterrows():
        a_idxs = comp_row["a_idxs"]
        b_idxs = comp_row["b_idxs"]

        if len(a_idxs) == 0 or len(b_idxs) == 0:
            max_iou_list.append(None)
            mean_iou_list.append(None)
            max_overlap_a_list.append(None)
            max_overlap_b_list.append(None)
            continue

        sub = strong_pairs_df[
            strong_pairs_df["idx_a"].isin(a_idxs) &
            strong_pairs_df["idx_b"].isin(b_idxs)
        ]

        if len(sub) == 0:
            max_iou_list.append(None)
            mean_iou_list.append(None)
            max_overlap_a_list.append(None)
            max_overlap_b_list.append(None)
            continue

        max_iou_list.append(sub["iou"].max())
        mean_iou_list.append(sub["iou"].mean())
        max_overlap_a_list.append(sub["overlap_a"].max())
        max_overlap_b_list.append(sub["overlap_b"].max())

    components_df["max_iou"] = max_iou_list
    components_df["mean_iou"] = mean_iou_list
    components_df["max_overlap_a"] = max_overlap_a_list
    components_df["max_overlap_b"] = max_overlap_b_list

    return components_df


def norm_text(x: str) -> str:
    """Нормализует текст для сравнения адресов."""
    if pd.isna(x):
        return ""
    x = str(x).lower().strip()
    x = x.replace("ё", "е")
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def extract_house_number(address: str) -> str:
    """Извлекает номер дома из адреса."""
    address = norm_text(address)
    m = re.search(r"\b\d+[а-яa-z]?\b", address)
    return m.group(0) if m else ""


def validate_component_addresses_flexible(
    components_df: pd.DataFrame,
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Проверяет согласованность адресов в компонентах."""
    components_df = components_df.copy()

    strong_list = []
    weak_list = []
    conflict_list = []

    for _, comp in components_df.iterrows():
        a_idxs = comp["a_idxs"]
        b_idxs = comp["b_idxs"]

        a_nums = set(A.loc[a_idxs, "house_a"].dropna()) if len(a_idxs) else set()
        b_nums = set(B.loc[b_idxs, "house_b"].dropna()) if len(b_idxs) else set()

        if len(a_nums) == 0 or len(b_nums) == 0:
            strong = False
            weak = False
            conflict = False
        else:
            inter = a_nums.intersection(b_nums)
            weak = len(inter) > 0
            strong = weak
            conflict = len(inter) == 0

        strong_list.append(strong)
        weak_list.append(weak)
        conflict_list.append(conflict)

    components_df["address_match_strong"] = strong_list
    components_df["address_match_weak"] = weak_list
    components_df["address_conflict_flag"] = conflict_list

    return components_df


def add_component_review_flags(
    components_df: pd.DataFrame,
    conf_review_thr: float = 0.4
) -> pd.DataFrame:
    """Добавляет флаги на объекты, требующие проверки."""
    components_df = components_df.copy()

    components_df["needs_review"] = False

    components_df.loc[components_df["address_conflict_flag"], "needs_review"] = True

    weak_mask = (
        components_df["max_iou"].notna() &
        (components_df["max_iou"] < conf_review_thr)
    )
    components_df.loc[weak_mask, "needs_review"] = True

    complex_mask = components_df["match_type"].isin(["1:N", "N:1", "N:N"])
    components_df.loc[complex_mask, "needs_review"] = True

    return components_df


def build_component_geometry(
    components_df: pd.DataFrame,
    A: gpd.GeoDataFrame,
    B: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Строит геометрию для каждой компоненты."""
    geometries = []

    for _, row in components_df.iterrows():
        a_idxs = row["a_idxs"]
        b_idxs = row["b_idxs"]

        geom = None

        if len(b_idxs) > 0:
            geom = unary_union(B.loc[b_idxs].geometry)
        elif len(a_idxs) > 0:
            geom = unary_union(A.loc[a_idxs].geometry)

        geometries.append(geom)

    components_df = components_df.copy()
    components_df["geometry"] = geometries

    return components_df
