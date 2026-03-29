import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def geometry_to_wkt(value) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return value.wkt if hasattr(value, "wkt") else str(value)


def save_csv(df: pd.DataFrame, path: Path | str, index: bool = False) -> None:
    out = df.copy()
    if "geometry" in out.columns:
        out["geometry"] = out["geometry"].apply(geometry_to_wkt)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=index)


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower().strip().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_house_number(address) -> str:
    address = norm_text(address)
    match = re.search(r"\b\d+[а-яa-z]?\b", address)
    return match.group(0) if match else ""


def safe_mean(values: Iterable[float]) -> float:
    cleaned = [x for x in values if pd.notna(x)]
    if not cleaned:
        return np.nan
    return float(np.mean(cleaned))


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [
        (value, weight)
        for value, weight in zip(values, weights)
        if pd.notna(value) and pd.notna(weight) and weight > 0
    ]
    if not pairs:
        return np.nan
    vals = np.array([value for value, _ in pairs], dtype=float)
    wts = np.array([weight for _, weight in pairs], dtype=float)
    return float(np.average(vals, weights=wts))


def first_notna(series: pd.Series):
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return series.iloc[0]


def mean_notna(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) == 0:
        return np.nan
    return float(series.mean())
