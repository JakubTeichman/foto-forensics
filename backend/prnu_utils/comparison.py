# comparison.py
"""
Porównania PRNU:
 - ncc (globalne)
 - blockwise_ncc (lokalne)
 - multiscale_ncc (przez kilka rozdzielczości)
 - compare_prnu_paths: obsługa ważonego uśredniania referencji
 - compare_prnu_with_urls: pobiera pliki tymczasowo (wymaga io_helpers.download_to_tempfile)
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np

from .prnu_extraction import extract_prnu_from_path, extract_prnu_from_bytes
from .io_helpers import download_to_tempfile 

def _safe_zero_mean(a: np.ndarray) -> np.ndarray:
    a = a - a.mean()
    return a


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cross-correlation (skalar) dla dwóch map tej samej wielkości.
    Jeśli rozmiary się różnią — dokonujemy centralnego cropu do wspólnego najmniejszego rozmiaru.
    """
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    a = _safe_zero_mean(a)
    b = _safe_zero_mean(b)
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-12
    return float(((a * b).sum()) / denom)


def blockwise_ncc(a: np.ndarray, b: np.ndarray, block_size: int = 64, min_std: float = 1e-6) -> float:
    """
    Dzieli obrazy na bloki i liczy NCC per blok, następnie średnia.
    Pomaga wykryć lokalne dopasowania i ograniczyć wpływ przycięć/kompresji.
    """
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    scores = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block_a = a[i:i + block_size, j:j + block_size]
            block_b = b[i:i + block_size, j:j + block_size]
            if block_a.size == 0 or block_b.size == 0:
                continue
            if block_a.std() < min_std or block_b.std() < min_std:
                continue
            scores.append(ncc(block_a, block_b))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def multiscale_ncc(a: np.ndarray, b: np.ndarray, scales: Optional[List[float]] = None) -> float:
    """
    Liczy NCC po kilku skalach (resize) i zwraca ważoną średnią.
    scales: lista współczynników skalowania (np. [1.0, 0.5, 0.25])
    """
    import cv2 

    if scales is None:
        scales = [1.0, 0.5, 0.25]
    scores = []
    weights = []
    for s in scales:
        if s == 1.0:
            aa = a
            bb = b
        else:
            h = max(1, int(a.shape[0] * s))
            w = max(1, int(a.shape[1] * s))
            aa = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            bb = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        score = ncc(aa, bb)
        scores.append(score)
        weights.append(s)
    weights = np.array(weights, dtype=np.float32)
    if weights.sum() == 0:
        return float(np.mean(scores))
    return float(np.dot(scores, weights) / weights.sum())

def _image_quality_weight(img_prnu: np.ndarray) -> float:
    """
    Prosta heurystyka jakości obrazu do wagi przy uśrednianiu referencji.
    Tutaj wykorzystujemy odchylenie standardowe mapy (im większe, tym więcej informacji).
    Możemy to rozszerzyć o ostrość, SNR, rozdzielczość itp.
    """
    s = float(np.nanstd(img_prnu))
    return max(1e-3, s)


def _weighted_average(refs: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """
    Ważone uśrednianie referencji. refs: lista map PRNU o (najczęściej) tych samych rozmiarach.
    Jeżeli rozmiary różne — przycinamy do min(h),min(w).
    """
    if not refs:
        raise ValueError("refs must not be empty")
    min_h = min(r.shape[0] for r in refs)
    min_w = min(r.shape[1] for r in refs)
    trimmed = [r[:min_h, :min_w] for r in refs]
    arr = np.stack(trimmed, axis=0)
    if weights is None:
        weights = np.array([1.0] * len(refs), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape[0] != arr.shape[0]:
            raise ValueError("weights length mismatch")
    weights = weights / (weights.sum() + 1e-12)
    avg = np.tensordot(weights, arr, axes=(0, 0))
    avg = avg - avg.mean()
    std = avg.std()
    if std < 1e-8:
        std = 1.0
    avg = avg / std
    return avg.astype(np.float32)


def compare_prnu_paths(evidence_path: str,
                       reference_paths: List[str],
                       *,
                       use_weighted_average: bool = True,
                       block_size: int = 64,
                       multiscale: bool = True) -> dict:
    """
    Porównanie: evidence_path vs list of local reference files.
    Zwraca słownik z wynikami: ncc_global, ncc_blockwise, ncc_multiscale (opcjonalnie), refs_count
    """
    ev = extract_prnu_from_path(evidence_path)
    refs = [extract_prnu_from_path(p) for p in reference_paths]
    if use_weighted_average:
        weights = [_image_quality_weight(r) for r in refs]
        avg_ref = _weighted_average(refs, weights=weights)
    else:
        avg_ref = _weighted_average(refs, weights=None)

    min_h = min(ev.shape[0], avg_ref.shape[0])
    min_w = min(ev.shape[1], avg_ref.shape[1])
    ev_c = ev[:min_h, :min_w]
    ref_c = avg_ref[:min_h, :min_w]

    res = {}
    res["ncc_global"] = ncc(ev_c, ref_c)
    res["ncc_blockwise"] = blockwise_ncc(ev_c, ref_c, block_size=block_size)
    if multiscale:
        res["ncc_multiscale"] = multiscale_ncc(ev_c, ref_c)
    res["refs_count"] = len(reference_paths)
    return res


def compare_prnu_with_urls(evidence_path: str,
                           reference_urls: List[str],
                           *,
                           use_weighted_average: bool = True,
                           block_size: int = 64,
                           multiscale: bool = True) -> dict:
    """
    Pobiera pliki referencyjne tymczasowo i deleguje do compare_prnu_paths.
    Wymaga: io_helpers.download_to_tempfile(url) -> zwraca ścieżkę lokalną.
    """
    tmp_paths = []
    try:
        for url in reference_urls:
            p = download_to_tempfile(url)
            tmp_paths.append(p)
        return compare_prnu_paths(evidence_path, tmp_paths,
                                  use_weighted_average=use_weighted_average,
                                  block_size=block_size,
                                  multiscale=multiscale)
    finally:
        for p in tmp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
