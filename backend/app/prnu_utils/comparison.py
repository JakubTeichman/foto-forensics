import numpy as np
from typing import List
from .prnu_extraction import extract_prnu_from_path, extract_prnu_from_bytes
from .io_helpers import download_to_tempfile
from pathlib import Path

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation scalar (same-size central crop if needed)."""
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a*a).sum()) * np.sqrt((b*b).sum())) + 1e-12
    return float(((a*b).sum()) / denom)

def compare_prnu_paths(evidence_path: str, reference_paths: List[str]) -> dict:
    """Case 1: evidence file path + list of local reference file paths (1..n)."""
    ev = extract_prnu_from_path(evidence_path)
    refs = [extract_prnu_from_path(p) for p in reference_paths]
    avg_ref = np.mean(refs, axis=0)
    score = ncc(ev, avg_ref)
    return {"ncc": score, "refs_count": len(reference_paths)}

def compare_prnu_with_urls(evidence_path: str, reference_urls: List[str]) -> dict:
    """Case 1 alternative: references provided as URLs (downloaded on the fly)."""
    # download refs to temp
    tmp_paths = []
    try:
        for url in reference_urls:
            p = download_to_tempfile(url)
            tmp_paths.append(p)
        res = compare_prnu_paths(evidence_path, tmp_paths)
    finally:
        for p in tmp_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
    return res
