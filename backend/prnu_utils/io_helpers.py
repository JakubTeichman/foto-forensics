import requests
from pathlib import Path
import tempfile
import shutil
from typing import Tuple

def download_to_tempfile(url: str) -> str:
    """Pobiera URL i zapisuje do pliku tymczasowego; zwraca ścieżkę."""
    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(url).suffix or ".jpg")
    with open(tmp.name, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return tmp.name

def save_npy_to_tempfile(npy_bytes: bytes) -> str:
    """Zapisuje dane npy (bytes) do pliku tymczasowego i zwraca ścieżkę."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
    with open(tmp.name, "wb") as f:
        f.write(npy_bytes)
    return tmp.name
