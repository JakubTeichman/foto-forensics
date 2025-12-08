import shutil
from pathlib import Path

def clear_folder(p: str):
    path = Path(p)
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def ensure_folders(base_upload: str = "uploads"):
    Path(base_upload).mkdir(exist_ok=True)
    Path(base_upload + "/evidence").mkdir(exist_ok=True)
    Path(base_upload + "/reference").mkdir(exist_ok=True)
