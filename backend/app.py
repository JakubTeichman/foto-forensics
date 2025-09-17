import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from prnu_utils import ensure_folders, clear_folder
from prnu_utils.comparison import compare_prnu_paths, compare_prnu_with_urls, ncc
from prnu_utils.prnu_extraction import extract_prnu_from_path
from prnu_utils.io_helpers import download_to_tempfile, save_npy_to_tempfile
from prnu_utils.db import get_connection, insert_device, insert_device_images, insert_fingerprint_url, get_devices_with_fingerprints, get_device_images
import numpy as np
import tempfile
import requests
import shutil
from pathlib import Path

# Config DB from env
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "mysql"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASS", "root"),
    "database": os.environ.get("DB_NAME", "forensics"),
}

UPLOAD_BASE = os.environ.get("UPLOAD_BASE", "uploads")
EVIDENCE_DIR = os.path.join(UPLOAD_BASE, "evidence")
REFERENCE_DIR = os.path.join(UPLOAD_BASE, "reference")
FINGERPRINT_TMP = os.path.join(UPLOAD_BASE, "fingerprints_tmp")

ensure_folders(UPLOAD_BASE)
Path(FINGERPRINT_TMP).mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# -----------------------------
# CASE 1 — ad-hoc compare (1..n references) — no DB writes
# -----------------------------
@app.route("/api/compare", methods=["POST"])
def api_compare():
    """
    FormData:
    - evidence: file (mandatory)
    - references: file[] OR reference_urls: json array of URLs (one of them required)
    - denoise (optional) - not used in prototype
    """
    # prepare folders
    clear_folder(EVIDENCE_DIR)
    clear_folder(REFERENCE_DIR)

    if "evidence" not in request.files:
        return jsonify({"error": "Missing evidence file"}), 400
    evidence = request.files["evidence"]
    ev_filename = secure_filename(evidence.filename)
    ev_path = os.path.join(EVIDENCE_DIR, ev_filename)
    evidence.save(ev_path)

    # accept either uploaded reference files or list of reference_urls in form
    reference_paths = []
    if "references" in request.files:
        refs = request.files.getlist("references")
        for f in refs:
            fn = secure_filename(f.filename)
            p = os.path.join(REFERENCE_DIR, fn)
            f.save(p)
            reference_paths.append(p)
    elif request.form.get("reference_urls"):
        # JSON array as string or comma-separated
        import json
        urls = json.loads(request.form.get("reference_urls"))
        # download to temp files
        for url in urls:
            p = download_to_tempfile(url)
            reference_paths.append(p)
    else:
        return jsonify({"error": "No references provided (files or reference_urls)"}), 400

    # do comparison
    result = compare_prnu_paths(ev_path, reference_paths)

    # cleanup downloaded refs if any (if they were URLs)
    # we always clear reference dir at start, but remove temp files downloaded outside reference dir
    # attempt to delete any tmp files sitting in system tmp
    for p in reference_paths:
        try:
            if str(Path(p)).startswith(tempfile.gettempdir()):
                Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    return jsonify(result)

# -----------------------------
# CASE 2 — admin adds device (uploads or gives URLs) -> compute avg fingerprint and store
# DB stores: device record, device_images (urls) and fingerprint_url (link to npy in cloud) OR local path for now
# -----------------------------
@app.route("/api/add_device", methods=["POST"])
def api_add_device():
    """
    FormData:
    - name (device name) [required]
    - model (optional)
    - description (optional)
    - references (file[] optional) OR reference_urls (json list of URLs)
    - fingerprint_upload_url (optional) - if you already have .npy stored in cloud, provide link
    Behaviour:
    - If references given -> compute avg fingerprint -> save npy temporarily -> (optionally upload to cloud)
    - Insert device row and either insert device_images (urls) and fingerprint_url (url or local tempfile path)
    """
    name = request.form.get("name")
    if not name:
        return jsonify({"error":"Missing name"}), 400
    model = request.form.get("model", "")
    description = request.form.get("description", "")

    # connect DB
    conn = get_connection(DB_CONFIG)
    device_id = insert_device(conn, name, model, description)

    # collect image URLs to store in device_images
    image_urls = []

    # 1) if uploaded files, save them locally and (optionally) upload to cloud manually later
    if "references" in request.files:
        refs = request.files.getlist("references")
        # save files to reference dir and (optionally) upload to cloud manually
        saved_paths = []
        for f in refs:
            fn = secure_filename(f.filename)
            p = os.path.join(REFERENCE_DIR, fn)
            f.save(p)
            saved_paths.append(p)
            image_urls.append(p)  # note: this is local path; admin should upload to cloud and replace with URL
        # compute fingerprint from saved_paths
        fps = [extract_prnu_from_path(p) for p in saved_paths]
        avg_fp = np.mean(fps, axis=0)
        # save avg fingerprint to temporary npy (you can upload this file to cloud manually)
        tmp_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        np.save(tmp_fp.name, avg_fp)
        tmp_fp_path = tmp_fp.name
        # for now, store path in fingerprint_url column (recommended: upload to cloud and store real URL)
        insert_device_images(conn, device_id, image_urls)
        insert_fingerprint_url(conn, device_id, tmp_fp_path)
        conn.close()
        return jsonify({"status":"ok","device_id": device_id, "fingerprint_stored": tmp_fp_path})

    # 2) if reference URLs provided
    if request.form.get("reference_urls"):
        import json
        urls = json.loads(request.form.get("reference_urls"))
        # download images, compute avg fingerprint
        tmp_paths = []
        fps = []
        try:
            for url in urls:
                path = download_to_tempfile(url)
                tmp_paths.append(path)
                fps.append(extract_prnu_from_path(path))
            avg_fp = np.mean(fps, axis=0)
            tmp_fp = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
            np.save(tmp_fp.name, avg_fp)
            # store urls into device_images (so we have source URLs)
            insert_device_images(conn, device_id, urls)
            insert_fingerprint_url(conn, device_id, tmp_fp.name)
            return jsonify({"status":"ok","device_id": device_id, "fingerprint_stored": tmp_fp.name})
        finally:
            for p in tmp_paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass

    # 3) if already have fingerprint url (.npy)
    if request.form.get("fingerprint_upload_url"):
        fp_url = request.form.get("fingerprint_upload_url")
        insert_device_images(conn, device_id, [])  # no images in this case
        insert_fingerprint_url(conn, device_id, fp_url)
        conn.close()
        return jsonify({"status":"ok","device_id": device_id, "fingerprint_url": fp_url})

    conn.close()
    return jsonify({"error":"No references or fingerprint provided"}), 400

# -----------------------------
# CASE 3 — search DB with evidence image
# -----------------------------
@app.route("/api/search_device", methods=["POST"])
def api_search_device():
    """
    FormData:
    - evidence (file) required
    - topk (optional)
    Behaviour:
    - compute PRNU from evidence
    - load all fingerprints from DB (fingerprint_url may be local path or HTTP URL)
    - compare using ncc and return topk sorted
    """
    if "evidence" not in request.files:
        return jsonify({"error":"Missing evidence"}), 400

    topk = int(request.form.get("topk", 5))
    evidence = request.files["evidence"]
    ev_path = os.path.join(EVIDENCE_DIR, secure_filename(evidence.filename))
    evidence.save(ev_path)
    ev_fp = extract_prnu_from_path(ev_path)

    conn = get_connection(DB_CONFIG)
    rows = get_devices_with_fingerprints(conn)
    conn.close()

    results = []
    for r in rows:
        fp_url = r.get("fingerprint_url")
        # if fp_url is local file path -> load numpy
        try:
            if str(fp_url).startswith("http://") or str(fp_url).startswith("https://"):
                resp = requests.get(fp_url, timeout=20)
                resp.raise_for_status()
                import io
                fp = np.load(io.BytesIO(resp.content), allow_pickle=False)
            else:
                # local path
                fp = np.load(fp_url)
        except Exception as e:
            # skip if cannot load fingerprint
            continue

        score = ncc(ev_fp, fp)
        results.append({
            "device_id": r.get("device_id"),
            "name": r.get("name"),
            "model": r.get("model"),
            "fingerprint_url": fp_url,
            "score": float(score)
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:topk]
    return jsonify({"top": results})
