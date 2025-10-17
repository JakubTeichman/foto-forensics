import mysql.connector
import io
import numpy as np
from typing import List, Dict

def get_connection(cfg: Dict):
    return mysql.connector.connect(**cfg)

# Config example: {"host":"mysql","user":"root","password":"root","database":"forensics"}

def insert_device(conn, name: str, model: str, description: str) -> int:
    cur = conn.cursor()
    cur.execute("INSERT INTO devices (name, model, description) VALUES (%s,%s,%s)", (name, model, description))
    conn.commit()
    device_id = cur.lastrowid
    cur.close()
    return device_id

def insert_device_images(conn, device_id: int, urls: List[str]):
    cur = conn.cursor()
    for url in urls:
        cur.execute("INSERT INTO device_images (device_id, image_url) VALUES (%s,%s)", (device_id, url))
    conn.commit()
    cur.close()

def insert_fingerprint_url(conn, device_id: int, fp_url: str):
    cur = conn.cursor()
    cur.execute("INSERT INTO fingerprints (device_id, fingerprint_url) VALUES (%s,%s)", (device_id, fp_url))
    conn.commit()
    cur.close()

def get_devices_with_fingerprints(conn) -> List[Dict]:
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT d.id as device_id, d.name, d.model, f.fingerprint_url
        FROM devices d
        JOIN fingerprints f ON f.device_id = d.id
    """)
    rows = cur.fetchall()
    cur.close()
    return rows

def get_device_images(conn, device_id: int):
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT image_url FROM device_images WHERE device_id=%s", (device_id,))
    rows = cur.fetchall()
    cur.close()
    return [r['image_url'] for r in rows]
