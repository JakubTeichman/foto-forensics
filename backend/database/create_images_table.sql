CREATE TABLE images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  device_brand TEXT,
  device_model TEXT,
  device_serial TEXT,
  image_url TEXT,
  role TEXT DEFAULT 'other'
);