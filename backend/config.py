import os

class Config:
    DB_USER = os.getenv("DB_USER", "forensics_user")
    DB_PASS = os.getenv("DB_PASS", "forensics_pass")
    DB_HOST = os.getenv("DB_HOST", "mysql")
    DB_NAME = os.getenv("DB_NAME", "forensics")

    SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
