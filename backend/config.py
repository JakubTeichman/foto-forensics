import os

class Config:
    DATABASE_URI = os.getenv("DATABASE_URI", "mysql+pymysql://forensics_user:forensics_pass@mysql/forensics")
    NOISEPRINT_MODEL_PATH = os.getenv("NOISEPRINT_MODEL_PATH", "/app/noiseprint/weights/model_noiseprint.pth")
