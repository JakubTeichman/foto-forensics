from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    SQLALCHEMY_DATABASE_URI = os.getenv('FLASK_DB_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
