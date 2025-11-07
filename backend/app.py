from flask import Flask, jsonify
from flask_cors import CORS
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os
import pymysql
from config import Config

# Importy blueprintów
from routes.analyze_routes import analyze_bp
from routes.compare_routes import compare_bp
from routes.steganography_routes import steganography_bp
from routes.other_routes import others_bp
from routes.noiseprint_routes import noiseprint_bp
from routes.add_reference import add_reference_bp

# ---- 🔧 Inicjalizacja ----
db = SQLAlchemy()
mail = Mail()

def create_app():
    # 🌐 Inicjalizacja aplikacji
    app = Flask(__name__)
    CORS(app)

    # 📁 Wczytanie zmiennych środowiskowych
    load_dotenv()

    # 💾 Konfiguracja bazy danych
    DB_USER = os.getenv("DB_USER", "forensics_user")
    DB_PASS = os.getenv("DB_PASS", "forensics_pass")
    DB_HOST = os.getenv("DB_HOST", "mysql")
    DB_NAME = os.getenv("DB_NAME", "forensics")

    DATABASE_URI = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URI
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 📬 Konfiguracja maila
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER')
    app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')
    app.config['MAIL_DEFAULT_SENDER'] = ('Foto Forensics', 'fotoforensics3@gmail.com')

    # 🔗 Inicjalizacja rozszerzeń
    db.init_app(app)
    mail.init_app(app)

    # 🧱 Tworzenie tabel w bazie, jeśli nie istnieją
    with app.app_context():
        from models.model import Image 
        db.create_all()

    # 🧪 Endpoint testowy do sprawdzenia połączenia z bazą
    @app.route("/test-db")
    def test_db():
        try:
            conn = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASS,
                database=DB_NAME,
                cursorclass=pymysql.cursors.DictCursor
            )
            with conn.cursor() as cursor:
                cursor.execute("SELECT DATABASE() AS db_name;")
                result = cursor.fetchone()
            conn.close()
            return jsonify({"status": "success", "connected_to": result["db_name"]})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    # 📦 Rejestracja blueprintów
    app.register_blueprint(add_reference_bp)
    app.register_blueprint(analyze_bp, url_prefix="/analyze")
    app.register_blueprint(steganography_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(others_bp)
    app.register_blueprint(noiseprint_bp, url_prefix="/noiseprint")

    return app


# 🚀 Uruchomienie aplikacji
if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
