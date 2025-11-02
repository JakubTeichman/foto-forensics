from flask import Flask
from flask_cors import CORS
import os
from routes.analyze_routes import analyze_bp
from routes.compare_routes import compare_bp
from routes.steganography_routes import steganography_bp
from routes.other_routes import others_bp
from extensions import mail

def create_app():
    app = Flask(__name__)
    CORS(app)

     # Konfiguracja maila
    app.config['MAIL_SERVER'] = 'smtp.gmail.com'
    app.config['MAIL_PORT'] = 587
    app.config['MAIL_USE_TLS'] = True
    app.config['MAIL_USERNAME'] = os.getenv('EMAIL_USER') 
    app.config['MAIL_PASSWORD'] = os.getenv('EMAIL_PASSWORD')         
    app.config['MAIL_DEFAULT_SENDER'] = ('Foto Forensics', 'fotoforensics3@gmail.com')

    mail.init_app(app)

    # Rejestrujemy główny blueprint z endpointami API
    app.register_blueprint(analyze_bp, url_prefix="/analyze")
    app.register_blueprint(steganography_bp)   
    app.register_blueprint(compare_bp)
    app.register_blueprint(others_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
