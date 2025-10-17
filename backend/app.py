from flask import Flask
from flask_cors import CORS
from routes.analyze_routes import analyze_bp
from routes.compare_routes import compare_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Rejestrujemy główny blueprint z endpointami API
    app.register_blueprint(analyze_bp, url_prefix="/analyze")
    app.register_blueprint(compare_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
