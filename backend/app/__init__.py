from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from .config import Config
from .routes import main
from .demo import demo  # importuj blueprint

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    app.config.from_object(Config)
    app.register_blueprint(main)
    app.register_blueprint(demo)  # rejestruj blueprint
    db.init_app(app)

    return app
