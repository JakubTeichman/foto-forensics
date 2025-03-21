from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from .config import Config
from .routes import main

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    
    app.config.from_object(Config)
    app.register_blueprint(main)
    db.init_app(app)

    return app
