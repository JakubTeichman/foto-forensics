from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy() 

class Photo(db.Model):
    __tablename__ = 'photos'
    id = db.Column(db.Integer, primary_key=True)