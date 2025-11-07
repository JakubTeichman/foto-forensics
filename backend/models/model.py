from app import db

class Image(db.Model):
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    device_brand = db.Column(db.String(100))
    device_model = db.Column(db.String(100))
    device_serial = db.Column(db.String(100))
    image_url = db.Column(db.String(255))
    role = db.Column(db.String(50), default='other')
