from flask import Blueprint, jsonify

main = Blueprint('main', __name__)

@main.route('/')
def home():
    from app.models import Photo
    return jsonify({"message": "Foto-Forensic API działa!"})

