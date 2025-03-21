from flask import Blueprint, jsonify
from app.models import Photo

main = Blueprint('main', __name__)

@main.route('/')
def home():
    return jsonify({"message": "Foto-Forensic API działa!"})

