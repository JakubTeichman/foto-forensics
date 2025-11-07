# add_reference.py
from flask import Blueprint, request, jsonify, current_app
from io import BytesIO
import numpy as np
import cv2
from PIL import Image
import base64
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# jeśli w projekcie jest już SQLAlchemy z app, można użyć tej konfiguracji zamiast tworzyć nową
Base = declarative_base()

add_reference_bp = Blueprint('add_reference', __name__)

# --------------------------
# Model SQLAlchemy
# --------------------------
class DeviceReference(Base):
    __tablename__ = 'device_references'
    id = sa.Column(sa.Integer, primary_key=True, autoincrement=True)
    manufacturer = sa.Column(sa.String(100), nullable=False)
    model = sa.Column(sa.String(100), nullable=False)
    noiseprint = sa.Column(sa.LargeBinary, nullable=False)  # skompresowane .npz
    num_images = sa.Column(sa.Integer, nullable=False)
    created_at = sa.Column(sa.TIMESTAMP, server_default=sa.text('CURRENT_TIMESTAMP'))

# --------------------------
# Helpers: serializacja
# --------------------------
def serialize_noiseprint(np_array: np.ndarray) -> bytes:
    """
    Zapisuje numpy array do skompresowanego .npz w pamięci i zwraca bytes.
    """
    mem = BytesIO()
    # np.savez_compressed zachowuje format, po wczytaniu użyj np.load(BytesIO)
    np.savez_compressed(mem, noise=np_array)
    mem.seek(0)
    return mem.read()

def deserialize_noiseprint(blob: bytes) -> np.ndarray:
    mem = BytesIO(blob)
    loaded = np.load(mem)
    return loaded['noise']

# --------------------------
# Wczytywanie konfiguracji bazy (przykład)
# --------------------------
def get_db_session():
    """
    Tworzy sesję SQLAlchemy na podstawie konfiguracji app.config['DATABASE_URI']
    Jeśli w Twojej aplikacji już jest session, zmodyfikuj żeby użyć jej.
    """
    database_uri = current_app.config.get('DATABASE_URI')  # np. 'mysql+pymysql://user:pass@host/db'
    engine = sa.create_engine(database_uri, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session(), engine

# --------------------------
# Funkcja pomocnicza: konwersja obrazu -> tensor/np
# (użyj swojej image_to_tensor/generate_noiseprint jeśli chcesz)
# --------------------------
def image_to_gray_array(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

# Zamiast tego możesz importować i używać generate_noiseprint z noiseprint_bp
from noiseprint.model import FullConvNet as NoiseprintModel
import torch

# load model once (jeżeli już ładujesz w innym miejscu, lepiej współdzielić)
# Ścieżka do wag - dopasuj jeśli już masz model ładowany globalnie
MODEL_PATH = os.getenv('NOISEPRINT_MODEL_PATH', '/app/noiseprint/weights/model_noiseprint.pth')

# Uwaga: jeśli model ładuje się gdzie indziej w aplikacji, usuń to i importuj istniejący obiekt
noise_model = NoiseprintModel()
noise_model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
noise_model.eval()

def pil_to_tensor(pil_img: Image.Image):
    img = pil_img.convert('L')
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor

def generate_noiseprint_from_pil(pil_img: Image.Image):
    t = pil_to_tensor(pil_img)
    with torch.no_grad():
        out = noise_model(t).squeeze().cpu().numpy()
    return out

# --------------------------
# Endpoint: dodaj referencję (min 3 obrazy)
# --------------------------
@add_reference_bp.route('/add_reference', methods=['POST'])
def add_reference():
    # pola: manufacturer, model, files: images[]
    manufacturer = request.form.get('manufacturer')
    model_name = request.form.get('model')

    if not manufacturer or not model_name:
        return jsonify({'error': 'manufacturer and model are required'}), 400

    files = request.files.getlist('images')
    if not files or len(files) < 3:
        return jsonify({'error': 'At least 3 images are required as reference'}), 400

    # generate noiseprints for each image
    noiseprints = []
    for f in files:
        try:
            pil = Image.open(f.stream)
            np_noise = generate_noiseprint_from_pil(pil)
            noiseprints.append(np_noise)
        except Exception as e:
            return jsonify({'error': f'Failed to process one image: {str(e)}'}), 400

    # upewnij się, że wszystkie noiseprinty maja taki sam kształt; jeśli nie, możemy resize
    shapes = [n.shape for n in noiseprints]
    if len(set(shapes)) != 1:
        # prosty sposób: resize każdy do kształtu pierwszego
        target_shape = shapes[0]
        resized = []
        for n in noiseprints:
            resized.append(cv2.resize(n, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR))
        noiseprints = resized

    # average
    mean_np = np.mean(noiseprints, axis=0).astype(np.float32)

    # serialize
    blob = serialize_noiseprint(mean_np)

    # save to DB
    session, engine = get_db_session()
    try:
        # upewnij się, że tabela istnieje (pierwsze uruchomienie)
        Base.metadata.create_all(engine)

        ref = DeviceReference(
            manufacturer=manufacturer,
            model=model_name,
            noiseprint=blob,
            num_images=len(files)
        )
        session.add(ref)
        session.commit()
        return jsonify({'message': 'Reference added', 'id': ref.id}), 201
    except Exception as e:
        session.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()

# --------------------------
# Endpoint: porównaj pojedyncze zdjęcie z referencjami w DB
# zwraca top N wyników (domyślnie 5)
# --------------------------
@add_reference_bp.route('/compare_with_db', methods=['POST'])
def compare_with_db():
    top_n = int(request.form.get('top_n', 5))
    evidence_file = request.files.get('evidence')
    if evidence_file is None:
        return jsonify({'error': 'evidence file required'}), 400

    # generate evidence noiseprint
    try:
        pil_e = Image.open(evidence_file.stream)
        evidence_np = generate_noiseprint_from_pil(pil_e)
    except Exception as e:
        return jsonify({'error': f'Failed to process evidence: {str(e)}'}), 400

    session, engine = get_db_session()
    try:
        rows = session.query(DeviceReference).all()
        results = []
        for row in rows:
            ref_np = deserialize_noiseprint(row.noiseprint)
            # jeśli rozmiary różne -> resize ref do evidence
            if ref_np.shape != evidence_np.shape:
                ref_resized = cv2.resize(ref_np, (evidence_np.shape[1], evidence_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            else:
                ref_resized = ref_np

            # szybka miara podobieństwa: pearson corr (flatten)
            a = evidence_np.flatten()
            b = ref_resized.flatten()
            # zabezpieczenie przed stałą tablicą
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                corr = 0.0
            else:
                corr = float(np.corrcoef(a, b)[0,1])

            results.append({
                'id': row.id,
                'manufacturer': row.manufacturer,
                'model': row.model,
                'num_images': row.num_images,
                'correlation': corr
            })

        # posortuj malejąco po korelacji i zwróć top_n
        results_sorted = sorted(results, key=lambda x: x['correlation'], reverse=True)[:top_n]
        return jsonify({'results': results_sorted})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        session.close()
