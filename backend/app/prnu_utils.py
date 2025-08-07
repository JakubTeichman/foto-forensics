import os
import numpy as np
from skimage import io, color, img_as_float
from skimage.restoration import denoise_wavelet
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter

def extract_noise(image):
    """
    Prosta ekstrakcja szumu: denoise (wavelet) i różnica.
    Zwraca znormalizowany szum.
    """
    # jeśli kolorowe, weź luminancję
    if image.ndim == 3:
        image_gray = color.rgb2gray(image)
    else:
        image_gray = image

    image_gray = img_as_float(image_gray)
    denoised = denoise_wavelet(image_gray, multichannel=False, convert2ycbcr=False, rescale_sigma=True)
    noise = image_gray - denoised
    # Znormalizuj (zero mean, unit variance)
    noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-12)
    return noise

def build_fingerprint(noise_list):
    """
    Uśrednia listę szumów w fingerprint.
    """
    stacked = np.stack(noise_list, axis=0)
    fingerprint = np.mean(stacked, axis=0)
    # normalizacja
    fingerprint = (fingerprint - np.mean(fingerprint)) / (np.std(fingerprint) + 1e-12)
    return fingerprint

def peak_correlation_energy(query_noise, fingerprint):
    """
    Upraszczona wersja PCE: szukamy korelacji i liczymy stosunek szczytu do średniej energii reszty.
    """
    corr = correlate2d(query_noise, fingerprint, mode='same')
    peak = np.max(np.abs(corr))
    # energia tła: średnia z corr bez okolicy szczytu
    h, w = corr.shape
    cy, cx = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
    # maska wycinająca okolicę 11x11 wokół szczytu
    radius = 5
    mask = np.ones_like(corr, dtype=bool)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    mask[y0:y1, x0:x1] = False
    background = corr[mask]
    energy_bg = np.mean(background**2) + 1e-12
    pce_value = (peak**2) / energy_bg
    return pce_value, corr

def process_device_folder(path, reference_filenames):
    noises = []
    for fname in reference_filenames:
        img = io.imread(os.path.join(path, fname))
        noise = extract_noise(img)
        noises.append(noise)
    fingerprint = build_fingerprint(noises)
    return fingerprint

def classify_query(query_path, fingerprints):
    img = io.imread(query_path)
    query_noise = extract_noise(img)
    results = {}
    for device_name, fp in fingerprints.items():
        pce_value, _ = peak_correlation_energy(query_noise, fp)
        results[device_name] = pce_value
    # wybierz największe
    best = max(results, key=results.get)
    return best, results

def main():
    # konfiguracja: katalogi i które pliki są referencyjne
    devices = {
        "device1": {
            "path": "dataset/device1",
            "reference": ["ref1.jpg", "ref2.jpg", "ref3.jpg"]  # podstawowe zdjęcia do fingerprintu
        },
        "device2": {
            "path": "dataset/device2",
            "reference": ["ref1.jpg", "ref2.jpg", "ref3.jpg"]
        }
    }

    # budowa fingerprintów
    fingerprints = {}
    for name, info in devices.items():
        fp = process_device_folder(info["path"], info["reference"])
        fingerprints[name] = fp
        print(f"[INFO] Zbudowano fingerprint dla {name}")

    # klasyfikacja przykładowych query
    queries = [
        ("dataset/device1/query1.jpg", "device1"),
        ("dataset/device2/query1.jpg", "device2"),
        # możesz dodać więcej
    ]

    for qpath, true_label in queries:
        predicted, scores = classify_query(qpath, fingerprints)
        print(f"\nZapytanie: {qpath}")
        print(f"Prawdziwe źródło: {true_label}, Predykcja: {predicted}")
        for d, val in scores.items():
            print(f"  {d}: PCE={val:.2f}")

if __name__ == "__main__":
    main()
