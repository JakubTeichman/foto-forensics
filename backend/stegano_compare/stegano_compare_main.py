import logging
import numpy as np
import matplotlib.pyplot as plt
import os
from .utils import load_image_safe, save_heatmap_to_base64
from .difference_metrics import compute_differences
from .residual_analysis import analyze_residuals
from .lsb_analysis import analyze_lsb
from .dct_analysis import compare_dct
from .model_detector import detect_anomaly

logging.basicConfig(level=logging.INFO, format="%(message)s")

def make_heatmap_from_map(diff_map, title="diff", cmap="hot"):
    """Return matplotlib figure for diff_map (0..1)"""
    fig, ax = plt.subplots(figsize=(6,4))
    im = ax.imshow(diff_map, cmap=cmap)
    ax.axis('off')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def analyze_images(original_path_or_file, suspicious_path_or_file, return_images_base64=True):
    """
    Full pipeline:
      - load
      - compute mse, ssim, diff_map
      - residual analysis
      - lsb analysis
      - dct analysis
      - detection (RF if exists else heuristic)
      - generate heatmaps (diff_map, residual_diff_map, lsb_diff_map)
    Returns: dict with metrics, score, and base64 heatmaps (if requested)
    """
    orig = load_image_safe(original_path_or_file)
    susp = load_image_safe(suspicious_path_or_file)

    report = {"status":"OK", "notes":[]}

        # 1) differences
    mse, ssim_val, diff_map = compute_differences(orig, susp)
    report.update({
        "mse": float(mse) if not np.isnan(mse) else 0.0,
        "ssim": float(ssim_val) if not np.isnan(ssim_val) else 0.0
    })

    # 2) residuals
    residual_diff_mean, residual_diff_map = analyze_residuals(orig, susp)
    report["residual_diff_mean"] = float(residual_diff_mean) if not np.isnan(residual_diff_mean) else 0.0

    # 3) lsb
    lsb_prop, lsb_diff_map = analyze_lsb(orig, susp)
    report["lsb_prop"] = float(lsb_prop) if lsb_prop is not None and not np.isnan(lsb_prop) else 0.0

    # 4) dct
    dct_score, dct_orig_stats, dct_susp_stats = compare_dct(orig, susp)
    report["dct_score"] = float(dct_score) if not np.isnan(dct_score) else 0.0
    report["dct_orig_stats"] = dct_orig_stats
    report["dct_susp_stats"] = dct_susp_stats

    # 5) detection
    score, method = detect_anomaly(
        report["mse"], report["ssim"], report["residual_diff_mean"],
        report["lsb_prop"], report["dct_score"]
    )
    report["stego_score"] = float(score) if not np.isnan(score) else 0.0
    report["detector_method"] = method


    # 6) heatmaps (base64)
    heatmaps = {}
    if return_images_base64:
        try:
            fig1 = make_heatmap_from_map(diff_map, title="Pixel absolute difference")
            heatmaps["diff_map"] = save_heatmap_to_base64(fig1)
            plt.close(fig1)

            fig2 = make_heatmap_from_map(residual_diff_map, title="Residual difference (HPF)")
            heatmaps["residual_diff_map"] = save_heatmap_to_base64(fig2)
            plt.close(fig2)

            fig3 = make_heatmap_from_map(lsb_diff_map, title="LSB differences", cmap="gray")
            heatmaps["lsb_diff_map"] = save_heatmap_to_base64(fig3)
            plt.close(fig3)
        except Exception as e:
            logging.error(f"Heatmap generation failed: {e}")
            report["notes"].append(f"Heatmap error: {e}")

    report["heatmaps_base64"] = heatmaps
    report["notes"].append("Analysis complete.")
    return report

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 3:
        print("Usage: python stegano_compare_main.py <original> <suspicious>")
        sys.exit(1)
    res = analyze_images(sys.argv[1], sys.argv[2], True)
    print(json.dumps(res, indent=2))
