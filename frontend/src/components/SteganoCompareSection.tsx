import React, { useState } from "react";
import { Activity, Loader2, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import HeatmapViewer from "./HeatmapViewer";

interface SteganoReport {
  status?: string;
  mse: number | null;
  ssim: number | null;
  lsb_diff: number | null;
  residual_diff: number | null;
  stego_probability: number | null;
  heatmap_diff?: string | null;
  heatmap_residual?: string | null;
  score?: number | null;
  threshold?: number | null;
  stego_detected?: boolean | null;
  heatmap_siamese?: string | null;
}

interface SteganoCompareSectionProps {
  originalFile: File | null;
  suspiciousFile: File | null;
}

// Zaktualizowane stałe dla normalizacji wyniku sieci Syjamskiej
// Zakres surowego wyniku: 0.290 do 0.310
const THRESHOLD = 0.303; // Środek, odpowiada 50%
// Steepness K (około 400) dobrze mapuje zakres 0.290-0.310 na ~1%-~99%
const STEEPNESS_K = 450; 

const normalizeSiameseScore = (rawScore: number | null): number => {
    if (rawScore === null || isNaN(rawScore)) return 0.5;

    // Obliczanie wartości sigmoidalnej
    const normalized = 1 / (1 + Math.exp(-STEEPNESS_K * (rawScore - THRESHOLD)));

    // Zabezpieczenie przed wartościami spoza zakresu [0, 1]
    return Math.max(0, Math.min(1, normalized));
};


const SteganoCompareSection: React.FC<SteganoCompareSectionProps> = ({
  originalFile,
  suspiciousFile,
}) => {
  const [reportAlgo, setReportAlgo] = useState<SteganoReport | null>(null);
  const [reportSiamese, setReportSiamese] = useState<SteganoReport | null>(null);
  const [loadingAlgo, setLoadingAlgo] = useState(false);
  const [loadingSiamese, setLoadingSiamese] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [algoDone, setAlgoDone] = useState(false);
  const [siameseDone, setSiameseDone] = useState(false);
  const [tooDifferent, setTooDifferent] = useState(false);


  // ensure base64 has data URI prefix
  const ensureDataUri = (s?: string | null) => {
    if (!s) return null;
    if (s.startsWith("data:image/")) return s;
    return `data:image/png;base64,${s}`;
  };

  const pickNumber = (src: any, keys: string[]): number | null => {
    for (const k of keys) {
      if (src && Object.prototype.hasOwnProperty.call(src, k)) {
        const v = src[k];
        if (v === null || v === undefined) return null;
        const num = Number(v);
        if (!Number.isFinite(num)) return null;
        return num;
      }
    }
    return null;
  };

  const mapBackendToReport = (data: any): SteganoReport => {
    console.debug("raw stegano response:", data);

    const heatmapCandidates = (obj: any, keys: string[]) => {
      for (const k of keys) {
        if (!obj) continue;
        if (obj[k]) return obj[k];
      }
      return null;
    };

    const heatTop = (d: any) =>
      heatmapCandidates(d, [
        "heatmap_diff",
        "heatmap_residual",
        "diff_map",
        "residual_diff_map",
        "lsb_diff_map",
      ]);

    const heatmapDiffRaw =
      heatTop(data) && typeof heatTop(data) === "string" ? heatTop(data) : null;

    const hb = data?.heatmaps_base64 ?? data?.heatmaps ?? null;
    const diffFromHb =
      hb && (hb.diff_map || hb.diff || hb.heatmap_diff || hb.structural || hb["diff_map"])
        ? hb.diff_map ?? hb.diff ?? hb.heatmap_diff ?? hb.structural ?? hb["diff_map"]
        : null;
    const residualFromHb =
      hb && (hb.residual_diff_map || hb.residual || hb.heatmap_residual)
        ? hb.residual_diff_map ?? hb.residual ?? hb.heatmap_residual
        : null;

    const mse = pickNumber(data, ["mse", "MSE"]);
    const ssim = pickNumber(data, ["ssim", "SSIM", "ssim_val"]);
    const lsb =
      pickNumber(data, ["lsb_diff", "lsb_prop", "lsb", "lsb_propensity", "lsb_probability"]) ??
      null;
    const residual =
      pickNumber(data, [
        "residual_diff",
        "residual_diff_mean",
        "residual",
        "resid_diff",
        "residual_mean",
      ]) ?? null;

    const prob =
      pickNumber(data, [
        "stego_probability",
        "stego_score",
        "prob",
        "probability",
        "probability_stego",
        "score",
      ]) ?? null;

    const stegoProbability =
      pickNumber(data, ["stego_probability", "stego_score", "prob", "probability", "score"]) ?? prob ?? null;

    const heatmap_diff_raw =
      (typeof data?.heatmap_diff === "string" && data.heatmap_diff) ||
      (typeof data?.diff_map === "string" && data.diff_map) ||
      diffFromHb ||
      null;

    const heatmap_residual_raw =
      (typeof data?.heatmap_residual === "string" && data.heatmap_residual) ||
      (typeof data?.residual_diff_map === "string" && data.residual_diff_map) ||
      residualFromHb ||
      null;

    const mapped: SteganoReport = {
      mse: mse,
      ssim: ssim,
      lsb_diff: lsb,
      residual_diff: residual,
      stego_probability: stegoProbability,
      heatmap_diff: ensureDataUri(heatmap_diff_raw),
      heatmap_residual: ensureDataUri(heatmap_residual_raw),
      score: pickNumber(data, ["score"]) ?? null,
      threshold: pickNumber(data, ["threshold"]) ?? null,
      stego_detected:
        data && (data.stego_detected === true || data.stego_detected === false)
          ? Boolean(data.stego_detected)
          : null,
      heatmap_siamese: ensureDataUri(data?.heatmap_siamese ?? null),
    };

    console.debug("mapped stegano report:", mapped);
    return mapped;
  };

  const handleAnalyze = async () => {
    if (!originalFile || !suspiciousFile) {
      setError("Please provide both Original and Suspicious images.");
      return;
    }

    setLoadingAlgo(true);
    setError(null);

    const formData = new FormData();
    formData.append("original", originalFile);
    formData.append("suspicious", suspiciousFile);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_BASE}/stegano/compare`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Server error during analysis.");
      }
      const data = await res.json();
      const mapped = mapBackendToReport(data);
      setReportAlgo(mapped);
      setAlgoDone(true);

      // sprawdzamy Twój "trzeci przypadek" – TREŚCIOWA RÓŻNICA
      if (mapped.mse !== null && mapped.ssim !== null) {
        if (mapped.mse > 0.02 || mapped.ssim < 0.70) {
          setTooDifferent(true);
        }
      }

    } catch (err: any) {
      console.error("stegano/compare error:", err);
      setError(err?.message || "Unknown error");
    } finally {
      setLoadingAlgo(false);
    }
  };

  const handleSiameseAnalyze = async () => {
    if (!originalFile || !suspiciousFile) {
      setError("Please provide both Original and Suspicious images.");
      return;
    }

    setLoadingSiamese(true);
    setError(null);

    const formData = new FormData();
    formData.append("original", originalFile);
    formData.append("suspicious", suspiciousFile);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_BASE}/stegano/siamese`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || "Server error during Siamese analysis.");
      }
      const data = await res.json();
      const mappedSiamese = mapBackendToReport(data);
      setReportSiamese(mappedSiamese);
      setSiameseDone(true);
    } catch (err: any) {
      console.error("stegano/siamese error:", err);
      setError(err?.message || "Unknown error (siamese)");
    } finally {
      setLoadingSiamese(false);
    }
  };

  const formatValue = (val: number | null) => (val !== null && !isNaN(val) ? val.toFixed(4) : "–");

  const probColor = (p: number) =>
    p > 0.75 ? "from-red-500 to-red-700" : p > 0.5 ? "from-yellow-500 to-yellow-700" : "from-teal-500 to-green-500";

  // Obliczenie znormalizowanego wyniku dla sieci syjamskiej
  const rawSiameseScore = reportSiamese?.score ?? null;
  const normalizedScore = normalizeSiameseScore(rawSiameseScore);
  
  // Wartość logiczna dla paska postępu/koloru
  const barValue = normalizedScore;


  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        <Activity className="w-5 h-5 text-teal-400" />
        Steganographic Image Comparison
      </h3>

      <div className="space-y-6">
        {/* Primary analyze */}
        {!algoDone && (<button
          onClick={handleAnalyze}
          disabled={loadingAlgo}
          className={`w-full py-3 text-lg font-semibold rounded-xl transition duration-300 ${
            loadingAlgo
              ? "bg-teal-800/50 cursor-not-allowed text-gray-300"
              : "bg-gradient-to-r from-teal-500 to-green-500 hover:opacity-90 text-white shadow-lg shadow-green-900/20"
          }`}
        >
          {loadingAlgo ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="animate-spin w-5 h-5" /> Analyzing...
            </span>
          ) : (
            "Run Steganographic Analysis"
          )}
        </button>
      )}


        {error && <div className="p-4 bg-red-900/40 border border-red-700 rounded-lg text-red-300">{error}</div>}

        {/* Algorithmic results - BEZ ZMIAN W WYŚWIETLANIU WYNIKÓW */}
        {/* --- INTERPRETATION MESSAGE BASED ON MSE + SSIM --- */}
        {reportAlgo && (() => {
          const mse = reportAlgo.mse ?? null;
          const ssim = reportAlgo.ssim ?? null;

          if (mse === null || ssim === null) return null;

          // 1. IDENTYCZNE OBRAZY
          if (mse < 0.00001 && ssim > 0.9999) {
            return (
              <div className="p-4 bg-blue-900/40 border border-blue-700 text-blue-300 rounded-lg flex gap-3">
                <CheckCircle className="w-6 h-6 text-blue-300" />
                <div>
                  <p className="font-semibold text-lg">Images appear identical</p>
                  <p className="text-sm">
                    The content of both images is effectively the same — algorithmic differences will be minimal.
                  </p>
                </div>
              </div>
            );
          }

          // 2. PRAWIE IDENTYCZNE
          if (mse < 0.001 && ssim > 0.98) {
            return (
              <div className="p-4 bg-emerald-900/40 border border-emerald-700 text-emerald-300 rounded-lg flex gap-3">
                <CheckCircle className="w-6 h-6 text-emerald-300" />
                <div>
                  <p className="font-semibold text-lg">Images are nearly identical</p>
                  <p className="text-sm">
                    These images differ only slightly — subtle artifacts may still be detectable.
                  </p>
                </div>
              </div>
            );
          }

          // 3. ZBYT RÓŻNE TREŚCIOWO
          if (mse > 0.01 && ssim < 0.80) {
            return (
              <div className="p-4 bg-red-900/40 border border-red-700 text-red-300 rounded-lg flex gap-3">
                <AlertTriangle className="w-6 h-6 text-red-300" />
                <div>
                  <p className="font-semibold text-lg">Images differ significantly</p>
                  <p className="text-sm">
                    The images appear to contain very different content — reliable steganographic comparison may not be possible.
                  </p>
                </div>
              </div>
            );
          }

          return null;
        })()}

        {reportAlgo && !tooDifferent &&(
          <div className="space-y-6">
            <h4 className="text-md font-semibold text-teal-300 mb-2">Algorithmic Analysis</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-gray-300">
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">MSE</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(reportAlgo.mse)}</p>
              </div>
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">SSIM</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(reportAlgo.ssim)}</p>
              </div>
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">LSB Diff</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(reportAlgo.lsb_diff)}</p>
              </div>
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Residual Diff</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(reportAlgo.residual_diff)}</p>
              </div>
            </div>

            {/* Heatmaps (side by side) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {reportAlgo.heatmap_diff ? (
                  <HeatmapViewer title="Structural Difference Heatmap" imageData={reportAlgo.heatmap_diff} />
                ) : (
                  <div className="text-xs text-gray-500 italic">No structural heatmap.</div>
                )}

                {reportAlgo.heatmap_residual ? (
                  <HeatmapViewer title="Residual Difference Heatmap" imageData={reportAlgo.heatmap_residual} />
                ) : (
                  <div className="text-xs text-gray-500 italic">No residual heatmap.</div>
                )}
              </div>

          </div>
        )}

        {/* Siamese section - WPROWADZONE ZMIANY DLA NORMALIZACJI SCORE */}
        {reportSiamese && (
          <div className="pt-6 border-t border-gray-800">
            <h4 className="text-md font-semibold text-teal-300 mb-3">Siamese Network Analysis</h4>
            <div className="space-y-3">
              <div className="bg-gray-800/40 border border-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400">Siamese score (Normalized)</p>
                    <p className="text-2xl font-bold text-teal-300">
                      {/* Używamy znormalizowanego wyniku */}
                      {((normalizedScore) * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">Decision</p>
                    {/* Decyzja może być oparta na znormalizowanym wyniku > 0.5 (50%) */}
                    <p className={`font-semibold ${normalizedScore > 0.5 ? "text-red-400" : "text-green-400"}`}>
                      {normalizedScore > 0.5 ? "Possible Stego" : "Likely Clean"}
                    </p>
                  </div>
                </div>

                <div className="mt-3">
                  <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden border border-gray-700">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        barValue > 0.75
                          ? "bg-gradient-to-r from-red-500 to-red-700"
                          : barValue > 0.5
                          ? "bg-gradient-to-r from-yellow-500 to-yellow-700"
                          : "bg-gradient-to-r from-green-500 to-teal-600"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, barValue * 100))}%` }}
                    />
                  </div>
                  {/* Wyświetlamy informację o progu 50% */}
                  <p className="text-xs text-gray-400 italic mt-1">
                    Threshold (raw score {THRESHOLD.toFixed(3)}): 50%
                    {rawSiameseScore !== null && (
                        <span className="ml-2 text-gray-500">
                            (Raw value: {rawSiameseScore.toFixed(4)})
                        </span>
                    )}
                  </p>
                </div>

                <div className="mt-4">
                  {reportSiamese.heatmap_siamese ? (
                    <HeatmapViewer title="Siamese Activation Heatmap" imageData={reportSiamese.heatmap_siamese} />
                  ) : (
                    <div className="text-xs text-gray-500 italic">No Siamese heatmap available.</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Run Siamese button */}
        {reportAlgo && !siameseDone && !tooDifferent && (
          <button
            onClick={handleSiameseAnalyze}
            disabled={loadingSiamese}
            className={`w-full py-3 text-md font-semibold rounded-xl transition duration-300 ${
              loadingSiamese
                ? "bg-gray-700 cursor-not-allowed text-gray-400"
                : "bg-gradient-to-r from-teal-500 to-cyan-600 hover:opacity-90 text-white shadow-lg"
            }`}
          >
            {loadingSiamese ? (
              <span className="flex items-center justify-center gap-2">
                <Loader2 className="animate-spin w-5 h-5" /> Running Siamese Analysis...
              </span>
            ) : (
              "Analyze with Siamese Network"
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default SteganoCompareSection;