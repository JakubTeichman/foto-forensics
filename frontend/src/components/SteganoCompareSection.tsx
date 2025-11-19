// SteganoCompareSection.tsx
import React, { useState } from "react";
import { Activity, Loader2, AlertTriangle, CheckCircle, XCircle } from "lucide-react";
import HeatmapViewer from "./HeatmapViewer";

interface SteganoReport {
  status: string;
  mse: number | null;
  ssim: number | null;
  lsb_diff: number | null;
  residual_diff: number | null;
  stego_probability: number | null;
  heatmap_diff?: string;
  heatmap_residual?: string;
  score?: number;
  threshold?: number;
  stego_detected?: boolean;
  heatmap_siamese?: string;
}

interface SteganoCompareSectionProps {
  originalFile: File | null;
  suspiciousFile: File | null;
}

const SteganoCompareSection: React.FC<SteganoCompareSectionProps> = ({
  originalFile,
  suspiciousFile,
}) => {
  const [report, setReport] = useState<SteganoReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSiamese, setLoadingSiamese] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // poprawia base64 (dodaje prefix jeśli brak)
  const ensureDataUri = (s?: string | null) => {
    if (!s) return null;
    if (s.startsWith("data:image/")) return s;
    // spróbuj PNG, jeśli nie zadziała backend może wysłać jpg — frontend img element poradzi sobie
    return `data:image/png;base64,${s}`;
  };

  const handleAnalyze = async () => {
    if (!originalFile || !suspiciousFile) {
      setError("Please provide both Original and Suspicious images.");
      return;
    }

    setLoading(true);
    setError(null);
    setReport(null);

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

      // napraw base64 heatmap (może przyjść bez prefixu)
      data.heatmap_diff = ensureDataUri(data.heatmap_diff);
      data.heatmap_residual = ensureDataUri(data.heatmap_residual);

      setReport(data);
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
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
      data.heatmap_siamese = ensureDataUri(data.heatmap_siamese);

      // scalamy wynik (zachowujemy poprzednie pola)
      setReport((prev) => ({ ...(prev || {}), ...(data || {}) }));
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoadingSiamese(false);
    }
  };

  const formatValue = (val: number | null) =>
    val !== null && !isNaN(val) ? val.toFixed(4) : "–";

  // similarityLevel decyduje czy pokazać siamese button
  let similarityLevel: "high" | "medium" | "low" | null = null;
  if (report && report.mse !== null && report.ssim !== null) {
    if (report.mse < 0.001 && report.ssim > 0.98) similarityLevel = "high";
    else if (report.mse < 0.01 && report.ssim > 0.9) similarityLevel = "medium";
    else similarityLevel = "low";
  }

  const levelConfig = {
    high: {
      icon: <CheckCircle className="w-5 h-5 text-green-400" />,
      text: "Images are nearly identical.",
      bg: "bg-green-900/30 border-green-700 text-green-300",
    },
    medium: {
      icon: <AlertTriangle className="w-5 h-5 text-yellow-400" />,
      text: "Images are similar, but some differences detected.",
      bg: "bg-yellow-900/30 border-yellow-700 text-yellow-300",
    },
    low: {
      icon: <XCircle className="w-5 h-5 text-red-400" />,
      text: "Images differ significantly — detailed comparison unreliable.",
      bg: "bg-red-900/30 border-red-700 text-red-300",
    },
  };

  // probability bar color helper
  const probColor = (p: number) =>
    p > 0.75 ? "from-red-500 to-red-700" : p > 0.5 ? "from-yellow-500 to-yellow-700" : "from-teal-500 to-green-500";

  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        <Activity className="w-5 h-5 text-teal-400" />
        Steganographic Image Comparison
      </h3>

      <div className="space-y-6">
        {/* Primary analysis button */}
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className={`w-full py-3 text-lg font-semibold rounded-xl transition duration-300 ${
            loading ? "bg-teal-800/50 cursor-not-allowed text-gray-300" : "bg-gradient-to-r from-teal-500 to-green-500 hover:opacity-90 text-white shadow-lg shadow-green-900/20"
          }`}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <Loader2 className="animate-spin w-5 h-5" /> Analyzing...
            </span>
          ) : (
            "Run Steganographic Analysis"
          )}
        </button>

        {error && <div className="p-4 bg-red-900/40 border border-red-700 rounded-lg text-red-300">{error}</div>}

        {/* Show primary results */}
        {report && (
          <div className="space-y-6">
            {similarityLevel && (
              <div className={`p-3 border rounded-lg flex items-center gap-3 ${levelConfig[similarityLevel].bg}`}>
                {levelConfig[similarityLevel].icon}
                <div className="text-sm">
                  <div className="font-medium">{levelConfig[similarityLevel].text}</div>
                  <div className="text-xs text-gray-400">MSE: {formatValue(report.mse)} · SSIM: {formatValue(report.ssim)}</div>
                </div>
              </div>
            )}

            {/* If too different — show message and stop (no siamese) */}
            {similarityLevel === "low" && (
              <div className="p-3 bg-gray-800/40 border border-gray-700 rounded-lg text-yellow-200">
                Images differ substantially — further Siamese analysis is disabled.
              </div>
            )}

            {/* Metrics grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-gray-300">
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">MSE</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(report.mse)}</p>
              </div>

              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">SSIM</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(report.ssim)}</p>
              </div>

              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">LSB Diff</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(report.lsb_diff)}</p>
              </div>

              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <p className="text-sm text-gray-400">Residual Diff</p>
                <p className="text-xl font-semibold text-teal-400">{formatValue(report.residual_diff)}</p>
              </div>

              {/* Probability bar (original method) */}
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg lg:col-span-2">
                <p className="text-sm text-gray-400">Steganography Probability</p>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden mt-2">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${probColor(report.stego_probability ?? 0)} transition-all duration-700`}
                    style={{ width: `${Math.min(100, Math.max(0, (report.stego_probability ?? 0) * 100))}%` }}
                  />
                </div>
                <p className="mt-1 text-sm text-gray-300">{((report.stego_probability ?? 0) * 100).toFixed(2)}%</p>
              </div>
            </div>

            {/* Heatmaps (structural / residual) */}
            <div className="space-y-4">
              {report.heatmap_diff ? (
                <HeatmapViewer title="Structural Difference Heatmap" imageData={report.heatmap_diff} />
              ) : (
                <div className="text-xs text-gray-500 italic">No structural heatmap.</div>
              )}

              {report.heatmap_residual ? (
                <HeatmapViewer title="Residual Difference Heatmap" imageData={report.heatmap_residual} />
              ) : (
                <div className="text-xs text-gray-500 italic">No residual heatmap.</div>
              )}
            </div>
          </div>
        )}

        {/* SIAMESE section — pokazujemy dopiero PO pierwszej analizie i tylko gdy similarity !== low */}
        {report && similarityLevel !== "low" && (
          <div className="pt-6 border-t border-gray-800">
            <h4 className="text-md font-semibold text-teal-300 mb-3">Siamese Network Analysis</h4>

            <div className="space-y-3">
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

              {/* Siamese results panel (confidence + heatmap + threshold) */}
              {report.score !== undefined && (
                <div className="bg-gray-800/40 border border-gray-700 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-400">Siamese score</p>
                      <p className="text-2xl font-bold text-teal-300">{(report.score * 100).toFixed(2)}%</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-400">Decision</p>
                      <p className={`font-semibold ${report.stego_detected ? "text-red-400" : "text-green-400"}`}>
                        {report.stego_detected ? "Possible Stego" : "Likely Clean"}
                      </p>
                    </div>
                  </div>

                  {/* confidence bar (siamese) */}
                  <div className="mt-3">
                    <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden border border-gray-700">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          (report.score ?? 0) > 0.75 ? "bg-gradient-to-r from-red-500 to-red-700" : (report.score ?? 0) > 0.5 ? "bg-gradient-to-r from-yellow-500 to-yellow-700" : "bg-gradient-to-r from-green-500 to-teal-600"
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, (report.score ?? 0) * 100))}%` }}
                      />
                    </div>
                    {report.threshold !== undefined && (
                      <p className="text-xs text-gray-400 italic mt-1">Threshold: {(report.threshold).toFixed(4)}</p>
                    )}
                  </div>

                  {/* Siamese heatmap */}
                  <div className="mt-4">
                    {report.heatmap_siamese ? (
                      <HeatmapViewer title="Siamese Activation Heatmap" imageData={report.heatmap_siamese} />
                    ) : (
                      <div className="text-xs text-gray-500 italic">No Siamese heatmap available.</div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SteganoCompareSection;
