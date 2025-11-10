import React, { useState } from "react";
import { Activity, Loader2 } from "lucide-react";
import HeatmapViewer from "./HeatmapViewer";

interface SteganoReport {
  status: string;
  mse: number | null;
  ssim: number | null;
  lsb_diff: number | null;
  residual_diff: number | null;
  stego_probability: number | null;
  integrity_status?: string;
  notes: string[];
  heatmap_diff?: string;
  heatmap_residual?: string;
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
  const [error, setError] = useState<string | null>(null);

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
      if (!res.ok) throw new Error("Server error during analysis.");
      const data = await res.json();
      setReport(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatValue = (val: number | null) =>
    val !== null && !isNaN(val) ? val.toFixed(4) : "–";

  const probability = report?.stego_probability ?? 0;
  const probabilityColor =
    probability > 0.75
      ? "from-red-500 to-red-700"
      : probability > 0.5
      ? "from-yellow-500 to-yellow-700"
      : "from-teal-500 to-green-500";

  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        <Activity className="w-5 h-5 text-teal-400" />
        Steganographic Image Comparison
      </h3>

      <div className="space-y-6">
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className={`w-full py-3 text-lg font-semibold rounded-xl transition duration-300 ${
            loading
              ? "bg-teal-800/50 cursor-not-allowed text-gray-300"
              : "bg-gradient-to-r from-teal-500 to-green-500 hover:opacity-90 text-white shadow-lg shadow-green-900/20"
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

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-900/40 border border-red-700 rounded-lg text-red-300">
            {error}
          </div>
        )}

        {/* Report */}
        {report && (
          <div className="space-y-8 animate-fadeIn">
            {report.integrity_status === "failed" && (
              <div className="p-4 bg-yellow-900/30 border border-yellow-700 text-yellow-300 rounded-lg">
                Integrity check <strong>failed</strong> — detailed steganographic comparison triggered.
              </div>
            )}

            {/* Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-gray-300">
              {[
                { label: "MSE", value: report.mse },
                { label: "SSIM", value: report.ssim },
                { label: "LSB Diff", value: report.lsb_diff },
                { label: "Residual Diff", value: report.residual_diff },
              ].map((metric, i) => (
                <div
                  key={i}
                  className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg"
                >
                  <p className="text-sm text-gray-400">{metric.label}</p>
                  <p className="text-xl font-semibold text-teal-400">
                    {formatValue(metric.value)}
                  </p>
                </div>
              ))}

              {/* Stego Probability */}
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg col-span-full">
                <span className="block text-sm text-gray-400 mb-1">
                  Steganography Probability
                </span>
                <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                  <div
                    className={`h-3 rounded-full bg-gradient-to-r ${probabilityColor} transition-all duration-700`}
                    style={{ width: `${probability * 100}%` }}
                  />
                </div>
                <p className="mt-1 text-sm text-gray-300">
                  {(probability * 100).toFixed(2)}%
                </p>
              </div>
            </div>

            {/* Heatmaps */}
            <div className="space-y-6">
              {report.heatmap_diff && (
                <HeatmapViewer
                  title="Structural Difference Heatmap"
                  imageData={report.heatmap_diff}
                />
              )}
              {report.heatmap_residual && (
                <HeatmapViewer
                  title="Residual Difference Heatmap"
                  imageData={report.heatmap_residual}
                />
              )}
              {!report.heatmap_diff && !report.heatmap_residual && (
                <p className="text-gray-500 text-center italic">
                  No heatmaps generated for these images.
                </p>
              )}
            </div>

            {/* Notes */}
            {report.notes?.length > 0 && (
              <div className="bg-gray-800/60 border border-gray-700 p-4 rounded-lg">
                <h4 className="text-lg font-semibold text-teal-400 mb-2">
                  Analysis Notes
                </h4>
                <ul className="list-disc list-inside text-gray-400 space-y-1">
                  {report.notes.map((note, idx) => (
                    <li key={idx}>{note}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SteganoCompareSection;
