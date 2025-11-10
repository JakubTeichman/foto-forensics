import React, { useState } from "react";

interface SteganoReportProps {
  image: File | null;
}

const SteganoReport: React.FC<SteganoReportProps> = ({ image }) => {
  const [result, setResult] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const analyzeStegano = async () => {
    if (!image) {
      setError("Please select an image before analysis.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_BASE}/stegano/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setResult(data);
      } else {
        setError(data.error || "Steganography analysis failed.");
      }
    } catch (err) {
      console.error("Error analyzing steganography:", err);
      setError("Failed to connect to backend.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const extractMethods = (resp: any) => {
    if (!resp) return {};
    if (resp.report?.analysis_results) return resp.report.analysis_results;
    if (resp.details?.methods_results) return resp.details.methods_results;
    if (resp.details && typeof resp.details === "object") {
      const maybe = Object.entries(resp.details).every(
        ([_, v]) => v && typeof v === "object" && ("score" in v || "detected" in v)
      );
      if (maybe) return resp.details;
    }
    if (resp.analysis_results) return resp.analysis_results;
    return {};
  };

  const methodsResults = extractMethods(result);

  // 🔹 Obliczanie procentu wykryć
  const totalMethods = Object.keys(methodsResults).length;
  const detectedCount = Object.values(methodsResults).filter((m: any) => m?.detected).length;
  const detectionRatio = totalMethods > 0 ? detectedCount / totalMethods : 0;

  // 🔹 Opisowy poziom wykrycia
  let detectionLabel = "No hidden data detected.";
  let detectionColor = "text-green-400";
  if (detectionRatio >= 0.25 && detectionRatio <= 0.5) {
    detectionLabel = "Possibility of hidden data.";
    detectionColor = "text-orange-400";
  } else if (detectionRatio > 0.5) {
    detectionLabel = "Hidden data detected!";
    detectionColor = "text-red-400";
  } else if (detectionRatio > 0 && detectionRatio < 0.25) {
    detectionLabel = "Low possibility of hidden data.";
    detectionColor = "text-yellow-400";
  }

  return (
    <div>
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        Steganography Analysis
      </h3>

      {!image && (
        <p className="text-gray-400 italic text-center mb-4">
          No image selected for analysis.
        </p>
      )}

      {/* 🔹 Przycisk analizy */}
      {!result && (
      <div className="flex justify-left mb-4">
        <button
          onClick={analyzeStegano}
          disabled={!image || isAnalyzing}
          className={`px-5 py-2 rounded-lg font-semibold transition-all duration-300 ${
            !image
              ? "bg-gray-700 text-gray-400 cursor-not-allowed"
              : isAnalyzing
              ? "bg-gray-600 text-gray-300 cursor-wait"
              : "bg-teal-600 hover:bg-teal-700 text-white shadow-md"
          }`}
        >
          {isAnalyzing ? "Analyzing..." : "Run Steganography Analysis"}
        </button>
      </div>
      )}

      {isAnalyzing && (
        <div className="flex flex-col items-center py-4">
          <div className="w-6 h-6 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm mt-3 text-gray-400">
            Generating steganography analysis...
          </p>
        </div>
      )}

      {error && <p className="text-red-500 mt-4 font-medium text-center">⚠️ {error}</p>}

      {result && (
        <div
          className={`mt-5 bg-gray-900 border border-gray-800 rounded-xl p-5 transition-all duration-500`}
        >
          <h4 className="text-lg font-semibold text-teal-400 mb-3 text-left">
            Steganography Detection Report
          </h4>

          <div className="mb-3 text-left">
            <div className={`font-semibold ${detectionColor}`}>{detectionLabel}</div>
            <div className="text-xs text-gray-500 mt-1">
              Methods run: {totalMethods} — positives: {detectedCount} (
              {(detectionRatio * 100).toFixed(1)}%)
            </div>
          </div>

          {Object.keys(methodsResults).length > 0 ? (
            <>
              <table className="w-full text-sm border-collapse mt-2">
                <thead>
                  <tr className="border-b border-gray-700 text-gray-300">
                    <th className="text-left py-1">Method</th>
                    <th className="text-left py-1">Score</th>
                    <th className="text-left py-1">Detection Status</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(methodsResults).map(([method, data]: any) => {
                    const scoreValue =
                      typeof data?.score_calibrated === "number"
                        ? data.score_calibrated
                        : typeof data?.score_raw === "number"
                        ? data.score_raw
                        : typeof data?.score === "number"
                        ? data.score
                        : parseFloat(
                            data?.score_calibrated || data?.score_raw || data?.score
                          ) || 0;

                    const detected = !!data?.detected;

                    return (
                      <tr key={method} className="border-b border-gray-800 text-gray-300">
                        <td className="py-1 font-medium">{method}</td>
                        <td className="py-1">{scoreValue.toFixed(3)}</td>
                        <td
                          className={`py-1 font-semibold ${
                            detected ? "text-red-400" : "text-green-400"
                          }`}
                        >
                          {detected ? "Detected" : "Not detected"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {/* 🔥 Heatmapa — zawsze pokazuj jeśli jest dostępna */}
              {result.average_heatmap_base64 && (
                <div className="mt-6 text-center">
                  <h5 className="text-md font-semibold text-teal-400 mb-3">
                    Aggregated Steganalysis Heatmap
                  </h5>
                  <div className="flex justify-center">
                    <img
                      src={`data:image/png;base64,${result.average_heatmap_base64}`}
                      alt="Aggregated Steganalysis Heatmap"
                      className="rounded-xl shadow-lg border border-gray-800 max-w-full h-auto"
                      style={{
                        width: "80%",
                        maxWidth: "480px",
                        transition: "transform 0.3s ease",
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.transform = "scale(1.03)")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.transform = "scale(1.0)")
                      }
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Average of all individual method heatmaps.
                  </p>
                </div>
              )}
            </>
          ) : (
            <p className="text-gray-500 italic mt-2 text-center">
              No per-method details available.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SteganoReport;
