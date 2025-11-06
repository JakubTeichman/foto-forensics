import React, { useState, useEffect } from "react";

interface SteganoReportProps {
  image: File | null;
}

const SteganoReport: React.FC<SteganoReportProps> = ({ image }) => {
  const [result, setResult] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 🔹 Automatyczna analiza po wczytaniu obrazu
  useEffect(() => {
    if (!image) return;

    const analyzeStegano = async () => {
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

    analyzeStegano();
  }, [image]);

  // 🔹 Pomocnicza funkcja – wyciąga wyniki metod niezależnie od struktury odpowiedzi
  const extractMethods = (resp: any) => {
    if (!resp) return {};

    if (resp.report?.analysis_results && typeof resp.report.analysis_results === "object") {
      return resp.report.analysis_results;
    }
    if (resp.details?.methods_results && typeof resp.details.methods_results === "object") {
      return resp.details.methods_results;
    }
    if (resp.details && typeof resp.details === "object") {
      const maybe = Object.entries(resp.details).every(
        ([_, v]) => v && typeof v === "object" && ("score" in v || "detected" in v)
      );
      if (maybe) return resp.details;
    }
    if (resp.analysis_results && typeof resp.analysis_results === "object") {
      return resp.analysis_results;
    }

    return {};
  };

  const methodsResults = extractMethods(result);

  return (
    <div>
      {!result && (
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        Steganography Analysis
      </h3>)}

      {!image && (
        <p className="text-gray-400 italic text-center">
          No image selected for analysis.
        </p>
      )}

      {isAnalyzing && (
        <div className="flex flex-col items-center py-6">
          <div className="w-6 h-6 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm mt-3 text-gray-400">
            Generating steganography analysis...
          </p>
        </div>
      )}

      {error && (
        <p className="text-red-500 mt-4 font-medium text-center">⚠️ {error}</p>
      )}

      {result && (
        <div className="mt-5 bg-gray-900 border border-gray-800 rounded-xl p-4 transition-all duration-300">
          <h4 className="text-lg font-semibold text-teal-400 mb-3 text-left ">
            Steganography Detection Report
          </h4>

          <div className="mb-3 text-left">
            {result.hidden_detected ? (
              <div className="text-red-400 font-semibold">
                Hidden data detected!
                {result.detected_methods?.length > 0 && (
                  <span className="text-gray-400 ml-2">
                    ({result.detected_methods.join(", ")})
                  </span>
                )}
              </div>
            ) : (
              <div className="text-green-400 font-semibold">
                No hidden data detected.
              </div>
            )}
            <div className="text-xs text-gray-500 mt-1">
              Methods run: {result.total_methods ?? Object.keys(methodsResults).length}
              {result.positive_count !== undefined
                ? ` — positives: ${result.positive_count}`
                : ""}
            </div>
          </div>

          {Object.keys(methodsResults).length > 0 ? (
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
