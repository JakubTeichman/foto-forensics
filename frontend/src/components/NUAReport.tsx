import React, { useEffect, useState } from "react";

interface NUAReportProps {
  imageFile: File | null; // Image passed from parent component
}

const NUAReport: React.FC<NUAReportProps> = ({ imageFile }) => {
  const [result, setResult] = useState<{ detected: boolean; confidence: number } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!imageFile) return;

    const analyzeNUA = async () => {
      setLoading(true);
      setResult(null);

      const formData = new FormData();
      formData.append("file", imageFile);

      try {
        const response = await fetch("http://localhost:5000/analyze/nua", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        setResult(data);
      } catch (error) {
        console.error("Error during NUA analysis:", error);
      } finally {
        setLoading(false);
      }
    };

    analyzeNUA();
  }, [imageFile]);

  return (
    <div className="w-full bg-gray-900 text-gray-100 border border-gray-700 rounded-2xl shadow-lg p-4 flex flex-col items-center justify-center">
      <h2 className="text-lg font-semibold mb-2 text-center">📊 NUA Report</h2>

      {loading ? (
        <div className="flex flex-col items-center py-4">
          <div className="w-5 h-5 border-2 border-gray-300 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm mt-2 text-gray-400">Analyzing...</p>
        </div>
      ) : result ? (
        <div
          className={`w-full text-center py-3 rounded-lg transition-all duration-300 ${
            result.detected
              ? "bg-red-900/30 border border-red-700"
              : "bg-green-900/30 border border-green-700"
          }`}
        >
          <p className="text-sm font-medium">
            Result:{" "}
            <span
              className={`font-semibold ${
                result.detected ? "text-red-400" : "text-green-400"
              }`}
            >
              {result.detected ? "NUA detected" : "No NUA detected"}
            </span>
          </p>
          <p className="text-sm text-gray-400">
            Confidence: {(result.confidence * 100).toFixed(2)}%
          </p>
        </div>
      ) : (
        <p className="text-sm text-gray-400 py-2">No data available</p>
      )}
    </div>
  );
};

export default NUAReport;
