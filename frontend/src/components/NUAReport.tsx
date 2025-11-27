import React, { useEffect, useState, useRef } from "react";
import { Info } from "lucide-react";

interface NUAReportProps {
  imageFile: File | null;
  referenceImages?: File[];
}

interface NUAResult {
  detected: boolean;
  confidence: number;
}

const NUAReport: React.FC<NUAReportProps> = ({ imageFile, referenceImages = [] }) => {
  const [mainResult, setMainResult] = useState<NUAResult | null>(null);
  const [referenceResults, setReferenceResults] = useState<NUAResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);

  const lastHash = useRef<string | null>(null);

  useEffect(() => {
    if (!imageFile) return;

    const hash =
      imageFile.name +
      "-" +
      imageFile.size +
      "-" +
      referenceImages.map((f) => f.name + "-" + f.size).join(",");

    if (hash === lastHash.current) return;
    lastHash.current = hash;

    const analyzeNUA = async () => {
      setLoading(true);
      setError(null);
      setMainResult(null);
      setReferenceResults([]);

      try {
        const mainFormData = new FormData();
        mainFormData.append("file", imageFile);

        const mainResponse = await fetch(`${process.env.REACT_APP_API_BASE}/analyze/nua`, {
          method: "POST",
          body: mainFormData,
        });

        if (!mainResponse.ok) {
          throw new Error("Failed to analyze the evidence image.");
        }

        const mainData = await mainResponse.json();
        setMainResult(mainData);

        if (referenceImages.length > 0) {
          const results: NUAResult[] = [];

          for (const refImg of referenceImages) {
            const formData = new FormData();
            formData.append("file", refImg);

            const res = await fetch(`${process.env.REACT_APP_API_BASE}/analyze/nua`, {
              method: "POST",
              body: formData,
            });

            if (!res.ok) continue;
            const data = await res.json();
            results.push(data);
          }

          setReferenceResults(results);
        }
      } catch (err: any) {
        console.error("Error during NUA analysis:", err);
        setError(err.message || "Unexpected error occurred during NUA analysis.");
      } finally {
        setLoading(false);
      }
    };

    analyzeNUA();
  }, [imageFile, referenceImages]);

  const detectedCount = referenceResults.filter((r) => r.detected).length;
  const total = referenceResults.length;
  const detectedRatio = total > 0 ? detectedCount / total : 0;

  // 🔹 Kolor dla reference set
  let refBgColor = "bg-green-900/30 border-green-700";
  if (detectedRatio >= 0.4 && detectedRatio <= 0.6) {
    refBgColor = "bg-orange-900/30 border-orange-700";
  } else if (detectedRatio > 0.6) {
    refBgColor = "bg-red-900/30 border-red-700";
  }

  // ===============================
  // 🔹 Dynamiczne wyjaśnienia NUA
  // ===============================

  const explanationNoNUA =
    "The absence of detected non-unique artefacts (NUA) suggests that the image was captured without photographic modes or post-processing procedures that could negatively affect the extraction of features such as PRNU or noiseprint.";

  const explanationDetectedNUA =
    "The detection of NUA indicates the possibility that the image was taken using photographic modes or subjected to post-processing, which may interfere with accurate and reliable extraction of features such as PRNU or noiseprint. This means that correlation results based on these features may be less reliable.";

  const mixedCase = referenceResults.some((r) => r.detected) &&
                     referenceResults.some((r) => !r.detected);

  const finalExplanation =
    mixedCase
      ? `${explanationNoNUA}\n\n${explanationDetectedNUA}`
      : mainResult?.detected
      ? explanationDetectedNUA
      : explanationNoNUA;

  return (
    <div className="mt-6 mb-4 bg-gray-900 border border-teal-800 rounded-xl shadow-lg p-5 w-full relative">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2 mr-1">
          NUA Report
        </h2>

        {/* 🔹 Ikonka informacji */}
        <button
          onClick={() => setShowInfo(!showInfo)}
          className="text-teal-400 hover:text-teal-200 transition"
        >
          <Info size={20} />
        </button>
      </div>

      {/* 🔹 POPUP z wyjaśnieniem */}
      {showInfo && (
        <div className="absolute right-5 top-14 bg-gray-800 border border-teal-700 p-4 rounded-lg w-80 shadow-xl z-50 text-sm text-gray-300 whitespace-pre-wrap">
          {finalExplanation}
        </div>
      )}

      {loading ? (
        <div className="flex flex-col items-center py-6">
          <div className="w-6 h-6 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm mt-3 text-gray-400">Analyzing NUA patterns...</p>
        </div>
      ) : error ? (
        <p className="text-sm text-red-400 text-center py-3">⚠️ {error}</p>
      ) : mainResult ? (
        <>
          {/* 🔹 Wynik zdjęcia dowodowego */}
          <div
            className={`w-full text-center py-4 rounded-lg border transition-all duration-300 ${
              mainResult.detected
                ? "bg-red-900/30 border-red-700"
                : "bg-green-900/30 border-green-700"
            }`}
          >
            <p className="text-base font-medium">
              Evidence Image:{" "}
              <span className="font-semibold text-gray-300">
                {imageFile?.name ?? "Unnamed image"}
              </span>
            </p>
            <p className="text-sm mt-2">
              Result:{" "}
              <span
                className={`font-semibold ${
                  mainResult.detected ? "text-red-400" : "text-green-400"
                }`}
              >
                {mainResult.detected ? "NUA Detected" : "No NUA Detected"}
              </span>
            </p>
            <p className="text-sm text-gray-400 mt-1">
              Confidence: {(mainResult.confidence).toFixed(2)}%
            </p>
          </div>

          {/* 🔹 Wyniki zdjęć referencyjnych */}
          {referenceImages.length > 0 && (
            <div
              className={`mt-5 w-full text-center py-4 rounded-lg border transition-all duration-300 ${refBgColor}`}
            >
              {referenceResults.length > 0 ? (
                <>
                  <p className="text-base font-medium text-teal-300">
                    Reference Set Analysis
                  </p>
                  <p className="text-sm text-gray-400 mt-1">
                    NUA detected in {detectedCount}/{referenceResults.length} reference images
                  </p>
                </>
              ) : (
                <p className="text-sm text-gray-400">
                  Analyzing reference images...
                </p>
              )}
            </div>
          )}
        </>
      ) : (
        <p className="text-sm text-gray-400 text-center py-3">No data available</p>
      )}
    </div>
  );
};

export default NUAReport;
