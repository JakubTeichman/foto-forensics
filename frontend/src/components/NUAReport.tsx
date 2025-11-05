import React, { useEffect, useState, useRef } from "react";

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

  // 🔹 Trzymamy poprzedni hash, żeby wykrywać realne zmiany
  const lastHash = useRef<string | null>(null);

  useEffect(() => {
    if (!imageFile) return;

    // 🔹 Tworzymy hash (unikalny identyfikator kombinacji plików)
    const hash =
      imageFile.name +
      "-" +
      imageFile.size +
      "-" +
      referenceImages.map((f) => f.name + "-" + f.size).join(",");

    // Jeśli hash się nie zmienił → nie analizujemy ponownie
    if (hash === lastHash.current) return;
    lastHash.current = hash;

    const analyzeNUA = async () => {
      setLoading(true);
      setError(null);
      setMainResult(null);
      setReferenceResults([]);

      try {
        // 🔹 Analiza głównego (dowodowego) zdjęcia
        const mainFormData = new FormData();
        mainFormData.append("file", imageFile);

        const mainResponse = await fetch("http://localhost:5000/analyze/nua", {
          method: "POST",
          body: mainFormData,
        });

        if (!mainResponse.ok) {
          throw new Error("Failed to analyze the evidence image.");
        }

        const mainData = await mainResponse.json();
        setMainResult(mainData);

        // 🔹 Analiza zdjęć referencyjnych — tylko jeśli istnieją
        if (referenceImages.length > 0) {
          const results: NUAResult[] = [];

          for (const refImg of referenceImages) {
            const formData = new FormData();
            formData.append("file", refImg);

            const res = await fetch("http://localhost:5000/analyze/nua", {
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

  return (
    <div className="mt-6 mb-4 bg-gray-900 border border-teal-800 rounded-xl shadow-lg p-5 w-full">
      <h2 className="text-lg font-semibold mb-3 text-center text-teal-400">
        NUA Report
      </h2>

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
              Confidence: {(mainResult.confidence * 100).toFixed(2)}%
            </p>
          </div>

          {/* 🔹 Wyniki zdjęć referencyjnych */}
          {referenceImages.length > 0 && (
            <div className="mt-5 w-full text-center py-4 rounded-lg border border-teal-700 bg-gray-800/40">
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
        <p className="text-sm text-gray-400 text-center py-3">
          No data available
        </p>
      )}
    </div>
  );
};

export default NUAReport;
