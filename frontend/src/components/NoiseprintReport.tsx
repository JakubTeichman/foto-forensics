import React, { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

interface NoiseprintReportProps {
  imageFile: File | null;
}

const NoiseprintReport: React.FC<NoiseprintReportProps> = ({ imageFile }) => {
  const [noiseprint, setNoiseprint] = useState<string | null>(null);
  const [stats, setStats] = useState<{ [key: string]: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 🆕 stan dla porównania z bazą danych
  const [dbLoading, setDbLoading] = useState(false);
  const [dbMatches, setDbMatches] = useState<any[] | null>(null);
  const [bestMatch, setBestMatch] = useState<any | null>(null);

  const handleGenerate = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/generate`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Failed to generate noiseprint (status: ${response.status})`);
      }

      const data = await response.json();
      setNoiseprint(`data:image/png;base64,${data.noiseprint}`);
      setStats(data.stats);
    } catch (err: any) {
      console.error("Error generating noiseprint:", err);
      setError(err.message || "Unexpected error occurred while generating noiseprint.");
    } finally {
      setLoading(false);
    }
  };

  const handleCompareWithDB = async () => {
    if (!imageFile) return;
    setDbLoading(true);
    setDbMatches(null);
    setBestMatch(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("evidence", imageFile);

      const response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/compare_with_db`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setDbMatches(data.matches || []);
      setBestMatch(data.best_match || null);
    } catch (err) {
      console.error(err);
      setError("Error comparing noiseprint with database.");
    } finally {
      setDbLoading(false);
    }
  };

  const getSimColor = (val: number) => {
    if (val < 0.4) return "text-red-500";
    if (val < 0.6) return "text-orange-400";
    if (val < 0.75) return "text-yellow-400";
    if (val < 0.9) return "text-lime-400";
    return "text-green-400";
  };

  useEffect(() => {
    if (!imageFile) {
      setNoiseprint(null);
      setStats(null);
      setError(null);
      setDbMatches(null);
      setBestMatch(null);
    }
  }, [imageFile]);

  return (
    <div className="bg-gray-900 border border-emerald-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-6 text-emerald-400 flex items-center gap-2 text-center">
        Noiseprint Report
      </h3>

      {!imageFile ? (
        <p className="text-gray-400 italic text-center">No image selected.</p>
      ) : (
        <>
          {/* 🟩 Wyśrodkowane przyciski */}
          <div className="flex justify-center gap-6 mb-6">
            <button
              onClick={handleGenerate}
              disabled={loading}
              className="px-6 py-3 bg-gradient-to-r from-teal-500 to-emerald-400 hover:from-teal-600 hover:to-emerald-500 disabled:bg-gray-700 text-black font-bold rounded-xl shadow-lg transition-all transform hover:scale-[1.03]"
            >
              {loading ? "Generating..." : "Generate Noiseprint"}
            </button>

            <button
              onClick={handleCompareWithDB}
              disabled={dbLoading}
              className="px-6 py-3 bg-gradient-to-r from-emerald-600 to-green-400 hover:from-emerald-700 hover:to-green-500 disabled:bg-gray-700 text-black font-bold rounded-xl shadow-lg transition-all transform hover:scale-[1.03]"
            >
              {dbLoading ? "Checking..." : "Compare with Database"}
            </button>
          </div>

          {loading && (
            <div className="flex items-center justify-center gap-2 text-gray-400 mt-3">
              <Loader2 className="h-5 w-5 animate-spin text-emerald-400" />
              <span>Generating noiseprint...</span>
            </div>
          )}

          {error && <p className="text-red-400 text-sm text-center mt-3">⚠️ {error}</p>}

          {noiseprint && !loading && (
            <div className="flex flex-col md:flex-row gap-6 mt-6">
              <div className="flex-1">
                <h4 className="text-emerald-400 mb-2 font-medium">Noiseprint Image:</h4>
                <img
                  src={noiseprint}
                  alt="Noiseprint visualization"
                  className="rounded-lg border border-gray-700 shadow-md"
                />
              </div>

              {stats && (
                <div className="flex-1 bg-gray-800/60 rounded-lg p-4 text-gray-300">
                  <h4 className="text-emerald-400 font-medium mb-3">Noiseprint Parameters:</h4>
                  <ul className="space-y-1 text-sm">
                    {Object.entries(stats).map(([key, value]) => (
                      <li key={key}>
                        <span className="text-emerald-400 font-semibold capitalize">{key}:</span>{" "}
                        {value.toFixed(4)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* 🟢 Sekcja wyników z bazy danych */}
          {dbMatches && (
            <div className="mt-12 border-t border-gray-700 pt-6">
              <h4 className="text-xl font-semibold text-emerald-400 mb-4">
                Database Comparison Results
              </h4>

              {bestMatch ? (
                <div className="bg-gray-800/50 rounded-lg p-4 border border-emerald-700 mb-4">
                  <p className="text-emerald-300 font-bold mb-2">🏆 Best Match:</p>
                  <p className="text-gray-200">
                    {bestMatch.manufacturer} {bestMatch.model}
                  </p>
                  <p className="text-gray-400 text-sm">
                    Similarity:{" "}
                    <span className={`font-semibold ${getSimColor(bestMatch.similarity)}`}>
                      {bestMatch.similarity}
                    </span>
                  </p>
                </div>
              ) : (
                <p className="text-gray-500 italic">No similar reference found.</p>
              )}

              <div className="overflow-x-auto max-h-64 mt-4">
                <table className="w-full text-sm text-gray-300 border border-gray-700 rounded-lg">
                  <thead className="bg-gray-800 text-gray-400 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left">ID</th>
                      <th className="px-3 py-2 text-left">Manufacturer</th>
                      <th className="px-3 py-2 text-left">Model</th>
                      <th className="px-3 py-2 text-left">Images</th>
                      <th className="px-3 py-2 text-left">Similarity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dbMatches.map((m) => (
                      <tr
                        key={m.id}
                        className="border-t border-gray-700 hover:bg-gray-800/70"
                      >
                        <td className="px-3 py-2">{m.id}</td>
                        <td className="px-3 py-2">{m.manufacturer}</td>
                        <td className="px-3 py-2">{m.model}</td>
                        <td className="px-3 py-2">{m.num_images}</td>
                        <td className={`px-3 py-2 font-semibold ${getSimColor(m.similarity)}`}>
                          {m.similarity}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default NoiseprintReport;
