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

  const [dbLoading, setDbLoading] = useState(false);
  const [dbMatches, setDbMatches] = useState<any[] | null>(null);
  const [bestMatch, setBestMatch] = useState<any | null>(null);

  const handleGenerate = async () => {
    if (!imageFile) return null;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      // Wprowadzam pętlę retry z wykładniczym czasem oczekiwania dla stabilności
      const maxRetries = 3;
      let response: Response | undefined; // JAWNA DEKLARACJA TYPU
      for (let i = 0; i < maxRetries; i++) {
          try {
              response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/generate`, {
                  method: "POST",
                  body: formData,
              });
              if (response.ok) break;
          } catch (e) {
              if (i < maxRetries - 1) {
                  const delay = Math.pow(2, i) * 1000;
                  await new Promise(resolve => setTimeout(resolve, delay));
              } else {
                  throw new Error(`Failed to generate noiseprint after ${maxRetries} attempts.`);
              }
          }
      }

      if (!response!.ok) {
        throw new Error(`Failed to generate noiseprint (status: ${response!.status})`);
      }

      const data = await response!.json();

      const base64 = `data:image/png;base64,${data.noiseprint}`;
      setNoiseprint(base64);
      setStats(data.stats);

      return base64;       // <-- ZWRACAMY KONKRETNĄ WARTOŚĆ
    } catch (err: any) {
      console.error("Error generating noiseprint:", err);
      setError(err.message || "Unexpected error occurred while generating noiseprint.");
      return null;
    } finally {
      setLoading(false);
    }
  };


  const handleCompareWithDB = async () => {
    if (!imageFile) return;

    let np = noiseprint;

    if (!np) {
      // Jeśli noiseprint nie jest jeszcze wygenerowany, wygeneruj go
      np = await handleGenerate();
      if (!np) return;  // <-- Zabezpieczenie przed błędem generowania
    }

    setDbLoading(true);
    setDbMatches(null);
    setBestMatch(null);
    setError(null);

    try {
      const formData = new FormData();
      // Konwersja base64 na Blob do przesłania
      const blob = await fetch(np).then((res) => res.blob());
      formData.append("noiseprint", blob, "noiseprint.png");

      // Wprowadzam pętlę retry z wykładniczym czasem oczekiwania dla stabilności
      const maxRetries = 3;
      let response: Response | undefined; // JAWNA DEKLARACJA TYPU
      for (let i = 0; i < maxRetries; i++) {
          try {
              response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/compare_with_db`, {
                  method: "POST",
                  body: formData,
              });
              if (response.ok) break;
          } catch (e) {
              if (i < maxRetries - 1) {
                  const delay = Math.pow(2, i) * 1000;
                  await new Promise(resolve => setTimeout(resolve, delay));
              } else {
                  throw new Error(`Failed to compare with database after ${maxRetries} attempts.`);
              }
          }
      }

      if (!response!.ok) throw new Error(`HTTP ${response!.status}`);

      const data = await response!.json();
      setDbMatches(data.matches || []);
      setBestMatch(data.best_match || null);
      
    } catch (err) {
      console.error(err);
      setError("Error comparing noiseprint with database.");
    } finally {
      setDbLoading(false);
    }
  };

  /**
   * Zaktualizowana funkcja do kolorowania na skali 0-100 (Nowy format: transformed_score).
   * Stara funkcja operowała na skali 0-1.
   */
  const getSimColor = (val: number) => {
    if (val < 50) return "text-red-400"; // Poniżej losowego szumu
    if (val < 60) return "text-orange-400"; // Niskie/średnie
    if (val < 75) return "text-yellow-400"; // Średnie/dobre
    if (val < 90) return "text-lime-400"; // Wysokie
    return "text-green-400"; // Bardzo wysokie
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

          {(loading || dbLoading) && (
            <div className="flex items-center justify-center gap-2 text-gray-400 mt-3">
              <Loader2 className="h-5 w-5 animate-spin text-emerald-400" />
              <span>{loading ? "Generating noiseprint..." : "Comparing with database..."}</span>
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

          {dbMatches && (
            <div className="mt-12 border-t border-gray-700 pt-6">
              <h4 className="text-xl font-semibold text-emerald-400 mb-4">
                Database Comparison Results
              </h4>

              {bestMatch ? (
                <div className="bg-gray-800/50 rounded-lg p-4 border border-emerald-700 mb-4">
                  <p className="text-emerald-300 font-bold mb-2">Best Match:</p>
                  <p className="text-gray-200">
                    {bestMatch.manufacturer} {bestMatch.model}
                  </p>
                  <p className="text-gray-400 text-sm">
                    Similarity (0-100%):{" "}
                    {/* ZMIANA 1: Użycie transformed_score zamiast similarity */}
                    <span className={`font-semibold ${getSimColor(bestMatch.transformed_score)}`}>
                      {bestMatch.transformed_score.toFixed(2)}%
                    </span>
                    <span className="ml-3 text-xs text-gray-500">
                      (Raw Corr: {bestMatch.raw_correlation.toFixed(4)})
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
                      <th className="px-3 py-2 text-left">Similarity (0-100%)</th>
                      <th className="px-3 py-2 text-left">Raw Corr. (-1 do 1)</th>
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
                        {/* ZMIANA 2: Użycie transformed_score zamiast similarity */}
                        <td className={`px-3 py-2 font-semibold ${getSimColor(m.transformed_score)}`}>
                          {m.transformed_score.toFixed(2)}%
                        </td>
                        <td className="px-3 py-2 text-gray-400">
                          {m.raw_correlation.toFixed(4)}
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