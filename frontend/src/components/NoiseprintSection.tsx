import React, { useState } from "react";
import { Loader2, Info } from "lucide-react";

const noiseprintInterpretation = [
  {
    range: "< 0.20",
    label: "No similarity",
    desc: "The noiseprints are completely different — the images originate from different sensors.",
  },
  {
    range: "0.20 – 0.45",
    label: "Low similarity",
    desc: "Some minor shared characteristics are visible, but there is no evidence of device consistency.",
  },
  {
    range: "0.45 – 0.65",
    label: "Moderate correlation",
    desc: "The noiseprint shows noticeable similarity, but it is not strong enough to draw a definitive conclusion.",
  },
  {
    range: "0.65 – 0.80",
    label: "Strong correlation",
    desc: "The images likely come from the same device — interpretation should still be cautious.",
  },
  {
    range: "0.80 – 0.90",
    label: "Very strong match",
    desc: "The noiseprints are nearly identical — high reliability of device consistency.",
  },
  {
    range: "> 0.90",
    label: "Practically certain match",
    desc: "Extremely strong fit — the images originate from the same sensor.",
  },
];

interface NoiseprintSectionProps {
  evidenceImage: File;
  referenceImages: File[];
}

const NoiseprintSection: React.FC<NoiseprintSectionProps> = ({
  evidenceImage,
  referenceImages,
}) => {
  const [loading, setLoading] = useState(false);
  const [embeddingSim, setEmbeddingSim] = useState<number | null>(null);
  const [evidenceNoiseprint, setEvidenceNoiseprint] = useState<string | null>(null);
  const [meanNoiseprint, setMeanNoiseprint] = useState<string | null>(null);
  const [statsEvidence, setStatsEvidence] = useState<any | null>(null);
  const [statsMean, setStatsMean] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleNoiseprintCompare = async () => {
    if (!evidenceImage || referenceImages.length === 0) return;
    setLoading(true);
    setError(null);
    setEmbeddingSim(null);
    try {
      const formData = new FormData();
      formData.append("evidence", evidenceImage);
      referenceImages.forEach((file) => formData.append("references", file));

      const response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/compare`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setEvidenceNoiseprint(
        data.evidence_noiseprint
          ? `data:image/png;base64,${data.evidence_noiseprint}`
          : null
      );

      setMeanNoiseprint(
        data.mean_reference_noiseprint
          ? `data:image/png;base64,${data.mean_reference_noiseprint}`
          : null
      );

      setStatsEvidence(data.stats_evidence ?? null);
      setStatsMean(data.stats_mean ?? null);

      // embedding similarity
      if (referenceImages.length > 0) {
        const formDataEmb = new FormData();
        formDataEmb.append("evidence", evidenceImage);
        formDataEmb.append("reference", referenceImages[0]);

        const embResponse = await fetch(
          `${process.env.REACT_APP_API_BASE}/noiseprint/compare_embedding`,
          {
            method: "POST",
            body: formDataEmb,
          }
        );

        if (embResponse.ok) {
          const embData = await embResponse.json();
          setEmbeddingSim(embData.similarity_score);
        } else {
          setEmbeddingSim(null);
        }
      }
    } catch (err) {
      console.error(err);
      setError("Error computing Noiseprint embedding similarity.");
    } finally {
      setLoading(false);
    }
  };

  // --- NEW STATE ---
  const [showInterpretation, setShowInterpretation] = useState(false);

  const getRangeColor = (range: string) => {
    if (range.includes("< 0.20")) return "text-red-400";
    if (range.includes("0.20")) return "text-orange-400";
    if (range.includes("0.45")) return "text-yellow-400";
    if (range.includes("0.65")) return "text-lime-400";
    if (range.includes("0.80")) return "text-green-400";
    if (range.includes("> 0.90")) return "text-emerald-400";
    return "text-gray-300";
  };

  // Determine color class for a numeric similarity score
  const getSimColor = (score: number) => {
    if (score < 0.20) return "text-red-400";
    if (score < 0.45) return "text-orange-400";
    if (score < 0.65) return "text-yellow-400";
    if (score < 0.80) return "text-lime-400";
    if (score < 0.90) return "text-green-400";
    return "text-emerald-400";
  };


  return (
    <div className="mt-6 bg-gray-900/80 rounded-2xl p-8 border border-green-800 shadow-lg shadow-green-900/30 backdrop-blur-md relative">
      {loading && (
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm flex flex-col items-center justify-center z-10 rounded-2xl">
          <Loader2 className="h-10 w-10 text-green-400 animate-spin mb-3" />
          <p className="text-green-300 text-sm tracking-widest">
            Generating noiseprints...
          </p>
        </div>
      )}

      <h3 className="text-2xl font-semibold mb-6 text-green-400 tracking-wide">
        Noiseprint Embedding Similarity
      </h3>

      <button
        onClick={handleNoiseprintCompare}
        disabled={loading}
        className={`${
          loading
            ? "bg-gray-700 cursor-not-allowed opacity-70"
            : "bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500 shadow-lg shadow-green-800/30 hover:shadow-green-600/40"
        } text-black font-extrabold px-8 py-3 rounded-xl text-lg transition-all transform hover:scale-[1.03]`}
      >
        {loading ? "Processing..." : "Generate Noiseprints"}
      </button>

      {error && <p className="text-red-400 text-center mb-4 mt-4">{error}</p>}

      {!loading && (evidenceNoiseprint || meanNoiseprint) && (
        <div className="mt-10">
          <div className="flex flex-col md:flex-row justify-between items-start gap-6">
            <div className="flex-1 text-center">
              <h4 className="text-gray-300 mb-3 font-semibold">Evidence Noiseprint</h4>
              {evidenceNoiseprint ? (
                <img
                  src={evidenceNoiseprint}
                  alt="Evidence Noiseprint"
                  className="rounded-lg border border-gray-700 mx-auto max-h-72 object-contain"
                />
              ) : (
                <p className="text-gray-500 italic text-sm">No noiseprint generated.</p>
              )}
              {statsEvidence && (
                <div className="mt-2 text-sm text-gray-400 space-y-1">
                  <p>Mean: {statsEvidence.mean?.toFixed(4)}</p>
                  <p>Std: {statsEvidence.std?.toFixed(4)}</p>
                  <p>Energy: {statsEvidence.energy?.toFixed(4)}</p>
                  <p>Entropy: {statsEvidence.entropy?.toFixed(4)}</p>
                </div>
              )}
            </div>

            <div className="flex-1 text-center">
              <h4 className="text-gray-300 mb-3 font-semibold">Mean Reference Noiseprint</h4>
              {meanNoiseprint ? (
                <img
                  src={meanNoiseprint}
                  alt="Mean Reference Noiseprint"
                  className="rounded-lg border border-gray-700 mx-auto max-h-72 object-contain"
                />
              ) : (
                <p className="text-gray-500 italic text-sm">
                  No mean reference noiseprint available.
                </p>
              )}
              {statsMean && (
                <div className="mt-2 text-sm text-gray-400 space-y-1">
                  <p>Mean: {statsMean.mean?.toFixed(4)}</p>
                  <p>Std: {statsMean.std?.toFixed(4)}</p>
                  <p>Energy: {statsMean.energy?.toFixed(4)}</p>
                  <p>Entropy: {statsMean.entropy?.toFixed(4)}</p>
                </div>
              )}
            </div>
          </div>

          {embeddingSim !== null && (
            <div className="text-center mt-10 space-y-3">
              <div className="flex items-center justify-center gap-2">
                <p className={`text-5xl font-extrabold ${getSimColor(embeddingSim)} drop-shadow-md`}>
                  {embeddingSim.toFixed(3)}
                </p>

                {/* --- Info icon toggle --- */}
                <button
                  onClick={() => setShowInterpretation(!showInterpretation)}
                  className="p-1 hover:bg-gray-700 rounded-full transition"
                >
                  <Info className="text-blue-300" size={26} />
                </button>
              </div>

              <p className="text-gray-400 text-lg">Embedding Similarity Score</p>
              <p className="text-gray-500 mt-2 text-sm italic">
                Higher values indicate stronger evidence of same device origin.
              </p>

              {/* --- INTERPRETATION BLOCK (togglable) --- */}
              {showInterpretation && (
                <div className="mt-6 bg-gray-800/70 p-5 rounded-xl border border-gray-700 text-left">
                  <div className="flex items-center gap-2 mb-3">
                    <Info className="text-blue-300" size={20} />
                    <p className="text-blue-300 font-semibold tracking-wide">
                      Interpretation of correlation score
                    </p>
                  </div>

                  <ul className="space-y-3">
                    {noiseprintInterpretation.map((item) => (
                      <li key={item.range}>
                        <p className="text-gray-300">
                          <span className={`font-semibold ${getRangeColor(item.range)}`}>
                            {item.range} — {item.label}
                          </span>
                          <br />
                          <span className="text-gray-400 text-sm">{item.desc}</span>
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

        </div>
      )}
    </div>
  );
};

export default NoiseprintSection;
