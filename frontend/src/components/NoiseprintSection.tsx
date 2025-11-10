import React, { useState } from 'react';

interface NoiseprintSectionProps {
  evidenceImage: File;
  referenceImages: File[];
  noiseData?: any;
  noiseError?: string | null;
}

const NoiseprintSection: React.FC<NoiseprintSectionProps> = ({ evidenceImage, referenceImages }) => {
  const [loading, setLoading] = useState(false);
  const [corr, setCorr] = useState<number | null>(null);
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
    try {
      // === 1️⃣ Klasyczne porównanie ===
      const formData = new FormData();
      formData.append('evidence', evidenceImage);
      referenceImages.forEach((file) => formData.append('references', file));

      const response = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/compare`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();

      setCorr(data.mean_correlation ?? data.peak_correlation ?? null);
      setEvidenceNoiseprint(`data:image/png;base64,${data.evidence_noiseprint}`);
      setMeanNoiseprint(`data:image/png;base64,${data.mean_reference_noiseprint ?? ''}`);
      setStatsEvidence(data.stats_evidence);
      setStatsMean(data.stats_mean ?? null);

      // === 2️⃣ Embedding similarity (dla 1. referencji) ===
      if (referenceImages.length > 0) {
        const formDataEmb = new FormData();
        formDataEmb.append('evidence', evidenceImage);
        formDataEmb.append('reference', referenceImages[0]); // porównanie z pierwszym plikiem

        const embResponse = await fetch(`${process.env.REACT_APP_API_BASE}/noiseprint/compare_embedding`, {
          method: 'POST',
          body: formDataEmb,
        });

        if (embResponse.ok) {
          const embData = await embResponse.json();
          setEmbeddingSim(embData.similarity_score);
        } else {
          setEmbeddingSim(null);
        }
      }
    } catch (err) {
      console.error(err);
      setError('Error computing Noiseprint correlation.');
    } finally {
      setLoading(false);
    }
  };

  const getCorrColor = (val: number) => {
    if (val < 0.3) return 'text-red-500';
    if (val < 0.5) return 'text-orange-400';
    if (val < 0.7) return 'text-yellow-400';
    if (val < 0.85) return 'text-lime-400';
    return 'text-green-400';
  };

  const getSimColor = (val: number) => {
    if (val < 0.4) return 'text-red-500';
    if (val < 0.6) return 'text-orange-400';
    if (val < 0.75) return 'text-yellow-400';
    if (val < 0.9) return 'text-lime-400';
    return 'text-green-400';
  };

  return (
    <div className="mt-12 bg-gray-900/80 rounded-2xl p-8 border border-green-800 shadow-lg shadow-green-900/30 backdrop-blur-md">
      <h3 className="text-2xl font-semibold mb-6 text-green-400 tracking-wide">
        Noiseprint Comparison Report + Embedding Similarity
      </h3>

      {/* 🔘 Przycisk */}
      <div className="flex justify-left mb-8">
        <button
          onClick={handleNoiseprintCompare}
          disabled={loading}
          className={`${
            loading
              ? 'bg-gray-700 cursor-not-allowed opacity-70'
              : 'bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500 shadow-lg shadow-green-800/30 hover:shadow-green-600/40'
          } text-black font-extrabold px-8 py-3 rounded-xl text-lg transition-all transform hover:scale-[1.03]`}
        >
          {loading ? 'Processing...' : 'Generate Noiseprints'}
        </button>
      </div>

      {error && <p className="text-red-400 text-center mb-4">{error}</p>}

      {(corr !== null || embeddingSim !== null) && (
        <div className="mt-10">
          <div className="flex flex-col md:flex-row justify-between items-start gap-6">
            {/* Evidence */}
            <div className="flex-1 text-center">
              <h4 className="text-gray-300 mb-3 font-semibold">Evidence Noiseprint</h4>
              {evidenceNoiseprint && (
                <img
                  src={evidenceNoiseprint}
                  alt="Evidence Noiseprint"
                  className="rounded-lg border border-gray-700 mx-auto max-h-72 object-contain"
                />
              )}
              {statsEvidence && (
                <div className="mt-2 text-sm text-gray-400 space-y-1">
                  <p>Mean: {statsEvidence.mean.toFixed(4)}</p>
                  <p>Std: {statsEvidence.std.toFixed(4)}</p>
                  <p>Energy: {statsEvidence.energy.toFixed(4)}</p>
                  <p>Entropy: {statsEvidence.entropy.toFixed(4)}</p>
                </div>
              )}
            </div>

            {/* Reference */}
            <div className="flex-1 text-center">
              <h4 className="text-gray-300 mb-3 font-semibold">Mean Reference Noiseprint</h4>
              {meanNoiseprint && (
                <img
                  src={meanNoiseprint}
                  alt="Mean Reference Noiseprint"
                  className="rounded-lg border border-gray-700 mx-auto max-h-72 object-contain"
                />
              )}
              {statsMean && (
                <div className="mt-2 text-sm text-gray-400 space-y-1">
                  <p>Mean: {statsMean.mean.toFixed(4)}</p>
                  <p>Std: {statsMean.std.toFixed(4)}</p>
                  <p>Energy: {statsMean.energy.toFixed(4)}</p>
                  <p>Entropy: {statsMean.entropy.toFixed(4)}</p>
                </div>
              )}
            </div>
          </div>

          {/* Wyniki */}
          <div className="text-center mt-10 space-y-4">
            {corr !== null && (
              <p className={`text-4xl font-bold ${getCorrColor(corr)} drop-shadow-md`}>
                {corr.toFixed(3)} <span className="text-gray-400 text-lg">Correlation</span>
              </p>
            )}
            {embeddingSim !== null && (
              <p className={`text-3xl font-bold ${getSimColor(embeddingSim)} drop-shadow-sm`}>
                {embeddingSim.toFixed(3)}{' '}
                <span className="text-gray-400 text-lg">Embedding Similarity</span>
              </p>
            )}
            <p className="text-gray-400 mt-2 text-sm italic">
              Higher values indicate stronger evidence of same device origin.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default NoiseprintSection;
