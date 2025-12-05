import React, { useState } from "react";
import { Loader2, Info } from "lucide-react";

// Interpretation table for NCC correlation (transformed to 0-100 scale)
// Note that the original interpretation (0-1) is still used for color determination
const noiseprintInterpretation = [
  {
    range: "< 0.40 (NCC)", // NCC ~0.20-0.45, score ~10-40
    label: "No Similarity",
    desc: "The noiseprints are completely different — the images originate from different sensors.",
    color: "text-red-400",
  },
  {
    range: "0.40 – 0.60 (NCC)", // NCC ~0.45-0.65, score ~40-60
    label: "Low Correlation",
    desc: "Some minor common features are visible, but there is no evidence of device consistency.",
    color: "text-orange-400",
  },
  {
    range: "0.60 – 0.70 (NCC)", // NCC ~0.65-0.80, score ~60-80
    label: "Moderate Correlation",
    desc: "The noiseprint shows noticeable similarity, but it is not strong enough to draw a definitive conclusion.",
    color: "text-yellow-400",
  },
  {
    range: "0.70 – 0.85 (NCC)", // NCC ~0.80-0.90, score ~80-90
    label: "Strong Correlation",
    desc: "The images likely come from the same device — interpretation should be cautious.",
    color: "text-lime-400",
  },
  {
    range: "> 0.85 (NCC)", // NCC > 0.90, score > 90
    label: "Very Strong Match",
    desc: "The noiseprints are almost identical — high reliability of device consistency.",
    color: "text-emerald-400",
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
  // Updated state for the averaged correlation score from multiple references (0-100)
  const [meanScore, setMeanScore] = useState<number | null>(null); 
  // Updated state for the correlation score with the first reference (0-100)
  const [transformedEmbeddingScore, setTransformedEmbeddingScore] = useState<number | null>(null); 
  
  const [evidenceNoiseprint, setEvidenceNoiseprint] = useState<string | null>(null);
  // Updated state name for the averaged noiseprint
  const [meanReferenceNoiseprint, setMeanReferenceNoiseprint] = useState<string | null>(null);
  
  const [statsEvidence, setStatsEvidence] = useState<any | null>(null);
  const [statsMean, setStatsMean] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleNoiseprintCompare = async () => {
    if (!evidenceImage || referenceImages.length === 0) {
      setError("Please provide an evidence image and at least one reference image.");
      return;
    }
    
    setLoading(true);
    setError(null);
    setMeanScore(null);
    setTransformedEmbeddingScore(null);
    setMeanReferenceNoiseprint(null);
    setStatsMean(null);
    
    try {
      // Step 1: Comparison with multiple references (setting mean_score and mean_reference_noiseprint)
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

      // Fix: Using the new backend field for the averaged noiseprint
      setMeanReferenceNoiseprint(
        data.mean_reference_noiseprint
          ? `data:image/png;base64,${data.mean_reference_noiseprint}`
          : null
      );

      setStatsEvidence(data.stats_evidence ?? null);
      setStatsMean(data.stats_mean ?? null);
      setMeanScore(data.mean_score ?? null); // Main score (0-100)

      // Step 2: Comparison of "embedding similarity" (correlation evidence vs first reference)
      if (referenceImages.length > 0) {
        const formDataEmb = new FormData();
        formDataEmb.append("evidence", evidenceImage);
        // We only use the first reference for "embedding" (as in the original logic)
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
          // Fix: Using the new backend field: transformed_similarity_score (0-100 scale)
          const similarityScore = typeof embData.transformed_similarity_score === 'number' ? embData.transformed_similarity_score : null;
          setTransformedEmbeddingScore(similarityScore);
        } else {
          setTransformedEmbeddingScore(null);
        }
      }
    } catch (err) {
      console.error(err);
      setError("Error processing Noiseprints. Please check if the files are correct.");
    } finally {
      setLoading(false);
    }
  };

  const [showInterpretation, setShowInterpretation] = useState(false);

  // Function to determine color based on the correlation score (0-100)
  const getScoreColor = (score: number | null) => {
    if (score === null) return "text-gray-500";
    if (score < 40) return "text-red-400"; // < 0.40 NCC
    if (score < 60) return "text-orange-400"; // 0.40 - 0.60 NCC
    if (score < 70) return "text-yellow-400"; // 0.60 - 0.70 NCC
    if (score < 85) return "text-lime-400"; // 0.70 - 0.85 NCC
    return "text-emerald-400"; // > 0.85 NCC
  };

  const CorrelationResult: React.FC<{ score: number | null, title: string }> = ({ score, title }) => (
    <div className="flex flex-col items-center justify-center p-4 bg-gray-800/50 rounded-lg shadow-inner">
      <p className="text-gray-400 text-sm mb-1">{title}</p>
      <div className={`text-4xl font-extrabold ${getScoreColor(score)}`}>
        {score !== null ? `${score.toFixed(2)}%` : "N/A"}
      </div>
      <p className="text-xs text-gray-500 mt-2">
        {score !== null ? (score < 40 ? "No Consistency" : score >= 85 ? "High Confidence" : "Requires Analysis") : "No Data"}
      </p>
    </div>
  );

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
        Noiseprint and Correlation Analysis
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
        {loading ? "Processing..." : "Run Noiseprint Analysis"}
      </button>

      {error && <p className="text-red-400 text-center mb-4 mt-4">{error}</p>}

      {/* Fix: Logic for displaying results based on meanScore or transformedEmbeddingScore */}
      {(meanScore !== null || transformedEmbeddingScore !== null) && (
        <div className="mt-8">
          <h4 className="text-xl font-medium text-green-300 mb-4">Correlation Results (0-100 Scale)</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <CorrelationResult 
                score={meanScore} 
                title={`AVERAGE CORRELATION (Evidence vs ${referenceImages.length} References)`} 
            />
            {referenceImages.length > 0 && (
                <CorrelationResult 
                    score={transformedEmbeddingScore} 
                    title={`CORRELATION WITH FIRST REFERENCE`} 
                />
            )}
          </div>
        </div>
      )}

      {/* Displaying Noiseprints */}
      <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-8">
        {evidenceNoiseprint && (
          <div className="flex flex-col items-center bg-gray-800/50 p-4 rounded-xl">
            <h4 className="text-lg font-medium text-white mb-3">Evidence Noiseprint</h4>
            <img 
              src={evidenceNoiseprint} 
              alt="Evidence Noiseprint" 
              className="w-full h-auto rounded-lg shadow-xl border border-gray-700 max-w-lg" 
            />
            {statsEvidence && (
              <p className="text-xs text-gray-400 mt-2">Mean: {statsEvidence.mean.toFixed(4)}, Std Dev: {statsEvidence.std.toFixed(4)}</p>
            )}
          </div>
        )}

        {/* Fix: Displaying the AVERAGED Reference Noiseprint */}
        {meanReferenceNoiseprint && (
          <div className="flex flex-col items-center bg-gray-800/50 p-4 rounded-xl">
            <h4 className="text-lg font-medium text-white mb-3">Averaged Reference Noiseprint</h4>
            <img 
              src={meanReferenceNoiseprint} 
              alt="Mean Reference Noiseprint" 
              className="w-full h-auto rounded-lg shadow-xl border border-gray-700 max-w-lg" 
            />
            {statsMean && (
              <p className="text-xs text-gray-400 mt-2">Mean: {statsMean.mean.toFixed(4)}, Std Dev: {statsMean.std.toFixed(4)}</p>
            )}
          </div>
        )}
      </div>

      {/* Interpretation Section */}
      {(meanScore !== null || transformedEmbeddingScore !== null) && (
        <div className="mt-8 border-t border-gray-700 pt-6">
          <button 
            onClick={() => setShowInterpretation(!showInterpretation)}
            className="flex items-center text-green-400 hover:text-green-300 transition-colors"
          >
            <Info className="h-5 w-5 mr-2" />
            <span className="font-semibold">
              {showInterpretation ? "Hide" : "Show"} Correlation Interpretation Table (NCC)
            </span>
          </button>
          
          {showInterpretation && (
            <div className="mt-4 space-y-3 p-4 bg-gray-800/50 rounded-lg">
              {noiseprintInterpretation.map((item, index) => (
                <div key={index} className="flex flex-col sm:flex-row sm:items-center p-3 border-b border-gray-700 last:border-b-0">
                  <div className={`sm:w-1/4 font-mono text-sm ${item.color}`}>
                    {item.range}
                  </div>
                  <div className="sm:w-1/4 font-semibold text-white mt-1 sm:mt-0">
                    {item.label}
                  </div>
                  <div className="sm:w-1/2 text-gray-400 text-sm mt-1 sm:mt-0">
                    {item.desc}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NoiseprintSection;