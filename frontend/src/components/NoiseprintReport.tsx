import React, { useEffect, useState } from "react";

interface NoiseprintReportProps {
  imageFile: File | null;
}

const NoiseprintReport: React.FC<NoiseprintReportProps> = ({ imageFile }) => {
  const [noiseprint, setNoiseprint] = useState<string | null>(null);
  const [stats, setStats] = useState<{ [key: string]: number } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    if (imageFile) {
      handleGenerate();
    } else {
      setNoiseprint(null);
      setStats(null);
      setError(null);
    }
  }, [imageFile]);

  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        Noiseprint Report
      </h3>

      {!imageFile ? (
        <p className="text-gray-400 italic">No image selected.</p>
      ) : loading ? (
        <div className="flex items-center gap-2 text-gray-400 mt-3">
          <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
          <span>Generating noiseprint...</span>
        </div>
      ) : error ? (
        <p className="text-red-400 text-sm mt-2">⚠️ {error}</p>
      ) : (
        noiseprint && (
          <div className="flex flex-col md:flex-row gap-6 mt-6">
            <div className="flex-1">
              <h4 className="text-teal-400 mb-2 font-medium">Noiseprint Image:</h4>
              <img
                src={noiseprint}
                alt="Noiseprint visualization"
                className="rounded-lg border border-gray-700 shadow-md"
              />
            </div>

            {stats && (
              <div className="flex-1 bg-gray-800/60 rounded-lg p-4 text-gray-300">
                <h4 className="text-teal-400 font-medium mb-3">Noiseprint Parameters:</h4>
                <ul className="space-y-1 text-sm">
                  {Object.entries(stats).map(([key, value]) => (
                    <li key={key}>
                      <span className="text-teal-400 font-semibold capitalize">{key}:</span>{" "}
                      {value.toFixed(4)}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )
      )}
    </div>
  );
};

export default NoiseprintReport;
