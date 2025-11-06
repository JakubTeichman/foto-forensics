import React, { useState } from 'react';

interface SteganoReportProps {
  image: File | null;
}

const SteganoReport: React.FC<SteganoReportProps> = ({ image }) => {
  const [result, setResult] = useState<any | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyzeStegano = async () => {
    if (!image) return;

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', image);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_BASE}/stegano/analyze`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        setResult(data);
      } else {
        setError(data.error || 'Steganography analysis failed.');
      }
    } catch (err) {
      console.error('Error analyzing steganography:', err);
      setError('Failed to connect to backend.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // helper: extract method-results object robustly from different response shapes
  const extractMethods = (resp: any) => {
    if (!resp) return {};

    if (resp.report?.analysis_results && typeof resp.report.analysis_results === 'object') {
      return resp.report.analysis_results;
    }

    if (resp.details?.methods_results && typeof resp.details.methods_results === 'object') {
      return resp.details.methods_results;
    }

    if (resp.details && typeof resp.details === 'object') {
      const maybe = Object.entries(resp.details).every(([_, v]) =>
        v && typeof v === 'object' && ('score' in v || 'detected' in v)
      );
      if (maybe) return resp.details;
    }

    if (resp.analysis_results && typeof resp.analysis_results === 'object') {
      return resp.analysis_results;
    }

    return {};
  };

  const methodsResults = extractMethods(result);

  return (
    <div className="bg-gray-800 border border-teal-800 rounded-xl bg-opacity-70 rounded-lg p-6 mt-6">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        Steganography Analysis
      </h3>

      {!image && (
        <p className="text-gray-400 italic">No image selected for analysis.</p>
      )}

      {image && !result && !isAnalyzing && (
        <button
          onClick={handleAnalyzeStegano}
          className="bg-gradient-to-r from-teal-500 to-green-500 hover:from-teal-600 hover:to-green-600 text-black font-bold px-6 py-2 rounded-xl transition-all duration-300"
        >
          Analyze Steganography
        </button>
      )}

      {isAnalyzing && (
        <div className="flex items-center gap-2 text-gray-400 mt-3">
          <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin"></div>
          <span>Analyzing image for hidden content...</span>
        </div>
      )}

      {error && (
        <p className="text-red-500 mt-4 font-medium">Error: {error}</p>
      )}

      {result && (
        <div className="mt-5 bg-[#1a1a1a] border border-gray-700 rounded-xl p-4">
          <h4 className="text-lg font-semibold text-teal-400 mb-3">
            Steganography Detection Report
          </h4>

          {/* summary */}
          <div className="mb-3">
            {result.hidden_detected ? (
              <div className="text-red-400 font-semibold">
                Hidden data detected!
                {result.detected_methods?.length > 0 && (
                  <span className="text-gray-400 ml-2">
                    ({result.detected_methods.join(', ')})
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
              {result.positive_count !== undefined ? ` — positives: ${result.positive_count}` : ''}
            </div>
          </div>

          {/* table of methods */}
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
                  // ✅ uwzględniamy różne możliwe pola z backendu
                  const scoreValue =
                    typeof data?.score_calibrated === 'number'
                      ? data.score_calibrated
                      : typeof data?.score_raw === 'number'
                      ? data.score_raw
                      : typeof data?.score === 'number'
                      ? data.score
                      : parseFloat(data?.score_calibrated || data?.score_raw || data?.score) || 0;

                  const detected = !!data?.detected;

                  return (
                    <tr
                      key={method}
                      className="border-b border-gray-800 text-gray-300"
                    >
                      <td className="py-1 font-medium">{method}</td>
                      <td className="py-1">{scoreValue.toFixed(3)}</td>
                      <td
                        className={`py-1 font-semibold ${
                          detected ? 'text-red-400' : 'text-green-400'
                        }`}
                      >
                        {detected ? 'Detected' : 'Not detected'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : (
            <p className="text-gray-500 italic mt-2">
              No per-method details available.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SteganoReport;
