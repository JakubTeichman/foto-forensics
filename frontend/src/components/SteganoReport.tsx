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
    formData.append('file', image); // ✅ Zmienione na 'file' — backend teraz to odbierze

    try {
      const res = await fetch('http://localhost:5000/stegano/analyze', {
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

  return (
    <div className="bg-[#0f0f0f] border border-gray-800 rounded-2xl p-6 mt-6 text-gray-200 shadow-lg">
      <h3 className="text-2xl font-semibold text-teal-400 mb-4">
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
        <p className="text-red-500 mt-4 font-medium">❌ {error}</p>
      )}

      {result && (
        <div className="mt-5 bg-[#1a1a1a] border border-gray-700 rounded-xl p-4">
          <h4 className="text-lg font-semibold text-teal-400 mb-2">
            Analysis Result
          </h4>
          {result.hidden_detected ? (
            <div className="text-green-400">
              ✅ Hidden data detected!
              <p className="text-gray-300 mt-1 text-sm">
                Method: {result.method || 'Unknown'}
              </p>
              {result.details && (
                <pre className="mt-2 text-sm text-gray-400 whitespace-pre-line">
                  {result.details}
                </pre>
              )}
            </div>
          ) : (
            <div className="text-gray-300">
              🔍 No hidden content found in this image.
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SteganoReport;
