import React, { useState, useEffect, useMemo } from "react";

// Definicje typów dla bardziej czytelnego kodu
interface SteganoMethodResult {
  method: string;
  score: number; // Surowy wynik (0-1)
  score_percent: number; // Wynik skalowany (0-100)
  detected: boolean;
  details: Record<string, any>;
}

// Zaktualizowany interfejs, aby pasował do struktury zwracanej przez endpoint
interface SteganoReportResponse {
  status: string;
  hidden_detected: boolean; // Top-level
  method: string;
  details: {
    methods_results: Record<string, SteganoMethodResult>;
    average_heatmap_base64?: string;
    [key: string]: any; // Dopuszczamy inne pola z 'results'
  };
  methods_results: Record<string, SteganoMethodResult>; // Duplikat na najwyższym poziomie
}

interface SteganoReportProps {
  image: File | null;
}

// =========================================================================
// 🎨 Progress Bar Component
// =========================================================================
const ProgressBar: React.FC<{ percentage: number; color: string }> = ({
  percentage,
  color,
}) => {
  // Ograniczamy procent do wyświetlania, na wszelki wypadek
  const displayPercentage = Math.max(0, Math.min(100, percentage));
  const barWidth = `${Math.round(displayPercentage)}%`;

  return (
    <div className="w-full bg-gray-700 rounded-full h-8 overflow-hidden shadow-inner">
      <div
        className={`h-full text-base font-bold text-gray-900 text-center flex items-center justify-center transition-all duration-700 ease-out ${color}`}
        style={{ width: barWidth }}
      >
        {displayPercentage.toFixed(1)}%
      </div>
    </div>
  );
};


const SteganoReport: React.FC<SteganoReportProps> = ({ image }) => {
  const [result, setResult] = useState<SteganoReportResponse | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Zakładamy, że wynik metody CNN jest pod kluczem "ensemble_C1"
  const cnnResult: SteganoMethodResult | null = useMemo(() => {
    // Poprawiona ścieżka dostępu: wyniki są zagnieżdżone w 'details.methods_results'
    const results = result?.details?.methods_results;
    if (results && results.ensemble_C1) {
      return results.ensemble_C1;
    }
    return null;
  }, [result]);

  const scorePercent = cnnResult?.score_percent ?? 0;
  const rawScore = cnnResult?.score ?? 0;


  const analyzeStegano = async () => {
    if (!image) {
      setError("Please select an image before analysis.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", image);

    try {
      // Wysłanie zapytania do API (zakładamy endpoint /stegano/analyze)
      const res = await fetch(`/stegano/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        // Poprawione sprawdzenie: szukamy 'ensemble_C1' w 'details.methods_results'
        const cnnData = data?.details?.methods_results?.ensemble_C1;
        
        if (cnnData) {
          setResult(data as SteganoReportResponse); // Ustawiamy całą odpowiedź z API
        } else {
             // W przypadku błędu backendu lub nieprawidłowej struktury
             setError("Analysis structure error: 'ensemble_C1' result missing or invalid. Check backend logs for full details.");
        }
        
      } else {
        setError(data.error || "Steganography analysis failed.");
      }
    } catch (err) {
      console.error("Error analyzing steganography:", err);
      setError("Failed to connect to backend or process data.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 🔹 Określenie koloru i etykiety na podstawie wyniku procentowego (po angielsku)
  const getDetectionStatus = (percentage: number) => {
    let label = "No Steganography Detected";
    let colorClass = "bg-green-600";
    let textColor = "text-green-400";
    let emoji = "✅";

    if (percentage >= 80) {
      label = "VERY HIGH Probability of Hidden Data";
      colorClass = "bg-red-700";
      textColor = "text-red-400";
      emoji = "🚨";
    } else if (percentage >= 50) {
      label = "High Probability of Hidden Data";
      colorClass = "bg-red-600";
      textColor = "text-red-400";
      emoji = "🔥";
    } else if (percentage >= 20) {
      label = "Moderate Probability of Hidden Data";
      colorClass = "bg-yellow-500";
      textColor = "text-yellow-400";
      emoji = "⚠️";
    } else if (percentage > 0) {
      label = "Low Probability of Hidden Data";
      colorClass = "bg-green-500";
      textColor = "text-green-500";
      emoji = "🔎";
    }
    
    return { label, colorClass, textColor, emoji };
  };

  const { label, colorClass, textColor, emoji } = getDetectionStatus(scorePercent);
  // Status detected jest używany głównie dla celów backendowych (czy przekroczono 0.1)
  const detected = cnnResult?.detected ?? false; 


  return (
    <div className="w-full p-4 md:p-6 bg-gray-900 rounded-xl shadow-2xl">
      <h3 className="text-xl font-semibold mb-6 text-teal-400 text-center">
        CNN Steganography Detector
      </h3>

      {!image && (
        <p className="text-gray-400 italic text-center mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
          Please select an image to start the analysis.
        </p>
      )}

      {/* 🔹 Analysis Button (zgodnie z życzeniem, mechanizm się nie zmienia) */}
      {image && !result && (
        <div className="flex justify-center mb-8">
          <button
            onClick={analyzeStegano}
            disabled={isAnalyzing}
            className={`w-full max-w-md px-6 py-3 rounded-xl font-bold transition-all duration-300 transform hover:scale-[1.02] shadow-xl ${
              isAnalyzing
                ? "bg-gray-600 text-gray-300 cursor-wait"
                : "bg-teal-600 hover:bg-teal-700 text-white "
            }`}
          >
            {isAnalyzing ? "Analysis in progress..." : "Run Steganography Analysis"}
          </button>
        </div>
      )}

      {isAnalyzing && (
        <div className="flex flex-col items-center py-8 bg-gray-800 rounded-xl">
          <div className="w-8 h-8 border-4 border-teal-400 border-t-transparent rounded-full animate-spin"></div>
          <p className="text-sm mt-3 text-gray-400">
            Analyzing image using CNN model...
          </p>
        </div>
      )}

      {error && <p className="text-red-500 mt-4 font-medium text-center p-3 bg-red-900 bg-opacity-30 rounded-lg border border-red-700">⚠️ {error}</p>}

      {cnnResult && (
        <div className={`mt-5 bg-gray-900 rounded-xl p-6`}>
          
          {/* Main Result and Progress Bar */}
          <div className="text-center mb-6">
            <p className={`text-xl font-extrabold ${textColor} mb-6 flex justify-center items-center gap-3`}>
              {label}
            </p>
            
            {/* Progress Bar (Główna wizualizacja) */}
            <ProgressBar percentage={scorePercent} color={colorClass} />
            
            <p className="text-xs text-gray-500 mt-3 italic">
              Raw Model Score (0.0 - 1.0): {rawScore.toFixed(4)}
            </p>
          </div>
          
          {/* Rerun Button */}
           <div className="flex justify-center mt-6 pt-4 border-t border-gray-700">
             <button
              onClick={() => { setResult(null); setError(null); analyzeStegano(); }}
              className="px-4 py-2 text-sm text-teal-300 bg-gray-900 rounded-lg hover:bg-gray-700 transition-colors"
            >
              Run Analysis Again
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default SteganoReport;