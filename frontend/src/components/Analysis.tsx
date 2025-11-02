import React, { useState } from 'react';
import MetadataChart from './MetadataChart';
import GpsMap from './GpsMap';
import CheckSumBox from './CheckSumPanel';
import SteganoReport from './SteganoReport';
import jsPDF from "jspdf";
import html2canvas from "html2canvas";

interface AnalysisResults {
  metadata: { [key: string]: any };
  manipulationScore: number;
  regions: { x: number; y: number; width: number; height: number; confidence: number }[];
}

interface AnalysisProps {
  setActiveTab?: (tab: string) => void;
}

// ✅ Funkcja pomocnicza — konwersja [39, 28, 689/25] na np. 39.478
const convertDMS = (arr: any[]): number => {
  if (!Array.isArray(arr) || arr.length < 3) return NaN;
  const [deg, min, sec] = arr;
  const secVal = typeof sec === 'string' && sec.includes('/')
    ? (() => {
        const [a, b] = sec.split('/').map(Number);
        return a / b;
      })()
    : Number(sec);
  return deg + min / 60 + secVal / 3600;
};

const Analysis: React.FC<AnalysisProps> = ({ setActiveTab }) => {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [hexData, setHexData] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showReport, setShowReport] = useState(false);
  const [isLoadingFullHex, setIsLoadingFullHex] = useState(false);
  const [isFullHexVisible, setIsFullHexVisible] = useState(false);

  const generatePDF = async () => {
  const reportElement = document.getElementById("analysis-report");
  if (!reportElement) {
    alert("⚠️ No report to export!");
    return;
  }

  // 🔹 Zapamiętaj aktualny stan sekcji (np. czy HEX rozwinięty)
  const prevScrollStates = Array.from(reportElement.querySelectorAll(".overflow-auto")).map((el) => ({
    el,
    originalMaxHeight: (el as HTMLElement).style.maxHeight,
    originalOverflow: (el as HTMLElement).style.overflow,
  }));

  // 🔹 Tymczasowe rozwinięcie wszystkich przewijalnych sekcji
  prevScrollStates.forEach(({ el }) => {
    (el as HTMLElement).style.maxHeight = "none";
    (el as HTMLElement).style.overflow = "visible";
  });

  // 🔹 Ukryj elementy interaktywne (przyciski, navbar, footer itp.)
  const hiddenElements = Array.from(document.querySelectorAll("button, nav, footer"));
  hiddenElements.forEach((el) => ((el as HTMLElement).style.display = "none"));

  // 🔹 Zrób zrzut widoku jako canvas
  const canvas = await html2canvas(reportElement, {
    scale: 2,
    useCORS: true,
    backgroundColor: "#1a1a1a",
  } as any);

  // 🔹 Przywróć układ po renderze
  prevScrollStates.forEach(({ el, originalMaxHeight, originalOverflow }) => {
    (el as HTMLElement).style.maxHeight = originalMaxHeight;
    (el as HTMLElement).style.overflow = originalOverflow;
  });
  hiddenElements.forEach((el) => ((el as HTMLElement).style.display = ""));

  // 🔹 Konwertuj na PDF
  const imgData = canvas.toDataURL("image/png");
  const pdf = new jsPDF("p", "mm", "a4");

  const pageWidth = pdf.internal.pageSize.getWidth();
  const pageHeight = pdf.internal.pageSize.getHeight();
  const imgWidth = pageWidth;
  const imgHeight = (canvas.height * imgWidth) / canvas.width;

  let position = 0;
  if (imgHeight > pageHeight) {
    // jeśli zrzut jest dłuższy niż strona A4 → podziel na strony
    let heightLeft = imgHeight;
    let y = 0;
    while (heightLeft > 0) {
      pdf.addImage(imgData, "PNG", 0, y ? -y : 0, imgWidth, imgHeight);
      heightLeft -= pageHeight;
      y += pageHeight;
      if (heightLeft > 0) pdf.addPage();
    }
  } else {
    pdf.addImage(imgData, "PNG", 0, 0, imgWidth, imgHeight);
  }

  // 🔹 Zapisz PDF
  pdf.save(`FotoForensics_Report_${selectedFile?.name || "image"}.pdf`);
};


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setShowReport(false);
    setIsFullHexVisible(false);
    setHexData(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch('http://localhost:5000/analyze/metadata', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (res.ok) {
        const hexReader = new FileReader();
        hexReader.onload = (event) => {
          const buffer = event.target?.result as ArrayBuffer;
          const bytes = new Uint8Array(buffer);
          const hexString = Array.from(bytes)
            .slice(0, 4096)
            .map((b) => b.toString(16).padStart(2, '0'))
            .join(' ');
          setHexData(hexString);
        };
        hexReader.readAsArrayBuffer(selectedFile);

        setAnalysisResults({
          metadata: data,
          manipulationScore: 0,
          regions: [],
        });

        setShowReport(true);
      } else {
        alert(data.error || 'Error analyzing image');
      }
    } catch (error) {
      console.error('Analysis error:', error);
      alert('Failed to analyze image');
    } finally {
      setIsAnalyzing(false);
    }

    if (setActiveTab) setActiveTab('results');
  };

  const handleShowFullHex = () => {
    if (!selectedFile) return;
    setIsLoadingFullHex(true);
    const reader = new FileReader();
    reader.onload = (event) => {
      const buffer = event.target?.result as ArrayBuffer;
      const bytes = new Uint8Array(buffer);
      const hexString = Array.from(bytes)
        .map((b) => b.toString(16).padStart(2, '0'))
        .join(' ');
      setHexData(hexString);
      setIsFullHexVisible(true);
      setIsLoadingFullHex(false);
    };
    reader.readAsArrayBuffer(selectedFile);
  };

  const handleCopyHex = () => {
    if (hexData) {
      navigator.clipboard.writeText(hexData);
      alert('✅ HEX copied to clipboard!');
    }
  };
// ✅ Wyodrębnienie GPS (obsługa różnych formatów: zagnieżdżone lub płaskie)
let lat: number | null = null;
let lon: number | null = null;

if (analysisResults?.metadata) {
  // Szukamy danych GPS zarówno w zagnieżdżonych jak i płaskich strukturach
  const metadata = analysisResults.metadata;
  const gps =
    metadata['GPS Info'] && Object.keys(metadata['GPS Info']).length > 0
      ? metadata['GPS Info']
      : metadata;

  const rawLat = gps['GPS GPSLatitude'] || gps['GPSLatitude'];
  const rawLon = gps['GPS GPSLongitude'] || gps['GPSLongitude'];
  const latRef = gps['GPS GPSLatitudeRef'] || gps['GPSLatitudeRef'];
  const lonRef = gps['GPS GPSLongitudeRef'] || gps['GPSLongitudeRef'];

  if (rawLat && rawLon) {
    try {
      // 🔹 Parsowanie tekstu "[39, 28, 689/25]" na tablicę [39, 28, 689/25]
      const parseCoord = (coordStr: string): number[] => {
        const cleaned = coordStr.replace(/\s/g, '').replace(/^\[|\]$/g, '');
        return cleaned.split(',').map((val) => {
          if (val.includes('/')) {
            const [num, den] = val.split('/').map(Number);
            return num / den;
          }
          return Number(val);
        });
      };

      const latArr = Array.isArray(rawLat) ? rawLat : parseCoord(rawLat);
      const lonArr = Array.isArray(rawLon) ? rawLon : parseCoord(rawLon);

      lat = convertDMS(latArr);
      lon = convertDMS(lonArr);

      if (latRef === 'S') lat = -lat;
      if (lonRef === 'W') lon = -lon;

      console.log('📍 Wynik GPS (Decimal Degrees):', { lat, lon });
    } catch (err) {
      console.error('❌ Błąd parsowania GPS:', err);
    }
  }
}


  return (
    <div className="max-w-6xl mx-auto">
      {/* 🔹 Nagłówek — pokazuje się tylko, gdy nie ma uploadu */}
      { !selectedFile && (
        <div className="text-center mb-12 mt-8">
          <h2 className="text-5xl font-bold mb-4 pb-1 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
            Image Analysis
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Upload your image to begin forensic examination for authenticity and hidden traces.
          </p>
        </div>
      )}

      {/* --- Sekcja wyboru pliku --- */}
      {!selectedFile && (
        <div
          className="bg-gray-900 rounded-xl p-8 border border-teal-800 mt-6 mb-4"
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            const file = e.dataTransfer.files[0];
            if (file) handleFileChange({ target: { files: [file] } } as any);
          }}
        >
          <div className="flex justify-between items-center mb-3">
            <h3 className="text-xl font-medium text-teal-400">Select Image File</h3>
            <button onClick={() => setIsUploadOpen(false)} className="text-gray-400 hover:text-white">
              <i className="fas fa-times"></i>
            </button>
          </div>

          <div className="border-2 border-dashed border-teal-700 rounded-lg p-8 text-center hover:border-teal-500 transition-colors">
            <input
              type="file"
              id="fileUpload"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
            <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center">
              <i className="fas fa-file-image text-4xl text-teal-500 mb-4"></i>
              <span className="text-lg mb-2 text-gray-200">Drag and drop or click to browse</span>
              <span className="text-sm text-gray-400">Maximum file size: 50MB</span>
            </label>
          </div>
        </div>
      )}


      {/* --- Sekcja analizy --- */}
      {selectedFile && (
        <div className="bg-gray-900 rounded-xl p-6 border border-teal-800 mt-6 mb-4">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-medium">Image Analysis</h3>
            <button
              onClick={() => {
                setSelectedFile(null);
                setPreviewUrl(null);
                setAnalysisResults(null);
                setIsUploadOpen(false);
                setHexData(null);
                setShowReport(false);
                if (setActiveTab) setActiveTab('upload');
              }}
              className="text-gray-400 hover:text-white"
            >
              <i className="fas fa-times"></i>
            </button>
          </div>

          <div id="analysis-report" className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* --- LEWA KOLUMNA — podgląd + HEX --- */}
            <div>
              <div className="bg-black bg-opacity-50 rounded-lg overflow-hidden">
                {previewUrl && (
                  <img
                    src={previewUrl}
                    alt="Selected"
                    className="w-full h-auto object-contain max-h-[400px]"
                  />
                )}
              </div>

              {/* ✅ Sumy kontrolne pliku */}
              <CheckSumBox files={selectedFile ? [selectedFile] : []} />

              {/* ✅ HEX VIEW */}
              {showReport && hexData && (
                <div className="bg-gray-950 mt-4 p-3 rounded-lg text-xs text-gray-300 font-mono border border-gray-800">
                  <h4 className="text-teal-400 font-medium mb-2">
                    {isFullHexVisible ? "Full Hexadecimal View" : "Hexadecimal View (first 4KB)"}
                  </h4>

                  <div className="font-mono text-xs max-h-96 overflow-y-auto bg-gray-950 rounded-md border border-gray-800 p-2">
                    <div className="flex text-gray-400 font-semibold mb-1 sticky top-0 bg-gray-950 pb-1">
                      <div className="w-4/5">Hexadecimal</div>
                      <div className="w-1/5 text-right">ASCII</div>
                    </div>
                    <div className="divide-y divide-gray-800">
                      {(() => {
                        const bytes = hexData.split(" ").filter((b) => b.length > 0);
                        const lines = [];
                        for (let i = 0; i < bytes.length; i += 16) {
                          const chunk = bytes.slice(i, i + 16);
                          const ascii = chunk
                            .map((b) => {
                              const code = parseInt(b, 16);
                              return code >= 32 && code <= 126 ? String.fromCharCode(code) : ".";
                            })
                            .join("");
                          lines.push(
                            <div key={i} className="flex justify-between text-gray-300">
                              <div className="w-4/5">{chunk.join(" ")}</div>
                              <div className="w-1/5 text-right text-teal-400">{ascii}</div>
                            </div>
                          );
                        }
                        return lines;
                      })()}
                    </div>
                  </div>

                  <div className="flex justify-center gap-3 mt-3">
                    {!isFullHexVisible && (
                      <button
                        onClick={handleShowFullHex}
                        disabled={isLoadingFullHex}
                        className={`${
                          isLoadingFullHex
                            ? "bg-gray-700 cursor-not-allowed"
                            : "bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500"
                        } text-black font-semibold px-4 py-2 rounded-lg text-sm transition-all`}
                      >
                        {isLoadingFullHex ? "Loading full HEX..." : "Show full HEX"}
                      </button>
                    )}
                    <button
                      onClick={handleCopyHex}
                      className="bg-teal-700 hover:bg-teal-600 text-white font-semibold px-4 py-2 rounded-lg text-sm transition-all"
                    >
                      <i className="fas fa-copy mr-2"></i> Copy HEX
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* --- PRAWA KOLUMNA — przycisk Analyze i raport --- */}
            <div className="flex flex-col justify-start">
              {/* 🔹 Przyciski i podsumowanie pliku */}
              {!showReport && (
                <div className="flex flex-col items-center justify-center mt-10 gap-6">
                  <div className="text-center text-gray-400">
                    <span className="block font-semibold text-white text-lg">{selectedFile.name}</span>
                    <span className="text-sm">{Math.round(selectedFile.size / 1024)} KB</span>
                  </div>

                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className={`${
                      isAnalyzing
                        ? "bg-gray-700 cursor-not-allowed"
                        : "bg-gradient-to-r from-teal-400 to-green-400 hover:from-teal-500 hover:to-green-500"
                    } text-black font-bold text-lg px-10 py-4 rounded-xl shadow-lg hover:shadow-teal-500/30 transform hover:scale-105 transition-all duration-300`}
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center gap-3">
                        <i className="fas fa-spinner fa-spin text-xl"></i> Analyzing...
                      </span>
                    ) : (
                      <span className="flex items-center gap-3">
                        <i className="fas fa-microscope text-xl"></i> Start Analysis
                      </span>
                    )}
                  </button>
                </div>
              )}

              {/* --- Raport metadanych --- */}
              {showReport && analysisResults && (
                <div className="bg-gray-800 border border-teal-800 rounded-xl bg-opacity-70 rounded-lg p-6 mt-6">
                  {(() => {
                    const exifCount = Object.keys(analysisResults.metadata["EXIF Data"] || {}).length;
                    const gpsCount = Object.entries(analysisResults.metadata["GPS Info"] || {}).filter(
                      ([key, value]) =>
                        key.toUpperCase().includes("GPS") &&
                        value &&
                        value !== "0" &&
                        value !== "[]" &&
                        value !== "No GPS data detected"
                    ).length;

                    if (exifCount === 0 && gpsCount === 0) return null;
                    return <MetadataChart exifCount={exifCount} gpsCount={gpsCount} />;
                  })()}

                  <h4 className="text-lg font-medium mb-4 text-teal-400 mt-6">Metadata Report</h4>

                  <div className="grid grid-cols-2 gap-2 mb-6">
                    {["Filename", "File Size", "Format", "Mode", "Resolution"].map((label) => (
                      <div key={label} className="bg-gray-900 p-2 rounded-md text-sm text-gray-300">
                        <span className="text-teal-400 font-medium mr-2">{label}:</span>
                        {label === "Filename"
                          ? selectedFile.name
                          : String(analysisResults.metadata[label] || "N/A")}
                      </div>
                    ))}
                  </div>

                  <div className="flex flex-col gap-4 mb-6">
                    {["EXIF Data", "GPS Info"].map((section) => (
                      <div
                        key={section}
                        className="bg-gray-900 rounded-md p-3 max-h-56 overflow-auto scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-gray-900"
                      >
                        <h5 className="text-teal-400 font-medium mb-2">{section}</h5>
                        {typeof analysisResults.metadata[section] === "object" &&
                        Object.keys(analysisResults.metadata[section]).length > 0 ? (
                          <table className="w-full text-xs text-gray-300">
                            <tbody>
                              {Object.entries(analysisResults.metadata[section]).map(([key, value]) => (
                                <tr key={key} className="border-b border-gray-800">
                                  <td className="text-gray-400 pr-2">{key}</td>
                                  <td className="text-white">{String(value)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        ) : (
                          <p className="text-gray-500 italic">No {section.toLowerCase()} found.</p>
                        )}
                      </div>
                    ))}

                    {lat !== null && lon !== null && !isNaN(lat) && !isNaN(lon) && (
                      <div className="mt-4">
                        <h5 className="text-teal-400 font-medium mb-2">Location Preview</h5>
                        <div className="h-64 w-full rounded-lg overflow-hidden border border-teal-700">
                          <GpsMap lat={lat} lon={lon} />
                        </div>
                      </div>
                    )}

                    {selectedFile && <SteganoReport image={selectedFile} />}

                      {showReport && (
                        <div className="flex justify-end mb-4">
                          <button
                            onClick={generatePDF}
                            className="bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500 text-black font-semibold px-5 py-2 rounded-lg text-sm shadow-md hover:shadow-teal-500/40 transition-all"
                          >
                            <i className="fas fa-file-pdf mr-2"></i> Download PDF Report
                          </button>
                        </div>
                      )}

                  </div>
                </div>
              )}
            </div>
          </div>

        </div>
      )}
    </div>
  );
};

export default Analysis;
