import React, { useState } from 'react';
import MetadataChart from './MetadataChart';

interface AnalysisResults {
  metadata: { [key: string]: any };
  manipulationScore: number;
  regions: { x: number; y: number; width: number; height: number; confidence: number }[];
}

interface AnalysisProps {
  setActiveTab?: (tab: string) => void;
}

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
        // ✅ Generowanie HEX (pierwsze 4KB)
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

  // ✅ Wczytanie pełnego zapisu HEX po kliknięciu przycisku
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

  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-3xl font-bold mb-8 text-teal-400">Image Analysis</h2>

      {/* Upload panel */}
      {!isUploadOpen && !selectedFile && (
        <div className="flex flex-col items-center justify-center bg-gray-900 bg-opacity-50 rounded-xl p-12 border-2 border-dashed border-teal-700">
          <i className="fas fa-cloud-upload-alt text-5xl text-teal-500 mb-4"></i>
          <h3 className="text-xl font-medium mb-2">Upload an image for forensic analysis</h3>
          <p className="text-gray-400 mb-6 text-center max-w-lg">
            We support JPEG, PNG, TIFF, and RAW formats. Your image will be analyzed for manipulation, metadata, and authenticity.
          </p>
          <button
            onClick={() => setIsUploadOpen(true)}
            className="bg-teal-600 hover:bg-teal-700 px-6 py-3 rounded-lg transition-colors"
          >
            <i className="fas fa-upload mr-2"></i> Upload Image
          </button>
        </div>
      )}

      {/* File selection */}
      {isUploadOpen && !selectedFile && (
        <div className="bg-gray-900 rounded-xl p-8 border border-teal-800">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-medium">Select Image File</h3>
            <button onClick={() => setIsUploadOpen(false)} className="text-gray-400 hover:text-white">
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="border-2 border-dashed border-teal-700 rounded-lg p-8 text-center">
            <input type="file" id="fileUpload" className="hidden" accept="image/*" onChange={handleFileChange} />
            <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center">
              <i className="fas fa-file-image text-4xl text-teal-500 mb-4"></i>
              <span className="text-lg mb-2">Drag and drop or click to browse</span>
              <span className="text-sm text-gray-400">Maximum file size: 50MB</span>
            </label>
          </div>
        </div>
      )}

      {/* Analysis view */}
      {selectedFile && (
        <div className="bg-gray-900 rounded-xl p-6 border border-teal-800">
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

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ✅ Left column: image + hex view */}
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

              {/* ✅ Hex view (po analizie) */}
              {showReport && hexData && (
                <div className="bg-gray-950 mt-4 p-3 rounded-lg text-xs text-gray-300 max-h-72 overflow-auto font-mono border border-gray-800">
                  <h4 className="text-teal-400 font-medium mb-2">
                    {isFullHexVisible ? 'Full Hexadecimal View' : 'Hexadecimal View (first 4KB)'}
                  </h4>
                  <pre className="whitespace-pre-wrap break-all">{hexData}</pre>
                </div>
              )}

              {/* ✅ Przycisk POD oknem z hexem */}
              {showReport && hexData && !isFullHexVisible && (
                <div className="text-center mt-3">
                  <button
                    onClick={handleShowFullHex}
                    disabled={isLoadingFullHex}
                    className={`${
                      isLoadingFullHex
                        ? 'bg-gray-700 cursor-not-allowed'
                        : 'bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500'
                    } text-black font-semibold px-4 py-2 rounded-lg text-sm transition-all`}
                  >
                    {isLoadingFullHex ? 'Loading full HEX...' : 'Show full HEX'}
                  </button>
                </div>
              )}

              {/* ✅ File info + analyze (tylko przed analizą) */}
              {!showReport && (
                <div className="mt-4 flex justify-between items-center">
                  <div className="text-sm text-gray-400">
                    <span className="font-medium text-white">{selectedFile.name}</span>
                    <div>{Math.round(selectedFile.size / 1024)} KB</div>
                  </div>

                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className={`${
                      isAnalyzing
                        ? 'bg-gray-600 cursor-not-allowed'
                        : 'bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500'
                    } text-black font-bold px-6 py-2 rounded-lg transition-all`}
                  >
                    {isAnalyzing ? (
                      <span className="flex items-center gap-2">
                        <i className="fas fa-spinner fa-spin"></i> Analyzing...
                      </span>
                    ) : (
                      <>
                        <i className="fas fa-microscope mr-2"></i> Analyze
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {/* ✅ Right column: metadata report */}
            {showReport && analysisResults && (
              <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6">
                <MetadataChart
                  exifCount={Object.keys(analysisResults.metadata['EXIF Data'] || {}).length}
                  gpsCount={Object.keys(analysisResults.metadata['GPS Info'] || {}).length}
                />

                <h4 className="text-lg font-medium mb-4 text-teal-400 mt-6">
                  Metadata Report
                </h4>

                <div className="grid grid-cols-2 gap-2 mb-6">
                  {['Filename', 'File Size', 'Format', 'Mode', 'Resolution'].map((label) => (
                    <div key={label} className="bg-gray-900 p-2 rounded-md text-sm text-gray-300">
                      <span className="text-teal-400 font-medium mr-2">{label}:</span>
                      {label === 'Filename'
                        ? selectedFile.name
                        : String(analysisResults.metadata[label] || 'N/A')}
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-2 gap-4 mb-6">
                  {['EXIF Data', 'GPS Info'].map((section) => (
                    <div key={section} className="bg-gray-900 rounded-md p-3 max-h-56 overflow-auto">
                      <h5 className="text-teal-400 font-medium mb-2">{section}</h5>
                      {typeof analysisResults.metadata[section] === 'object' &&
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
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Analysis;
