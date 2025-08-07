import React, { useState } from 'react';
import ManipulationChart from './ManipulationChart';

interface AnalysisResults {
  metadata: { [key: string]: string };
  manipulationScore: number;
  regions: { x: number; y: number; width: number; height: number; confidence: number }[];
}

interface AnalysisProps {
  setActiveTab?: (tab: string) => void; // funkcja opcjonalna do zmiany aktywnej zakładki
}

const Analysis: React.FC<AnalysisProps> = ({ setActiveTab }) => {
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = () => {
    // Symulacja wyników analizy – zastąp własnym API
    setAnalysisResults({
      metadata: {
        'Camera Model': 'Canon EOS 5D Mark IV',
        'Date Taken': '2025-05-28 14:32:45',
        Resolution: '5472 x 3648',
        'File Size': '12.4 MB',
        'Color Space': 'sRGB',
        Exposure: '1/125 sec at f/4.0',
        ISO: '400',
        GPS: 'Removed',
        Software: 'Adobe Photoshop CC 2024',
      },
      manipulationScore: 78,
      regions: [
        { x: 120, y: 80, width: 200, height: 150, confidence: 0.92 },
        { x: 450, y: 320, width: 180, height: 120, confidence: 0.87 },
      ],
    });

    // Po analizie przełącz zakładkę (jeśli setActiveTab podano)
    if (setActiveTab) {
      setActiveTab('results'); // nazwa zakładki przykładowa, dostosuj wg potrzeb
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h2 className="text-3xl font-bold mb-8 text-teal-400">Image Analysis</h2>

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

      {isUploadOpen && !selectedFile && (
        <div className="bg-gray-900 rounded-xl p-8 border border-teal-800">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-medium">Select Image File</h3>
            <button
              onClick={() => setIsUploadOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="border-2 border-dashed border-teal-700 rounded-lg p-8 text-center">
            <input
              type="file"
              id="fileUpload"
              className="hidden"
              accept="image/*"
              onChange={handleFileChange}
            />
            <label htmlFor="fileUpload" className="cursor-pointer flex flex-col items-center">
              <i className="fas fa-file-image text-4xl text-teal-500 mb-4"></i>
              <span className="text-lg mb-2">Drag and drop or click to browse</span>
              <span className="text-sm text-gray-400">Maximum file size: 50MB</span>
            </label>
          </div>
        </div>
      )}

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
                // Możesz tu też wywołać setActiveTab('upload') jeśli chcesz resetować zakładkę
                if (setActiveTab) setActiveTab('upload');
              }}
              className="text-gray-400 hover:text-white"
            >
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
              <div className="mt-4 flex justify-between">
                <div className="text-sm text-gray-400">
                  <span className="font-medium text-white">{selectedFile.name}</span>
                  <div>{Math.round(selectedFile.size / 1024)} KB</div>
                </div>
                {!analysisResults && (
                  <button
                    onClick={handleAnalyze}
                    className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-6 py-2 rounded-lg hover:from-teal-600 hover:to-green-500 transition-all"
                  >
                    <i className="fas fa-microscope mr-2"></i> Analyze
                  </button>
                )}
              </div>
            </div>

            {analysisResults && (
              <div className="bg-gray-800 bg-opacity-70 rounded-lg p-6">
                <h4 className="text-lg font-medium mb-4 text-teal-400">Analysis Results</h4>

                <div className="mb-6">
                  <ManipulationChart manipulationScore={analysisResults.manipulationScore} />
                </div>

                <div className="grid grid-cols-2 gap-x-4 gap-y-2 mb-6">
                  {Object.entries(analysisResults.metadata).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-400">{key}:</span>
                      <span className="text-white font-medium">{value}</span>
                    </div>
                  ))}
                </div>

                <div>
                  <h5 className="text-md font-medium mb-2 text-teal-400">Detected Manipulations</h5>
                  <div className="bg-gray-900 rounded p-3 text-sm">
                    {analysisResults.regions.map((region, index) => (
                      <div key={index} className="mb-2 last:mb-0">
                        <div className="flex justify-between mb-1">
                          <span>Region {index + 1}</span>
                          <span className="text-green-400">
                            {Math.round(region.confidence * 100)}% confidence
                          </span>
                        </div>
                        <div className="text-gray-400 text-xs">
                          Position: x:{region.x}, y:{region.y}, w:{region.width}, h:{region.height}
                        </div>
                      </div>
                    ))}
                  </div>
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
