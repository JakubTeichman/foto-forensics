import React, { useEffect, useState } from 'react';
import ImageUploader from './ImageUploader';
import ChecksumPanel from './CheckSumPanel';
import NUAReport from './NUAReport';

interface CheckProps {
  setActiveTab: (tab: string) => void;
}

const Check: React.FC<CheckProps> = ({ setActiveTab }) => {
  const [image1, setImage1] = useState<File | null>(null);
  const [images2, setImages2] = useState<File[]>([]);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrls2, setPreviewUrls2] = useState<string[]>([]);
  const [similarity, setSimilarity] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<boolean>(false);
  const [denoiseMethod, setDenoiseMethod] = useState<string>('bm3d');
  const [loading, setLoading] = useState<boolean>(false);

  const bothUploaded = image1 && images2.length > 0;

  useEffect(() => {
    setActiveTab('check');
  }, [setActiveTab]);

  const handleCompare = async () => {
    if (!image1 || images2.length === 0) return;

    setSimilarity(null);
    setError(null);
    setWarning(false);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('image1', image1);
      images2.forEach((file) => formData.append('images2', file));
      formData.append('denoise_method', denoiseMethod);

      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
      const response = await fetch(`${apiUrl}/compare-multiple`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();

      if (data.similarity !== undefined) {
        const numericValue = parseFloat(data.similarity);
        setSimilarity(numericValue);
        if (data.size_warning) setWarning(true);
      } else {
        setError('Nie udało się obliczyć korelacji.');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Wystąpił błąd podczas porównywania obrazów.');
    } finally {
      setLoading(false);
    }
  };

  // 🎨 Funkcja określająca kolor PCE w zależności od wartości
  const getPceColor = (value: number) => {
    if (value < 20) return 'text-red-500';
    if (value < 45) return 'text-orange-500';
    if (value < 55) return 'text-amber-400';
    if (value < 80) return 'text-yellow-400';
    if (value < 95) return 'text-lime-400';
    if (value < 200) return 'text-green-400';
    return 'text-emerald-400';
  };

  return (
    <div className="max-w-6xl mx-auto mt-8">
      {/* 🔹 Nagłówek */}
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 pb-1 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
          Image Comparison
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Compare and analyze multiple images to identify photo origin. 
        </p>
      </div>

      {/* 🔹 Uploadery */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <ImageUploader
          title="Original Image"
          previewUrl={previewUrl1}
          setPreviewUrl={setPreviewUrl1}
          setSelectedFile={setImage1}
          id="file1"
        />

        <ImageUploader
          title="Reference Images"
          multiple
          previewUrls={previewUrls2}
          setPreviewUrls={setPreviewUrls2}
          setSelectedFiles={setImages2}
          id="file2"
          maxVisible={8}
        />
      </div>

      {/* 🔹 Panel raportu */}
      {bothUploaded && (
        <div className="mt-10 bg-gray-900/80 rounded-2xl p-8 border border-green-800 shadow-lg shadow-green-900/20 backdrop-blur-md">
          <h3 className="text-2xl font-semibold mb-6 text-green-400 tracking-wide">
            Comparison Report
          </h3>

          {/* 🔧 Metoda odszumiania */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Select Denoising Method:
            </label>
            <select
              value={denoiseMethod}
              onChange={(e) => setDenoiseMethod(e.target.value)}
              className="bg-gray-800/90 text-gray-200 border border-green-800 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-400 focus:outline-none w-full"
            >
              <option value="bm3d">BM3D (High Accuracy)</option>
              <option value="wavelet">Wavelet (Fast)</option>
            </select>
          </div>

          {/* ⚠️ Ostrzeżenie */}
          {warning && (
            <div className="mt-6 mb-8 p-5 rounded-2xl bg-amber-900/20 border border-amber-500/40 text-amber-300 text-sm font-medium backdrop-blur-md shadow-inner shadow-amber-900/30">
              <span className="block text-base font-semibold mb-2 text-amber-400">
                Resolution Mismatch Detected
              </span>
              Uploaded images have <strong>different resolutions</strong>, which may reduce precision
              of the <span className="text-green-300 font-semibold">PRNU correlation</span> results.
            </div>
          )}

          {/* 📊 Wyniki */}
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-8">
            <div className="space-y-2 text-gray-300 text-sm md:text-base">
              <p>
                Resolution Match:{' '}
                <span className={warning ? 'text-red-500 font-semibold' : 'text-green-400 font-semibold'}>
                  {warning ? 'No' : 'Yes'}
                </span>
              </p>
              <p>
                Denoising Method:{' '}
                <span className="text-teal-400 font-semibold">{denoiseMethod.toUpperCase()}</span>
              </p>
            </div>

            <div className="flex flex-col items-center justify-center min-w-[240px] min-h-[80px]">
              {loading ? (
                <div className="flex flex-col items-center">
                  <div className="w-10 h-10 border-4 border-t-green-400 border-gray-700 rounded-full animate-spin"></div>
                  <p className="text-green-400 mt-3 text-sm font-medium">Analyzing...</p>
                </div>
              ) : (
                <>
                  {similarity !== null ? (
                    <p className={`text-3xl font-bold ${getPceColor(similarity)} drop-shadow-md`}>
                      {similarity.toFixed(3)} <span className="text-gray-400 text-lg">PCE</span>
                    </p>
                  ) : (
                    <p className="text-gray-500 italic text-sm">No data yet</p>
                  )}
                  {error && <p className="text-red-400 font-semibold mt-2">{error}</p>}
                </>
              )}
            </div>

            {/* 🔘 Przyciski */}
            <div className="flex justify-center">
              <button
                onClick={handleCompare}
                disabled={loading}
                className={`${
                  loading
                    ? 'bg-gray-700 cursor-not-allowed opacity-70'
                    : 'bg-gradient-to-r from-teal-500 to-green-400 hover:from-teal-600 hover:to-green-500 shadow-lg shadow-green-800/30 hover:shadow-green-600/40'
                } text-black font-extrabold px-8 py-3 rounded-xl text-lg transition-all transform hover:scale-[1.03]`}
              >
                {loading ? 'Processing...' : 'Compare'}
              </button>
            </div>
          </div>
        </div>
      )}
        
      {image1 && <NUAReport imageFile={image1} referenceImages={images2} />}

      {/* ✅ Panel sum kontrolnych */}
      {image1 && <ChecksumPanel files={[image1]} />}
    </div>
  );
};

export default Check;
