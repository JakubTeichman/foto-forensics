import React, { useEffect, useState } from 'react';
import ImageUploader from './ImageUploader';

interface CheckProps {
  setActiveTab: (tab: string) => void;
}

const Check: React.FC<CheckProps> = ({ setActiveTab }) => {
  const [image1, setImage1] = useState<File | null>(null);
  const [images2, setImages2] = useState<File[]>([]);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrls2, setPreviewUrls2] = useState<string[]>([]);
  const [similarity, setSimilarity] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<boolean>(false);
  const [denoiseMethod, setDenoiseMethod] = useState<string>('bm3d');

  const bothUploaded = image1 && images2.length > 0;

  useEffect(() => {
    setActiveTab('check');
  }, [setActiveTab]);

  const handleCompare = async () => {
    if (!image1 || images2.length === 0) return;

    setSimilarity(null);
    setError(null);
    setWarning(false);

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
        const formatted = parseFloat(data.similarity).toFixed(5);
        setSimilarity(`${formatted} (PCE)`);
        if (data.size_warning) setWarning(true);
      } else {
        setError('Nie udało się obliczyć korelacji.');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Wystąpił błąd podczas porównywania obrazów.');
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      {/* Nagłówek */}
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
          Image Comparison
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Compare and analyze multiple images to detect differences and verify authenticity.
        </p>
      </div>

      {/* Uploader */}
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

      {/* Panel wyników */}
      {bothUploaded && (
        <div className="mt-8 bg-gray-900/70 rounded-xl p-6 border border-green-900 shadow-lg shadow-green-900/20 backdrop-blur-md">
          <h3 className="text-xl font-medium mb-6 text-green-400">Comparison Report</h3>

          {/* Select metody denoisingu */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Select Denoising Method:
            </label>
            <select
              value={denoiseMethod}
              onChange={(e) => setDenoiseMethod(e.target.value)}
              className="bg-gray-800/80 text-gray-200 border border-green-800 rounded-lg px-3 py-2 focus:ring-2 focus:ring-green-400 focus:outline-none w-full"
            >
              <option value="bm3d">BM3D (High Accuracy)</option>
              <option value="wavelet">Wavelet (Fast)</option>
            </select>
          </div>

          {/* Ostrzeżenie o rozdzielczości */}
          {warning && (
            <div className="mt-6 mb-8 p-5 rounded-2xl bg-emerald-900/20 border border-emerald-500/40 text-emerald-300 text-sm font-medium backdrop-blur-md shadow-inner shadow-emerald-900/30">
              <span className="block text-base font-semibold mb-2 text-emerald-400">
                ⚠ Resolution Mismatch Detected
              </span>
              The uploaded images have <strong>different resolutions</strong>, which may slightly reduce
              the precision of the <span className="text-green-300 font-semibold">PRNU similarity</span> result.
              <br />
              For optimal results, use images captured in the <strong>same resolution and camera settings</strong>.
            </div>
          )}


          {/* Wyniki i przycisk */}
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-6">
            <div className="space-y-2 text-gray-300">
              <p>Resolution Match: {warning ? '❌ No' : '✅ Yes'}</p>
              <p>
                Denoising Method: <span className="text-green-400">{denoiseMethod.toUpperCase()}</span>
              </p>
            </div>

            <div className="space-y-2 text-gray-300">
              <p>
                Similarity Score (PRNU):{' '}
                <span className="font-semibold text-green-400">
                  {similarity ?? '---'}
                </span>
              </p>
              {error && <p className="text-red-400 font-semibold">{error}</p>}
            </div>

            <div>
              <button
                onClick={handleCompare}
                className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-6 py-2 rounded-lg hover:from-teal-600 hover:to-green-500 transition-all"
              >
                Compare
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Check;
