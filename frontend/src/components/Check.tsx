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

  const bothUploaded = image1 && images2.length > 0;

  useEffect(() => {
    setActiveTab('check');
  }, [setActiveTab]);

  const handleCompare = async () => {
    if (!image1 || images2.length === 0) return;

    setSimilarity(null);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image1', image1);

      // dodajemy wiele obrazów referencyjnych
      images2.forEach((file, index) => {
        formData.append('images2', file);
      });

      const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

      const response = await fetch(`${apiUrl}/compare-multiple`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);

      const data = await response.json();

      if (data.similarity !== undefined) {
        const formatted = parseFloat(data.similarity).toFixed(5);
        setSimilarity(`${formatted} (Pearson)`);
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
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
          Image Comparison
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Compare and analyze multiple images to detect differences and verify authenticity.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Obraz dowodowy (1 plik) */}
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
          maxVisible={8}  // np. pokazujemy 4 miniatury, reszta w +X
        />

      </div>

      {bothUploaded && (
        <div className="mt-8 bg-gray-900 bg-opacity-50 rounded-xl p-6 border border-green-900">
          <h3 className="text-xl font-medium mb-4 text-green-400">Comparison Report</h3>
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-6">
            <div className="space-y-2 text-gray-300">
              <p>File Size Difference: —</p>
              <p>Resolution Match: —</p>
              <p>Color Profile: —</p>
            </div>
            <div className="space-y-2 text-gray-300">
              <p>Modified Regions: —</p>
              <p>Compression Level: —</p>
              <p>
                Similarity Score (PRNU):{' '}
                <span className="font-semibold text-green-400">
                  {similarity ?? '---'}
                </span>
              </p>
              {error && (
                <p className="text-red-400 font-semibold">{error}</p>
              )}
            </div>
            <div>
              <button
                onClick={handleCompare}
                className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-6 py-2 rounded-lg hover:from-teal-600 hover:to-green-500 transition-all"
              >
                <i className="fas fa-balance-scale mr-2"></i> Porównaj
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Check;
