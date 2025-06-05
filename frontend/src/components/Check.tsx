import React, { useState } from 'react';
import ImageUploader from './ImageUploader';

const Check: React.FC = () => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrl2, setPreviewUrl2] = useState<string | null>(null);
  const [similarity, setSimilarity] = useState<string | null>(null);

  const bothUploaded = image1 && image2;

  const handleCompare = async () => {
    if (!image1 || !image2) return;

    try {
      const formData = new FormData();
      formData.append('image1', image1);
      formData.append('image2', image2);

      const response = await fetch('/compare', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSimilarity(`${(data.similarity * 100).toFixed(2)}%`);
    } catch (error) {
      console.error('Error:', error);
      setSimilarity('Błąd podczas przesyłania plików');
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
        <ImageUploader
          title="Original Image"
          previewUrl={previewUrl1}
          setPreviewUrl={setPreviewUrl1}
          setSelectedFile={setImage1}
          id="file1"
        />
        <ImageUploader
          title="Comparison Image"
          previewUrl={previewUrl2}
          setPreviewUrl={setPreviewUrl2}
          setSelectedFile={setImage2}
          id="file2"
        />
      </div>

      {bothUploaded && (
        <div className="mt-8 bg-gray-900 bg-opacity-50 rounded-xl p-6 border border-green-900">
          <h3 className="text-xl font-medium mb-4 text-green-400">Comparison Report</h3>
          <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-6">
            <div className="space-y-2 text-gray-300">
              <p>File Size Difference: 1.2 MB</p>
              <p>Resolution Match: 100%</p>
              <p>Color Profile: sRGB</p>
            </div>
            <div className="space-y-2 text-gray-300">
              <p>Modified Regions: None detected</p>
              <p>Compression Level: Similar</p>
              <p>Similarity Score (PRNU): {similarity ?? '---'}</p>
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
