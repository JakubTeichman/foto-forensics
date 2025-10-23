import React, { useEffect, useState } from 'react';
import ImageUploader from './ImageUploader';

interface SteganoCompareProps {
  setActiveTab: (tab: string) => void;
}

const SteganoCompare: React.FC<SteganoCompareProps> = ({ setActiveTab }) => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrl2, setPreviewUrl2] = useState<string | null>(null);

  useEffect(() => {
    setActiveTab('stegano');
  }, [setActiveTab]);

  const bothUploaded = image1 && image2;

  return (
    <div className="max-w-6xl mx-auto mt-8">
      {/* Nagłówek */}
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 pb-1 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
          Steganography Comparison
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Upload two images to begin analyzing possible hidden data differences.
        </p>
      </div>

      {/* Uploader */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <ImageUploader
          title="Original Image"
          previewUrl={previewUrl1}
          setPreviewUrl={setPreviewUrl1}
          setSelectedFile={setImage1}
          id="stegano-file1"
        />

        <ImageUploader
          title="Suspicious Image"
          previewUrl={previewUrl2}
          setPreviewUrl={setPreviewUrl2}
          setSelectedFile={setImage2}
          id="stegano-file2"
        />
      </div>

      {/* Placeholder na przyszły raport */}
      {bothUploaded && (
        <div className="mt-12 bg-gray-900/70 rounded-xl p-6 border border-green-900 shadow-lg shadow-green-900/20 backdrop-blur-md text-center text-gray-300">
          <p className="text-green-400 font-semibold text-lg">
            ✅ Both images uploaded successfully!
          </p>
          <p className="text-gray-400 mt-2">
            The steganography comparison module will appear here soon.
          </p>
        </div>
      )}
    </div>
  );
};

export default SteganoCompare;
