import React, { useEffect, useState } from 'react';
import ImageUploader from './ImageUploader';
import CheckSumPanel from './CheckSumPanel';
import SteganoCompareSection from './SteganoCompareSection'; // 🔹 importujemy nasz nowy komponent

interface SteganoCompareProps {
  setActiveTab: (tab: string) => void;
}

const SteganoCompare: React.FC<SteganoCompareProps> = ({ setActiveTab }) => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrl2, setPreviewUrl2] = useState<string | null>(null);
  const [integrityResult, setIntegrityResult] = useState<'passed' | 'failed' | null>(null);

  useEffect(() => {
    setActiveTab('stegano');
  }, [setActiveTab]);

  const bothUploaded = image1 && image2;

  // 🔹 Funkcja do liczenia SHA-256
  const calculateHash = async (file: File): Promise<string> => {
    const buffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  };

  // 🔹 Automatyczne porównywanie hashy
  useEffect(() => {
    const computeIntegrity = async () => {
      if (image1 && image2) {
        const [hash1, hash2] = await Promise.all([calculateHash(image1), calculateHash(image2)]);
        setIntegrityResult(hash1 === hash2 ? 'passed' : 'failed');
      } else {
        setIntegrityResult(null);
      }
    };
    computeIntegrity();
  }, [image1, image2]);

  return (
    <div className="max-w-6xl mx-auto mt-8">
      {/* Header */}
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 pb-1 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
          Steganography Comparison
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Upload two images to analyze potential hidden data differences and verify integrity.
        </p>
      </div>

      {/* Uploaders */}
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

      {/* Integrity Report + Checksum Panel */}
      {bothUploaded && (
        <div className="mt-12 space-y-8">
          {/* Integrity Report */}
          {integrityResult && (
            <div
              className={`rounded-xl p-6 text-center border shadow-lg backdrop-blur-md transition-all duration-500 ${
                integrityResult === 'passed'
                  ? 'bg-green-900/30 border-green-700 text-green-400 shadow-green-900/20'
                  : 'bg-red-900/30 border-red-700 text-red-400 shadow-red-900/20'
              }`}
            >
              <h3 className="text-2xl font-semibold mb-2">Integrity Report</h3>
              {integrityResult === 'passed' ? (
                <p className="text-lg">Integrity verified — the files are identical.</p>
              ) : (
                <p className="text-lg">
                  Integrity check failed — the files differ. Launching steganographic analysis...
                </p>
              )}
            </div>
          )}

          {/* 🔹 SteganoCompareSection (tylko jeśli integrity failed) */}
            {integrityResult === 'failed' && (
              <div className="mt-8">
                <SteganoCompareSection
                  originalFile={image1}
                  suspiciousFile={image2}
                />
              </div>
            )}
            
          {/* CheckSumPanel */}
          <CheckSumPanel files={[image1, image2].filter(Boolean) as File[]} />

          
        </div>
      )}
    </div>
  );
};

export default SteganoCompare;
