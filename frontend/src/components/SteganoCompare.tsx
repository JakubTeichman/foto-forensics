import React, { useEffect, useState } from 'react';
import ImageUploader from './ImageUploader';
import CheckSumPanel from './CheckSumPanel';

interface SteganoCompareProps {
  setActiveTab: (tab: string) => void;
}

const SteganoCompare: React.FC<SteganoCompareProps> = ({ setActiveTab }) => {
  const [image1, setImage1] = useState<File | null>(null);
  const [image2, setImage2] = useState<File | null>(null);
  const [previewUrl1, setPreviewUrl1] = useState<string | null>(null);
  const [previewUrl2, setPreviewUrl2] = useState<string | null>(null);
  const [checksums, setChecksums] = useState<{ [key: string]: string }>({});
  const [integrityResult, setIntegrityResult] = useState<'passed' | 'failed' | null>(null);
  const [loadingChecksums, setLoadingChecksums] = useState(false);

  useEffect(() => {
    setActiveTab('stegano');
  }, [setActiveTab]);

  const bothUploaded = image1 && image2;

 const handleStartCalculation = () => {
  setIntegrityResult(null);
  // pokaż spinner tylko jeśli trwa dłużej niż 300 ms
  const timer = setTimeout(() => setLoadingChecksums(true), 300);
  (handleStartCalculation as any).timer = timer; // przechowaj referencję
};

const handleChecksumsCalculated = (newChecksums: { [key: string]: string }) => {
  if ((handleStartCalculation as any).timer) {
    clearTimeout((handleStartCalculation as any).timer);
  }
  setLoadingChecksums(false);
  setChecksums(newChecksums);

  const keys = Object.keys(newChecksums);
  if (keys.length === 2) {
    const [sum1, sum2] = Object.values(newChecksums);
    setIntegrityResult(sum1 === sum2 ? 'passed' : 'failed');
  } else {
    setIntegrityResult(null);
  }
};


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

      {/* Checksum and Integrity Section */}
      {bothUploaded && (
        <div className="mt-12 space-y-8">
          {/* Integrity report or loader */}
          <div
            className={`rounded-xl p-6 text-center border shadow-lg backdrop-blur-md transition-all duration-500 ${
              integrityResult === 'passed'
                ? 'bg-green-900/30 border-green-700 text-green-400 shadow-green-900/20 opacity-100'
                : integrityResult === 'failed'
                ? 'bg-red-900/30 border-red-700 text-red-400 shadow-red-900/20 opacity-100'
                : 'bg-gray-900/30 border-gray-700 text-gray-300 shadow-gray-900/20 opacity-90'
            }`}
          >
            {!loadingChecksums && integrityResult && (
              <>
                <h3 className="text-2xl font-semibold mb-2">Integrity Report</h3>
                {integrityResult === 'passed' ? (
                  <p className="text-lg">Integrity verified — the files are identical.</p>
                ) : (
                  <p className="text-lg">Integrity failed — the files differ.</p>
                )}
              </>
            )}

            {loadingChecksums && (
              <div className="flex flex-col items-center justify-center space-y-3">
                {/* 🔹 Spinner */}
                <div className="w-10 h-10 border-4 border-teal-500 border-t-transparent rounded-full animate-spin"></div>
                <h3 className="text-2xl font-semibold text-teal-400">Analyzing Integrity...</h3>
                <p className="text-sm text-gray-400">Please wait while checksums are calculated.</p>
              </div>
            )}
          </div>

          {/* Checksum results */}
          <CheckSumPanel
            files={[image1, image2].filter(Boolean) as File[]}
            onChecksumsCalculated={handleChecksumsCalculated}
            onStartCalculation={handleStartCalculation} // ✅ nowy callback
          />
        </div>
      )}
    </div>
  );
};

export default SteganoCompare;
