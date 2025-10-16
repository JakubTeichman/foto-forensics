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
    const [warning, setWarning] = useState<boolean>(false); // 🔴 Dodane

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

        images2.forEach((file) => {
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

          // 🔴 Jeśli backend zwróci ostrzeżenie o rozdzielczości
          if (data.size_warning) {
            setWarning(true);
          }
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

        {bothUploaded && (
          <div className="mt-8 bg-gray-900 bg-opacity-50 rounded-xl p-6 border border-green-900">
            <h3 className="text-xl font-medium mb-4 text-green-400">Comparison Report</h3>

            {warning && (
              <div className="mt-4 mb-4 p-4 rounded-2xl bg-red-500/10 border border-red-500/30 text-red-400 backdrop-blur-sm shadow-md shadow-red-900/20 text-sm font-medium flex items-start space-x-2">
                <span className="text-lg leading-none">⚠️</span>
                <span>
                  The uploaded images have <strong>different resolutions</strong>. This may slightly affect the accuracy of the similarity result.<br />
                  For best results, use reference images captured in the <strong>same resolution</strong>.
                </span>
              </div>
            )}


            <div className="flex flex-col md:flex-row md:justify-between md:items-center gap-6">
              <div className="space-y-2 text-gray-300">
                <p>File Size Difference: —</p>
                <p>Resolution Match: {warning ? '❌ No' : '✅ Yes'}</p>
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
