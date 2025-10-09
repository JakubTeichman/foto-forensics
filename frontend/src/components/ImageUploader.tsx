import React from 'react';

interface Props {
  title: string;
  previewUrl?: string | null;            // dla single
  previewUrls?: string[];                 // dla multiple
  setPreviewUrl?: (url: string | null) => void;
  setPreviewUrls?: (urls: string[]) => void;
  setSelectedFile?: (file: File | null) => void;
  setSelectedFiles?: (files: File[]) => void;
  id: string;
  multiple?: boolean;
  maxVisible?: number;                    // ile miniatur pokazać od razu
}

const ImageUploader: React.FC<Props> = ({
  title,
  previewUrl,
  previewUrls,
  setPreviewUrl,
  setPreviewUrls,
  setSelectedFile,
  setSelectedFiles,
  id,
  multiple = false,
  maxVisible = 9, // domyślnie 9, żeby zrobić 3x3 grid
}) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;

    if (multiple) {
      const files = Array.from(e.target.files);
      setSelectedFiles && setSelectedFiles(files);

      const urls: string[] = [];
      files.forEach(file => {
        const reader = new FileReader();
        reader.onloadend = () => {
          urls.push(reader.result as string);
          if (urls.length === files.length) setPreviewUrls && setPreviewUrls(urls);
        };
        reader.readAsDataURL(file);
      });
    } else {
      const file = e.target.files[0];
      if (!file) return;

      setSelectedFile && setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl && setPreviewUrl(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
      <h3 className="text-2xl font-medium mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
        {title}
      </h3>
      <div className="border-2 border-dashed border-teal-800/30 rounded-xl p-4 text-center bg-black/20">
        {multiple ? (
          previewUrls && previewUrls.length > 0 ? (
            <div className="grid grid-cols-3 gap-2 justify-center">
              {previewUrls.slice(0, maxVisible).map((url, index) => (
                <img
                  key={index}
                  src={url}
                  alt={`${title} ${index + 1}`}
                  className="w-20 h-20 object-cover rounded-lg"
                />
              ))}
              {previewUrls.length > maxVisible && (
                <div className="w-20 h-20 flex items-center justify-center bg-gray-700 text-white font-semibold rounded-lg text-sm">
                  +{previewUrls.length - maxVisible}
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <i className="fas fa-cloud-upload-alt text-4xl text-green-500 mb-4"></i>
              <input type="file" id={id} className="hidden" onChange={handleChange} multiple />
              <label htmlFor={id} className="cursor-pointer text-gray-300 hover:text-white">
                Upload {title} (Multiple)
              </label>
            </div>
          )
        ) : previewUrl ? (
          <img src={previewUrl} alt={title} className="max-h-[300px] mx-auto" />
        ) : (
          <div className="flex flex-col items-center">
            <i className="fas fa-cloud-upload-alt text-4xl text-green-500 mb-4"></i>
            <input type="file" id={id} className="hidden" onChange={handleChange} />
            <label htmlFor={id} className="cursor-pointer text-gray-300 hover:text-white">
              Upload {title}
            </label>
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;
