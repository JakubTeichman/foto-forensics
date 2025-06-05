import React from 'react';

interface Props {
  title: string;
  previewUrl: string | null;
  setPreviewUrl: (url: string | null) => void;
  setSelectedFile: (file: File | null) => void;
  id: string;
}

const ImageUploader: React.FC<Props> = ({ title, previewUrl, setPreviewUrl, setSelectedFile, id }) => {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
      <h3 className="text-2xl font-medium mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">{title}</h3>
      <div className="border-2 border-dashed border-teal-800/30 rounded-xl p-8 text-center bg-black/20">
        {previewUrl ? (
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
