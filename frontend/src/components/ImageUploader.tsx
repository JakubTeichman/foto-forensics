import React, { useState, DragEvent } from "react";

interface Props {
  title: string;
  previewUrl?: string | null;
  previewUrls?: string[];
  setPreviewUrl?: (url: string | null) => void;
  setPreviewUrls?: (urls: string[]) => void;
  setSelectedFile?: (file: File | null) => void;
  setSelectedFiles?: (files: File[]) => void;
  id: string;
  multiple?: boolean;
  maxVisible?: number;
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
  maxVisible = 9,
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleFiles = (files: FileList) => {
    if (multiple) {
      const fileArray = Array.from(files);
      setSelectedFiles && setSelectedFiles(fileArray);

      const urls: string[] = [];
      fileArray.forEach((file) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          urls.push(reader.result as string);
          if (urls.length === fileArray.length) {
            setPreviewUrls && setPreviewUrls(urls);
          }
        };
        reader.readAsDataURL(file);
      });
    } else {
      const file = files[0];
      if (!file) return;
      setSelectedFile && setSelectedFile(file);

      const reader = new FileReader();
      reader.onloadend = () => setPreviewUrl && setPreviewUrl(reader.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) handleFiles(e.target.files);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  return (
    <div className="bg-gray-900 rounded-xl p-8 border border-teal-800 shadow-lg">
      <h3 className="text-2xl font-medium mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
        {title}
      </h3>

      {/* Drag & Drop Area */}
      <div
        className={`border-2 border-dashed rounded-xl p-10 text-center transition-colors duration-200 ${
          isDragging
            ? "border-teal-500 bg-teal-900/20"
            : "border-teal-700 hover:border-teal-500 bg-gray-800/40"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {multiple ? (
          previewUrls && previewUrls.length > 0 ? (
            <div className="grid grid-cols-3 gap-2 justify-center">
              {previewUrls.slice(0, maxVisible).map((url, index) => (
                <img
                  key={index}
                  src={url}
                  alt={`${title} ${index + 1}`}
                  className="w-20 h-20 object-cover rounded-lg border border-gray-700"
                />
              ))}
              {previewUrls.length > maxVisible && (
                <div className="w-20 h-20 flex items-center justify-center bg-gray-700 text-white font-semibold rounded-lg text-sm">
                  +{previewUrls.length - maxVisible}
                </div>
              )}
            </div>
          ) : (
            <label htmlFor={id} className="cursor-pointer flex flex-col items-center">
              <i className="fas fa-file-image text-5xl text-teal-500 mb-4"></i>
              <span className="text-lg mb-2">Drag & drop or click to upload images</span>
              <span className="text-sm text-gray-400">You can select multiple files</span>
              <input
                type="file"
                id={id}
                multiple
                accept="image/*"
                className="hidden"
                onChange={handleChange}
              />
            </label>
          )
        ) : previewUrl ? (
          <img
            src={previewUrl}
            alt={title}
            className="max-h-[300px] mx-auto rounded-lg border border-gray-700"
          />
        ) : (
          <label htmlFor={id} className="cursor-pointer flex flex-col items-center">
            <i className="fas fa-file-image text-5xl text-teal-500 mb-4"></i>
            <span className="text-lg mb-2">Drag & drop or click to upload image</span>
            <span className="text-sm text-gray-400">Supported: JPG, PNG, TIFF, RAW</span>
            <input
              type="file"
              id={id}
              accept="image/*"
              className="hidden"
              onChange={handleChange}
            />
          </label>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;
