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

  const handleDrop = (e: DragEvent<HTMLDivElement | HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement | HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement | HTMLLabelElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  return (
    <div
      className="relative rounded-2xl p-8 border border-teal-400/30 shadow-[0_0_20px_rgba(0,255,200,0.15)]
      bg-gradient-to-br from-gray-900/40 to-black/60 backdrop-blur-xl overflow-hidden"
    >
      {/* Glass reflection effect */}
      <div className="absolute inset-0 bg-gradient-to-tl from-teal-500/10 to-transparent pointer-events-none" />
      <div className="absolute -inset-[2px] bg-[radial-gradient(circle_at_20%_20%,rgba(0,255,200,0.15),transparent_60%)] pointer-events-none rounded-2xl" />

      <h3 className="text-2xl font-semibold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-300 to-green-400 drop-shadow-lg tracking-wide relative z-10">
        {title}
      </h3>

      {/* Drag & Drop Area */}
      <div
        className={`relative border-2 border-dashed rounded-xl p-10 text-center transition-all duration-300 
        ${
          isDragging
            ? "border-teal-400/80 bg-teal-500/10 scale-[1.02] shadow-[0_0_30px_rgba(0,255,180,0.3)]"
            : "border-teal-600/40 hover:border-teal-400/80 bg-gray-900/30"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {multiple ? (
          previewUrls && previewUrls.length > 0 ? (
            <div className="grid grid-cols-3 gap-3 justify-center relative z-10">
              {previewUrls.slice(0, maxVisible).map((url, index) => (
                <div
                  key={index}
                  className="relative overflow-hidden rounded-lg border border-teal-700/50 bg-gray-800/40 backdrop-blur-sm"
                >
                  <img
                    src={url}
                    alt={`${title} ${index + 1}`}
                    className="w-full h-24 object-cover rounded-md hover:scale-105 transition-transform"
                  />
                </div>
              ))}
              {previewUrls.length > maxVisible && (
                <div className="w-full h-24 flex items-center justify-center bg-gray-800/60 text-teal-300 font-semibold rounded-md text-sm border border-teal-700/40 backdrop-blur-sm">
                  +{previewUrls.length - maxVisible}
                </div>
              )}
            </div>
          ) : (
            <label
              htmlFor={id}
              className="cursor-pointer flex flex-col items-center w-full relative z-10"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <i className="fas fa-cloud-upload-alt text-6xl text-teal-400 mb-4 drop-shadow-[0_0_10px_rgba(0,255,200,0.3)]"></i>
              <span className="text-lg mb-2 text-teal-200">Drag & drop or click to upload</span>
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
          <div className="relative rounded-lg overflow-hidden border border-teal-600/40 bg-gray-900/40 backdrop-blur-sm shadow-lg">
            <img
              src={previewUrl}
              alt={title}
              className="max-h-[300px] mx-auto rounded-lg object-contain"
            />
          </div>
        ) : (
          <label
            htmlFor={id}
            className="cursor-pointer flex flex-col items-center w-full relative z-10"
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <i className="fas fa-cloud-upload-alt text-6xl text-teal-400 mb-4 drop-shadow-[0_0_10px_rgba(0,255,200,0.3)]"></i>
            <span className="text-lg mb-2 text-teal-200">Drag & drop or click to upload</span>
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
