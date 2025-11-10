import React from "react";

interface HeatmapViewerProps {
  title: string;
  imageData: string; // Base64 string
}

const HeatmapViewer: React.FC<HeatmapViewerProps> = ({ title, imageData }) => {
  return (
    <div className="bg-neutral-800 border border-neutral-700 mt-4 rounded-2xl overflow-hidden">
      <div className="border-b border-neutral-700 px-3 py-2">
        <h3 className="text-sm text-gray-300">{title}</h3>
      </div>
      <div className="flex justify-center items-center p-3">
        <img
          src={`data:image/png;base64,${imageData}`}
          alt={title}
          className="rounded-lg shadow-md max-w-full h-auto"
        />
      </div>
    </div>
  );
};

export default HeatmapViewer;
