// HeatmapViewer.tsx
import React, { useState } from "react";

interface HeatmapViewerProps {
  title: string;
  imageData?: string | null; // base64 or data URI
}

const detectMime = (b64?: string | null) => {
  if (!b64) return "image/png";
  if (b64.startsWith("data:image/")) {
    const m = b64.match(/^data:(image\/[a-zA-Z0-9.+-]+);base64,/);
    return m ? m[1] : "image/png";
  }
  // fallback guess by header bytes (rare; we just return png)
  return "image/png";
};

const toDataUri = (b64?: string | null) => {
  if (!b64) return null;
  if (b64.startsWith("data:image/")) return b64;
  // assume png
  return `data:image/png;base64,${b64}`;
};

const HeatmapViewer: React.FC<HeatmapViewerProps> = ({ title, imageData }) => {
  const [open, setOpen] = useState(false);
  const dataUri = toDataUri(imageData ?? null);

  return (
    <>
      <div className="bg-neutral-800 border border-neutral-700 mt-4 rounded-2xl overflow-hidden">
        <div className="border-b border-neutral-700 px-3 py-2 flex items-center justify-between">
          <h3 className="text-sm text-gray-300">{title}</h3>
          <button
            onClick={() => setOpen(true)}
            className="text-xs text-teal-300 hover:underline"
            aria-label={`Open ${title}`}
            disabled={!dataUri}
          >
            Open
          </button>
        </div>

        <div className="flex justify-center items-center p-3">
          {dataUri ? (
            <img
              src={dataUri}
              alt={title}
              className="rounded-lg shadow-md max-w-full h-auto cursor-pointer"
              onClick={() => setOpen(true)}
              onError={(e) => {
                // fallback: if img load fails, show text
                const el = e.currentTarget as HTMLImageElement;
                el.style.display = "none";
              }}
            />
          ) : (
            <div className="text-xs text-gray-500 italic">No heatmap available</div>
          )}
        </div>
      </div>

      {/* Modal */}
      {open && dataUri && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="bg-gray-900 border border-neutral-700 rounded-2xl overflow-auto max-w-4xl w-full p-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm text-gray-200">{title}</h3>
              <button className="text-sm text-teal-300" onClick={() => setOpen(false)}>Close</button>
            </div>
            <div className="flex justify-center">
              <img src={dataUri} alt={title} className="max-h-[78vh] rounded-lg" />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default HeatmapViewer;
