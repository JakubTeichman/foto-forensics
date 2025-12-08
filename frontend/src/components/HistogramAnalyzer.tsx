// 📁 src/components/HistogramAnalyzer.tsx
import React, { useState, useEffect, useRef } from "react";

interface HistogramAnalyzerProps {
  image: File;
}

const HistogramAnalyzer: React.FC<HistogramAnalyzerProps> = ({ image }) => {
  const [activeTab, setActiveTab] = useState<"rgb" | "luminance">("rgb");
  const [histData, setHistData] = useState<any>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Oblicz histogramy
  useEffect(() => {
    if (!image) return;

    const img = new Image();
    img.src = URL.createObjectURL(image);
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      const { data } = ctx.getImageData(0, 0, img.width, img.height);

      const rHist = new Array(256).fill(0);
      const gHist = new Array(256).fill(0);
      const bHist = new Array(256).fill(0);
      const lHist = new Array(256).fill(0);

      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const luminance = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
        rHist[r]++;
        gHist[g]++;
        bHist[b]++;
        lHist[luminance]++;
      }

      const maxCount = Math.max(...rHist, ...gHist, ...bHist, ...lHist);
      setHistData({ rHist, gHist, bHist, lHist, maxCount });
    };
  }, [image]);

  // Render histogramu na <canvas>
  useEffect(() => {
    if (!histData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { rHist, gHist, bHist, lHist, maxCount } = histData;
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0f0f0f";
    ctx.fillRect(0, 0, width, height);

    const drawLine = (arr: number[], color: string) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.2;
      for (let i = 0; i < 256; i++) {
        const x = (i / 255) * width;
        const y = height - (arr[i] / maxCount) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    if (activeTab === "rgb") {
      drawLine(rHist, "#ff4d4d");
      drawLine(gHist, "#4dff4d");
      drawLine(bHist, "#4d94ff");
    } else {
      // Luminancja z tłem tonalnym
      const shadowEnd = width * 0.33;
      const midtoneEnd = width * 0.67;

      ctx.fillStyle = "rgba(50,50,50,0.2)";
      ctx.fillRect(0, 0, shadowEnd, height);
      ctx.fillStyle = "rgba(80,80,80,0.2)";
      ctx.fillRect(shadowEnd, 0, midtoneEnd - shadowEnd, height);
      ctx.fillStyle = "rgba(120,120,120,0.2)";
      ctx.fillRect(midtoneEnd, 0, width - midtoneEnd, height);

      drawLine(lHist, "#00e0ff");
    }
  }, [histData, activeTab]);

  return (
    <div className="bg-[#0f0f0f] border border-gray-800 text-gray-200 rounded-2xl shadow-lg overflow-hidden">
      <div className="border-b border-gray-800 flex">
        {["rgb", "luminance"].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as "rgb" | "luminance")}
            className={`flex-1 py-3 text-sm font-medium transition-all duration-200 ${
              activeTab === tab
                ? "border-b-2 border-cyan-400 text-cyan-400"
                : "text-gray-500 hover:text-gray-300"
            }`}
          >
            {tab === "rgb" ? "RGB Histogram" : "Luminance Histogram"}
          </button>
        ))}
      </div>

      <div className="p-4">
        <canvas ref={canvasRef} width={600} height={240} className="w-full" />
        {activeTab === "luminance" && (
          <div className="flex justify-between text-xs text-gray-500 mt-2 px-1">
            <span>Shadows</span>
            <span>Midtones</span>
            <span>Highlights</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default HistogramAnalyzer;
