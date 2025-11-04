import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

interface HomeProps {
  setActiveTab: (tab: string) => void;
}

const Home: React.FC<HomeProps> = ({ setActiveTab }) => {
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const goToAnalysis = () => {
    setActiveTab('analysis');
    navigate('/analysis');
  };

  const goToAbout = () => {
    setActiveTab('about');
    navigate('/about');
  };

  // Efekt Matrixa – ciemny i subtelny
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);
    const letters = '01<>/\\{}[]#%$&@ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    const fontSize = 16;
    const columns = Math.floor(width / fontSize);
    const drops = Array(columns).fill(1);

    const draw = () => {
      ctx.fillStyle = 'rgba(0, 0, 0, 0.08)';
      ctx.fillRect(0, 0, width, height);

      ctx.fillStyle = 'rgba(0, 255, 128, 0.35)';
      ctx.font = `${fontSize}px monospace`;

      for (let i = 0; i < drops.length; i++) {
        const text = letters[Math.floor(Math.random() * letters.length)];
        const x = i * fontSize;
        const y = drops[i] * fontSize;

        ctx.fillText(text, x, y);

        if (y > height && Math.random() > 0.975) drops[i] = 0;
        drops[i]++;
      }
    };

    const interval = setInterval(draw, 45);

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', handleResize);
    return () => {
      clearInterval(interval);
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  return (
    <div
      className="relative min-h-screen text-white flex items-center justify-center overflow-hidden m-0 p-0
      bg-[radial-gradient(ellipse_at_top_left,rgba(0,255,150,0.15)_0%,rgba(0,150,100,0.1)_40%,black_100%)]"
      // jeśli chcesz całkowicie czarne tło, usuń linię z bg-[radial-gradient(...)]
    >
      {/* Matrix */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none opacity-90"
      />

      {/* Główna zawartość */}
      <main className="relative z-10 text-center max-w-5xl px-4">
        <h1 className="text-7xl font-bold mb-8 leading-tight">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
            FotoForensics
          </span>
          <br />
          <span className="inline-block mt-2">Image Analysis Tool</span>
        </h1>

        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12">
          Powerful forensics platform for everyone. Detect manipulations and hidden data, check photo source
          analyze metadata, check photo location, verify integrity.
        </p>

        <div className="flex flex-wrap justify-center gap-6">
          <button
            onClick={goToAnalysis}
            className="bg-gradient-to-r from-teal-400 to-green-400 text-black font-medium py-4 px-8 rounded-full hover:opacity-90 transition-all text-lg"
          >
            Start Free Analysis
          </button>

          <button
            onClick={goToAbout}
            className="bg-white/10 backdrop-blur-sm text-white font-medium py-4 px-8 rounded-full hover:bg-white/20 transition-all text-lg"
          >
            Learn More
          </button>
        </div>
      </main>

      <style>
        {`
          footer {
            margin-top: 0 !important;
          }
        `}
      </style>
    </div>
  );
};

export default Home;
