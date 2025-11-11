import React, { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

interface HomeProps {
  setActiveTab: (tab: string) => void;
}

const Home: React.FC<HomeProps> = ({ setActiveTab }) => {
  const navigate = useNavigate();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [title, setTitle] = useState("FOTOFORENSICS");
  const [subtitle, setSubtitle] = useState("Image Analysis Tool");

  const goToAnalysis = () => {
    setActiveTab("analysis");
    navigate("/analysis");
  };

  const goToAbout = () => {
    setActiveTab("about");
    navigate("/about");
  };

  // 🔢 Efekt Matrixa (tylko na stronie głównej)
  useEffect(() => {
  const canvas = canvasRef.current;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // ✅ Ostre krawędzie i lepszy kontrast
  ctx.imageSmoothingEnabled = false;
  ctx.shadowBlur = 0;

  let width = (canvas.width = window.innerWidth);
  let height = (canvas.height = window.innerHeight);
  const letters = "01<>/\\{}[]#%$&@ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");
  const fontSize = 18; // lekko większa czcionka = lepsza czytelność
  const columns = Math.floor(width / fontSize);
  const drops = Array(columns).fill(1);

  const draw = () => {
    // 🔧 Mniejsze rozmazanie tła
    ctx.fillStyle = "rgba(0, 0, 0, 0.25)";
    ctx.fillRect(0, 0, width, height);

    // 🔥 Jaśniejsze, wyraźniejsze znaki
    ctx.fillStyle = "rgba(0, 255, 180, 0.85)";
    ctx.font = `${fontSize}px 'Share Tech Mono', monospace`;
    ctx.textBaseline = "top";

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
  window.addEventListener("resize", handleResize);

  return () => {
    clearInterval(interval);
    window.removeEventListener("resize", handleResize);
  };
}, []);


  // 🧩 Efekt szyfrowania tekstu (cyfrowy reveal)
const scrambleText = (
  text: string,
  setText: React.Dispatch<React.SetStateAction<string>>
) => {
  const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*";
  let iteration = 0;

  const interval = setInterval(() => {
    setText((prev: string) =>
      prev
        .split("")
        .map((_: string, i: number) => {
          if (i < iteration) {
            return text[i];
          }
          return chars[Math.floor(Math.random() * chars.length)];
        })
        .join("")
    );

    iteration += 0.5;
    if (iteration >= text.length) clearInterval(interval);
  }, 50);
};


  useEffect(() => {
    scrambleText("FOTOFORENSICS", setTitle);
    setTimeout(() => scrambleText("Image Analysis Tool", setSubtitle), 800);
  }, []);

  return (
    <div
      className="relative min-h-screen text-white flex items-center justify-center overflow-hidden m-0 p-0 
      bg-[radial-gradient(ellipse_at_top_left,rgba(0,255,180,0.2)_0%,rgba(0,150,120,0.15)_40%,rgba(0,0,0,0.9)_100%)]"
    >
      {/* Matrix Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none opacity-90"
      />

      {/* Szklany blur warstwa */}
      <div className="absolute inset-0 backdrop-blur-[2px] bg-black/10" />

      {/* Główna treść */}
      <main className="relative z-10 text-center max-w-5xl px-6">
        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-5xl md:text-7xl font-bold mb-8 leading-tight font-mono tracking-wide"
        >
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 drop-shadow-[0_0_10px_rgba(0,255,180,0.6)]">
            {title}
          </span>
          <br />
          <span className="inline-block mt-3 text-gray-300">{subtitle}</span>
        </motion.h1>

        <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-12 font-light">
          Powerful forensic platform to detect image manipulation, analyze
          metadata, and verify authenticity with modern AI-based tools.
        </p>

        <div className="flex flex-wrap justify-center gap-6">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
            onClick={goToAnalysis}
            className="bg-gradient-to-r from-teal-400 to-green-400 text-black font-semibold py-4 px-8 rounded-full hover:opacity-90 transition-all text-lg shadow-lg"
          >
            Start Free Analysis
          </motion.button>

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.98 }}
            onClick={goToAbout}
            className="bg-white/10 backdrop-blur-md text-white font-medium py-4 px-8 rounded-full hover:bg-white/20 transition-all text-lg"
          >
            Learn More
          </motion.button>
        </div>
      </main>

      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

          footer {
            margin-top: 0 !important;
          }

          h1, p, button {
            user-select: none;
          }
        `}
      </style>
    </div>
  );
};

export default Home;
