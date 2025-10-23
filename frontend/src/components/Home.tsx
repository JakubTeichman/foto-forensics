import React from 'react';
import { useNavigate } from 'react-router-dom';

interface HomeProps {
  setActiveTab: (tab: string) => void;
}

const Home: React.FC<HomeProps> = ({ setActiveTab }) => {
  const navigate = useNavigate();

  const goToAnalysis = () => {
    setActiveTab('analysis');
    navigate('/analysis');
  };

  const goToAbout = () => {
    setActiveTab('about');
    navigate('/about');
  };

  return (
    <div className="relative min-h-[calc(100vh-0px)] bg-black text-white flex items-center justify-center overflow-hidden">
      {/* Tło gradientowe */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 w-[900px] h-[900px] -translate-x-1/2 -translate-y-1/2 bg-gradient-to-r from-teal-500/20 to-green-400/20 rounded-full blur-[160px]" />
      </div>

      {/* Zawartość główna */}
      <div className="relative z-10 text-center max-w-5xl px-4">
        <h1 className="text-7xl font-bold mb-8 leading-tight">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
            Advanced
          </span>
          <br />
          <span className="inline-block mt-2">Image Analysis Tool</span>
        </h1>

        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12">
          Powerful forensics platform for professionals. Detect manipulations, analyze metadata, and verify authenticity with precision.
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
      </div>
    </div>
  );
};

export default Home;
