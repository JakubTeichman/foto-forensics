import React from 'react';
import { useNavigate } from 'react-router-dom';

interface AboutProps {
  setActiveTab: (tab: string) => void;
}

const About: React.FC<AboutProps> = ({ setActiveTab }) => {
  const navigate = useNavigate();

  const handleStartAnalysis = () => {
    setActiveTab('analysis');
    navigate('/analysis');
  };

  const modules = [
    {
      icon: 'fas fa-project-diagram',
      title: 'Image Comparison',
      description: `Compare an evidence image with reference photos to confirm if they come from the same device.

Use reference images with identical parameters for best accuracy.

Denoising methods:
• BM3D – more precise, slower  
• Wavelet – faster, slightly less accurate.`,
      link: '/compare'
    },
    {
      icon: 'fas fa-microscope',
      title: 'Image Analysis',
      description: `Explore binary and metadata structure. Inspect hexadecimal and ASCII data, check EXIF and GPS info, visualize locations, and detect steganography.`,
      link: '/analysis'
    },
    {
      icon: 'fas fa-link',
      title: 'Steganography + Integrity',
      description: `Compare two images to verify integrity and reveal hidden data or possible tampering.`,
      link: '/stegano-compare'
    }
  ];

  return (
    <div className="relative flex flex-col items-center overflow-hidden min-h-screen pt-24 pb-24 text-white bg-black">
      {/* 🔹 Zawartość */}
      <div className="relative z-10 w-full flex flex-col items-center">
        {/* Nagłówek */}
        <div className="relative max-w-4xl text-center mb-16">
          <h2 className="text-3xl font-bold mb-6 text-teal-400 leading-tight">
            Advanced Photo Forensics Analysis
          </h2>
          <p className="text-lg text-gray-300 mb-8">
            Detect image manipulation, analyze metadata, and verify authenticity with cutting-edge forensic tools.
          </p>
          <button
            onClick={handleStartAnalysis}
            className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-8 py-3 rounded-full hover:from-teal-600 hover:to-green-500 transition-all"
          >
            Start Analysis
          </button>
        </div>

        {/* Trzy karty */}
        <div className="relative grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl mb-20">
          {[
            {
              icon: 'fas fa-search',
              title: 'Detect Manipulation',
              description: 'Identify altered regions, clone stamps, and digital inconsistencies with high precision.'
            },
            {
              icon: 'fas fa-code',
              title: 'Metadata Analysis',
              description: 'Extract EXIF and GPS data to confirm image source, parameters, and authenticity.'
            },
            {
              icon: 'fas fa-shield-alt',
              title: 'Forensic Reports',
              description: 'Generate structured forensic documentation for professional and legal use.'
            }
          ].map((card, i) => (
            <div
              key={i}
              className="bg-gray-900/60 p-6 rounded-xl border border-teal-800 hover:border-teal-500 hover:bg-gray-900/70 transition-all"
            >
              <div className="text-teal-400 text-4xl mb-4">
                <i className={card.icon}></i>
              </div>
              <h3 className="text-xl font-bold mb-3 text-white">{card.title}</h3>
              <p className="text-gray-400">{card.description}</p>
            </div>
          ))}
        </div>

        {/* Moduły */}
        <div className="relative flex flex-col space-y-8 w-full max-w-5xl px-4">
          {modules.map((mod, i) => (
            <div
              key={i}
              className="flex flex-col md:flex-row items-center md:items-start bg-gray-900/50 p-6 rounded-2xl border border-teal-700 hover:border-teal-500 shadow-md hover:shadow-teal-600/20 transition-all"
            >
              <div className="text-teal-400 text-4xl mb-4 md:mb-0 md:mr-6 shrink-0">
                <i className={mod.icon}></i>
              </div>
              <div className="flex flex-col text-left md:text-justify">
                <h3 className="text-2xl font-bold mb-3 text-white">{mod.title}</h3>
                <p className="text-gray-400 mb-6 whitespace-pre-line leading-relaxed">
                  {mod.description}
                </p>
                <button
                  onClick={() => navigate(mod.link)}
                  className="self-start bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-6 py-2 rounded-full hover:from-teal-600 hover:to-green-500 transition-all"
                >
                  Go to Module
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Marginesy nav/footer */}
      <style>
        {`
          nav {
            margin-bottom: 0 !important;
          }
          footer {
            margin-top: 0 !important;
          }
        `}
      </style>
    </div>
  );
};

export default About;
