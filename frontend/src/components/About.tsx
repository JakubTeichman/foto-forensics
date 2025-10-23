import React from 'react';
import { useNavigate } from 'react-router-dom';

interface AboutProps {
  setActiveTab: (tab: string) => void;
}

const About: React.FC<AboutProps> = ({ setActiveTab }) => {
  const navigate = useNavigate(); // <--- DODAJ TO

  const handleStartAnalysis = () => {
    setActiveTab('analysis');  // zostaw, żeby zachować spójność z resztą logiki
    navigate('/analysis');     // <--- TO DODAJE PRAWDZIWE PRZEKIEROWANIE
  };


  return (
    <div className="flex flex-col items-center mt-8"> {/* 🔹 odstęp od góry */}
      <div className="max-w-4xl text-center mb-12">
        <h2 className="text-3xl font-bold mb-6 text-teal-400 leading-tight">
          Advanced Photo Forensics Analysis
        </h2>
        <p className="text-lg text-gray-300 mb-8">
          Detect image manipulation, analyze metadata, and verify authenticity with our state-of-the-art forensic tools.
        </p>
        <button
          onClick={handleStartAnalysis}
          className="bg-gradient-to-r from-teal-500 to-green-400 text-black font-bold px-8 py-3 rounded-full hover:from-teal-600 hover:to-green-500 transition-all"
        >
          Start Analysis
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 w-full max-w-5xl">
        {[
          {
            icon: 'fas fa-search',
            title: 'Detect Manipulation',
            description: 'Identify altered regions, clone stamps, and other digital manipulations with precision.'
          },
          {
            icon: 'fas fa-code',
            title: 'Metadata Analysis',
            description: 'Extract and analyze hidden EXIF data to verify image origin and authenticity.'
          },
          {
            icon: 'fas fa-shield-alt',
            title: 'Forensic Reports',
            description: 'Generate detailed forensic reports suitable for legal and professional use.'
          }
        ].map((card, i) => (
          <div key={i} className="bg-gray-900 bg-opacity-60 p-6 rounded-xl border border-teal-800">
            <div className="text-teal-400 text-4xl mb-4">
              <i className={card.icon}></i>
            </div>
            <h3 className="text-xl font-bold mb-3 text-white">{card.title}</h3>
            <p className="text-gray-400">{card.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default About;
