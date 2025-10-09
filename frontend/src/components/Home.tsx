import React from 'react';

interface HomeProps {
  setActiveTab: (tab: string) => void;
}

const Home: React.FC<HomeProps> = ({ setActiveTab }) => {
  return (
    <div className="relative min-h-screen bg-black text-white flex flex-col items-center justify-center px-4">
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-to-r from-teal-500/20 to-green-400/20 rounded-full blur-[120px]" />
      </div>
      <div className="text-center max-w-5xl mx-auto pt-20">
        <h1 className="text-7xl font-bold mb-8">
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">Advanced</span>
          <br />
          <span className="inline-block mt-2">Image Analysis Tool</span>
        </h1>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12">
          Powerful forensics platform for professionals. Detect manipulations, analyze metadata, and verify authenticity with precision.
        </p>
        <div className="flex flex-wrap justify-center gap-6">
          <button
            onClick={() => setActiveTab('check')}
            className="bg-gradient-to-r from-teal-400 to-green-400 text-black font-medium py-4 px-8 rounded-full hover:opacity-90 transition-all text-lg"
          >
            Start Free Analysis
          </button>
          <button
            onClick={() => setActiveTab('about')}
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
