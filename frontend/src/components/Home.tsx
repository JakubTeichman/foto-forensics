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
      <nav className="absolute top-0 left-0 w-full py-6 px-8">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center space-x-2">
            <div className="relative w-12 h-12">
              <div className="absolute inset-0 bg-gradient-to-r from-teal-400 to-green-400 rounded-lg transform rotate-45"></div>
              <div className="absolute inset-[2px] bg-black rounded-lg transform rotate-45 flex items-center justify-center">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 font-bold text-xl transform -rotate-45">F</span>
              </div>
            </div>
            <span className="text-xl font-bold">Foto Forensics</span>
          </div>
          <button className="px-6 py-2 bg-white/10 backdrop-blur-sm rounded-full text-sm font-medium hover:bg-white/20 transition-all">
            Sign In
          </button>
        </div>
      </nav>
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
      <div className="absolute bottom-12 left-0 w-full">
        <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8 px-4">
          {[
            {
              icon: 'fas fa-shield-alt',
              title: 'Secure Analysis',
              description: 'Your data is protected with enterprise-grade encryption and security measures.'
            },
            {
              icon: 'fas fa-bolt',
              title: 'Real-time Results',
              description: 'Get instant analysis results powered by advanced AI algorithms.'
            },
            {
              icon: 'fas fa-chart-line',
              title: 'Detailed Reports',
              description: 'Comprehensive forensic reports with visual indicators and metrics.'
            }
          ].map((feature, i) => (
            <div key={i} className="bg-white/5 backdrop-blur-sm p-6 rounded-2xl border border-white/10">
              <i className={`${feature.icon} text-2xl text-teal-400 mb-4`} />
              <h3 className="text-lg font-medium mb-2">{feature.title}</h3>
              <p className="text-gray-400 text-sm">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Home;
