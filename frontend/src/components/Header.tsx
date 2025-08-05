import React from 'react';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black bg-opacity-95 border-b border-green-900">
      <div className="container mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          <div
            className="flex items-center cursor-pointer hover:opacity-90 transition-opacity"
            onClick={() => setActiveTab('home')}
          >
            <div className="w-10 h-10 rounded-full bg-gradient-to-r from-green-600 to-teal-500 flex items-center justify-center">
              <span className="font-bold text-black text-lg">F</span>
            </div>
            <div className="ml-3">
              <span className="text-xl font-bold text-teal-500">FOTO</span>
              <span className="ml-2 text-xl font-bold text-green-400">FORENSICS</span>
            </div>
          </div>
          <nav className="flex-1 ml-8">
            <ul className="flex space-x-8">
              {['Check', 'About', 'Collaborate', 'Contact'].map((item) => (
                <li key={item}>
                  <button
                    onClick={() => setActiveTab(item.toLowerCase())}
                    className={`px-3 py-2 !rounded-button whitespace-nowrap cursor-pointer relative ${
                      activeTab === item.toLowerCase()
                        ? 'text-green-400 font-medium after:content-[""] after:absolute after:bottom-0 after:left-0 after:w-full after:h-0.5 after:bg-green-400'
                        : 'text-gray-400 hover:text-green-400 transition-colors'
                    }`}
                  >
                    {item}
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      </div>
    </header>
  );
};

export default Header;
