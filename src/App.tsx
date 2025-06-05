import React, { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './components/Home';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');

  const renderTab = () => {
    switch (activeTab) {
      case 'home':
        return <Home />;
      // Add other cases for different tabs
      default:
        return <Home />;
    }
  };

  return (
    <div className="min-h-screen bg-black text-white">
      <Header activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="pt-16">{renderTab()}</main>
      <Footer />
    </div>
  );
};

export default App;