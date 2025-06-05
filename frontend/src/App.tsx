import React, { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './components/Home';
import Check from './components/Check';
import About from './components/About';
import Collaborate from './components/Collaborate';
import Contact from './components/Contact';
import Analysis from './components/Analysis';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('home');

  const renderTab = () => {
    switch (activeTab) {
      case 'home': return <Home setActiveTab={setActiveTab} />;
      case 'check': return <Check />;
      case 'about': return <About setActiveTab={setActiveTab} />;
      case 'collaborate': return <Collaborate />;
      case 'contact': return <Contact />;
      case 'analysis': return <Analysis />;
      default: return <Home setActiveTab={setActiveTab} />;
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
