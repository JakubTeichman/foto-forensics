import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './components/Home';
import Check from './components/Check';
import About from './components/About';
import Collaborate from './components/Collaborate';
import Contact from './components/Contact';
import Analysis from './components/Analysis';
import SteganoCompare from './components/SteganoCompare'; 

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');

  return (
    <div className="min-h-screen bg-black text-white">
      <Header />
      <main className="pt-16">
        <Routes>
          <Route path="/" element={<Home setActiveTab={setActiveTab} />} />
          <Route path="/about" element={<About setActiveTab={setActiveTab} />} />
          <Route path="/compare" element={<Check setActiveTab={setActiveTab} />} />
          <Route path="/collaborate" element={<Collaborate setActiveTab={setActiveTab} />} />
          <Route path="/contact" element={<Contact setActiveTab={setActiveTab} />} />
          <Route path="/analysis" element={<Analysis setActiveTab={setActiveTab} />} />
          <Route path="/stegano-compare" element={<SteganoCompare setActiveTab={setActiveTab} />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
};

export default App;
