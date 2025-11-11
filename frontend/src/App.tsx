import React, { useEffect, useRef, useState } from 'react';
import { Routes, Route, useLocation } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import Home from './components/Home';
import Check from './components/Check';
import About from './components/About';
import Collaborate from './components/Collaborate';
import Contact from './components/Contact';
import Analysis from './components/Analysis';
import SteganoCompare from './components/SteganoCompare';
import AddReference from './components/AddReference';

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('home');
  const location = useLocation();
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (controllerRef.current) {
      console.log('Aborting pending requests due to route change.');
      controllerRef.current.abort();
    }
    controllerRef.current = new AbortController();

    return () => {
      if (controllerRef.current) controllerRef.current.abort();
    };
  }, [location.pathname]);

  return (
    <div className="relative min-h-screen bg-black text-white overflow-hidden">

      {/* 🧭 Główna zawartość */}
      <div className="relative z-10 flex flex-col min-h-screen">
        <Header />
        <main className="flex-grow pt-16">
          <Routes>
            <Route path="/" element={<Home setActiveTab={setActiveTab} />} />
            <Route path="/about" element={<About setActiveTab={setActiveTab} />} />
            <Route path="/compare" element={<Check setActiveTab={setActiveTab} />} />
            <Route path="/collaborate" element={<Collaborate setActiveTab={setActiveTab} />} />
            <Route path="/contact" element={<Contact setActiveTab={setActiveTab} />} />
            <Route path="/add-reference" element={<AddReference />} />
            <Route
              path="/analysis"
              element={
                <Analysis
                  setActiveTab={setActiveTab}
                  abortSignal={controllerRef.current?.signal}
                />
              }
            />
            <Route
              path="/stegano-compare"
              element={<SteganoCompare setActiveTab={setActiveTab} />}
            />
          </Routes>
        </main>
        <Footer />
      </div>
    </div>
  );
};

export default App;
