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

  // Shared AbortController ref
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    // Abort previous pending requests when changing route
    if (controllerRef.current) {
      console.log('Aborting pending requests due to route change.');
      controllerRef.current.abort();
    }

    // Create a new controller for the new route
    controllerRef.current = new AbortController();

    // Cleanup on unmount
    return () => {
      if (controllerRef.current) {
        controllerRef.current.abort();
      }
    };
  }, [location.pathname]);

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
          <Route path="/add-reference" element={<AddReference />} />
          {/* Pass signal as prop */}
          <Route
            path="/analysis"
            element={<Analysis setActiveTab={setActiveTab} abortSignal={controllerRef.current?.signal} />}
          />
          <Route
            path="/stegano-compare"
            element={<SteganoCompare setActiveTab={setActiveTab}/>}
          />
        </Routes>
      </main>
      <Footer /> 
    </div>
  );
};

export default App;
