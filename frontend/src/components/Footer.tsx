import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white/5 backdrop-blur-sm py-6 mt-8 border-t border-white/10">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-4 md:mb-0">
            <div className="flex items-center">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-teal-500 to-green-400 flex items-center justify-center mr-2">
                <span className="font-bold text-black text-sm">F</span>
              </div>
              <span className="text-lg font-semibold text-teal-400">FOTO FORENSICS</span>
            </div>
            <p className="text-gray-400 mt-1 max-w-md text-sm leading-relaxed">
              Advanced image analysis tools for professionals in digital forensics, law enforcement, and media verification.
            </p>
          </div>

          <div className="flex space-x-5">
            {['twitter', 'linkedin', 'github', 'youtube'].map(icon => (
              <a
                key={icon}
                href="#"
                className="text-gray-400 hover:text-teal-400 transition-colors"
              >
                <i className={`fab fa-${icon} text-lg`}></i>
              </a>
            ))}
          </div>
        </div>

        <div className="border-t border-gray-800 mt-6 pt-4 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-500 text-xs">
            © 2025 FOTO FORENSICS. All rights reserved.
          </p>
          <div className="flex mt-3 md:mt-0">
            <a href="#" className="text-gray-500 hover:text-gray-300 text-xs mx-2">
              Privacy Policy
            </a>
            <a href="#" className="text-gray-500 hover:text-gray-300 text-xs mx-2">
              Terms of Service
            </a>
            <a href="#" className="text-gray-500 hover:text-gray-300 text-xs mx-2">
              Contact Us
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
