import React from 'react';

const Footer: React.FC = () => {
  return (
    <footer className="bg-white/5 backdrop-blur-sm py-6 mt-12 border-t border-white/10">
      <div className="container mx-auto px-6">
        <div className="flex flex-col md:flex-row justify-between items-center">
          <div className="mb-2 md:mb-0">
            <div className="flex items-center">
              <div className="w-10 h-10 rounded-full bg-gradient-to-r from-teal-500 to-green-400 flex items-center justify-center mr-3">
                <span className="font-bold text-black">F</span>
              </div>
              <span className="text-xl font-bold text-teal-400">FOTO FORENSICS</span>
            </div>
            <p className="text-gray-400 mt-2 max-w-md">
              Image analysis tool for digital forensics.
            </p>
          </div>
          <div className="flex space-x-5">
            {[
              { iconClass: 'fab fa-linkedin', url: 'https://www.linkedin.com/in/jakub-teichman-ba0a0224a/' },
              { iconClass: 'fab fa-github', url: 'https://github.com/JakubTeichman' }
            ].map(({ iconClass, url }) => (
              <a
                key={iconClass}
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-teal-400 transition-colors"
              >
                <i className={`fab fa-${iconClass} text-lg`}></i>
              </a>
            ))}
          </div>

        </div>
        <div className="border-t border-gray-800 mt-3 pt-3 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-500 text-sm">
            © 2025 FOTO FORENSICS
          </p>
          <div className="flex mt-3 md:mt-0">
            <a href="/contact" className="text-gray-500 hover:text-gray-300 text-sm mx-3">Contact Us</a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
