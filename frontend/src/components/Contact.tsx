import React, { FormEvent } from 'react';

interface ContactProps {
  setActiveTab: (tab: string) => void;
}

const Contact: React.FC<ContactProps> = ({ setActiveTab }) => {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    // Tu możesz dodać logikę wysłania wiadomości np. fetch/axios

    // Po wysłaniu formularza zmień zakładkę
    setActiveTab('nextTab'); // podmień na właściwą nazwę zakładki docelowej
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
          Contact Us
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Get in touch with our team for support, inquiries, or partnership opportunities.
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
          <h3 className="text-xl font-medium mb-6">Get in Touch</h3>
          <form className="space-y-4" onSubmit={handleSubmit}>
            {['Your Name', 'Email Address', 'Subject'].map((label, i) => (
              <div key={i}>
                <label className="block text-sm font-medium mb-2">{label}</label>
                <input
                  type="text"
                  className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500"
                />
              </div>
            ))}
            <div>
              <label className="block text-sm font-medium mb-2">Message</label>
              <textarea
                rows={4}
                className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500"
              ></textarea>
            </div>
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-teal-400 to-green-400 text-black font-medium py-3 rounded-lg hover:from-teal-500 hover:to-green-500 transition-all"
            >
              Send Message
            </button>
          </form>
        </div>

        <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
          <h3 className="text-2xl font-medium mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">Contact Information</h3>
          <div className="space-y-6">
            {[
              {
                icon: 'fas fa-map-marker-alt',
                label: 'Address',
                text: '123 Tech Boulevard, Suite 456\nSan Francisco, CA 94107',
              },
              {
                icon: 'fas fa-envelope',
                label: 'Email',
                text: 'info@fotoforensics.com\nsupport@fotoforensics.com',
              },
              {
                icon: 'fas fa-phone-alt',
                label: 'Phone',
                text: '+1 (555) 123-4567\n+1 (555) 987-6543',
              },
            ].map((item, i) => (
              <div key={i} className="flex items-start">
                <div className="text-green-400 text-xl mr-4">
                  <i className={item.icon}></i>
                </div>
                <div>
                  <h4 className="font-medium mb-1">{item.label}</h4>
                  <p className="text-gray-400 whitespace-pre-line">{item.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;
