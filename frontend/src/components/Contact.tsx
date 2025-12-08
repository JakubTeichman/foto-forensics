import React, { FormEvent, useState } from 'react';

interface ContactProps {
  setActiveTab: (tab: string) => void;
}

const Contact: React.FC<ContactProps> = ({ setActiveTab }) => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: '',
  });

  const [status, setStatus] = useState<{ type: 'success' | 'error' | ''; message: string }>({
    type: '',
    message: '',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    // Basic validation
    if (!formData.name || !formData.email || !formData.subject || !formData.message) {
      setStatus({ type: 'error', message: 'Please fill in all fields before submitting.' });
      return;
    }

    try {
      const response = await fetch('/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send message.');
      }

      setStatus({ type: 'success', message: 'Your message has been sent successfully!' });
      setFormData({ name: '', email: '', subject: '', message: '' });

      // ✅ optionally: redirect or change tab
      // setActiveTab('home');

    } catch (err: any) {
      setStatus({
        type: 'error',
        message: err.message || 'Something went wrong. Please try again later.',
      });
    }
  };

  return (
    <div className="max-w-4xl mx-auto mt-8">
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
          Contact Us
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Get in touch for support or partnership opportunities.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* FORM SECTION */}
        <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
          <h3 className="text-xl font-medium mb-6">Get in Touch</h3>

          <form className="space-y-4" onSubmit={handleSubmit}>
            {['name', 'email', 'subject'].map((field, i) => (
              <div key={i}>
                <label className="block text-sm font-medium mb-2 capitalize">
                  {field === 'email' ? 'Email Address' : `Your ${field}`}
                </label>
                <input
                  type={field === 'email' ? 'email' : 'text'}
                  name={field}
                  value={(formData as any)[field]}
                  onChange={handleChange}
                  className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500"
                />
              </div>
            ))}

            <div>
              <label className="block text-sm font-medium mb-2">Message</label>
              <textarea
                rows={4}
                name="message"
                value={formData.message}
                onChange={handleChange}
                className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500"
              ></textarea>
            </div>

            {status.message && (
              <div
                className={`text-sm p-3 rounded-lg ${
                  status.type === 'success'
                    ? 'bg-green-900/40 text-green-300 border border-green-700'
                    : 'bg-red-900/40 text-red-300 border border-red-700'
                }`}
              >
                {status.message}
              </div>
            )}

            <button
              type="submit"
              className="w-full bg-gradient-to-r from-teal-400 to-green-400 text-black font-medium py-3 rounded-lg hover:from-teal-500 hover:to-green-500 transition-all"
            >
              Send Message
            </button>
          </form>
        </div>

        {/* CONTACT INFO SECTION */}
        <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
          <h3 className="text-2xl font-medium mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
            Contact Information
          </h3>
          <div className="space-y-6">
            {[
              {
                icon: 'fas fa-map-marker-alt',
                label: 'Address',
                text: 'Kraków\nPoland',
              },
              {
                icon: 'fas fa-envelope',
                label: 'Email',
                text: 'fotoforensics3@gmail.com',
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
