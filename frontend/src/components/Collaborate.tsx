import React, { useState, FormEvent } from 'react';

interface CollaborateProps {
  setActiveTab: (tab: string) => void;
}

const Collaborate: React.FC<CollaborateProps> = ({ setActiveTab }) => {
  const [formData, setFormData] = useState({
    name: '',
    deviceModel: '',
    imageCount: '',
    email: '',
    format: 'jpg',
  });

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    try {
      const response = await fetch("http://localhost:5000/collaborate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (response.ok) {
        alert(result.message);
        setActiveTab("nextTab"); // switch tab after successful submission
      } else {
        alert(result.error || "An error occurred while submitting the form.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Could not connect to the server.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto mt-8">
      <div className="text-center mb-12">
        <h2 className="text-5xl font-bold mb-4 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400 leading-tight">
          Collaborate with Us
        </h2>
        <p className="text-xl text-gray-400 max-w-2xl mx-auto">
          Join our network of forensic experts and contribute to advancing image analysis technology.
        </p>
      </div>
      <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10">
        <form className="space-y-6" onSubmit={handleSubmit}>
          {[
            { label: 'Name', type: 'text', value: formData.name, key: 'name' },
            { label: 'Device Model', type: 'text', value: formData.deviceModel, key: 'deviceModel' },
            { label: 'Image Count', type: 'number', value: formData.imageCount, key: 'imageCount' },
            { label: 'Email', type: 'email', value: formData.email, key: 'email' },
          ].map((field) => (
            <div key={field.key}>
              <label className="block text-sm font-medium mb-2">{field.label}</label>
              <input
                type={field.type}
                className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500"
                value={field.value}
                onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              />
            </div>
          ))}

          <div>
            <label className="block text-sm font-medium mb-2">Format</label>
            <div className="relative">
              <select
                className="w-full bg-black bg-opacity-50 border border-green-800 rounded-lg px-4 py-2 focus:outline-none focus:border-green-500 appearance-none"
                value={formData.format}
                onChange={(e) => setFormData({ ...formData, format: e.target.value })}
              >
                {['jpg', 'png', 'raw', 'tiff'].map((opt) => (
                  <option key={opt} value={opt}>{opt.toUpperCase()}</option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center px-2 pointer-events-none">
                <i className="fas fa-chevron-down text-green-500"></i>
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-gradient-to-r from-teal-400 to-green-400 text-black font-medium py-3 rounded-lg hover:from-teal-500 hover:to-green-500 transition-all"
          >
            Submit Request
          </button>
        </form>
      </div>
    </div>
  );
};

export default Collaborate;
