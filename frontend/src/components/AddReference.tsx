import React, { useState } from "react";
import ImageUploader from "./ImageUploader";

const AddReference: React.FC = () => {
  const [manufacturer, setManufacturer] = useState("");
  const [model, setModel] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const manufacturers = [
    "Apple",
    "Samsung",
    "Xiaomi",
    "Huawei",
    "Google",
    "OnePlus",
    "Sony",
    "Oppo",
    "Vivo",
    "Inne",
  ];

  const handleSubmit = async () => {
    setError(null);
    setSuccess(null);

    if (!manufacturer) {
      setError("Please select a manufacturer.");
      return;
    }

    if (!/^[\w\s-]{3,}$/.test(model)) {
      setError("Model name must be at least 3 characters and contain only letters, numbers, spaces, or dashes.");
      return;
    }

    if (images.length < 3) {
      setError("Please upload at least 3 reference images.");
      return;
    }

    const formData = new FormData();
    formData.append("manufacturer", manufacturer);
    formData.append("model", model);
    images.forEach((file) => formData.append("images", file));

    try {
      const response = await fetch("http://localhost:5000/add_reference", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        setSuccess("Reference successfully added!");
        setManufacturer("");
        setModel("");
        setImages([]);
        setPreviewUrls([]);
      } else {
        const data = await response.json();
        setError(data.message || "Something went wrong while adding reference.");
      }
    } catch (err) {
      setError("Server connection failed.");
    }
  };

  return (
    <div className="p-8 text-gray-100 bg-gray-950 rounded-2xl shadow-lg border border-teal-800 max-w-3xl mx-auto mt-10">
      <h2 className="text-3xl font-semibold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-green-400">
        Add Reference Device
      </h2>

      {/* Manufacturer dropdown */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2 text-gray-300">
          Manufacturer
        </label>
        <select
          value={manufacturer}
          onChange={(e) => setManufacturer(e.target.value)}
          className="w-full bg-gray-900 border border-teal-700 text-gray-100 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-teal-500"
        >
          <option value="">-- Select Manufacturer --</option>
          {manufacturers.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </div>

      {/* Model input */}
      <div className="mb-6">
        <label className="block text-sm font-medium mb-2 text-gray-300">
          Model
        </label>
        <input
          type="text"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="Enter model name (e.g. iPhone 14 Pro)"
          className="w-full bg-gray-900 border border-teal-700 text-gray-100 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-teal-500"
        />
      </div>

      {/* Image uploader */}
      <ImageUploader
        title="Upload Reference Images"
        id="reference-images"
        multiple
        setSelectedFiles={setImages}
        setPreviewUrls={setPreviewUrls}
        previewUrls={previewUrls}
      />

      {/* Error / success messages */}
      {error && <p className="mt-4 text-red-500 font-medium">{error}</p>}
      {success && <p className="mt-4 text-green-400 font-medium">{success}</p>}

      {/* Submit button */}
      <button
        onClick={handleSubmit}
        className="mt-6 bg-gradient-to-r from-teal-500 to-green-500 text-black font-semibold px-6 py-2 rounded-lg hover:from-teal-400 hover:to-green-400 transition-all"
      >
        Submit Reference
      </button>
    </div>
  );
};

export default AddReference;
