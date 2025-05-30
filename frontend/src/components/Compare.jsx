import React, { useState } from 'react';

function Compare() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    if (!image1 || !image2) return;

    try {
        const formData = new FormData();
        formData.append('image1', image1);
        formData.append('image2', image2);

        const response = await fetch('/compare', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setResult(data.similarity.toFixed(4));
    } catch (error) {
        console.error('Error:', error);
        setResult('Błąd podczas przesyłania plików');
    }
  };

  return (
    <div>
      <h2>Porównanie PRNU</h2>
      <input type="file" accept="image/*" onChange={(e) => setImage1(e.target.files[0])} />
      <input type="file" accept="image/*" onChange={(e) => setImage2(e.target.files[0])} />
      <button onClick={handleUpload}>Porównaj</button>
      {result && <p>Wynik podobieństwa: {result}</p>}
    </div>
  );
}

export default Compare;
