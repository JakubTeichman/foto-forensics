// src/components/GpsMap.tsx
import React from 'react';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';

// ✅ Fix marker icon issue (Leaflet domyślnie nie znajduje ikon w bundlu)
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl:
    'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface Props {
  lat: number;
  lon: number;
}

const GpsMap: React.FC<Props> = ({ lat, lon }) => {
  if (!lat || !lon || isNaN(lat) || isNaN(lon)) {
    return (
      <div className="text-gray-400 italic text-center mt-4">
        No valid GPS coordinates available.
      </div>
    );
  }

  return (
    <div className="w-full h-64 mt-4 rounded-lg overflow-hidden border border-gray-800 shadow-lg">
      <MapContainer
        center={[lat, lon]}
        zoom={13}
        style={{ width: '100%', height: '100%' }}
      >
        {/* 🌙 Dark mode map tiles */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a> contributors'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        <Marker position={[lat, lon]}>
          <Popup>
            <strong>GPS Coordinates:</strong> {lat.toFixed(6)}, {lon.toFixed(6)}
          </Popup>
        </Marker>
      </MapContainer>
    </div>
  );
};

export default GpsMap;
