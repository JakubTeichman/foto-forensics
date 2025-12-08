import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
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

// ✅ Komponent do odświeżenia widoku po zmianie współrzędnych
const RecenterMap: React.FC<{ lat: number; lon: number }> = ({ lat, lon }) => {
  const map = useMap();
  useEffect(() => {
    map.setView([lat, lon], 13);
  }, [lat, lon, map]);
  return null;
};

const GpsMap: React.FC<Props> = ({ lat, lon }) => {
  if (!lat || !lon || isNaN(lat) || isNaN(lon)) {
    return (
      <div className="text-gray-400 italic text-center mt-4">
        No valid GPS coordinates available.
      </div>
    );
  }

  return (
    <div
      className="
        relative 
        w-full 
        h-72 sm:h-80 md:h-96 
        mt-6
        rounded-2xl 
        overflow-hidden 
        border border-teal-700/50 
        bg-black/40 
        backdrop-blur-md 
        shadow-[0_0_25px_rgba(0,255,200,0.15)]
        z-0
      "
    >
      {/* ✅ Navbar zawsze nad mapą */}
      <div className="absolute top-0 left-0 right-0 h-0 z-[5]" />

      <MapContainer
        center={[lat, lon]}
        zoom={13}
        zoomControl={true}
        style={{
          width: '100%',
          height: '100%',
          position: 'absolute',
          inset: 0,
          zIndex: 1,
        }}
        className="rounded-2xl"
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

        <RecenterMap lat={lat} lon={lon} />
      </MapContainer>
    </div>
  );
};

export default GpsMap;
