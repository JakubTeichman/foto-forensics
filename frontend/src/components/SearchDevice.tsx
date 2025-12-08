import React, { useState } from "react";
import { apiSearchDevice } from "../api/prnu";

const SearchDevice: React.FC = () => {
  const [evidence, setEvidence] = useState<File | null>(null);
  const [results, setResults] = useState<any[]>([]);

  const submit = async () => {
    if (!evidence) return;
    const res = await apiSearchDevice(evidence);
    setResults(res.top || []);
  };

  return (
    <div>
      <h3>Search Device (Case 3)</h3>
      <input type="file" onChange={e=>setEvidence(e.target.files?.[0]||null)} />
      <button onClick={submit}>Search</button>
      <ul>{results.map((r,i)=> <li key={i}>{r.name} ({r.model}) — {r.score.toFixed(4)}</li>)}</ul>
    </div>
  );
};

export default SearchDevice;
