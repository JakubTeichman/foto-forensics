import React, { useState } from "react";
import { apiCompare } from "../api/prnu";

const PRNUCompare: React.FC = () => {
  const [evidence, setEvidence] = useState<File | null>(null);
  const [refs, setRefs] = useState<FileList | null>(null);
  const [refUrlsText, setRefUrlsText] = useState<string>("");
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!evidence) return;
    setLoading(true);
    setResult(null);
    try {
      const refUrls = refUrlsText ? refUrlsText.split("\\n").map(s => s.trim()).filter(Boolean) : null;
      const data = await apiCompare(refs, refUrls, evidence);
      setResult(data);
    } catch (e: any) {
      setResult({ error: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h3>Case 1 — compare (1..n references)</h3>
      <div>
        <label>Evidence</label>
        <input type="file" onChange={e => setEvidence(e.target.files?.[0] || null)} />
      </div>
      <div>
        <label>Reference files (multiple)</label>
        <input type="file" multiple onChange={e => setRefs(e.target.files)} />
      </div>
      <div>
        <label>Or reference URLs (one per line)</label>
        <textarea value={refUrlsText} onChange={e => setRefUrlsText(e.target.value)} rows={4}></textarea>
      </div>
      <button onClick={handleSubmit} disabled={loading}>Compare</button>

      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
};

export default PRNUCompare;
