import React, { useState } from "react";
import { apiAddDevice } from "../api/prnu";

const AddDevice: React.FC = () => {
  const [name, setName] = useState("");
  const [model, setModel] = useState("");
  const [description, setDescription] = useState("");
  const [refs, setRefs] = useState<FileList | null>(null);
  const [refUrlsText, setRefUrlsText] = useState("");
  const [result, setResult] = useState<any>(null);

  const submit = async () => {
    try {
      const urls = refUrlsText ? refUrlsText.split("\\n").map(s=>s.trim()).filter(Boolean) : null;
      const res = await apiAddDevice(name, model, description, refs, urls);
      setResult(res);
    } catch (e:any) {
      setResult({error: e.message});
    }
  };

  return (
    <div>
      <h3>Add Device (admin)</h3>
      <input placeholder="Name" value={name} onChange={e=>setName(e.target.value)} />
      <input placeholder="Model" value={model} onChange={e=>setModel(e.target.value)} />
      <textarea placeholder="Description" value={description} onChange={e=>setDescription(e.target.value)} />
      <input type="file" multiple onChange={e => setRefs(e.target.files)} />
      <textarea placeholder="Reference URLs (one per line)" value={refUrlsText} onChange={e=>setRefUrlsText(e.target.value)} />
      <button onClick={submit}>Add Device</button>
      {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
    </div>
  );
};

export default AddDevice;
