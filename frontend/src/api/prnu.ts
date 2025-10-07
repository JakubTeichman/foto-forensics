const API_BASE = import.meta.env.REACT_APP_API_BASE || "http://localhost:5000/api";

export async function apiCompare(filesRefs: FileList | null, referenceUrls: string[] | null, evidenceFile: File) {
  const fd = new FormData();
  fd.append("evidence", evidenceFile);
  if (filesRefs && filesRefs.length > 0) {
    Array.from(filesRefs).forEach(f => fd.append("references", f));
  } else if (referenceUrls && referenceUrls.length > 0) {
    fd.append("reference_urls", JSON.stringify(referenceUrls));
  } else {
    throw new Error("Provide reference files or reference URLs");
  }
  const res = await fetch(`${API_BASE}/compare`, { method: "POST", body: fd });
  return res.json();
}

export async function apiAddDevice(name: string, model: string, description: string, refs: FileList | null, refUrls: string[] | null, fingerprintUrl?: string) {
  const fd = new FormData();
  fd.append("name", name);
  fd.append("model", model || "");
  fd.append("description", description || "");
  if (refs && refs.length > 0) {
    Array.from(refs).forEach(f => fd.append("references", f));
  } else if (refUrls && refUrls.length > 0) {
    fd.append("reference_urls", JSON.stringify(refUrls));
  } else if (fingerprintUrl) {
    fd.append("fingerprint_upload_url", fingerprintUrl);
  } else {
    throw new Error("Provide refs files, refUrls or fingerprintUrl");
  }
  const res = await fetch(`${API_BASE}/add_device`, { method: "POST", body: fd });
  return res.json();
}

export async function apiSearchDevice(evidenceFile: File, topk = 5) {
  const fd = new FormData();
  fd.append("evidence", evidenceFile);
  fd.append("topk", String(topk));
  const res = await fetch(`${API_BASE}/search_device`, { method: "POST", body: fd });
  return res.json();
}
