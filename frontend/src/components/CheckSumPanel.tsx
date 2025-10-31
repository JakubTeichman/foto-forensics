// src/components/CheckSumPanel.tsx
import React, { useEffect, useState } from "react";

interface ChecksumResult {
  name: string;
  md5: string;
  sha1: string;
  sha256: string;
}

interface Props {
  files: File[]; // Można przekazać 1 lub więcej plików
}

const bufferToHex = (buffer: ArrayBuffer) =>
  Array.from(new Uint8Array(buffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

const digestSubtle = async (arrayBuffer: ArrayBuffer, algo: AlgorithmIdentifier) => {
  try {
    const hash = await crypto.subtle.digest(algo, arrayBuffer);
    return bufferToHex(hash);
  } catch (e) {
    return "N/A";
  }
};

const computeMD5Dynamic = async (arrayBuffer: ArrayBuffer): Promise<string> => {
  // próbujemy dynamicznie załadować spark-md5 (jeśli zainstalowany)
  try {
    // @ts-ignore dynamic import, może nie mieć typów
    const SparkMD5 = (await import("spark-md5")).default || (await import("spark-md5"));
    // spark-md5 obsługuje stringi lub ArrayBuffer (potrzebuje konwersji)
    // SparkMD5.ArrayBuffer.hash
    if (SparkMD5 && SparkMD5.ArrayBuffer && typeof SparkMD5.ArrayBuffer.hash === "function") {
      return SparkMD5.ArrayBuffer.hash(arrayBuffer);
    }
    // alternatywa: skonwertuj na Uint8Array -> string (wolniejsze)
    const u8 = new Uint8Array(arrayBuffer);
    return SparkMD5.ArrayBuffer.hash(u8 as any);
  } catch (err) {
    // brak spark-md5 — zwracamy N/A (bez crasha)
    return "N/A";
  }
};

const CheckSumPanel: React.FC<Props> = ({ files }) => {
  const [results, setResults] = useState<ChecksumResult[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const calculateChecksums = async () => {
      if (!files || files.length === 0) {
        setResults([]);
        return;
      }

      setBusy(true);
      const newResults: ChecksumResult[] = [];

      for (const file of files) {
        const arrayBuffer = await file.arrayBuffer();

        // SHA-1 i SHA-256 natywnie (crypto.subtle)
        const sha1 = await digestSubtle(arrayBuffer, "SHA-1");
        const sha256 = await digestSubtle(arrayBuffer, "SHA-256");

        // MD5 — próbujemy dynamicznie użyć spark-md5 (jeśli jest)
        const md5 = await computeMD5Dynamic(arrayBuffer);

        newResults.push({
          name: file.name,
          md5,
          sha1,
          sha256,
        });
      }

      setResults(newResults);
      setBusy(false);
    };

    calculateChecksums();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files?.length]); // odpalamy, gdy zmieni się lista plików

  if (!files || files.length === 0) return null;

  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        <i className="fas fa-fingerprint" />
        File Integrity Checksums
      </h3>

      {busy && <p className="text-sm text-gray-400 mb-3">Computing checksums…</p>}

      <div className="space-y-4">
        {results.map((r, idx) => (
          <div key={idx} className="bg-gray-800/60 rounded-lg p-4">
            <p className="text-lg font-medium text-white mb-2">{r.name}</p>
            <div className="text-sm text-gray-300 space-y-1 break-words">
              <p>
                <span className="text-teal-400 font-semibold">MD5:</span>{" "}
                <span className={r.md5 === "N/A" ? "text-yellow-300" : ""}>{r.md5}</span>
              </p>
              <p>
                <span className="text-teal-400 font-semibold">SHA-1:</span> {r.sha1}
              </p>
              <p>
                <span className="text-teal-400 font-semibold">SHA-256:</span> {r.sha256}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default CheckSumPanel;
