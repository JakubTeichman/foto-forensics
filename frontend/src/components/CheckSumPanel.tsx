import React, { useEffect, useState } from "react";

interface ChecksumResult {
  name: string;
  md5: string;
  sha1: string;
  sha256: string;
}

interface CheckSumPanelProps {
  files: File[];
  onChecksumsCalculated?: (checksums: { [key: string]: string }) => void;
  onStartCalculation?: () => void; // ✅ nowy prop
}

const bufferToHex = (buffer: ArrayBuffer) =>
  Array.from(new Uint8Array(buffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");

const digestSubtle = async (arrayBuffer: ArrayBuffer, algo: AlgorithmIdentifier) => {
  try {
    const hash = await crypto.subtle.digest(algo, arrayBuffer);
    return bufferToHex(hash);
  } catch {
    return "N/A";
  }
};

const computeMD5Dynamic = async (arrayBuffer: ArrayBuffer): Promise<string> => {
  try {
    // @ts-ignore dynamic import
    const SparkMD5 = (await import("spark-md5")).default || (await import("spark-md5"));
    if (SparkMD5?.ArrayBuffer?.hash) {
      return SparkMD5.ArrayBuffer.hash(arrayBuffer);
    }
    const u8 = new Uint8Array(arrayBuffer);
    return SparkMD5.ArrayBuffer.hash(u8 as any);
  } catch {
    return "N/A";
  }
};

const CheckSumPanel: React.FC<CheckSumPanelProps> = ({ files, onChecksumsCalculated, onStartCalculation }) => {
  const [results, setResults] = useState<ChecksumResult[]>([]);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const calculateChecksums = async () => {
      if (!files || files.length === 0) {
        setResults([]);
        return;
      }

      setBusy(true);
      if (onStartCalculation) onStartCalculation(); // ✅ uruchom spinner

      const newResults: ChecksumResult[] = [];

      for (const file of files) {
        const arrayBuffer = await file.arrayBuffer();

        const sha1 = await digestSubtle(arrayBuffer, "SHA-1");
        const sha256 = await digestSubtle(arrayBuffer, "SHA-256");
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

      if (onChecksumsCalculated) {
        const checksums: { [key: string]: string } = {};
        newResults.forEach((r) => {
          checksums[r.name] = r.sha256;
        });
        onChecksumsCalculated(checksums);
      }
    };

    calculateChecksums();
  }, [files, onChecksumsCalculated, onStartCalculation]);

  if (!files || files.length === 0) return null;

  return (
    <div className="bg-gray-900 border border-teal-800 rounded-xl p-6 mt-6 shadow-lg">
      <h3 className="text-xl font-semibold mb-4 text-teal-400 flex items-center gap-2">
        <i className="fas fa-fingerprint" />
        File Integrity Checksums
      </h3>

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
