//DEFINING THE LOOK OF THE APP

// app/page.tsx
"use client";
import { useState } from "react";

type FftResponse = {
  k: number; n: number; L: number;
  bin_index: number;
  ideal_bin_float: number;
  bin_offset_from_integer: number;
  Y_m_real: number; Y_m_imag: number; Y_m_abs: number;
  Y_m_norm_real: number; Y_m_norm_imag: number; Y_m_norm_abs: number;
  note: string;
};

export default function Page() {
  const [n, setN] = useState(4096);
  const [L, setL] = useState(2 * Math.PI);
  const [k, setK] = useState(14);
  const [data, setData] = useState<FftResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = async () => {
    try {
      setLoading(true); setErr(null);
      const url = new URL(`${process.env.NEXT_PUBLIC_API_URL}/fft`);
      url.search = new URLSearchParams({
        n: String(n),
        L: String(L),
        k: String(k),
      }).toString();

      const res = await fetch(url.toString());
      if (!res.ok) throw new Error(await res.text());
      const j = (await res.json()) as FftResponse;
      setData(j);
    } catch (e: any) {
      setErr(e.message ?? "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="p-6 space-y-4">
      <h1 className="text-2xl font-bold">Signal Lab UI</h1>

      <div className="grid grid-cols-3 gap-3 max-w-xl">
        <label className="flex flex-col">
          <span>N</span>
          <input className="border rounded p-2" type="number" value={n}
                 onChange={e => setN(parseInt(e.target.value || "0"))}/>
        </label>
        <label className="flex flex-col">
          <span>L</span>
          <input className="border rounded p-2" type="number" step="any" value={L}
                 onChange={e => setL(parseFloat(e.target.value || "0"))}/>
        </label>
        <label className="flex flex-col">
          <span>k</span>
          <input className="border rounded p-2" type="number" value={k}
                 onChange={e => setK(parseInt(e.target.value || "0"))}/>
        </label>
      </div>

      <button
        onClick={run}
        disabled={loading}
        className="px-4 py-2 rounded-xl shadow bg-black text-white disabled:opacity-50"
      >
        {loading ? "Computing…" : "Compute FFT bin"}
      </button>

      {err && <p className="text-red-600">{err}</p>}

      {data && (
        <pre className="p-4 rounded-xl bg-gray-100 overflow-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </main>
  );
}