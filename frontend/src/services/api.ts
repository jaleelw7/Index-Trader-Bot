import type { Probs, Candle } from "../types/api";
export type PredictionResponse = { ticker: string; asof: string; probs: Probs };
export type CandleResponse = { ticker: string; data: Candle[] };


function buildURL(
  path: string,
  params?: Record<string, string | number | undefined>
){
  const apiBase =  import.meta.env.VITE_API_BASE ?? window.location.origin;
  const url = new URL(path, apiBase)
  
  if (params){
    for (const [k, v] of Object.entries(params)){
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

async function getJSON<T>(path: string,
  params?: Record<string, string | number | undefined>,
  opts?: { signal?: AbortSignal }): Promise<T>{
  const apiUrl = buildURL(path, params); // Build URL
  // Fetch JSON response
  const res = await fetch(apiUrl, { signal: opts?.signal });
  const text = await res.text(); // Get body of JSON response
  // If the body cannot be read as JSON, read it as plain text
  let body: unknown = null;

  try{
    body = text ? JSON.parse(text): null;
  }
  catch{
    body = text;
  }

  if (!res.ok || (body as any)?.error) throw new Error((body as any)?.error || "Error fetching API JSON response");
  
  return body as T;
}

export function getPrediction(ticker: string, opts?: { signal?: AbortSignal }){
  return getJSON<PredictionResponse>("api/prediction", { ticker });
}

export function getCandles(ticker: string, period = "1mo", interval = "60m", opts?: { signal?: AbortSignal }){
  return getJSON<CandleResponse>("api/candles", { ticker, interval, period});
}