import type { Probs, Candle } from "../types/api";
export type PredictionResponse = { ticker: string; asof: string; probs: Probs };
export type CandleResponse = { ticker: string; data: Candle[] };


function buildURL(
  path: string,
  params?: Record<string, string | number | undefined>
){
  const apiBase = window.location.origin;
  const url = new URL(path, apiBase)
  
  if (params){
    for (const [k, v] of Object.entries(params)){
      if (v !== undefined) url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

async function getJSON<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T>{
  // Create controller for timeout
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  const apiUrl = buildURL(path, params); // Build URL
  // Fetch JSON response
  const res = await fetch(apiUrl, { signal: controller.signal }).finally(() => clearTimeout(timer));
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

export function getPrediction(ticker: string){
  return getJSON<PredictionResponse>("api/prediction", { ticker });
}

export function getCandles(ticker: string, period = "1mo", ){
  return getJSON<CandleResponse>("api/candles", { ticker, interval: "60m", period});
}