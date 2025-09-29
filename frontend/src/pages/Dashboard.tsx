import { use, useEffect, useState } from "react";
import { getPrediction, getCandles } from "../services/api";
import TickerIn from "../components/TickerIn";
import ProbBar from "../components/ProbBar";
import PriceGraph from "../components/PriceGraph";
import RangeButton from "../components/RangeButton";
import type { Probs, Candle } from "../types/api";
import { YFPRESETS, type RangeKey } from "./Ranges";

export default function Dashboard(){
  const [inputTicker, setInputTicker] = useState("SPY"); // User input
  const [ticker, setTicker] = useState<string | null>(null); // Last ticker submitted
  const [probs, setProbs] = useState<Probs | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [range, setRange] = useState<RangeKey>("1M");

  async function onSubmit(){
    if(!inputTicker) return;
    setLoading(true);
    setErr(null);

    try{
      setTicker(inputTicker);
      // prediction with timeout
      const predCtrl = new AbortController();
      const predTimer = setTimeout(() => predCtrl.abort(), 10_000);

      // candles with timeout
      const candlCtrl = new AbortController();
      const candlTimer = setTimeout(() => candlCtrl.abort(), 10_000);

      const [p, c] = await Promise.all([
        getPrediction(inputTicker, { signal: predCtrl.signal }),
        getCandles(inputTicker, YFPRESETS[range].period, YFPRESETS[range].interval, { signal: candlCtrl.signal })
      ]);
      setProbs(p.probs);
      setCandles(c.data);

      clearTimeout(candlTimer);
      clearTimeout(predTimer);
    }
    catch (e: any){
      if (e.name !== "AbortError") setErr(e.message || "Request failed");
      setProbs(null);
      setCandles([]);
    }
    finally{
      setLoading(false);
    }
  }

  // Refectch only the candles for the new period when the range changes
  useEffect(() => {
    if (!ticker) return;

    const rangeCtrl = new AbortController();
    const timer = setTimeout(() => rangeCtrl.abort(), 10_000);

    (async () => {
      if (!probs) return;
      setLoading(true);
      setErr(null);

      try{
        const c = await getCandles(ticker, YFPRESETS[range].period, YFPRESETS[range].interval, { signal: rangeCtrl.signal });
        setCandles(c.data);
      }
      catch (e: any){
        if (e.name !== "AbortError") setErr(e.message || "Candles request failed");
      }
      finally{
        if (!rangeCtrl.signal.aborted) setLoading(false);
      }
    })();
    return () => rangeCtrl.abort();
  }, [range, ticker]);

  return (
    <div className="conatiner">
      <h1>Torch Index</h1>
      <TickerIn ticker={inputTicker} setTicker={(v: string) => setInputTicker(v.toUpperCase())} onSubmit={onSubmit} loading={loading}/>
      <RangeButton range={range} setRange={setRange} disabled={loading}/>
      {err && <div className="error">{err}</div>}
      {probs && <ProbBar probs={probs}/>}
      <PriceGraph data={candles} range={range}/>
    </div>
  );
}