import { use, useEffect, useState } from "react";
import { getPrediction, getCandles } from "../services/api";
import TickerIn from "../components/TickerIn";
import ProbBar from "../components/ProbBar";
import PriceGraph from "../components/PriceGraph";
import RangeButton from "../components/RangeButton";
import type { Probs, Candle } from "../types/api";
import { YFPRESETS, type RangeKey } from "./Ranges";

export default function Dashboard(){
  const [ticker, setTicker] = useState("SPY");
  const [probs, setProbs] = useState<Probs | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [range, setRange] = useState<RangeKey>("1M");

  async function onSubmit(){
    setLoading(true);
    setErr(null);

    try{
      const [p, c] = await Promise.all([
        getPrediction(ticker),
        getCandles(ticker, YFPRESETS[range].period, YFPRESETS[range].interval)
      ]);
      setProbs(p.probs);
      setCandles(c.data);
    }
    catch (e: any){
      setErr(e.message || "Request failed");
      setProbs(null);
      setCandles([]);
    }
    finally{
      setLoading(false);
    }
  }

  // Refectch only the candles for the new period when the range changes
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (!probs) return;
      setLoading(true);
      setErr(null);

      try{
        const c = await getCandles(ticker, YFPRESETS[range].period, YFPRESETS[range].interval);
        if (!cancelled) setCandles(c.data);
      }
      catch (e: any){
        if (!cancelled) setErr(e.message || "Request failed");
      }
      finally{
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [range, ticker]);

  return (
    <div className="conatiner">
      <h1>Torch Index</h1>
      <TickerIn ticker={ticker} setTicker={setTicker} onSubmit={onSubmit} loading={loading}/>
      <RangeButton range={range} setRange={setRange} disabled={loading}/>
      {err && <div className="error">{err}</div>}
      {probs && <ProbBar probs={probs}/>}
      <PriceGraph data={candles} range={range}/>
    </div>
  );
}