import { use, useState } from "react";
import { getPrediction, getCandles } from "../services/api";
import TickerIn from "../components/TickerIn";
import ProbBar from "../components/ProbBar";
import PriceGraph from "../components/PriceGraph";
import type { Probs, Candle } from "../types/api";

export default function Dashboard(){
  const [ticker, setTicker] = useState("SPY");
  const [probs, setProbs] = useState<Probs | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function onSubmit(){
    setLoading(true);
    setErr(null);

    try{
      const [p, c] = await Promise.all([
        getPrediction(ticker),
        getCandles(ticker)
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

  return (
    <div className="conatiner">
      <h1>Torch Index</h1>
      <TickerIn ticker={ticker} setTicker={setTicker} onSubmit={onSubmit} loading={loading}/>
      {err && <div className="error">{err}</div>}
      {probs && <ProbBar probs={probs}/>}
      <PriceGraph data={candles}/>
    </div>
  );
}