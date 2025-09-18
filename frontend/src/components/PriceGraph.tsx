import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import type { Candle } from "../types/api";

export default function PriceGraph({ data }: { data: Candle[] }){
  return (
    <div className="card">
      <div className="label">Close Price</div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis dataKey="ts" tick={false}/>
            <YAxis domain={["auto", "auto"]}/>
            <Tooltip labelFormatter={(c) => new Date(c as string).toLocaleString()}/>
            <Line type="monotone" dataKey="close" dot={false}/>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}