import { useMemo } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import type { Candle } from "../types/api";

const tickFormat = new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit" });
const labelFormat = new Intl.DateTimeFormat(undefined, {
  year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute:"2-digit"
});

export default function PriceGraph({ data }: { data: Candle[] }){

  const numeric_ts = useMemo(
    () => data.map(d => ({ ...d, t: new Date(d.ts).getTime() })),
    [data]
  );

  return (
    <div className="card">
      <div className="label">Close Price</div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={numeric_ts}>
            <XAxis
              dataKey="t"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => tickFormat.format(new Date(t as number))}
              minTickGap={24}
            />
            <YAxis domain={["auto", "auto"]}/>
            <Tooltip labelFormatter={(t) => labelFormat.format(new Date(t as number))}/>
            <Line type="linear" dataKey="close" dot={false}/>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}