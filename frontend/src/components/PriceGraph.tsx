import { useMemo } from "react";
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";
import type { Candle } from "../types/api";
import type { RangeKey } from "../pages/Ranges";

type AxisFormat = {
  tickFormat: Intl.DateTimeFormat;
  labelFormat: Intl.DateTimeFormat;
  tickCount: number;
  minTickGap: number;
  angle: number;
}

function getAxisFormat(range: RangeKey): AxisFormat {
  switch (range) {
  case "1D":
    return {
      tickFormat: new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }),
      labelFormat: new Intl.DateTimeFormat(undefined, {
        year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit"
      }),
      tickCount: 8,
      minTickGap: 12,
      angle: 0,
    };
  case "5D":
    return {
      tickFormat: new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit", hour: "2-digit" }),
      labelFormat: new Intl.DateTimeFormat(undefined, {
        year: "numeric", month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit"
      }),
      tickCount: 8,
      minTickGap: 16,
      angle: -20,
    };
  case "1M":
  case "3M":
    return {
      tickFormat: new Intl.DateTimeFormat(undefined, { month: "short", day: "2-digit" }),
      labelFormat: new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "2-digit" }),
      tickCount: 7,
      minTickGap: 18,
      angle: -25,
    };
  default:
    return {
      tickFormat: new Intl.DateTimeFormat(undefined, { month: "short" }),
      labelFormat: new Intl.DateTimeFormat(undefined, { year: "numeric", month: "short", day: "2-digit" }),
      tickCount: 8,
      minTickGap: 20,
      angle: -30,
    };
  }
}
export default function PriceGraph({ data, range }: { data: Candle[]; range: RangeKey }){

  const numeric_ts = useMemo(
    () => data.map(d => ({ ...d, t: new Date(d.ts).getTime() })),
    [data]
  );
  const priceFormat = new Intl.NumberFormat(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})
  const axisFmt = useMemo(() => getAxisFormat(range), [range]);

  return (
    <div className="card">
      <div className="label">Close Price</div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={numeric_ts} margin={{ top: 10, right: 16, left: 8, bottom: 48 }}>
            <XAxis
              dataKey="t"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => axisFmt.tickFormat.format(new Date(t as number))}
              tickCount={axisFmt.tickCount}
              minTickGap={axisFmt.minTickGap}
              angle={axisFmt.angle}
              dy={8}
            />
            <YAxis 
              domain={["auto", "auto"]}
              tickFormatter={(n) => priceFormat.format(Number(n))}
            />
            <Tooltip 
              labelFormatter={(t) => axisFmt.labelFormat.format(new Date(t as number))}
              formatter={(n: number) => [
                priceFormat.format(Number(n)),
                "Close"
              ]}
            />
            <Line type="linear" dataKey="close" dot={false}/>
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}