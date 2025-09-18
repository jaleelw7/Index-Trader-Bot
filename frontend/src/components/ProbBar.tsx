import type { Probs } from "../types/api";

const order: (keyof Probs)[] = ["buy", "hold", "sell"]; // Class display order
// Class bar color
const color: Record<keyof Probs, string> = {
  buy: "#10b981",
  hold: "#6884ad",
  sell: "#ef4444",
};

export default function ProbBar({ probs }: {probs: Probs}){
  return (
    <div className="grid">
      {order.map(key => {
        const pct = (probs[key] * 100).toFixed(2);
        return (
          <div key={key} className="card">
            <div className="label">{key.toUpperCase()}</div>
            <div className="value">{pct}%</div>
            <div className="bar">
              <div className="barFill" style={{ width: `${pct}%`, background: color[key]}} />
            </div>
          </div>
        );
      })}
    </div>
  );
}