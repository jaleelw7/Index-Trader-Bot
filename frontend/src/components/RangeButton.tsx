import { YFPRESETS, type RangeKey } from "../pages/Ranges";

export default function RangeButton(
  { range, setRange, disabled }: 
  { range: RangeKey; setRange: (r: RangeKey) => void; disabled?: boolean }){
    const keys = Object.keys(YFPRESETS) as RangeKey[];
    return (
      <div className="range">
        {keys.map(k => (
          <button
            key={k}
            className={`btn-tab ${range === k ? "active" : ""}`}
            onClick={() => setRange(k)}
            disabled={disabled}
            title={`${YFPRESETS[k].period}`}
          >
            {k}
          </button>
        ))}
      </div>
    );
  }