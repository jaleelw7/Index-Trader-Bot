// Object to hold input
type TickerInProps = {
  ticker: string;
  setTicker: (val: string) => void;
  onSubmit: () => void;
  loading?: boolean;
};

export default function TickerIn({ ticker,setTicker, onSubmit, loading}: TickerInProps){
  return (
    <div className="row">
      <input
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase().trim())}
        placeholder="Ticker e.g. SPY"
        className="input"
      />
      <button onClick={onSubmit} disabled={loading} className="in_btn">
        {loading ? "Loading...": "Predict"}
      </button>
    </div>
  );
}