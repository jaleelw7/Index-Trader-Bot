export type RangeKey  = "1D" | "5D" | "1M" | "3M" | "6M" | "YTD" | "1Y";

export const YFPRESETS: Record<RangeKey, {period: string; interval: string;}> = {
  "1D":  { period: "1d",  interval: "5m"  },
  "5D":  { period: "5d",  interval: "15m" },
  "1M":  { period: "1mo", interval: "60m" },
  "3M":  { period: "3mo", interval: "1d" },
  "6M":  { period: "6mo", interval: "1d"  },
  "YTD": { period: "ytd", interval: "1d"  },
  "1Y":  { period: "1y",  interval: "1d"  }
};