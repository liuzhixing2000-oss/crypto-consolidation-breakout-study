#!/usr/bin/env python3
"""External replication of the frozen precious-metals 15m rule.

No parameter search is performed here.  The rule was fixed by the prior pilot:
8 trading hours of compression, first 15m retest after breakout, 2 ATR stop,
and an 8 trading-hour terminal exit.
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from precious_metals_multitimeframe import (
    PRIMARY_COST_BPS, features, get_json, range_state, session_name, yahoo_15m,
)


FROZEN_DURATION_H = 8
FROZEN_STOP_ATR = 2.0
FROZEN_HORIZON_H = 8
HISTORY_DAYS = int(os.getenv("PAXG_HISTORY_DAYS", "729"))
COSTS_BPS = [2.0, 5.0, 10.0, 20.0]
ASSETS = {
    "GC=F": "yahoo", "SI=F": "yahoo",
    "GLD": "yahoo", "SLV": "yahoo",
    "PAXGUSDT": "binance",
}


def binance_15m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, cursor, end_ms = [], int(start.timestamp()*1000), int(end.timestamp()*1000)
    while cursor < end_ms:
        url = "https://api.binance.com/api/v3/klines?" + urlencode({
            "symbol": symbol, "interval": "15m", "startTime": cursor,
            "endTime": end_ms, "limit": 1000})
        batch = get_json(url)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0])+1
        if len(rows) % 10000 == 0:
            print(f"  downloaded {len(rows):,} bars", flush=True)
        time.sleep(.04)
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    frame = pd.DataFrame(rows, columns=columns)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["time"] = pd.to_datetime(frame.open_time, unit="ms", utc=True)
    return frame.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("time").set_index("time").sort_index()


def evaluate_frozen(d: pd.DataFrame, symbol: str, breakout_loc: int, entry_loc: int,
                    boundary: float, side: str) -> dict | None:
    sign = 1 if side == "LONG" else -1
    entry, atr = float(d.close.iloc[entry_loc]), float(d.atr.iloc[entry_loc])
    risk = FROZEN_STOP_ATR*atr
    risk_pct = risk/entry
    if not (.0025 <= risk_pct <= .03):
        return None
    bars = FROZEN_HORIZON_H*4
    future = d.iloc[entry_loc+1:entry_loc+1+bars]
    if len(future) < bars:
        return None
    favorable = ((future.high-entry) if sign == 1 else (entry-future.low))/risk
    adverse = ((entry-future.low) if sign == 1 else (future.high-entry))/risk
    stopped_mask = adverse >= 1
    first_stop = int(np.argmax(stopped_mask.to_numpy())) if stopped_mask.any() else len(future)
    stopped = bool(stopped_mask.any())
    gross = -1.0 if stopped else float(sign*(future.close.iloc[-1]-entry)/risk)
    mfe = float(favorable.iloc[:first_stop].max()) if first_stop else 0.0
    return {"symbol": symbol, "breakout_time": d.index[breakout_loc],
            "entry_time": d.index[entry_loc], "side": side,
            "session": session_name(d.index[breakout_loc]), "entry": entry,
            "boundary": boundary, "risk_pct": risk_pct, "gross_r": gross,
            "mfe_r": mfe, "stopped": stopped}


def frozen_events(frame: pd.DataFrame, symbol: str) -> list[dict]:
    d, rows = features(frame), []
    state = range_state(d, FROZEN_DURATION_H*4)
    buffer = .10*d.atr
    masks = {
        "LONG": state.valid & (d.close > state.upper+buffer) &
                (d.close.shift(1) <= state.upper.shift(1)+buffer.shift(1)),
        "SHORT": state.valid & (d.close < state.lower-buffer) &
                 (d.close.shift(1) >= state.lower.shift(1)-buffer.shift(1)),
    }
    for side, mask in masks.items():
        sign, last = (1 if side == "LONG" else -1), None
        for ts in d.index[mask.fillna(False)]:
            if last is not None and ts-last < pd.Timedelta(hours=2):
                continue
            loc = d.index.get_loc(ts)
            boundary = float(state.upper.loc[ts] if sign == 1 else state.lower.loc[ts])
            retest = None
            for j in range(loc+1, min(loc+13, len(d))):
                tolerance = .10*float(d.atr.iloc[j])
                held = ((d.low.iloc[j] <= boundary+tolerance and d.close.iloc[j] >= boundary)
                        if sign == 1 else
                        (d.high.iloc[j] >= boundary-tolerance and d.close.iloc[j] <= boundary))
                if held:
                    retest = j
                    break
            if retest is not None:
                row = evaluate_frozen(d, symbol, loc, retest, boundary, side)
                if row:
                    row.update({"range_pct": float(state.width.loc[ts]/d.close.loc[ts]),
                                "efficiency": float(state.efficiency.loc[ts])})
                    rows.append(row)
            last = ts
    return rows


def stats(part: pd.DataFrame, cost_bps: float = PRIMARY_COST_BPS, exclude_top: int = 0) -> dict:
    if part.empty:
        return {"trades": 0}
    x = part.copy()
    x["net_r"] = x.gross_r-(cost_bps/10000)/x.risk_pct
    if exclude_top:
        x = x.drop(x.nlargest(min(exclude_top, len(x)), "net_r").index)
    if x.empty:
        return {"trades": 0}
    net = x.net_r.to_numpy()
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"trades": len(x), "avg_net_r": float(net.mean()),
            "median_net_r": float(np.median(net)), "win_rate": float((net > 0).mean()),
            "profit_factor": float(gains/losses) if losses else np.nan,
            "stop_rate": float(x.stopped.mean()), "hit_2r": float((x.mfe_r >= 2).mean()),
            "total_r": float(net.sum())}


def grouped(events: pd.DataFrame, columns: list[str], cost: float = PRIMARY_COST_BPS) -> pd.DataFrame:
    rows = []
    for keys, part in events.groupby(columns, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(columns, keys)), **stats(part, cost)})
    return pd.DataFrame(rows)


def main() -> None:
    output, cache = Path("precious_replication_results"), Path("precious_replication_data")
    output.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    all_rows, coverage = [], []
    for symbol, source in ASSETS.items():
        days = HISTORY_DAYS if source == "binance" else 59
        start = end-pd.Timedelta(days=days)
        print(f"[FROZEN REPLICATION] {symbol} ({source}, {days}d)", flush=True)
        path = cache/f"{symbol.replace('=','_')}_15m.csv.gz"
        try:
            data = (pd.read_csv(path, index_col="time", parse_dates=True) if path.exists()
                    else (binance_15m(symbol, start, end) if source == "binance"
                          else yahoo_15m(symbol, start, end)))
            if not path.exists():
                data.to_csv(path, compression="gzip")
            coverage.append({"symbol": symbol, "source": source, "bars": len(data),
                             "start": data.index.min(), "end": data.index.max(),
                             "calendar_days": (data.index.max()-data.index.min()).days})
            all_rows.extend(frozen_events(data, symbol))
        except Exception as exc:
            print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
    events = pd.DataFrame(all_rows)
    if events.empty:
        raise RuntimeError("No frozen-rule events were available")
    events["breakout_time"] = pd.to_datetime(events.breakout_time, utc=True)
    events["month"] = events.breakout_time.dt.strftime("%Y-%m")
    events.to_csv(output/"events.csv", index=False)
    pd.DataFrame(coverage).to_csv(output/"coverage.csv", index=False)
    symbol_rows = []
    for symbol, part in events.groupby("symbol", observed=True):
        for n in [0, 1, 3]:
            symbol_rows.append({"symbol": symbol, "test": f"EXCLUDE_TOP_{n}", **stats(part, exclude_top=n)})
    symbols = pd.DataFrame(symbol_rows)
    sides = grouped(events, ["symbol", "side"])
    sessions = grouped(events, ["symbol", "session"])
    months = grouped(events, ["symbol", "month"])
    costs = []
    for cost in COSTS_BPS:
        x = grouped(events, ["symbol"], cost); x.insert(1, "cost_bps", cost); costs.append(x)
    cost_frame = pd.concat(costs, ignore_index=True)
    symbols.to_csv(output/"symbol_robustness.csv", index=False)
    sides.to_csv(output/"side_robustness.csv", index=False)
    sessions.to_csv(output/"session_robustness.csv", index=False)
    months.to_csv(output/"monthly_robustness.csv", index=False)
    cost_frame.to_csv(output/"cost_stress.csv", index=False)
    report = ["# PRECIOUS-METALS FROZEN 15M EXTERNAL REPLICATION", "",
              "Frozen rule: 8h compression -> first 15m retest -> 2 ATR stop -> 8h exit.",
              "No parameter search is performed in this script.", "",
              "## Coverage", "", "```csv", pd.DataFrame(coverage).to_csv(index=False).strip(), "```", "",
              "## Symbol robustness", "", "```csv", symbols.to_csv(index=False).strip(), "```", "",
              "## Cost stress", "", "```csv", cost_frame.to_csv(index=False).strip(), "```", "",
              "## Pass rule", "",
              "A production candidate must be positive on independent gold proxies, remain positive at 20 bps, and remain positive after removing its three best trades."]
    text = "\n".join(report)
    (output/"validation_report.md").write_text(text, encoding="utf-8")
    print("\n===== FROZEN REPLICATION REPORT =====\n", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
