#!/usr/bin/env python3
"""48h hourly compression with 15m execution for liquid gold markets."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from precious_frozen_replication import binance_15m
from precious_metals_multitimeframe import get_json, session_name, yahoo_15m


HISTORY_DAYS = int(os.getenv("PAXG_HISTORY_DAYS", "729"))
ASSETS = {"GC=F": "yahoo", "SI=F": "yahoo", "PAXGUSDT": "binance"}
CONTEXT_H = 48
HORIZONS_H = [8, 12, 24]
COSTS_BPS = [5.0, 10.0, 20.0]
PRIMARY_COST_BPS = 10.0
MIN_THRESHOLD_H = int(os.getenv("MIN_THRESHOLD_H", "720"))


def hourly_bars(m15: pd.DataFrame) -> pd.DataFrame:
    """Label each 1h candle at its close; only completed 15m bars are included."""
    return m15.resample("1h", closed="left", label="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])


def hourly_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    prev = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    prior = d.close.shift(1)
    mid, std = prior.rolling(20).mean(), prior.rolling(20).std()
    d["bb_width"] = 4*std/mid
    d["bb_compression"] = d.bb_width/d.bb_width.shift(1).rolling(500, min_periods=150).median()
    d["volume_ratio"] = d.volume/d.volume.shift(1).rolling(48, min_periods=24).median().replace(0, np.nan)
    history_vol = d.volume_ratio.shift(1).expanding(min_periods=MIN_THRESHOLD_H)
    history_bb = d.bb_compression.shift(1).expanding(min_periods=MIN_THRESHOLD_H)
    d["vol40"], d["vol80"] = history_vol.quantile(.4), history_vol.quantile(.8)
    d["bb40"] = history_bb.quantile(.4)
    return d


def fifteen_atr(frame: pd.DataFrame) -> pd.Series:
    prev = frame.close.shift(1)
    tr = pd.concat([frame.high-frame.low, (frame.high-prev).abs(),
                    (frame.low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean()


def evaluate(m15: pd.DataFrame, atr15: pd.Series, event: dict, entry_loc: int,
             route: str, boundary: float, atr1h: float) -> list[dict]:
    entry, sign = float(m15.close.iloc[entry_loc]), (1 if event["side"] == "LONG" else -1)
    a15 = float(atr15.iloc[entry_loc])
    structure = (entry-(boundary-.25*atr1h) if sign == 1 else (boundary+.25*atr1h)-entry)
    stops = {"FIXED_2PCT": entry*.02, "ATR15_1.5": 1.5*a15,
             "ATR15_2": 2*a15, "STRUCTURE": structure}
    rows = []
    for stop_method, risk in stops.items():
        risk_pct = risk/entry
        if not np.isfinite(risk_pct) or not (.0025 <= risk_pct <= .03):
            continue
        for horizon_h in HORIZONS_H:
            bars = horizon_h*4
            future = m15.iloc[entry_loc+1:entry_loc+1+bars]
            if len(future) < bars:
                continue
            favorable = ((future.high-entry) if sign == 1 else (entry-future.low))/risk
            adverse = ((entry-future.low) if sign == 1 else (future.high-entry))/risk
            stopped_mask = adverse >= 1
            first_stop = int(np.argmax(stopped_mask.to_numpy())) if stopped_mask.any() else len(future)
            stopped = bool(stopped_mask.any())
            gross = -1.0 if stopped else float(sign*(future.close.iloc[-1]-entry)/risk)
            mfe = float(favorable.iloc[:first_stop].max()) if first_stop else 0.0
            rows.append({**event, "entry_time": m15.index[entry_loc]+pd.Timedelta(minutes=15),
                         "entry_route": route, "stop_method": stop_method,
                         "horizon_h": horizon_h, "entry": entry, "risk_pct": risk_pct,
                         "gross_r": gross, "mfe_r": mfe, "stopped": stopped})
    return rows


def detect(m15: pd.DataFrame, symbol: str) -> list[dict]:
    h1, atr15, rows = hourly_features(hourly_bars(m15)), fifteen_atr(m15), []
    upper = h1.high.shift(1).rolling(CONTEXT_H).max()
    lower = h1.low.shift(1).rolling(CONTEXT_H).min()
    width = upper-lower
    path = h1.close.diff().abs().shift(1).rolling(CONTEXT_H).sum()
    efficiency = (h1.close.shift(1)-h1.close.shift(CONTEXT_H)).abs()/path.replace(0, np.nan)
    valid = ((efficiency <= .32) & (width/h1.atr <= 9) & (width/h1.close <= .18) &
             h1.volume_ratio.between(h1.vol40, h1.vol80) & (h1.bb_compression <= h1.bb40))
    buffer = .10*h1.atr
    masks = {
        "LONG": valid & (h1.close > upper+buffer) &
                (h1.close.shift(1) <= upper.shift(1)+buffer.shift(1)),
        "SHORT": valid & (h1.close < lower-buffer) &
                 (h1.close.shift(1) >= lower.shift(1)-buffer.shift(1)),
    }
    for side, mask in masks.items():
        sign, last = (1 if side == "LONG" else -1), None
        for ts in h1.index[mask.fillna(False)]:
            if last is not None and ts-last < pd.Timedelta(hours=12):
                continue
            boundary = float(upper.loc[ts] if sign == 1 else lower.loc[ts])
            positions = np.flatnonzero(m15.index >= ts)
            if not len(positions):
                continue
            first = int(positions[0])
            event = {"symbol": symbol, "breakout_time": ts, "side": side,
                     "session": session_name(ts), "boundary": boundary,
                     "range_pct": float(width.loc[ts]/h1.close.loc[ts]),
                     "efficiency": float(efficiency.loc[ts])}
            # Original hourly entry at the completed breakout close.
            prior_positions = np.flatnonzero(m15.index < ts)
            if len(prior_positions):
                rows.extend(evaluate(m15, atr15, event, int(prior_positions[-1]),
                                     "H1_BREAKOUT_CLOSE", boundary, float(h1.atr.loc[ts])))
            # First completed 15m bar after the hourly close.
            rows.extend(evaluate(m15, atr15, event, first, "FIRST_15M", boundary, float(h1.atr.loc[ts])))
            held = []
            for j in range(first, min(first+2, len(m15))):
                held.append(m15.close.iloc[j] > boundary if sign == 1 else m15.close.iloc[j] < boundary)
            if len(held) == 2 and all(held):
                rows.extend(evaluate(m15, atr15, event, first+1, "HOLD_2X15M",
                                     boundary, float(h1.atr.loc[ts])))
            retest = None
            for j in range(first, min(first+13, len(m15))):
                tolerance = .10*float(atr15.iloc[j])
                ok = ((m15.low.iloc[j] <= boundary+tolerance and m15.close.iloc[j] >= boundary)
                      if sign == 1 else
                      (m15.high.iloc[j] >= boundary-tolerance and m15.close.iloc[j] <= boundary))
                if ok:
                    retest = j
                    rows.extend(evaluate(m15, atr15, event, j, "RETEST_15M",
                                         boundary, float(h1.atr.loc[ts])))
                    break
            if retest is not None:
                trigger = float(m15.high.iloc[retest] if sign == 1 else m15.low.iloc[retest])
                for j in range(retest+1, min(retest+13, len(m15))):
                    if (m15.close.iloc[j] > trigger if sign == 1 else m15.close.iloc[j] < trigger):
                        rows.extend(evaluate(m15, atr15, event, j, "RETEST_RESTART_15M",
                                             boundary, float(h1.atr.loc[ts])))
                        break
            last = ts
    return rows


def metrics(part: pd.DataFrame, cost_bps: float, exclude_top: int = 0) -> dict:
    if part.empty:
        return {"trades": 0}
    x = part.copy()
    x["net_r"] = x.gross_r-(cost_bps/10000)/x.risk_pct
    if exclude_top:
        x = x.drop(x.nlargest(min(exclude_top, len(x)), "net_r").index)
    if x.empty:
        return {"trades": 0}
    r = x.net_r.to_numpy(); pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(x), "avg_net_r": float(r.mean()),
            "median_net_r": float(np.median(r)), "win_rate": float((r > 0).mean()),
            "profit_factor": float(pos/neg) if neg else np.nan,
            "stop_rate": float(x.stopped.mean()), "hit_2r": float((x.mfe_r >= 2).mean()),
            "median_risk_pct": float(x.risk_pct.median()), "total_r": float(r.sum())}


def summarize(events: pd.DataFrame, columns: list[str], cost: float,
              exclude_top: int = 0) -> pd.DataFrame:
    rows = []
    for keys, part in events.groupby(columns, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(columns, keys)), **metrics(part, cost, exclude_top)})
    return pd.DataFrame(rows)


def main() -> None:
    output, cache = Path("precious_long_context_results"), Path("precious_long_context_data")
    output.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    all_rows, coverage = [], []
    for symbol, source in ASSETS.items():
        days = HISTORY_DAYS if source == "binance" else 59
        start = end-pd.Timedelta(days=days)
        print(f"[48H CONTEXT + 15M EXECUTION] {symbol} ({days}d)", flush=True)
        path = cache/f"{symbol.replace('=','_')}_15m.csv.gz"
        try:
            data = (pd.read_csv(path, index_col="time", parse_dates=True) if path.exists()
                    else (binance_15m(symbol, start, end) if source == "binance"
                          else yahoo_15m(symbol, start, end)))
            if not path.exists(): data.to_csv(path, compression="gzip")
            coverage.append({"symbol": symbol, "bars": len(data), "start": data.index.min(),
                             "end": data.index.max(), "days": (data.index.max()-data.index.min()).days})
            all_rows.extend(detect(data, symbol))
        except Exception as exc:
            print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
    events = pd.DataFrame(all_rows)
    if events.empty: raise RuntimeError("No long-context events")
    events["month"] = pd.to_datetime(events.breakout_time, utc=True).dt.strftime("%Y-%m")
    events.to_csv(output/"events.csv", index=False)
    pd.DataFrame(coverage).to_csv(output/"coverage.csv", index=False)
    route = summarize(events, ["symbol", "entry_route", "stop_method", "horizon_h"], PRIMARY_COST_BPS)
    route.to_csv(output/"route_stop_horizon.csv", index=False)
    robust = []
    for n in [0, 1, 3]:
        x = summarize(events, ["symbol", "entry_route", "stop_method", "horizon_h"],
                      PRIMARY_COST_BPS, n); x.insert(4, "exclude_top", n); robust.append(x)
    robust = pd.concat(robust, ignore_index=True)
    robust.to_csv(output/"outlier_robustness.csv", index=False)
    monthly = summarize(events, ["symbol", "month", "entry_route", "stop_method", "horizon_h"], PRIMARY_COST_BPS)
    monthly.to_csv(output/"monthly_results.csv", index=False)
    costs = []
    for cost in COSTS_BPS:
        x = summarize(events, ["symbol", "entry_route", "stop_method", "horizon_h"], cost)
        x.insert(0, "cost_bps", cost); costs.append(x)
    pd.concat(costs, ignore_index=True).to_csv(output/"cost_stress.csv", index=False)
    paxg = route[(route.symbol == "PAXGUSDT") & (route.trades >= 20)].copy()
    baseline = paxg[(paxg.entry_route == "H1_BREAKOUT_CLOSE") &
                    (paxg.stop_method == "FIXED_2PCT") & (paxg.horizon_h == 24)]
    best = paxg.sort_values(["avg_net_r", "profit_factor"], ascending=False).head(20)
    report = ["# 48H HOURLY CONTEXT + 15M EXECUTION", "",
              "The 48h context and compression policy are fixed. Execution routes and risk normalization are compared.", "",
              "## Coverage", "", "```csv", pd.DataFrame(coverage).to_csv(index=False).strip(), "```", "",
              "## Original 1h-style baseline on long PAXG sample", "", "```csv",
              baseline.to_csv(index=False).strip(), "```", "",
              "## PAXG cells with at least 20 events (diagnostic ranking)", "", "```csv",
              best.to_csv(index=False).strip(), "```", "",
              "A 15m route passes only if it beats the matched 1h baseline, survives 20 bps and exclusion of its three best trades."]
    text = "\n".join(report)
    (output/"validation_report.md").write_text(text, encoding="utf-8")
    print("\n===== LONG CONTEXT EXECUTION REPORT =====\n", flush=True); print(text, flush=True)


if __name__ == "__main__":
    main()
