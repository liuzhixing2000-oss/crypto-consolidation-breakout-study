#!/usr/bin/env python3
"""Exploratory 15m / 1h consolidation-breakout study for gold and silver.

The free Yahoo chart endpoint normally exposes only about 60 calendar days of
15-minute futures data.  This script therefore labels every result as a pilot
and keeps parameter selection separate from the chronological test half.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SYMBOLS = ["GC=F", "SI=F"]
LOOKBACK_DAYS = 59
PURE_WINDOWS_H = [5, 8, 12, 20, 32, 48]
HORIZONS_H = [2, 4, 8, 12, 24]
ATR_STOPS = [1.0, 1.5, 2.0]
COSTS_BPS = [2.0, 5.0, 10.0, 20.0]
PRIMARY_COST_BPS = 10.0


def get_json(url: str) -> object:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 precious-metals-study/1.0"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def yahoo_15m(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe="") + "?" +
           urlencode({"period1": int(start.timestamp()), "period2": int(end.timestamp()),
                      "interval": "15m", "includePrePost": "false", "events": "div,splits"}))
    payload = get_json(url)["chart"]
    if not payload.get("result"):
        raise RuntimeError(str(payload.get("error")))
    result = payload["result"][0]
    q = result["indicators"]["quote"][0]
    frame = pd.DataFrame({
        "time": pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    }).dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame.volume.fillna(0)
    return frame.drop_duplicates("time").set_index("time").sort_index()


def features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    prev = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    prior = d.close.shift(1)
    mid, std = prior.rolling(20).mean(), prior.rolling(20).std()
    d["bb_width"] = 4*std/mid
    d["bb_compression"] = d.bb_width / d.bb_width.shift(1).rolling(480, min_periods=120).median()
    d["volume_ratio"] = d.volume / d.volume.shift(1).rolling(192, min_periods=48).median().replace(0, np.nan)
    # Expanding historical thresholds: every decision uses strictly earlier bars.
    hist_vol = d.volume_ratio.shift(1).expanding(min_periods=480)
    hist_bb = d.bb_compression.shift(1).expanding(min_periods=480)
    d["vol40"], d["vol80"] = hist_vol.quantile(.4), hist_vol.quantile(.8)
    d["bb40"] = hist_bb.quantile(.4)
    return d


def session_name(ts: pd.Timestamp) -> str:
    hour = ts.hour
    if hour < 7:
        return "ASIA"
    if hour < 13:
        return "LONDON"
    if hour < 21:
        return "NEW_YORK"
    return "LATE"


def range_state(d: pd.DataFrame, bars: int) -> pd.DataFrame:
    upper = d.high.shift(1).rolling(bars).max()
    lower = d.low.shift(1).rolling(bars).min()
    width = upper-lower
    path = d.close.diff().abs().shift(1).rolling(bars).sum()
    efficiency = (d.close.shift(1)-d.close.shift(bars)).abs()/path.replace(0, np.nan)
    valid = ((efficiency <= .32) & (width/d.atr <= 9) & (width/d.close <= .18) &
             d.volume_ratio.between(d.vol40, d.vol80) & (d.bb_compression <= d.bb40))
    return pd.DataFrame({"upper": upper, "lower": lower, "width": width,
                         "efficiency": efficiency, "valid": valid}, index=d.index)


def evaluate(d: pd.DataFrame, event: dict, entry_loc: int, boundary: float,
             route: str) -> list[dict]:
    rows, sign = [], (1 if event["side"] == "LONG" else -1)
    entry, atr = float(d.close.iloc[entry_loc]), float(d.atr.iloc[entry_loc])
    stop_specs = {f"ATR_{m:g}": m*atr for m in ATR_STOPS}
    structural = (entry-(boundary-.25*atr) if sign == 1 else (boundary+.25*atr)-entry)
    if structural > 0:
        stop_specs["STRUCTURE_0.25ATR"] = structural
    for stop_method, risk in stop_specs.items():
        risk_pct = risk/entry
        if not (.0025 <= risk_pct <= .03):
            continue
        for horizon_h in HORIZONS_H:
            bars = horizon_h*4
            future = d.iloc[entry_loc+1:entry_loc+1+bars]
            if len(future) < bars:
                continue
            favorable = ((future.high-entry) if sign == 1 else (entry-future.low))/risk
            adverse = ((entry-future.low) if sign == 1 else (future.high-entry))/risk
            stopped_mask = adverse >= 1
            first_stop = int(np.argmax(stopped_mask.to_numpy())) if stopped_mask.any() else len(future)
            stopped = bool(stopped_mask.any())
            mfe = float(favorable.iloc[:first_stop].max()) if first_stop else 0.0
            gross = -1.0 if stopped else float(sign*(future.close.iloc[-1]-entry)/risk)
            rows.append({**event, "entry_time": d.index[entry_loc], "entry_route": route,
                         "stop_method": stop_method, "risk_pct": risk_pct,
                         "horizon_h": horizon_h, "gross_r": gross, "mfe_r": mfe,
                         "stopped": stopped})
    return rows


def detect(frame: pd.DataFrame, symbol: str) -> list[dict]:
    d, rows = features(frame), []
    specs = [("PURE_15M", h, h*4) for h in PURE_WINDOWS_H]
    specs.append(("ONE_HOUR_CONTEXT", 48, 48*4))
    for model, duration_h, bars in specs:
        state = range_state(d, bars)
        buffer = .10*d.atr
        masks = {
            "LONG": state.valid & (d.close > state.upper+buffer) &
                    (d.close.shift(1) <= state.upper.shift(1)+buffer.shift(1)),
            "SHORT": state.valid & (d.close < state.lower-buffer) &
                     (d.close.shift(1) >= state.lower.shift(1)-buffer.shift(1)),
        }
        for side, mask in masks.items():
            last = None
            for ts in d.index[mask.fillna(False)]:
                if last is not None and ts-last < pd.Timedelta(hours=max(3, duration_h/4)):
                    continue
                loc, sign = d.index.get_loc(ts), (1 if side == "LONG" else -1)
                boundary = float(state.upper.loc[ts] if side == "LONG" else state.lower.loc[ts])
                event = {"symbol": symbol, "model": model, "duration_h": duration_h,
                         "breakout_time": ts, "session": session_name(ts), "side": side,
                         "range_pct": float(state.width.loc[ts]/d.close.loc[ts]),
                         "efficiency": float(state.efficiency.loc[ts])}
                rows.extend(evaluate(d, event, loc, boundary, "DIRECT_15M"))
                # Two consecutive 15m closes beyond the original range.
                if loc+1 < len(d):
                    held = (d.close.iloc[loc+1] > boundary if sign == 1 else d.close.iloc[loc+1] < boundary)
                    if held:
                        rows.extend(evaluate(d, event, loc+1, boundary, "HOLD_2X15M"))
                # First retest within 3 hours, followed by a close beyond retest extreme.
                retest = None
                for j in range(loc+1, min(loc+13, len(d))):
                    tol = .10*float(d.atr.iloc[j])
                    ok = ((d.low.iloc[j] <= boundary+tol and d.close.iloc[j] >= boundary) if sign == 1
                          else (d.high.iloc[j] >= boundary-tol and d.close.iloc[j] <= boundary))
                    if ok:
                        retest = j
                        rows.extend(evaluate(d, event, j, boundary, "RETEST_15M"))
                        break
                if retest is not None:
                    trigger = float(d.high.iloc[retest] if sign == 1 else d.low.iloc[retest])
                    for j in range(retest+1, min(retest+13, len(d))):
                        if (d.close.iloc[j] > trigger if sign == 1 else d.close.iloc[j] < trigger):
                            rows.extend(evaluate(d, event, j, boundary, "RETEST_RESTART_15M"))
                            break
                last = ts
    return rows


def metric(part: pd.DataFrame, cost_bps: float) -> dict:
    if part.empty:
        return {"trades": 0}
    net = part.gross_r.to_numpy()-(cost_bps/10000)/part.risk_pct.to_numpy()
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    return {"trades": len(part), "avg_net_r": float(net.mean()),
            "median_net_r": float(np.median(net)), "win_rate": float((net > 0).mean()),
            "profit_factor": float(gains/losses) if losses else np.nan,
            "stop_rate": float(part.stopped.mean()), "hit_2r": float((part.mfe_r >= 2).mean()),
            "total_r": float(net.sum())}


def summarize(events: pd.DataFrame, columns: list[str], cost: float) -> pd.DataFrame:
    rows = []
    for keys, part in events.groupby(columns, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append({**dict(zip(columns, keys)), **metric(part, cost)})
    return pd.DataFrame(rows)


def frozen_candidates(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select one rule per model on calibration data, then evaluate it untouched."""
    keys = ["model", "duration_h", "entry_route", "stop_method", "horizon_h"]
    train = events[events["sample"] == "CALIBRATION"].copy()
    train["net"] = train.gross_r-(PRIMARY_COST_BPS/10000)/train.risk_pct
    # Selection score is clipped only for ranking, so one exceptional trend
    # cannot determine the rule. Reported test returns remain uncapped.
    train["selection_r"] = train.net.clip(-1.5, 3.0)
    ranked = []
    for values, part in train.groupby(keys, observed=True):
        counts = part.groupby("symbol").size()
        if len(part) < 12 or any(counts.get(s, 0) < 4 for s in SYMBOLS):
            continue
        ranked.append({**dict(zip(keys, values)), "calibration_trades": len(part),
                       "calibration_score": float(part.selection_r.mean()),
                       "calibration_median_r": float(part.net.median()),
                       "calibration_min_symbol_score": float(part.groupby("symbol").selection_r.mean().min())})
    ranked_frame = pd.DataFrame(ranked)
    if ranked_frame.empty:
        return ranked_frame, pd.DataFrame()
    # Cross-symbol agreement is required before average score breaks ties.
    ranked_frame = ranked_frame.sort_values(
        ["model", "calibration_min_symbol_score", "calibration_score"], ascending=[True, False, False])
    selected = ranked_frame.groupby("model", as_index=False).head(1)
    test_rows = []
    test = events[events["sample"] == "TEST"]
    for _, rule in selected.iterrows():
        mask = pd.Series(True, index=test.index)
        for key in keys:
            mask &= test[key] == rule[key]
        part = test[mask]
        test_rows.append({**{key: rule[key] for key in keys}, "scope": "BOTH", **metric(part, PRIMARY_COST_BPS)})
        for symbol in SYMBOLS:
            test_rows.append({**{key: rule[key] for key in keys}, "scope": symbol,
                              **metric(part[part.symbol == symbol], PRIMARY_COST_BPS)})
    return selected, pd.DataFrame(test_rows)


def main() -> None:
    output, cache = Path("precious_15m_results"), Path("precious_15m_data")
    output.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    start = end-pd.Timedelta(days=LOOKBACK_DAYS)
    all_rows, coverage = [], []
    for symbol in SYMBOLS:
        print(f"[PRECIOUS 15M] {symbol}", flush=True)
        path = cache/f"{symbol.replace('=','_')}_15m.csv.gz"
        try:
            data = (pd.read_csv(path, index_col="time", parse_dates=True) if path.exists()
                    else yahoo_15m(symbol, start, end))
            if not path.exists():
                data.to_csv(path, compression="gzip")
            coverage.append({"symbol": symbol, "bars": len(data), "start": data.index.min(),
                             "end": data.index.max(), "calendar_days": (data.index.max()-data.index.min()).days})
            all_rows.extend(detect(data, symbol))
        except Exception as exc:
            print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
    events = pd.DataFrame(all_rows)
    if events.empty:
        raise RuntimeError("No evaluable events. Check the Yahoo 15m data response.")
    events["breakout_time"] = pd.to_datetime(events.breakout_time, utc=True)
    split = events.breakout_time.min()+(events.breakout_time.max()-events.breakout_time.min())/2
    events["sample"] = np.where(events.breakout_time < split, "CALIBRATION", "TEST")
    events.to_csv(output/"events.csv", index=False)
    pd.DataFrame(coverage).to_csv(output/"data_coverage.csv", index=False)
    primary = summarize(events, ["sample", "symbol", "model", "duration_h", "entry_route",
                                 "stop_method", "horizon_h"], PRIMARY_COST_BPS)
    sessions = summarize(events[events["sample"] == "TEST"],
                         ["symbol", "session", "entry_route", "stop_method", "horizon_h"],
                         PRIMARY_COST_BPS)
    costs = []
    for cost in COSTS_BPS:
        x = summarize(events[events["sample"] == "TEST"],
                      ["model", "duration_h", "entry_route", "stop_method", "horizon_h"], cost)
        x.insert(0, "cost_bps", cost); costs.append(x)
    cost_frame = pd.concat(costs, ignore_index=True)
    selected, frozen_test = frozen_candidates(events)
    primary.to_csv(output/"primary_results.csv", index=False)
    sessions.to_csv(output/"session_results.csv", index=False)
    cost_frame.to_csv(output/"cost_stress.csv", index=False)
    selected.to_csv(output/"calibration_selection.csv", index=False)
    frozen_test.to_csv(output/"frozen_test_results.csv", index=False)
    report = ["# GOLD / SILVER 15M MULTI-TIMEFRAME PILOT", "",
              f"Coverage: {LOOKBACK_DAYS} calendar days; chronological split at {split}.",
              "Yahoo 15m history is short, so these results are exploratory and cannot confirm production edge.",
              f"Primary round-trip cost: {PRIMARY_COST_BPS:g} bps.", "",
              "## Data coverage", "", "```csv", pd.DataFrame(coverage).to_csv(index=False).strip(), "```", "",
              "## Rules selected using calibration half only", "", "```csv",
              selected.to_csv(index=False).strip(), "```", "",
              "## Untouched test-half results", "", "```csv",
              frozen_test.to_csv(index=False).strip(), "```", "",
              "## Interpretation guardrails", "",
              "- Do not select a live rule from this table alone; the same sample tested many cells.",
              "- A candidate must agree across GC and SI, survive cost stress, and retain sign by session.",
              "- Any surviving rule must be frozen and rerun on paid/archived multi-year 15m data."]
    text = "\n".join(report)
    (output/"validation_report.md").write_text(text, encoding="utf-8")
    print("\n===== PRECIOUS 15M PILOT REPORT =====\n", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
