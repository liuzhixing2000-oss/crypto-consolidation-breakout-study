#!/usr/bin/env python3
"""Cross-asset validation of consolidation breakouts on liquid markets."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


HISTORY_DAYS = 729
WINDOWS = [20, 24, 30, 36, 48, 72, 96, 120]
STOPS = [0.01, 0.015, 0.02]
HORIZONS = [6, 12, 24, 48]
COST_BPS = 20.0
ASSETS = {
    "CRYPTO": {"BTCUSDT": "binance", "ETHUSDT": "binance"},
    "METALS": {"GC=F": "yahoo", "SI=F": "yahoo"},
    "EQUITIES": {
        "TSLA": "yahoo", "NVDA": "yahoo", "AMD": "yahoo",
        "AAPL": "yahoo", "META": "yahoo", "AMZN": "yahoo",
        "COIN": "yahoo", "MSTR": "yahoo", "PLTR": "yahoo",
        "SNDK": "yahoo",
    },
}


def get_json(url: str) -> object:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 cross-asset-study/1.0"})
    for attempt in range(6):
        try:
            with urlopen(req, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def yahoo_hourly(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = ("https://query1.finance.yahoo.com/v8/finance/chart/" + quote(symbol, safe="") + "?" +
           urlencode({"period1": int(start.timestamp()), "period2": int(end.timestamp()),
                      "interval": "1h", "includePrePost": "false", "events": "div,splits"}))
    result = get_json(url)["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    frame = pd.DataFrame({
        "time": pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    }).dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame.volume.fillna(0)
    return frame.drop_duplicates("time").set_index("time").sort_index()


def binance_hourly(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows, cursor, end_ms = [], int(start.timestamp()*1000), int(end.timestamp()*1000)
    while cursor < end_ms:
        url = "https://fapi.binance.com/fapi/v1/klines?" + urlencode({
            "symbol": symbol, "interval": "1h", "startTime": cursor,
            "endTime": end_ms, "limit": 1500})
        batch = get_json(url)
        if not batch:
            break
        rows.extend(batch)
        cursor = int(batch[-1][0]) + 1
        time.sleep(.04)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
    frame = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame["time"] = pd.to_datetime(frame.open_time, unit="ms", utc=True)
    return frame.dropna(subset=["open", "high", "low", "close"]).drop_duplicates("time").set_index("time").sort_index()


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    prev = d.close.shift(1)
    tr = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    prior = d.close.shift(1)
    mid, std = prior.rolling(20).mean(), prior.rolling(20).std()
    d["bb_width"] = 4*std/mid
    d["bb_compression"] = d.bb_width / d.bb_width.shift(1).rolling(500, min_periods=150).median()
    d["volume_ratio"] = d.volume / d.volume.shift(1).rolling(48, min_periods=24).median().replace(0, np.nan)
    return d


def detect_and_evaluate(frame: pd.DataFrame, symbol: str, group: str) -> list[dict]:
    d, rows = add_features(frame), []
    for window in WINDOWS:
        upper = d.high.shift(1).rolling(window).max()
        lower = d.low.shift(1).rolling(window).min()
        width = upper-lower
        path = d.close.diff().abs().shift(1).rolling(window).sum()
        efficiency = (d.close.shift(1)-d.close.shift(window)).abs()/path.replace(0, np.nan)
        valid = (efficiency <= .32) & (width/d.atr <= 9) & (width/d.close <= .18)
        buffer = .10*d.atr
        masks = {
            "LONG": valid & (d.close > upper+buffer) & (d.close.shift(1) <= upper.shift(1)+buffer.shift(1)),
            "SHORT": valid & (d.close < lower-buffer) & (d.close.shift(1) >= lower.shift(1)-buffer.shift(1)),
        }
        last = {"LONG": None, "SHORT": None}
        for side, mask in masks.items():
            sign = 1 if side == "LONG" else -1
            for ts in d.index[mask.fillna(False)]:
                if last[side] is not None and ts-last[side] < pd.Timedelta(hours=12):
                    continue
                loc = d.index.get_loc(ts)
                entry = float(d.close.iloc[loc])
                for stop_pct in STOPS:
                    risk = entry*stop_pct
                    for horizon in HORIZONS:
                        future = d.iloc[loc+1:loc+1+horizon]
                        if len(future) < horizon:
                            continue
                        favorable = ((future.high-entry) if sign == 1 else (entry-future.low))/risk
                        adverse = ((entry-future.low) if sign == 1 else (future.high-entry))/risk
                        stopped_mask = adverse >= 1
                        first_stop = int(np.argmax(stopped_mask.to_numpy())) if stopped_mask.any() else len(future)
                        mfe = float(favorable.iloc[:first_stop].max()) if first_stop else 0.0
                        stopped = bool(stopped_mask.any())
                        gross = -1.0 if stopped else float(sign*(future.close.iloc[-1]-entry)/risk)
                        rows.append({
                            "asset_group": group, "symbol": symbol, "breakout_time": ts,
                            "side": side, "duration_h": window, "stop_pct": stop_pct,
                            "horizon_h": horizon, "entry": entry, "range_pct": float(width.loc[ts]/entry),
                            "efficiency": float(efficiency.loc[ts]), "volume_ratio": float(d.volume_ratio.loc[ts]),
                            "bb_compression": float(d.bb_compression.loc[ts]), "mfe_r": mfe,
                            "stopped": stopped, "gross_r": gross,
                        })
                last[side] = ts
    return rows


def metrics(x: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    if x.empty:
        return {"trades": 0}
    r = x.gross_r.to_numpy() - (cost_bps/10000)/x.stop_pct.to_numpy()
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(r), "symbols": x.symbol.nunique(), "avg_net_r": r.mean(),
            "median_net_r": float(np.median(r)), "win_rate": float((r > 0).mean()),
            "profit_factor": float(pos/neg) if neg else np.nan,
            "stop_rate": float(x.stopped.mean()), "hit_2r": float((x.mfe_r >= 2).mean()),
            "total_r": float(r.sum())}


def walk_forward(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = events[(events.duration_h == 48) & (events.stop_pct == .02) & (events.horizon_h == 24)].copy()
    selected, thresholds = [], []
    start, finish = base.breakout_time.min(), base.breakout_time.max()
    test_start, fold = start+pd.Timedelta(days=180), 0
    while test_start < finish:
        fold += 1
        test_end = test_start+pd.Timedelta(days=90)
        for group in ASSETS:
            train = base[(base.asset_group == group) & (base.breakout_time >= test_start-pd.Timedelta(days=180)) &
                         (base.breakout_time < test_start)]
            test = base[(base.asset_group == group) & (base.breakout_time >= test_start) &
                        (base.breakout_time < test_end)]
            if len(train) < 40 or test.empty:
                continue
            q = {"vol40": train.volume_ratio.quantile(.4), "vol80": train.volume_ratio.quantile(.8),
                 "bb40": train.bb_compression.quantile(.4)}
            thresholds.append({"fold": fold, "asset_group": group, "test_start": test_start,
                               "test_end": test_end, "train_events": len(train), **q})
            for policy, mask in {
                "BASELINE": pd.Series(True, index=test.index),
                "VOLUME_COMPRESSION": test.volume_ratio.between(q["vol40"], q["vol80"]) &
                                      (test.bb_compression <= q["bb40"]),
            }.items():
                part = test.loc[mask].copy()
                part["fold"], part["policy"] = fold, policy
                selected.append(part)
        test_start = test_end
    chosen = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    rows = []
    if not chosen.empty:
        for keys, part in chosen.groupby(["policy", "asset_group"], observed=True):
            rows.append({"policy": keys[0], "asset_group": keys[1], **metrics(part)})
        for keys, part in chosen.groupby(["policy", "asset_group", "symbol"], observed=True):
            rows.append({"policy": keys[0], "asset_group": keys[1], "symbol": keys[2], **metrics(part)})
    return pd.DataFrame(rows), pd.DataFrame(thresholds), chosen


def robustness(chosen: pd.DataFrame) -> pd.DataFrame:
    x = chosen[chosen.policy == "VOLUME_COMPRESSION"].copy()
    if x.empty:
        return pd.DataFrame()
    rows = []
    for group, part in x.groupby("asset_group", observed=True):
        rows.append({"test": "ALL", "asset_group": group, **metrics(part)})
        for side, p in part.groupby("side", observed=True):
            rows.append({"test": f"SIDE_{side}", "asset_group": group, **metrics(p)})
        contributions = part.assign(net=part.gross_r-(COST_BPS/10000)/part.stop_pct).groupby("symbol").net.sum()
        for n in (1, 2):
            keep = part[~part.symbol.isin(contributions.nlargest(n).index)]
            rows.append({"test": f"EXCLUDE_TOP_{n}", "asset_group": group, **metrics(keep)})
    return pd.DataFrame(rows)


def main() -> None:
    output, cache = Path("cross_asset_results"), Path("cross_asset_data")
    output.mkdir(exist_ok=True); cache.mkdir(exist_ok=True)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    start = end-pd.Timedelta(days=HISTORY_DAYS)
    all_rows = []
    for group, assets in ASSETS.items():
        for symbol, source in assets.items():
            print(f"[{group}] {symbol}", flush=True)
            path = cache/f"{symbol.replace('=','_')}_1h.csv.gz"
            try:
                data = (pd.read_csv(path, index_col="time", parse_dates=True) if path.exists()
                        else (binance_hourly(symbol, start, end) if source == "binance"
                              else yahoo_hourly(symbol, start, end)))
                if not path.exists():
                    data.to_csv(path, compression="gzip")
                if len(data) < 500:
                    print(f"  skip: only {len(data)} hourly bars", flush=True)
                    continue
                all_rows.extend(detect_and_evaluate(data, symbol, group))
            except Exception as exc:
                print(f"  skip: {type(exc).__name__}: {exc}", flush=True)
    events = pd.DataFrame(all_rows)
    events["breakout_time"] = pd.to_datetime(events.breakout_time, utc=True)
    events.to_csv(output/"events.csv", index=False)
    duration = []
    for keys, part in events.groupby(["asset_group", "duration_h", "stop_pct", "horizon_h"], observed=True):
        duration.append({"asset_group": keys[0], "duration_h": keys[1], "stop_pct": keys[2],
                         "horizon_h": keys[3], **metrics(part)})
    duration = pd.DataFrame(duration)
    validation, thresholds, chosen = walk_forward(events)
    stress = robustness(chosen)
    duration.to_csv(output/"duration_results.csv", index=False)
    validation.to_csv(output/"walk_forward_validation.csv", index=False)
    thresholds.to_csv(output/"walk_forward_thresholds.csv", index=False)
    stress.to_csv(output/"candidate_robustness.csv", index=False)
    report = ["# LIQUID CROSS-ASSET VALIDATION", "",
              "## 48h / 2% stop / 24 trading-hour hold: rolling OOS", "", "```csv",
              validation.to_csv(index=False).strip(), "```", "",
              "## Frozen VOLUME_COMPRESSION robustness", "", "```csv",
              stress.to_csv(index=False).strip(), "```", "",
              "## Duration comparison", "", "```csv",
              duration[(duration.stop_pct == .02) & (duration.horizon_h == 24)].to_csv(index=False).strip(),
              "```"]
    text = "\n".join(report)
    (output/"validation_report.md").write_text(text, encoding="utf-8")
    print("\n===== CROSS-ASSET VALIDATION REPORT =====\n")
    print(text)


if __name__ == "__main__":
    main()
