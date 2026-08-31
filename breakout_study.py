#!/usr/bin/env python3
"""Backtest consolidation duration vs breakout quality in risk (R) units."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pandas as pd


BASE_URL = "https://fapi.binance.com"
DURATION_BUCKETS = [
    (20, 36, "20-36h"),
    (36, 72, "36-72h"),
    (72, 120, "3-5d"),
    (120, 192, "5-8d"),
    (192, 336, "8-14d"),
]
WINDOWS = [20, 24, 30, 36, 48, 60, 72, 96, 120, 144, 168, 192, 240, 288, 336]
HORIZONS_H = [3, 6, 12, 24]


@dataclass(frozen=True)
class StopSpec:
    name: str
    range_fraction: float | None = None
    atr_multiple: float | None = None


STOP_SPECS = [
    StopSpec("half_range", range_fraction=0.50),
    StopSpec("full_range", range_fraction=1.00),
    StopSpec("atr_1_5", atr_multiple=1.50),
]


class BinanceData:
    def __init__(self, pause: float = 0.06):
        self.pause = pause

    def get(self, path: str, params: dict) -> object:
        for attempt in range(6):
            url = BASE_URL + path + ("?" + urlencode(params) if params else "")
            try:
                req = Request(url, headers={"User-Agent": "consolidation-study/1.0"})
                with urlopen(req, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8"))
                time.sleep(self.pause)
                return data
            except HTTPError as exc:
                if exc.code == 429:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"Rate limited: {path}")

    def top_usdt_perpetuals(self, n: int) -> list[str]:
        info = self.get("/fapi/v1/exchangeInfo", {})
        allowed = {
            x["symbol"] for x in info["symbols"]
            if x.get("contractType") == "PERPETUAL"
            and x.get("quoteAsset") == "USDT"
            and x.get("status") == "TRADING"
        }
        tickers = self.get("/fapi/v1/ticker/24hr", {})
        ranked = sorted(
            (x for x in tickers if x["symbol"] in allowed),
            key=lambda x: float(x.get("quoteVolume", 0)), reverse=True,
        )
        return [x["symbol"] for x in ranked[:n]]

    def klines(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
        rows: list[list] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self.get("/fapi/v1/klines", {
                "symbol": symbol, "interval": interval, "startTime": cursor,
                "endTime": end_ms, "limit": 1500,
            })
            if not batch:
                break
            rows.extend(batch)
            nxt = int(batch[-1][0]) + 1
            if nxt <= cursor:
                break
            cursor = nxt
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                "quote_volume", "trades", "taker_base", "taker_quote", "ignore"]
        df = pd.DataFrame(rows, columns=cols)
        if df.empty:
            return df
        for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        return df.drop_duplicates("time").set_index("time").sort_index()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df.close.shift(1)
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def bucket_for(hours: int) -> str:
    for lo, hi, label in DURATION_BUCKETS:
        if lo <= hours < hi or (label == "8-14d" and hours <= hi):
            return label
    return "other"


def detect_events(h1: pd.DataFrame, symbol: str) -> list[dict]:
    """Detect close-confirmed breakouts using only data known at that close."""
    d = h1.copy()
    d["atr"] = atr(d)
    d["atr_pct"] = d.atr / d.close
    events: list[dict] = []
    last_event: dict[tuple[int, str], pd.Timestamp] = {}
    for w in WINDOWS:
        shifted = d.shift(1)
        upper = shifted.high.rolling(w, min_periods=w).max()
        lower = shifted.low.rolling(w, min_periods=w).min()
        range_size = upper - lower
        range_atr = range_size / d.atr
        # A consolidation must be bounded relative to its own recent volatility,
        # show low directional efficiency, and retain most closes in the central band.
        net_move = (d.close.shift(1) - d.close.shift(w)).abs()
        path = d.close.diff().abs().shift(1).rolling(w).sum()
        efficiency = net_move / path.replace(0, np.nan)
        width_pct = range_size / d.close
        valid = (efficiency <= 0.32) & (range_atr <= 9.0) & (width_pct <= 0.18)
        buffer = 0.10 * d.atr
        up = valid & (d.close > upper + buffer) & (d.close.shift(1) <= upper.shift(1) + buffer.shift(1))
        dn = valid & (d.close < lower - buffer) & (d.close.shift(1) >= lower.shift(1) - buffer.shift(1))
        for side, mask in (("LONG", up), ("SHORT", dn)):
            for ts in d.index[mask.fillna(False)]:
                key = (w, side)
                if key in last_event and ts - last_event[key] < pd.Timedelta(hours=max(12, w // 2)):
                    continue
                row = d.loc[ts]
                events.append({
                    "symbol": symbol, "breakout_time": ts, "side": side,
                    "duration_h": w, "duration_bucket": bucket_for(w),
                    "entry": float(row.close), "upper": float(upper.loc[ts]),
                    "lower": float(lower.loc[ts]), "range_size": float(range_size.loc[ts]),
                    "range_pct": float(width_pct.loc[ts]), "atr_1h": float(row.atr),
                    "efficiency": float(efficiency.loc[ts]),
                })
                last_event[key] = ts
    return events


def evaluate_event(event: dict, m15: pd.DataFrame, fee_bps_roundtrip: float) -> list[dict]:
    start = event["breakout_time"] + pd.Timedelta(hours=1)
    path = m15.loc[(m15.index >= start) & (m15.index < start + pd.Timedelta(hours=max(HORIZONS_H)))]
    if len(path) < 8:
        return []
    sign = 1 if event["side"] == "LONG" else -1
    entry = event["entry"]
    out: list[dict] = []
    for spec in STOP_SPECS:
        risk = (event["range_size"] * spec.range_fraction if spec.range_fraction is not None
                else event["atr_1h"] * spec.atr_multiple)
        if not np.isfinite(risk) or risk <= 0:
            continue
        stop = entry - sign * risk
        fee_r = (fee_bps_roundtrip / 10000.0 * entry) / risk
        base = {**event, "stop_method": spec.name, "stop_price": stop,
                "risk_abs": risk, "risk_pct": risk / entry, "fee_r": fee_r}
        for horizon in HORIZONS_H:
            p = path.loc[path.index < start + pd.Timedelta(hours=horizon)]
            favorable = ((p.high-entry) if sign == 1 else (entry-p.low)) / risk
            adverse = ((entry-p.low) if sign == 1 else (p.high-entry)) / risk
            stopped = adverse >= 1.0
            first_stop_pos = int(np.argmax(stopped.to_numpy())) if stopped.any() else len(p)
            rec = dict(base)
            rec.update({
                "horizon_h": horizon,
                "mfe_r": float(favorable.max()), "mae_r": float(adverse.max()),
                "hit_1r": bool((favorable.iloc[:first_stop_pos+1] >= 1).any()),
                "hit_2r": bool((favorable.iloc[:first_stop_pos+1] >= 2).any()),
                "hit_3r": bool((favorable.iloc[:first_stop_pos+1] >= 3).any()),
                "stopped": bool(stopped.any()),
                "terminal_r": float(max(-1.0, sign * (p.close.iloc[-1]-entry) / risk) - fee_r),
            })
            out.append(rec)
    return out


def summarise(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    g = events.groupby(group_cols, dropna=False)
    return g.agg(
        samples=("symbol", "size"), symbols=("symbol", "nunique"),
        median_risk_pct=("risk_pct", "median"), median_mfe_r=("mfe_r", "median"),
        median_mae_r=("mae_r", "median"), hit_1r=("hit_1r", "mean"),
        hit_2r=("hit_2r", "mean"), hit_3r=("hit_3r", "mean"),
        stop_rate=("stopped", "mean"), avg_terminal_r=("terminal_r", "mean"),
        median_terminal_r=("terminal_r", "median"),
    ).reset_index()


def report(duration: pd.DataFrame, symbols: list[str], args: argparse.Namespace) -> str:
    lines = ["# 横盘—突破时长研究", "", f"标的数：{len(symbols)}；历史长度：{args.history_days}天。",
             f"往返成本假设：{args.fee_bps_roundtrip:.1f} bps。", "",
             "所有结果均以实际结构止损距离归一化为R；绝对涨跌幅不会直接被当成优势。", ""]
    if duration.empty:
        lines += ["未产生合格事件。请检查数据下载或适当放宽横盘条件。"]
    else:
        core = duration[(duration.horizon_h == 24) & (duration.stop_method == "half_range")]
        lines += ["## 24小时、半区间止损摘要", "", "```csv", core.to_csv(index=False).strip(), "```", ""]
    lines += ["## 判读原则", "", "- 至少保留100个独立事件后再比较时长组。",
              "- 优先看样本外的平均终值R、2R到达率、止损率和中位风险百分比。",
              "- 若更长横盘只有绝对MFE上升、但MFE/R和净R没有改善，则不支持延长扫描门槛。"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=50)
    p.add_argument("--symbol-list", default="")
    p.add_argument("--history-days", type=int, default=730)
    p.add_argument("--fee-bps-roundtrip", type=float, default=10.0)
    p.add_argument("--cache", default="data")
    p.add_argument("--output", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache, output = Path(args.cache), Path(args.output)
    cache.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    api = BinanceData()
    symbols = ([x.strip().upper() for x in args.symbol_list.split(",") if x.strip()]
               if args.symbol_list else api.top_usdt_perpetuals(args.symbols))
    end = pd.Timestamp.now(tz="UTC").floor("15min")
    start = end - pd.Timedelta(days=args.history_days)
    all_rows: list[dict] = []
    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}", flush=True)
        frames = {}
        for interval in ("1h", "15m"):
            fp = cache / f"{symbol}_{interval}_{args.history_days}d.csv.gz"
            if fp.exists():
                frames[interval] = pd.read_csv(fp, index_col="time", parse_dates=True)
            else:
                frames[interval] = api.klines(symbol, interval, int(start.timestamp()*1000), int(end.timestamp()*1000))
                frames[interval].to_csv(fp, compression="gzip")
        for ev in detect_events(frames["1h"], symbol):
            all_rows.extend(evaluate_event(ev, frames["15m"], args.fee_bps_roundtrip))
    events = pd.DataFrame(all_rows)
    events.to_csv(output / "events.csv", index=False)
    duration = summarise(events, ["duration_bucket", "duration_h", "stop_method", "horizon_h"])
    stops = summarise(events, ["stop_method", "horizon_h"])
    duration.to_csv(output / "duration_summary.csv", index=False)
    stops.to_csv(output / "stop_summary.csv", index=False)
    (output / "validation_report.md").write_text(report(duration, symbols, args), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args) | {"symbols_used": symbols}, indent=2), encoding="utf-8")
    print(f"Done: {len(events):,} evaluated paths -> {output}")


if __name__ == "__main__":
    main()
