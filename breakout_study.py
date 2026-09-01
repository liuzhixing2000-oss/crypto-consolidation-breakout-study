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
WINDOWS = [20, 24, 30, 36, 48]
HORIZONS_H = [3, 6, 12, 24]


@dataclass(frozen=True)
class StopSpec:
    name: str
    risk_pct: float


STOP_SPECS = [
    StopSpec("fixed_1pct", 0.010),
    StopSpec("fixed_1_5pct", 0.015),
    StopSpec("fixed_2pct", 0.020),
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


def add_context_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add features observable when each 1H candle closes; no future bars used."""
    d = d.copy()
    d["atr"] = atr(d)
    d["atr_pct"] = d.atr / d.close
    prior_close = d.close.shift(1)
    bb_mid = prior_close.rolling(20).mean()
    bb_std = prior_close.rolling(20).std()
    d["bb_width"] = (4.0 * bb_std) / bb_mid
    d["atr_compression"] = d.atr_pct / d.atr_pct.shift(1).rolling(720, min_periods=240).median()
    d["bb_compression"] = d.bb_width / d.bb_width.shift(1).rolling(720, min_periods=240).median()
    d["compression_slope"] = (
        d.atr.shift(1).rolling(12).mean() /
        d.atr.shift(13).rolling(24).mean()
    )

    # Resample first, then shift one complete 4H bucket to prevent using the
    # still-forming higher-timeframe candle.
    h4 = d[["open", "high", "low", "close", "volume"]].resample("4h").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    h4["ema20"] = h4.close.ewm(span=20, adjust=False).mean()
    h4["ema50"] = h4.close.ewm(span=50, adjust=False).mean()
    h4["trend_sign"] = np.sign(h4.ema20-h4.ema50).shift(1)
    h4["trend_strength"] = ((h4.ema20-h4.ema50).abs()/h4.close).shift(1)
    d["h4_trend_sign"] = h4.trend_sign.reindex(d.index, method="ffill")
    d["h4_trend_strength"] = h4.trend_strength.reindex(d.index, method="ffill")
    return d


def bucket_for(hours: int) -> str:
    for lo, hi, label in DURATION_BUCKETS:
        if lo <= hours < hi or (label == "8-14d" and hours <= hi):
            return label
    return "other"


def detect_events(h1: pd.DataFrame, symbol: str, symbol_rank: int = 0) -> list[dict]:
    """Detect close-confirmed breakouts using only data known at that close."""
    d = add_context_features(h1)
    candidates: list[dict] = []
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
                candle_range = max(float(row.high-row.low), 1e-12)
                direction = 1 if side == "LONG" else -1
                boundary = float(upper.loc[ts] if side == "LONG" else lower.loc[ts])
                breakout_distance = direction * (float(row.close)-boundary)
                candidates.append({
                    "symbol": symbol, "symbol_rank": symbol_rank,
                    "universe": "CORE_1_50" if symbol_rank <= 50 else "EXTERNAL_51_100",
                    "breakout_time": ts, "side": side,
                    "duration_h": w, "duration_bucket": bucket_for(w),
                    "entry": float(row.close), "upper": float(upper.loc[ts]),
                    "lower": float(lower.loc[ts]), "range_size": float(range_size.loc[ts]),
                    "range_pct": float(width_pct.loc[ts]), "atr_1h": float(row.atr),
                    "efficiency": float(efficiency.loc[ts]),
                    "atr_compression": float(row.atr_compression),
                    "bb_compression": float(row.bb_compression),
                    "compression_slope": float(row.compression_slope),
                    "volume_ratio": float(row.volume / d.volume.shift(1).rolling(48).median().loc[ts]),
                    "body_atr": float(abs(row.close-row.open) / row.atr),
                    "close_location": float((row.close-row.low)/candle_range if side == "LONG"
                                            else (row.high-row.close)/candle_range),
                    "breakout_atr": float(breakout_distance / row.atr),
                    "h4_aligned": bool(row.h4_trend_sign == direction),
                    "h4_trend_strength": float(row.h4_trend_strength),
                })
                last_event[key] = ts
    # One market break can satisfy several nested windows. Assign it only to
    # the longest valid window, then suppress same-direction repeats for 12h.
    if not candidates:
        return []
    unique: dict[tuple[pd.Timestamp, str], dict] = {}
    for event in candidates:
        key = (event["breakout_time"], event["side"])
        if key not in unique or event["duration_h"] > unique[key]["duration_h"]:
            unique[key] = event
    events: list[dict] = []
    last_kept: dict[str, pd.Timestamp] = {}
    for event in sorted(unique.values(), key=lambda x: x["breakout_time"]):
        side = event["side"]
        if side in last_kept and event["breakout_time"] - last_kept[side] < pd.Timedelta(hours=12):
            continue
        events.append(event)
        last_kept[side] = event["breakout_time"]
    return events


def evaluate_route(event: dict, future: pd.DataFrame, entry_time: pd.Timestamp,
                   entry: float, route: str, fee_bps_roundtrip: float) -> list[dict]:
    sign = 1 if event["side"] == "LONG" else -1
    out: list[dict] = []
    for spec in STOP_SPECS:
        risk = entry * spec.risk_pct
        stop = entry - sign * risk
        fee_r = (fee_bps_roundtrip / 10000.0 * entry) / risk
        base = {**event, "entry_route": route, "entry_time": entry_time,
                "route_entry": entry, "stop_method": spec.name, "stop_price": stop,
                "risk_abs": risk, "risk_pct": spec.risk_pct, "fee_r": fee_r}
        for horizon in HORIZONS_H:
            p = future.loc[(future.index > entry_time) &
                           (future.index <= entry_time + pd.Timedelta(hours=horizon))]
            if p.empty:
                continue
            favorable = ((p.high-entry) if sign == 1 else (entry-p.low)) / risk
            adverse = ((entry-p.low) if sign == 1 else (p.high-entry)) / risk
            stop_mask = adverse >= 1.0
            first_stop = int(np.argmax(stop_mask.to_numpy())) if stop_mask.any() else len(p)
            # Conservative intrabar rule: if stop and target occur in the same
            # 15m bar, count the stop first and exclude that bar from MFE/targets.
            before_stop = favorable.iloc[:first_stop]
            mfe = float(before_stop.max()) if len(before_stop) else 0.0
            mae = float(min(1.0, adverse.iloc[:first_stop+1].max()))
            stopped = bool(stop_mask.any())
            terminal = -1.0 if stopped else sign * (p.close.iloc[-1]-entry) / risk
            rec = dict(base)
            rec.update({"horizon_h": horizon, "mfe_r": mfe, "mae_r": mae,
                        "hit_1r": mfe >= 1, "hit_2r": mfe >= 2, "hit_3r": mfe >= 3,
                        "stopped": stopped, "gross_terminal_r": float(terminal),
                        "terminal_r": float(terminal-fee_r)})
            out.append(rec)
    return out


def evaluate_event(event: dict, m15: pd.DataFrame, fee_bps_roundtrip: float) -> list[dict]:
    start = event["breakout_time"] + pd.Timedelta(hours=1)
    future = m15.loc[(m15.index >= start) &
                     (m15.index <= start + pd.Timedelta(hours=36))]
    if len(future) < 8:
        return []
    out = evaluate_route(event, future, start-pd.Timedelta(minutes=15),
                         event["entry"], "DIRECT", fee_bps_roundtrip)
    # Retest: within 12h price touches the broken boundary and the 15m candle
    # closes back on the breakout side. Enter at that confirmation close.
    test = future.loc[future.index < start + pd.Timedelta(hours=12)]
    if event["side"] == "LONG":
        mask = (test.low <= event["upper"] + 0.10*event["atr_1h"]) & (test.close >= event["upper"])
    else:
        mask = (test.high >= event["lower"] - 0.10*event["atr_1h"]) & (test.close <= event["lower"])
    if mask.any():
        entry_time = test.index[int(np.argmax(mask.to_numpy()))]
        out += evaluate_route(event, future, entry_time, float(test.loc[entry_time, "close"]),
                              "RETEST", fee_bps_roundtrip)
    return out


def summarise(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    g = events.groupby(group_cols, dropna=False, observed=True)
    return g.agg(
        samples=("symbol", "size"), symbols=("symbol", "nunique"),
        median_risk_pct=("risk_pct", "median"), median_mfe_r=("mfe_r", "median"),
        median_mae_r=("mae_r", "median"), hit_1r=("hit_1r", "mean"),
        hit_2r=("hit_2r", "mean"), hit_3r=("hit_3r", "mean"),
        stop_rate=("stopped", "mean"), avg_terminal_r=("terminal_r", "mean"),
        median_terminal_r=("terminal_r", "median"),
    ).reset_index()


FEATURES = ["atr_compression", "bb_compression", "compression_slope",
            "volume_ratio", "body_atr", "close_location", "breakout_atr",
            "h4_trend_strength"]


def feature_study(events: pd.DataFrame) -> pd.DataFrame:
    """IS-defined quintiles, applied unchanged to OOS for the 48h baseline."""
    if events.empty:
        return pd.DataFrame()
    base = events[(events.duration_h == 48) & (events.entry_route == "DIRECT") &
                  (events.stop_method == "fixed_2pct") & (events.horizon_h == 24)].copy()
    rows: list[pd.DataFrame] = []
    for feature in FEATURES:
        train = pd.to_numeric(base.loc[base.sample_period == "IS", feature], errors="coerce").dropna()
        if train.nunique() < 5:
            continue
        edges = np.unique(train.quantile([0, .2, .4, .6, .8, 1]).to_numpy())
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        labels = [f"Q{i+1}" for i in range(len(edges)-1)]
        binned = pd.cut(pd.to_numeric(base[feature], errors="coerce"), edges,
                        labels=labels, include_lowest=True)
        temp = base.assign(feature=feature, feature_bin=binned)
        s = summarise(temp.dropna(subset=["feature_bin"]),
                      ["feature", "feature_bin", "sample_period"])
        rows.append(s)
    aligned = summarise(base.assign(feature="h4_aligned",
                                     feature_bin=base.h4_aligned.astype(str)),
                        ["feature", "feature_bin", "sample_period"])
    rows.append(aligned)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


POLICIES = ["BASELINE", "VOLUME", "BODY", "COMPRESSION", "VOLUME_BODY",
            "VOLUME_COMPRESSION", "BODY_COMPRESSION", "TRIPLE"]
COSTS_BPS = [6, 10, 15, 20, 25, 30]


def policy_masks(frame: pd.DataFrame, q: dict[str, float]) -> dict[str, pd.Series]:
    volume = frame.volume_ratio.between(q["vol40"], q["vol80"], inclusive="both")
    body = frame.body_atr >= q["body60"]
    compression = frame.bb_compression <= q["bb40"]
    yes = pd.Series(True, index=frame.index)
    return {"BASELINE": yes, "VOLUME": volume, "BODY": body,
            "COMPRESSION": compression, "VOLUME_BODY": volume & body,
            "VOLUME_COMPRESSION": volume & compression,
            "BODY_COMPRESSION": body & compression,
            "TRIPLE": volume & body & compression}


def risk_metrics(frame: pd.DataFrame, cost_bps: float) -> dict:
    if frame.empty:
        return {"trades": 0}
    x = frame.sort_values(["breakout_time", "symbol"]).copy()
    x["net_r"] = x.gross_terminal_r - (cost_bps/10000.0)/x.risk_pct
    r = x.net_r.to_numpy(float)
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[:-1]
    drawdown = equity-peak
    losses = r < 0
    longest = run = 0
    for loss in losses:
        run = run+1 if loss else 0
        longest = max(longest, run)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {"trades": len(x), "symbols": x.symbol.nunique(), "avg_net_r": r.mean(),
            "median_net_r": float(np.median(r)), "win_rate": float((r > 0).mean()),
            "profit_factor": float(pos/neg) if neg > 0 else np.nan,
            "max_drawdown_r": float(drawdown.min(initial=0)),
            "longest_loss_streak": longest, "stop_rate": float(x.stopped.mean()),
            "hit_2r": float(x.hit_2r.mean()), "total_r": float(r.sum())}


def cluster_bootstrap_ci(frame: pd.DataFrame, cost_bps: float = 20,
                         iterations: int = 1000) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    x = frame.copy()
    x["cluster"] = x.breakout_time.dt.floor("4h")
    x["net_r"] = x.gross_terminal_r - (cost_bps/10000.0)/x.risk_pct
    clusters = x.groupby("cluster", observed=True).net_r.mean().to_numpy()
    if len(clusters) < 20:
        return np.nan, np.nan
    rng = np.random.default_rng(260901)
    means = np.empty(iterations)
    for i in range(iterations):
        means[i] = rng.choice(clusters, size=len(clusters), replace=True).mean()
    return tuple(np.quantile(means, [.025, .975]))


def walk_forward_validation(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """180d train / 90d test. Thresholds come only from prior CORE events."""
    if events.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    base = events[(events.duration_h == 48) & (events.entry_route == "DIRECT") &
                  (events.stop_method == "fixed_2pct") & (events.horizon_h == 24)].copy()
    base = base.sort_values("breakout_time")
    start, finish = base.breakout_time.min(), base.breakout_time.max()
    test_start = start + pd.Timedelta(days=180)
    selected: list[pd.DataFrame] = []
    thresholds: list[dict] = []
    fold = 0
    while test_start < finish:
        fold += 1
        train_start, test_end = test_start-pd.Timedelta(days=180), test_start+pd.Timedelta(days=90)
        train = base[(base.universe == "CORE_1_50") &
                     (base.breakout_time >= train_start) & (base.breakout_time < test_start)]
        test = base[(base.breakout_time >= test_start) & (base.breakout_time < test_end)]
        if len(train) >= 200 and len(test) > 0:
            q = {"vol40": train.volume_ratio.quantile(.4), "vol80": train.volume_ratio.quantile(.8),
                 "body60": train.body_atr.quantile(.6), "bb40": train.bb_compression.quantile(.4)}
            thresholds.append({"fold": fold, "train_start": train_start, "test_start": test_start,
                               "test_end": test_end, "train_events": len(train), **q})
            for policy, mask in policy_masks(test, q).items():
                chosen = test.loc[mask].copy()
                chosen["fold"] = fold; chosen["policy"] = policy
                selected.append(chosen)
        test_start = test_end
    chosen = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    rows: list[dict] = []
    if not chosen.empty:
        for (fold, policy, universe), part in chosen.groupby(["fold", "policy", "universe"], observed=True):
            for cost in COSTS_BPS:
                rows.append({"fold": fold, "policy": policy, "universe": universe,
                             "cost_bps": cost, **risk_metrics(part, cost)})
    wf = pd.DataFrame(rows)

    aggregate: list[dict] = []
    stability: list[dict] = []
    if not chosen.empty:
        for (policy, universe), part in chosen.groupby(["policy", "universe"], observed=True):
            for cost in COSTS_BPS:
                aggregate.append({"policy": policy, "universe": universe, "cost_bps": cost,
                                  **risk_metrics(part, cost)})
            lo, hi = cluster_bootstrap_ci(part, 20)
            stability.append({"policy": policy, "universe": universe,
                              "cluster_ci95_low_20bps": lo, "cluster_ci95_high_20bps": hi,
                              "positive_fold_rate_20bps": float((wf[(wf.policy == policy) &
                                  (wf.universe == universe) & (wf.cost_bps == 20)].avg_net_r > 0).mean())})
    return wf, pd.DataFrame(aggregate), pd.DataFrame(stability), pd.DataFrame(thresholds), chosen


def selected_stability(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = events.copy()
    x["quarter"] = x.breakout_time.dt.tz_localize(None).dt.to_period("Q").astype(str)
    x["net_r_20bps"] = x.gross_terminal_r - (20/10000.0)/x.risk_pct
    symbol = x.groupby(["policy", "universe", "symbol"], observed=True).agg(
        trades=("symbol", "size"), avg_net_r=("net_r_20bps", "mean"),
        total_r=("net_r_20bps", "sum")).reset_index()
    quarter = x.groupby(["policy", "universe", "quarter"], observed=True).agg(
        trades=("symbol", "size"), avg_net_r=("net_r_20bps", "mean"),
        total_r=("net_r_20bps", "sum")).reset_index()
    return symbol, quarter


def report(duration: pd.DataFrame, features: pd.DataFrame, validation: pd.DataFrame,
           validation_stability: pd.DataFrame, thresholds: pd.DataFrame,
           symbols: list[str], args: argparse.Namespace) -> str:
    lines = ["# 横盘—突破时长研究", "", f"标的数：{len(symbols)}；历史长度：{args.history_days}天。",
             f"往返成本假设：{args.fee_bps_roundtrip:.1f} bps。", "",
             "所有结果均以实际结构止损距离归一化为R；绝对涨跌幅不会直接被当成优势。", ""]
    if duration.empty:
        lines += ["未产生合格事件。请检查数据下载或适当放宽横盘条件。"]
    else:
        core = duration[duration.horizon_h == 24]
        lines += ["## 24小时结果：直接突破 vs 回踩确认", "", "```csv",
                  core.to_csv(index=False).strip(), "```", ""]
    if not features.empty:
        lines += ["## 48小时直接突破、2%止损：质量特征分箱", "", "Q1–Q5边界只用IS计算，随后原样应用于OOS。", "",
                  "```csv", features.to_csv(index=False).strip(), "```", ""]
    if not validation.empty:
        core = validation[validation.cost_bps == 20]
        lines += ["## 冻结策略：滚动样本外与外部币种验证（20 bps）", "",
                  "```csv", core.to_csv(index=False).strip(), "```", ""]
    if not validation_stability.empty:
        lines += ["## 4小时事件簇Bootstrap与正窗口比例", "", "```csv",
                  validation_stability.to_csv(index=False).strip(), "```", ""]
    if not thresholds.empty:
        lines += ["## 各滚动窗口实际阈值", "", "```csv",
                  thresholds.to_csv(index=False).strip(), "```", ""]
    lines += ["## 判读原则", "", "- 至少保留100个独立事件后再比较时长组。",
              "- 优先看OOS的平均终值R、2R到达率、止损率和中位结果。",
              "- 若更长横盘只有绝对MFE上升、但MFE/R和净R没有改善，则不支持延长扫描门槛。"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=100)
    p.add_argument("--symbol-list", default="")
    p.add_argument("--history-days", type=int, default=730)
    p.add_argument("--min-history-days", type=int, default=365)
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
        if frames["1h"].empty or (frames["1h"].index.max()-frames["1h"].index.min()).days < args.min_history_days:
            print(f"  skip: less than {args.min_history_days} days history", flush=True)
            continue
        for ev in detect_events(frames["1h"], symbol, i):
            all_rows.extend(evaluate_event(ev, frames["15m"], args.fee_bps_roundtrip))
    events = pd.DataFrame(all_rows)
    if not events.empty:
        events["breakout_time"] = pd.to_datetime(events["breakout_time"], utc=True)
        cutoff = start + (end-start) * 0.70
        events["sample_period"] = np.where(events.breakout_time <= cutoff, "IS", "OOS")
    events.to_csv(output / "events.csv", index=False)
    duration = summarise(events, ["sample_period", "duration_h", "entry_route", "stop_method", "horizon_h"])
    stops = summarise(events, ["sample_period", "entry_route", "stop_method", "horizon_h"])
    features = feature_study(events)
    wf, validation, validation_stability, thresholds, chosen = walk_forward_validation(events)
    symbol_stability, quarter_stability = selected_stability(chosen)
    duration.to_csv(output / "duration_summary.csv", index=False)
    stops.to_csv(output / "stop_summary.csv", index=False)
    features.to_csv(output / "feature_summary.csv", index=False)
    wf.to_csv(output / "walk_forward_folds.csv", index=False)
    validation.to_csv(output / "policy_cost_validation.csv", index=False)
    validation_stability.to_csv(output / "cluster_validation.csv", index=False)
    thresholds.to_csv(output / "walk_forward_thresholds.csv", index=False)
    symbol_stability.to_csv(output / "policy_symbol_stability.csv", index=False)
    quarter_stability.to_csv(output / "policy_quarter_stability.csv", index=False)
    report_text = report(duration, features, validation, validation_stability,
                         thresholds, symbols, args)
    (output / "validation_report.md").write_text(report_text, encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(vars(args) | {"symbols_used": symbols}, indent=2), encoding="utf-8")
    print(f"Done: {len(events):,} evaluated paths -> {output}")
    print("\n===== VALIDATION REPORT =====\n")
    print(report_text)


if __name__ == "__main__":
    main()
