# Crypto Consolidation Breakout Study

独立研究项目：验证高流动性永续合约横盘多久后，突破的风险调整表现开始改善。

## 研究口径

- 1H K线识别横盘，长度从20小时到14天。
- 突破必须以1H收盘价有效越过此前区间。
- 15m K线评估突破后的逐路径结果。
- 结果以R倍数衡量，避免长周期因为绝对波幅更大而被高估。
- 同时测试三种结构止损：半区间、完整区间、ATR止损。
- 输出1R/2R/3R到达率、止损率、MFE、MAE、净期望及样本数。

## 快速运行

```bash
python -m pip install -r requirements.txt
python breakout_study.py --symbols 50 --history-days 730
```

先用少量标的验证：

```bash
python breakout_study.py --symbol-list BTCUSDT,ETHUSDT,SOLUSDT --history-days 365
```

结果写入 `results/`。程序只下载公开行情并回测，不发Telegram、不交易，也不会接触其他项目。

## 关键输出

- `events.csv`：每个突破事件和各止损方案的逐笔路径。
- `duration_summary.csv`：按横盘时长分组的风险调整结果。
- `stop_summary.csv`：不同止损逻辑的比较。
- `validation_report.md`：自动生成的中文摘要。

第一轮研究的目标不是直接选出交易参数，而是判断36–72小时、3–5天等时长组是否在样本外仍有更好的R期望。

## Railway

仓库已包含 `railway.json` 和 `Procfile`。连接仓库后默认启动命令为：

```bash
python breakout_study.py --symbols 50 --history-days 730 --output results --cache data
```

这是一次性研究任务，完成后正常退出，且不会循环执行或重复推送。
