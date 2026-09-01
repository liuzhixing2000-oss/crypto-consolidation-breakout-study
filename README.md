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
python breakout_study.py --symbols 100 --history-days 730 --min-history-days 365
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
- `feature_summary.csv`：48小时直接突破基线的压缩、突破质量与4H环境分箱。
- `walk_forward_folds.csv`：180天训练、90天测试的逐窗口结果。
- `policy_cost_validation.csv`：八个冻结策略在6–30 bps成本下的汇总。
- `cluster_validation.csv`：按4小时市场事件簇计算的Bootstrap区间。
- `walk_forward_thresholds.csv`：每个窗口仅由过去数据确定的实际阈值。
- `policy_symbol_stability.csv`：币种贡献与稳定性。
- `policy_quarter_stability.csv`：季度稳定性。
- `validation_report.md`：自动生成的中文摘要。

第一轮研究的目标不是直接选出交易参数，而是判断36–72小时、3–5天等时长组是否在样本外仍有更好的R期望。

## Railway

仓库已包含 `railway.json` 和 `Procfile`。连接仓库后默认启动命令为：

```bash
python breakout_study.py --symbols 100 --history-days 730 --min-history-days 365 --output results --cache data
```

这是一次性研究任务，完成后正常退出，且不会循环执行或重复推送。

## 黄金/白银15分钟多周期试验

`precious_metals_multitimeframe.py` 专门检验黄金和白银：

- 纯15分钟横盘：5、8、12、20、32、48小时；
- 48小时高周期背景配合15分钟突破确认；
- 直接突破、连续两根15分钟确认、回踩、回踩后重启；
- 1/1.5/2 ATR与结构失效止损；
- 2/4/8/12/24小时持有期；
- 亚洲、伦敦、纽约时段拆分，以及2–20 bps成本压力测试。

免费Yahoo 15分钟期货数据通常只有约60天，因此这一轮明确标记为探索性试验；任何候选参数都必须在更长的独立15分钟数据上再次验证。
