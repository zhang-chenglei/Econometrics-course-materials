# 第7讲案例：研发投入会提高企业绩效吗？

本案例使用一份**完全生成的**截面企业数据，练习工具变量（IV）与两阶段最小二乘（2SLS）。它对应教材第8章 §8.5，课堂上的入学抽签案例负责讲识别逻辑，本案例负责把 OLS、第一阶段、2SLS、弱工具、过度识别和 DWH 检验完整跑一遍。

**这不是现实企业数据，也不能用于评价真实研发回报。**

## 数据与数据生成过程

`data/firm_iv_case.csv` 包含 500 家虚构企业：

| 变量 | 含义 |
|---|---|
| `firm_id` | 企业编号 |
| `ROE` | 净资产收益率（%） |
| `RD` | 研发投入（百万元） |
| `ResDensity` | 科研资源密度（工具变量 Z1） |
| `University` | 高校数量（工具变量 Z2） |
| `size` | 企业规模（对数） |
| `age` | 企业年龄 |
| `noise_iv` | 与 `RD` 无关的随机变量，仅用于任务2的弱工具对照 |

数据生成过程（`scripts/generate_data.py`）中：

- 不可观测的 `ability` 同时影响 `RD` 和 `ROE`，因此 `RD` 内生；
- `ResDensity` 与 `University` 在 DGP 中被设为外生工具；
- `RD` 对 `ROE` 的真实结构系数为 **1.2**；
- `ability` 只在生成时使用，不写入 CSV。

## 软件与运行

需要 Python 3.11 或以上。Stata 17 或以上可选；有 Stata 时可以对照两套结果。

```sh
python3 -m pip install -r scripts/requirements.txt
python3 run_case.py --mode teaching
```

如果 Stata 不在默认位置：

```sh
python3 run_case.py --mode teaching --stata "/Applications/StataNow/StataMP.app/Contents/MacOS/stata-mp"
```

只运行 Python：

```sh
python3 run_case.py --mode teaching --no-stata
```

重新生成参考结果：

```sh
python3 run_case.py --mode reference
```

Python 结果写到 `output/python/`，Stata 结果写到 `output/stata/`。`reference/` 保存参考结果，`reference/key_numbers.json` 保存本页引用的核心数字。

## 四个任务

1. **OLS 与 IV**：先跑 OLS，再用 `ResDensity` 作为 `RD` 的工具跑 2SLS。比较系数，并结合 DGP 解释偏误方向。
2. **第一阶段与弱工具**：报告 `ResDensity` 的第一阶段系数、偏 F 和偏 R²；用 `noise_iv` 作为假工具对照，观察 2SLS 的不稳定。
3. **双工具与过度识别**：加入 `University`，比较单工具与双工具；运行过度识别检验。
4. **DWH 与综合判断**：运行 Durbin-Wu-Hausman 检验，汇总 OLS、单工具 IV、双工具 IV。

## 参考结果（来自固定数据）

| 项目 | 数值 |
|---|---:|
| 样本量 | 500 |
| DGP 真实 `RD` 效应 | 1.2 |
| OLS `RD` 系数（SE） | 1.4224（0.0578） |
| IV `RD` 系数（SE） | 1.1490（0.1104） |
| 第一阶段偏 F | 205.94 |
| 第一阶段偏 R² | 0.2572 |
| 弱工具（`noise_iv`）第一阶段 F | 0.339 |
| 弱工具 IV `RD` 系数（SE） | −0.576（3.914） |
| 过度识别统计量（p） | 0.184（0.668） |
| DWH F（p） | 8.681（约 0.003） |

点估计在 Stata 与 Python 中一致；DWH 的 p 值会因使用 F 分布或 χ² 分布而略有差异。

## 解释边界

- 本数据知道 DGP，所以可以判断 OLS 向上偏误；真实研究中不能只凭“IV 比 OLS 小”断言谁正确。
- 第一阶段强不等于工具外生；“F>10”只是特定简单设定下的粗略预警线。
- 过度识别检验不拒绝，不能证明两个工具都外生；拒绝也不能指出具体哪个工具坏了。
- DWH 不拒绝不等于 `RD` 外生；DWH 拒绝也不能反过来证明工具有效。
- 工具变量的可信度最终来自相关性机制和排除限制的制度与理论论证，而不是某个 p 值。

## 文件结构

```text
lecture07_instrumental_variables/
├── data/firm_iv_case.csv
├── scripts/
│   ├── generate_data.py
│   ├── analysis_python.py
│   ├── analysis_stata.do
│   └── requirements.txt
├── reference/
├── output/               # 运行后生成
├── run_case.py
└── README.md
```
