# 第六讲：温度冲击与经济增长

这是《计量经济学：理论与实践》第六讲的独立案例包。请保留整个目录；可以单独复制本目录到下载文件夹或桌面，无需保留课程仓库的外层目录。

课程页：[本讲讲义](https://zhang-chenglei.github.io/Econometrics-course/course/ch04.html)｜[案例实操](https://zhang-chenglei.github.io/Econometrics-course/course/lesson6_lab.html)｜[代码与结果分析](https://zhang-chenglei.github.io/Econometrics-course/course/lesson6_code.html)。

## 两套入口

- 教学入口用 Stata `reghdfe`，对照固定效应、标准误、异质性、趋势和滞后；Python 负责运行组织、数值汇总和绘图。
- 原包入口用作者的 Stata `cgmreg`，运行论文七张表、两幅图。只在临时副本中更换表3/6的旧导出语法，不改样本、估计、变量与检验。
- 这里没有仅用 Python 完成估计的替代版本。有 Python 但没有 Stata 时，可以阅读参考结果，不能声称已完成估计复现。

## 准备与运行

需要 Python 3.11 以上版本与有许可的 Stata。本次核验环境为 macOS、Stata/MP 19、`reghdfe` 6.13.1、`ftools` 2.50.0、`mmerge` 2.5.0、`cgmreg` 3.0.0、`outreg` 4.32、`outreg2` 2.3.2、`frmttable` 1.32。Python 绘图包版本固定在 `scripts/requirements.txt`。

进入本目录：

```sh
python3 -m pip install -r scripts/requirements.txt
python3 run_case.py --mode teaching --install
```

Windows 中可用 `python` 替换 `python3`。程序首先从 PATH 找 Stata，在 macOS 上也尝试默认应用目录；其他安装位置可用：

```sh
python3 run_case.py --stata "Stata可执行文件完整路径" --mode teaching --install
```

`--install` 仅在命令缺失时尝试安装，不覆盖已有版本。依赖安装需要网络；`cgmreg` 从作者官网获取，其余命令从 SSC 安装。已有依赖时可省略 `--install`。

原包或全部流程：

```sh
python3 run_case.py --mode original --install
python3 run_case.py --mode all --install
```

在 macOS 的本机环境已从独立下载目录完整验证；其他 Stata 版本或系统需要核对运行结果。本程序自动检查教学记录与参考值，但不会为消除差异而改样本或估计规格。

## 输入、输出与参考

| 目录 | 内容 |
|---|---|
| `original/` | 作者原始数据、代码与说明；运行中保持不变 |
| `scripts/` | 教学构建、绘图、依赖与原包入口 |
| `reference/` | 已核验教学结果，`original/` 子目录保存七张表、两幅图及19项核心对照 |
| `output/` | 自己新运行的结果；教学数据/图分别在 `data/`、`figures/`，原包输出在 `original/日期_时间/` |

教学运行检查 `output/teaching_check.json`，原包运行检查 `output/original/日期_时间/运行核验.json`。Stata 批处理退出码为0不足以说明 do-file 完成，因此入口还检查明确的完成标记、文件和输入校验值。报错时先看对应日志末尾。

所有生成文件写入本目录的 `output/` 或保留的临时运行副本，不写入课程仓库的外层目录。日志的运行记录保留协方差矩阵修正等提示；批处理许可证抬头留在本地 `output/stata_batch.log`，不要公开上传这份本地日志。

中文图形需要本机有中文字体：macOS 使用 Heiti SC；Windows 可用微软雅黑或黑体；Linux 可配置 Noto Sans CJK SC。缺字体导致的绘图失败不表示 Stata 估计失败，检查日志后配置字体再运行。

## 核对口径

教学完整面板6,802行，基准估计N=4,909；原包基准N=4,924。穷国总效应两套流程均约−1.394，但其标准误为教学0.389、原包0.408；原包无降水十阶系数和的p约0.055，教学为0.046587。单例处理、协方差修正与推断口径有差异，不要强行对齐星号。

已验证原包九个步骤全部完成，表2/3的19项核心数值与包内March2012论文稿按三位小数/整数吻合。未逐格核验所有刊发表格。原包导出表可能包含现代Stata显示的省略固定效应零占位行；它们不是新增的实质估计结果。

## 数据来源与署名

Dell, Melissa, Benjamin F. Jones, and Benjamin A. Olken. 2012. “Temperature Shocks and Economic Growth: Evidence from the Last Half Century.” *American Economic Journal: Macroeconomics*, 4(3): 66–95. DOI: [10.1257/mac.4.3.66](https://www.aeaweb.org/articles?id=10.1257/mac.4.3.66)。

原包：[openICPSR项目114251，V1](https://www.openicpsr.org/openicpsr/project/114251/version/V1/view)，`AEJ-Mac2010-0092-reproduction`。原始数据按记录中的CC-BY 4.0署名使用，原始代码许可见数据记录；本包保留作者的`original/readme.pdf`。教学改写与运行组织：张呈磊课程团队，2026-10-11。

这是真实公开研究数据，不是半合成模拟数据。短期天气回归不能不加条件地外推为长期气候政策预测。
