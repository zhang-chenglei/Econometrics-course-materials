"""第15章：把第12—14章输出整理为可发表式表图和教学报告。

本脚本不重新估计模型，而是读取前面章节已保存的机器可读结果，完成
“同一组数字、一次生成、多处引用”的可复现写作流程。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "共享代码"))

from common import (  # noqa: E402
    BLUE,
    GOLD,
    INK,
    export_table as save_table,
    load_data,
    save_figure,
    setup_plot_style,
    significance_stars,
)


CH14_TABLES = ROOT / "第14章" / "表格"
TABLE_DIR = ROOT / "第15章" / "表格"
FIGURE_DIR = ROOT / "第15章" / "图形"
NOTE_DIR = ROOT / "第15章" / "说明"
for directory in (TABLE_DIR, FIGURE_DIR, NOTE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GREEN = "#36856B"
RED = "#B64A4A"
setup_plot_style()


def read_result(name: str) -> pd.DataFrame:
    return pd.read_csv(CH14_TABLES / name, encoding="utf-8-sig")


def fmt_effect(value: float, se: float, p: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}{significance_stars(p)}\n({se:.{digits}f})"


twfe = read_result("表14-1_TWFE教学基准.csv")
staggered = read_result("表14-2_交错DID总体ATT.csv")
event = read_result("表14-3_动态事件研究.csv")
pretrend = read_result("表14-5_平行趋势联合检验.csv")
robust = read_result("表14-6_稳健性检验.csv")
hetero = read_result("表14-8_异质性直接差异检验.csv")
mechanism_twfe = read_result("表14-10_机制分析.csv")
mechanism = read_result("表14-10b_机制交错DID.csv")
data = load_data()


# ── 表15-1：TWFE教学基准（与教材中的四列表一致） ──────────────────────────────
def pick_twfe(outcome: str, controls: bool) -> pd.Series:
    token = "加入企业控制变量" if controls else "企业与年份固定效应"
    return twfe[(twfe["outcome"] == outcome) & twfe["label"].str.contains(token)].iloc[0]


rd0, rd1 = pick_twfe("rd_intensity_w", False), pick_twfe("rd_intensity_w", True)
pat0, pat1 = pick_twfe("ln_patent_invention", False), pick_twfe("ln_patent_invention", True)

table_14_1 = pd.DataFrame(
    {
        "项目": [
            "试验区处理",
            "企业固定效应",
            "年份固定效应",
            "企业特征控制",
            "城市聚类标准误",
            "观测数",
            "企业数",
        ],
        "(1) 研发投入强度": [
            fmt_effect(rd0.coefficient, rd0.std_error, rd0.p_value),
            "是",
            "是",
            "否",
            "是",
            f"{int(rd0.observations):,}",
            f"{int(rd0.firms):,}",
        ],
        "(2) 研发投入强度": [
            fmt_effect(rd1.coefficient, rd1.std_error, rd1.p_value),
            "是",
            "是",
            "是",
            "是",
            f"{int(rd1.observations):,}",
            f"{int(rd1.firms):,}",
        ],
        "(3) 发明专利申请": [
            fmt_effect(pat0.coefficient, pat0.std_error, pat0.p_value, 3),
            "是",
            "是",
            "否",
            "是",
            f"{int(pat0.observations):,}",
            f"{int(pat0.firms):,}",
        ],
        "(4) 发明专利申请": [
            fmt_effect(pat1.coefficient, pat1.std_error, pat1.p_value, 3),
            "是",
            "是",
            "是",
            "是",
            f"{int(pat1.observations):,}",
            f"{int(pat1.firms):,}",
        ],
    }
)
save_table(table_14_1, TABLE_DIR / "表15-1_TWFE教学基准最终版")


# ── 表15-2：正式主结果（从未处理组） ──────────────────────────────────────────
main_att = staggered[staggered["control_mode"] == "never"].copy()
outcome_names = {
    "rd_intensity_w": "研发投入强度",
    "ln_patent_invention": "发明专利申请（ln(1+件数)）",
}
main_att["结果变量"] = main_att["outcome"].map(outcome_names)
main_att["对照组"] = "从未处理企业"
main_att["总体ATT"] = main_att["att"].map(lambda x: f"{x:.4f}")
main_att["标准误"] = main_att["std_error"].map(lambda x: f"{x:.4f}")
main_att["95%置信区间"] = main_att.apply(
    lambda r: f"[{r.ci_low:.4f}, {r.ci_high:.4f}]", axis=1
)
main_att["联合平行趋势p值"] = main_att["pretrend_p_value"].map(lambda x: f"{x:.3f}")
table_14_2 = main_att[
    [
        "结果变量",
        "对照组",
        "总体ATT",
        "标准误",
        "95%置信区间",
        "联合平行趋势p值",
        "cities",
        "firms",
        "observations",
    ]
].rename(columns={"cities": "样本城市数", "firms": "企业数", "observations": "观测数"})
save_table(table_14_2, TABLE_DIR / "表15-2_交错DID主结果最终版")


# ── 表15-3：关键稳健性结果 ─────────────────────────────────────────────────────
selected_tokens = [
    "主设定",
    "加入企业控制变量",
    "批复当年作为处理起点",
    "排除2020年",
    "排除北京上海深圳",
    "政策前特征ATT加权",
]
short_robust = robust[
    robust["label"].map(lambda value: any(token in value for token in selected_tokens))
].copy()
short_robust["结果变量"] = short_robust["outcome"].map(
    {
        "rd_intensity_w": "研发投入强度",
        "ln_patent_invention": "发明专利申请",
    }
)
short_robust["规格"] = short_robust["label"].str.split("：").str[-1]
short_robust["估计值"] = short_robust["coefficient"].map(lambda x: f"{x:.4f}")
short_robust["标准误"] = short_robust["std_error"].map(lambda x: f"{x:.4f}")
short_robust["p值"] = short_robust["p_value"].map(lambda x: "<0.001" if x < 0.001 else f"{x:.3f}")
short_robust["95%置信区间"] = short_robust.apply(
    lambda r: f"[{r.ci_low:.4f}, {r.ci_high:.4f}]", axis=1
)
table_14_3 = short_robust[
    ["结果变量", "规格", "估计值", "标准误", "p值", "95%置信区间", "observations"]
].rename(columns={"observations": "观测数"})
save_table(table_14_3, TABLE_DIR / "表15-3_关键稳健性结果")


# ── 表15-4、15-5：异质性直接差异与机制证据 ────────────────────────────────────
table_14_4 = hetero.copy()
table_14_4["结果变量"] = table_14_4["outcome"].map(outcome_names)
table_14_4["组间差异"] = table_14_4["difference"].map(lambda x: f"{x:.4f}")
table_14_4["差异标准误"] = table_14_4["difference_se"].map(lambda x: f"{x:.4f}")
table_14_4["差异p值"] = table_14_4["difference_p_value"].map(
    lambda x: "<0.001" if x < 0.001 else f"{x:.3f}"
)
table_14_4 = table_14_4[
    [
        "dimension",
        "结果变量",
        "group0_label",
        "group1_label",
        "组间差异",
        "差异标准误",
        "差异p值",
    ]
].rename(
    columns={
        "dimension": "异质性维度",
        "group0_label": "基准组",
        "group1_label": "比较组",
    }
)
save_table(table_14_4, TABLE_DIR / "表15-4_异质性直接差异检验")

table_14_5 = mechanism.copy()
table_14_5["估计值"] = table_14_5["att"].map(lambda x: f"{x:.4f}")
table_14_5["标准误"] = table_14_5["std_error"].map(lambda x: f"{x:.4f}")
table_14_5["p值"] = table_14_5["p_value"].map(
    lambda x: "<0.001" if x < 0.001 else f"{x:.3f}"
)
table_14_5["标准化效应"] = table_14_5["standardized_effect"].map(lambda x: f"{x:.3f}")
table_14_5 = table_14_5[["label", "估计值", "标准误", "p值", "标准化效应"]].rename(
    columns={"label": "机制相关结果变量"}
)
save_table(table_14_5, TABLE_DIR / "表15-5_机制相关证据")


# ── 表15-6：全文数字一致性检查 ─────────────────────────────────────────────────
pre = data[
    (data["did_treated"] == 1)
    & (data["year"] < data["policy_first_full_year"])
]
rd_pre_mean = pre["rd_intensity_w"].mean()
patent_pct = (np.exp(main_att.loc[main_att["outcome"] == "ln_patent_invention", "att"].iloc[0]) - 1) * 100
rd_row = main_att[main_att["outcome"] == "rd_intensity_w"].iloc[0]
pat_row = main_att[main_att["outcome"] == "ln_patent_invention"].iloc[0]
audit = pd.DataFrame(
    [
        {
            "指标": "样本期",
            "正文统一值": f"{int(data.year.min())}—{int(data.year.max())}年",
            "来源": "最终分析数据",
        },
        {"指标": "观测数", "正文统一值": f"{len(data):,}", "来源": "最终分析数据"},
        {
            "指标": "企业数",
            "正文统一值": f"{data.firm_id.nunique():,}",
            "来源": "最终分析数据",
        },
        {
            "指标": "研发投入强度总体ATT",
            "正文统一值": f"{rd_row.att:.4f} [{rd_row.ci_low:.4f}, {rd_row.ci_high:.4f}]",
            "来源": "交错DID，从未处理组",
        },
        {
            "指标": "研发投入强度相对处理组政策前均值",
            "正文统一值": f"{rd_row.att / rd_pre_mean * 100:.1f}%",
            "来源": "总体ATT÷处理组政策前均值",
        },
        {
            "指标": "发明专利申请总体ATT",
            "正文统一值": f"{pat_row.att:.4f} [{pat_row.ci_low:.4f}, {pat_row.ci_high:.4f}]",
            "来源": "交错DID，从未处理组",
        },
        {
            "指标": "专利变换值的近似百分比换算",
            "正文统一值": f"{patent_pct:.1f}%",
            "来源": "100×(exp(ATT)-1)，仅作教学解释",
        },
    ]
)
save_table(audit, TABLE_DIR / "表15-6_全文关键数字一致性检查")


# ── 图15-1：两种估计量的主结果 ─────────────────────────────────────────────────
compare = pd.DataFrame(
    [
        {
            "outcome": "研发投入强度",
            "method": "TWFE教学基准",
            "estimate": rd0.coefficient,
            "low": rd0.ci_low,
            "high": rd0.ci_high,
        },
        {
            "outcome": "研发投入强度",
            "method": "交错DID主结果",
            "estimate": rd_row.att,
            "low": rd_row.ci_low,
            "high": rd_row.ci_high,
        },
        {
            "outcome": "发明专利申请",
            "method": "TWFE教学基准",
            "estimate": pat0.coefficient,
            "low": pat0.ci_low,
            "high": pat0.ci_high,
        },
        {
            "outcome": "发明专利申请",
            "method": "交错DID主结果",
            "estimate": pat_row.att,
            "low": pat_row.ci_low,
            "high": pat_row.ci_high,
        },
    ]
)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for axis, outcome in zip(axes, ["研发投入强度", "发明专利申请"]):
    part = compare[compare["outcome"] == outcome].reset_index(drop=True)
    y = np.arange(len(part))
    axis.errorbar(
        part["estimate"],
        y,
        xerr=[part["estimate"] - part["low"], part["high"] - part["estimate"]],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
    )
    axis.axvline(0, color=INK, linestyle="--", linewidth=1)
    axis.set_yticks(y, part["method"])
    axis.set_title(outcome)
    axis.set_xlabel("点估计与95%置信区间")
    axis.grid(axis="x", alpha=0.25)
fig.suptitle("人工智能试验区与企业创新：两种估计量的主结果", fontweight="bold")
fig.text(
    0.5,
    -0.01,
    "注：TWFE仅作教学基准；交错DID使用从未处理企业作对照并按处理组规模聚合。",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
save_figure(fig, FIGURE_DIR / "图15-1_主结果比较")


# ── 图15-2：最终版事件研究图 ───────────────────────────────────────────────────
event_main = event[event["control_mode"] == "never"].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for axis, (outcome, title, color) in zip(
    axes,
    [
        ("rd_intensity_w", "研发投入强度", BLUE),
        ("ln_patent_invention", "发明专利申请（ln(1+件数)）", GOLD),
    ],
):
    part = event_main[event_main["outcome"] == outcome].sort_values("event_time")
    axis.errorbar(
        part["event_time"],
        part["att"],
        yerr=1.96 * part["std_error"],
        fmt="o-",
        color=color,
        ecolor=color,
        capsize=3,
        linewidth=1.5,
    )
    axis.axhline(0, color=INK, linewidth=1, linestyle="--")
    axis.axvline(-0.5, color=RED, linewidth=1, linestyle=":")
    axis.set_title(title)
    axis.set_xlabel("相对处理年份（-1为基准期）")
    axis.set_ylabel("组别—时期ATT")
    axis.grid(alpha=0.22)
fig.suptitle("人工智能试验区对上市公司创新的动态影响", fontweight="bold")
fig.text(
    0.5,
    -0.01,
    "注：点为事件时间聚合ATT，线段为95%置信区间；对照组为从未处理企业，标准误按城市聚类。",
    ha="center",
    fontsize=9,
)
fig.tight_layout()
save_figure(fig, FIGURE_DIR / "图15-2_事件研究最终版")


# ── 图15-3：稳健性结果摘要 ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
for axis, (outcome, title, color) in zip(
    axes,
    [
        ("rd_intensity_w", "研发投入强度", BLUE),
        ("ln_patent_invention", "发明专利申请", GOLD),
    ],
):
    part = short_robust[short_robust["outcome"] == outcome].reset_index(drop=True)
    y = np.arange(len(part))
    axis.errorbar(
        part["coefficient"],
        y,
        xerr=[part["coefficient"] - part["ci_low"], part["ci_high"] - part["coefficient"]],
        fmt="o",
        color=color,
        ecolor=color,
        capsize=3,
    )
    axis.axvline(0, color=INK, linestyle="--", linewidth=1)
    axis.set_yticks(y, part["规格"])
    axis.invert_yaxis()
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.22)
fig.suptitle("关键稳健性检验的系数与95%置信区间", fontweight="bold")
fig.tight_layout()
save_figure(fig, FIGURE_DIR / "图15-3_稳健性结果摘要")


# ── 图15-4：机制证据（标准化） ─────────────────────────────────────────────────
fig, axis = plt.subplots(figsize=(8.5, 4.5))
plot_mech = mechanism.sort_values("standardized_effect").reset_index(drop=True)
y = np.arange(len(plot_mech))
colors = [GREEN if p < 0.05 else "#8A939B" for p in plot_mech["p_value"]]
for i, row in plot_mech.iterrows():
    axis.errorbar(
        row["standardized_effect"],
        i,
        xerr=1.96 * row["standardized_se"],
        fmt="o",
        color=colors[i],
        ecolor=colors[i],
        capsize=3,
    )
axis.axvline(0, color=INK, linestyle="--", linewidth=1)
axis.set_yticks(y, plot_mech["label"])
axis.set_xlabel("政策效应（以结果变量标准差计）")
axis.set_title("机制相关结果：标准化效应与95%置信区间", fontweight="bold")
axis.grid(axis="x", alpha=0.22)
fig.tight_layout()
save_figure(fig, FIGURE_DIR / "图15-4_机制证据摘要")


# ── 教学研究报告：所有数字均从输出表自动插入 ──────────────────────────────────
rd_pre_p = pretrend[
    (pretrend["outcome"] == "rd_intensity_w") & (pretrend["control_mode"] == "never")
]["pretrend_p_value"].iloc[0]
pat_pre_p = pretrend[
    (pretrend["outcome"] == "ln_patent_invention") & (pretrend["control_mode"] == "never")
]["pretrend_p_value"].iloc[0]
not_yet_pat_p = pretrend[
    (pretrend["outcome"] == "ln_patent_invention") & (pretrend["control_mode"] == "not_yet")
]["pretrend_p_value"].iloc[0]

report = f"""# 人工智能试验区与上市公司创新：半合成教学数据的完整报告示范

> **材料性质声明**：本报告使用的是“真实政策背景＋经清理与教学化处理的半合成企业面板”。政策事实可用于学习研究设计，所有回归数值只用于训练数据处理、交错DID估计和规范写作，不能作为对真实政策效果的经验判断，也不能用于政策建议。

## 摘要

本文以国家新一代人工智能创新发展试验区的分批批复为情境，演示如何研究区域人工智能政策与上市公司创新之间的关系。教学样本覆盖{int(data.year.min())}—{int(data.year.max())}年，共{data.firm_id.nunique():,}家企业、{len(data):,}个企业—年份观测。以从未处理企业为对照的交错DID结果显示，研发投入强度的总体ATT为{rd_row.att:.4f}（95%置信区间{rd_row.ci_low:.4f}—{rd_row.ci_high:.4f}），发明专利申请变换值的总体ATT为{pat_row.att:.4f}（95%置信区间{pat_row.ci_low:.4f}—{pat_row.ci_high:.4f}）。两项结果的政策前动态联合检验p值分别为{rd_pre_p:.3f}和{pat_pre_p:.3f}。稳健性、异质性和机制结果共同构成完整的教学证据链，但这些模式是为教学演示而保留的半合成特征，不具有现实政策含义。

**关键词**：人工智能；创新发展试验区；企业创新；交错双重差分；可复现研究

## 1. 研究问题与识别

研究问题是：试验区获批后，当地上市公司的创新投入和创新产出是否发生变化？2017年的全国规划并不构成城市间差异；2019—2021年具体试验区的分批批复才形成交错处理时点。主处理变量按企业政策前固定城市匹配，并从批复后的第一个完整年度开始取1。正式主估计使用从未处理企业作为对照，并聚合组别—时期ATT；TWFE仅用于展示传统教学基准。

这一设计依赖条件平行趋势、无提前反应、处理定义稳定和不存在严重跨城市溢出等假设。由于试验区不是随机获批，即使动态检验没有拒绝平行趋势，也不能将识别假设说成“已经证明”。

## 2. 数据与变量

样本为{int(data.year.min())}—{int(data.year.max())}年企业面板，包含{data.firm_id.nunique():,}家企业、{data.analysis_city_id.nunique():,}个城市和{len(data):,}个观测。结果变量为研发投入强度和发明专利申请数的 `ln(1+x)` 变换。企业层面控制变量包括规模、资产负债率、盈利能力、成长性、固定资产比重、现金流和企业年龄。完整变量、缺失率、样本筛选与平衡结果见第13章输出。

## 3. 主要结果

传统TWFE估计中，试验区处理与研发投入强度增加{rd0.coefficient:.4f}相关，城市聚类标准误为{rd0.std_error:.4f}；发明专利申请变换值的估计为{pat0.coefficient:.4f}，标准误为{pat0.std_error:.4f}。加入企业特征后，两项估计分别为{rd1.coefficient:.4f}和{pat1.coefficient:.4f}。

以从未处理企业为对照的交错DID主结果显示，研发投入强度总体ATT为{rd_row.att:.4f}，95%置信区间为[{rd_row.ci_low:.4f}, {rd_row.ci_high:.4f}]。该点估计约相当于处理组政策前研发投入强度均值的{rd_row.att / rd_pre_mean * 100:.1f}%。发明专利申请变换值的总体ATT为{pat_row.att:.4f}，95%置信区间为[{pat_row.ci_low:.4f}, {pat_row.ci_high:.4f}]；若仅按对数变换作近似换算，对应约{patent_pct:.1f}%的变化。后一个换算依赖函数形式，不应当替代对原始件数分布的展示。

## 4. 动态效应与稳健性

图15-2以处理前一年为基准，展示政策前后事件时间聚合ATT。研发投入和发明专利结果的政策前系数联合检验p值分别为{rd_pre_p:.3f}和{pat_pre_p:.3f}，在本教学样本中没有拒绝政策前动态共同为零。改用尚未处理企业作对照后，研发结果仍通过联合检验，但专利结果的p值为{not_yet_pat_p:.3f}。这说明研究结论的可信度不仅取决于点估计，也取决于对照组口径与识别诊断。

主要系数在加入企业特征、改变批复年度口径、排除2020年、排除北京上海深圳、控制教学用同期政策代理、剔除同省近邻代理以及政策前特征加权等规格下保持正向。同期政策、办公地址和邻近城市变量在现有数据中是明确标注的教学代理，不能冒充真实测量。因此，这些结果只能演示“怎样把识别威胁映射为检验”，不能声称已经排除现实中的同期政策和空间溢出。

## 5. 异质性与机制

异质性结论以直接交互检验为准，而不是比较“一组显著、另一组不显著”。研发投入强度方面，所有制、政策前规模和行业技术属性的组间差异达到5%显著水平，政策前创新基础差异不显著；专利结果方面，政策前创新基础和行业技术属性的直接差异显著，所有制与规模差异不显著。完整分组点估计与直接差异见表15-4。

机制相关估计显示，政府补助与研发人员结果显著为正，KZ和SA融资约束指标没有显著变化。规范的表述是：教学数据中的证据与“补助支持”和“研发人员增加”渠道一致，而没有观察到融资约束指标的清晰响应。仅凭这些两步结果不能宣称已经识别完整中介机制。

## 6. 结论、边界与可复现性

这套半合成教学结果形成了一个有意设计的完整练习：主要结果为正，以从未处理组为对照的政策前联合检验未拒绝系数共同为零，多数稳健性规格方向一致，异质性和机制检验则保留显著与不显著并存的情形。学生应学习如何根据输出写结论、如何报告不确定性、如何处理相互冲突的诊断，而不应记忆这些数值。

现实研究还需要重新核验数据许可、企业地址历史、同期政策、空间距离、政策执行强度和专利质量，并使用未经结果导向调整的原始数据重新估计。任何对真实人工智能试验区政策效果的判断，都必须来自真实数据、预先说明的处理规则和可以审计的复现材料。

## 表图索引

- 表15-1：TWFE教学基准最终版
- 表15-2：交错DID主结果最终版
- 表15-3：关键稳健性结果
- 表15-4：异质性直接差异检验
- 表15-5：机制相关证据
- 表15-6：全文关键数字一致性检查
- 图15-1：主结果比较
- 图15-2：事件研究最终版
- 图15-3：稳健性结果摘要
- 图15-4：机制证据摘要
"""
(NOTE_DIR / "第15章完整研究报告_半合成教学示范.md").write_text(report, encoding="utf-8")

print(ROOT / "第15章")
