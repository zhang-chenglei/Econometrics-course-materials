#!/usr/bin/env python3
r"""从真实复现数据与回归结果生成第六讲的全部教学图形。

数据源：
  data/analysis_panel.csv   分析样本（由 scripts/build_panel.do 导出）
  data/country_summary.csv  国家层面汇总
  data/model_summary.json   回归结果（数字权威）

运行：python3 scripts/generate_figures.py [图号 ...]
      不带参数画出全部；带数字只重画指定图，例如 `python3 scripts/generate_figures.py 5`

数据图追溯到上述文件；图10为明确标注的概念示意，数值不是回归估计。
字体纪律：中文标签只用 Heiti SC 覆盖得到的字形，数学符号一律走 mathtext，
          存图时断言“缺字形”警告为 0（见 [[matplotlib-heiti-unicode-subscript-loss]]）。
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "output" / "data"
FIG_DIR = ROOT / "output" / "figures"
PDF_DIR = FIG_DIR / "pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

NAVY = "#20364B"
SLATE = "#5F7D8C"
BRICK = "#A34A3A"
PAPER = "#F8F8F6"
MIST = "#E7EBEE"
BRICKPALE = "#F2E4E0"
SLATEPALE = "#DCE3E7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Heiti SC", "PingFang SC", "STHeiti", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "axes.edgecolor": SLATE,
        "axes.labelcolor": NAVY,
        "xtick.color": SLATE,
        "ytick.color": SLATE,
        "text.color": NAVY,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PANEL = pd.read_csv(DATA / "analysis_panel.csv")
COUNTRY = pd.read_csv(DATA / "country_summary.csv")
SUMMARY = json.loads((DATA / "model_summary.json").read_text(encoding="utf-8"))

GLYPH_WARNINGS: list[str] = []


def save(fig: plt.Figure, stem: str) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor=PAPER)
        fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor=PAPER)
    for w in caught:
        if "missing from font" in str(w.message).lower() or "Glyph" in str(w.message):
            GLYPH_WARNINGS.append(f"{stem}: {w.message}")
    plt.close(fig)


def clean(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=SLATEPALE, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def lab(step: dict, dp: int = 3) -> str:
    p = step["p"]
    s = "***" if p <= 0.01 else "**" if p <= 0.05 else "*" if p <= 0.10 else ""
    return f"{step['b']:.{dp}f}{s}"


def ci(step: dict, z: float = 1.96) -> tuple[float, float]:
    return step["b"] - z * step["se"], step["b"] + z * step["se"]


# ---------------------------------------------------------------------------
def fig01_cross_section_vs_within() -> None:
    """国家之间比较 vs 同一国家内部比较。"""
    panel = PANEL.dropna(subset=["g", "wtem"])
    cm = panel.groupby("cc_num")[["wtem", "g", "poor"]].mean()
    cm["poor"] = (cm["poor"] > 0.5).astype(int)

    dem = panel.copy()
    dem["wtem_c"] = dem["wtem"] - dem.groupby("cc_num")["wtem"].transform("mean")
    dem["g_c"] = dem["g"] - dem.groupby("cc_num")["g"].transform("mean")

    fig, axes = plt.subplots(1, 2, figsize=(13, 2.5))

    for grp, colour, name in [(0, SLATE, "富国"), (1, BRICK, "穷国")]:
        sub = cm[cm["poor"] == grp]
        axes[0].scatter(sub["wtem"], sub["g"], s=26, color=colour, alpha=0.75, label=name)
    b, a = np.polyfit(cm["wtem"], cm["g"], 1)
    xs = np.linspace(cm["wtem"].min(), cm["wtem"].max(), 50)
    axes[0].plot(xs, a + b * xs, color=NAVY, linewidth=2)
    axes[0].set_xlabel("国家平均气温（摄氏度）")
    axes[0].set_ylabel("国家平均增长率（百分点）")
    axes[0].set_title(f"① 国家之间比：斜率 {b:+.2f}")
    axes[0].legend(loc="upper right", fontsize=10)
    clean(axes[0])

    axes[1].scatter(dem["wtem_c"], dem["g_c"], s=8, color=SLATE, alpha=0.35)
    b2, a2 = np.polyfit(dem["wtem_c"], dem["g_c"], 1)
    xs2 = np.linspace(dem["wtem_c"].min(), dem["wtem_c"].max(), 50)
    axes[1].plot(xs2, a2 + b2 * xs2, color=BRICK, linewidth=2)
    axes[1].set_xlabel("气温减去本国均值（摄氏度）")
    axes[1].set_ylabel("增长率减去本国均值（百分点）")
    axes[1].set_title(f"② 同一国家内部比：斜率 {b2:+.2f}")
    clean(axes[1])

    fig.suptitle("比较对象不同，斜率就不同", fontsize=15, fontweight="bold", color=NAVY)
    fig.tight_layout()
    save(fig, "fig01_cross_section_vs_within")


def fig02_income_temperature() -> None:
    """横截面事实：热的国家往往更穷。"""
    cm = COUNTRY.dropna(subset=["mean_wtem", "lnrgdpl_t0"])
    fig, ax = plt.subplots(figsize=(9.6, 2.7))
    for grp, colour, name in [(0, SLATE, "富国"), (1, BRICK, "穷国")]:
        sub = cm[cm["poor"] == grp]
        ax.scatter(sub["mean_wtem"], sub["lnrgdpl_t0"], s=30, color=colour, alpha=0.8, label=name)
    b, a = np.polyfit(cm["mean_wtem"], cm["lnrgdpl_t0"], 1)
    xs = np.linspace(cm["mean_wtem"].min(), cm["mean_wtem"].max(), 50)
    ax.plot(xs, a + b * xs, color=NAVY, linewidth=2)
    ax.set_xlabel("1950—2003 年平均气温（摄氏度）")
    ax.set_ylabel(r"初始人均 GDP（对数）")
    ax.set_title("横截面事实：热的国家往往更穷")
    ax.text(
        0.97, 0.95, f"斜率 {b:+.3f}", transform=ax.transAxes, ha="right", va="top",
        fontsize=11, color=BRICK, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=10)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig02_income_temperature")


def fig03_ladder_coefficients() -> None:
    """固定效应阶梯：温度系数随固定效应变化。"""
    ladder = SUMMARY["ladder"]
    labels = [f"{name}（N={step['n']:,}）" for name, step in zip(
        ["无固定效应", "+国家", "+年份", "+地区×年份", "+穷国×年份"], ladder)]
    ys = np.arange(len(ladder))[::-1]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for y, step in zip(ys, ladder):
        lo, hi = ci(step)
        colour = BRICK if step["p"] <= 0.05 else SLATE
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.4, solid_capstyle="round")
        ax.scatter([step["b"]], [y], s=70, color=colour, zorder=5)
        ax.text(hi + 0.05, y, lab(step), va="center", fontsize=11, color=colour, fontweight="bold")
    ax.axvline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("气温对年增长率的估计系数（百分点 / 摄氏度），95% 区间")
    ax.set_title("每加一组固定效应，识别的比较来源就变一次")
    ax.set_xlim(-1.25, 0.6)
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig03_ladder_coefficients")


def fig04_region_year_shocks() -> None:
    """地区×年份冲击：不同地区有自己的年度波动。"""
    panel = PANEL.dropna(subset=["g", "wtem"]).copy()
    panel["wtem_dev"] = panel["wtem"] - panel.groupby("cc_num")["wtem"].transform("mean")
    region_year = (
        panel.groupby(["region_lab", "year"])["wtem_dev"].mean().unstack(level=0)
    )
    region_year = region_year.loc[1950:2003]

    names = {
        "_MENA": "中东与北非", "_SSAF": "撒哈拉以南非洲", "_LAC": "拉美与加勒比",
        "_WEOFF": "西欧及其衍生国", "_EECA": "东欧与中亚", "_SEAS": "亚洲与太平洋",
    }
    fig, ax = plt.subplots(figsize=(12.5, 2.5))
    for col in region_year.columns:
        ax.plot(region_year.index, region_year[col], linewidth=1.3, alpha=0.85,
                label=names.get(col, col))
    ax.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("年份")
    ax.set_ylabel("本国气温离本国均值的偏离（摄氏度）")
    ax.set_title("各地区的年度温度波动不同")
    low = float(np.nanmin(region_year.to_numpy()))
    high = float(np.nanmax(region_year.to_numpy()))
    ax.set_ylim(low - 0.15, high + 0.75)
    ax.legend(ncol=3, fontsize=10, loc="upper center", columnspacing=1.0, handlelength=1.4)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig04_region_year_shocks")


def fig05_se_comparison() -> None:
    """同一个系数，四种标准误。"""
    se = SUMMARY["std_errors"]
    order = ["se_ols", "se_robust", "se_cl_country", "se_cl_twoway"]
    labels = ["常规标准误", "异方差稳健", "按国家聚类", "双向聚类"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ys = np.arange(len(order))[::-1]
    for y, key in zip(ys, order):
        step = se[key]["poor"]
        lo, hi = ci(step)
        ax.plot([lo, hi], [y, y], color=BRICK, linewidth=2.6, solid_capstyle="round")
        ax.scatter([step["b"]], [y], s=70, color=BRICK, zorder=5)
        ax.text(hi + 0.06, y, f"标准误 {step['se']:.3f}　t = {step['t']:.2f}",
                ha="left", va="center", fontsize=10.5, color=NAVY)
    ax.axvline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(-3.0, 1.4)
    ax.set_xlabel("穷国相对富国的额外效应（百分点 / 摄氏度），95% 近似区间")
    ax.set_title("系数一动不动，不确定性一路变大")
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig05_se_comparison")


def fig06_poor_vs_rich() -> None:
    """穷国与富国的气温效应。"""
    it = SUMMARY["interaction"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    rows = [("富国", it["rich"], SLATE), ("穷国多出来的部分", it["poor_extra"], SLATE),
            ("穷国合计", it["poor_net"], BRICK)]
    ys = np.arange(len(rows))[::-1]
    for y, (name, step, colour) in zip(ys, rows):
        lo, hi = ci(step)
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.6, solid_capstyle="round")
        ax.scatter([step["b"]], [y], s=70, color=colour, zorder=5)
        ax.text(hi + 0.08, y, lab(step), va="center", fontsize=11, color=colour, fontweight="bold")
    ax.axvline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-2.9, 1.0)
    ax.set_xlabel("气温对年增长率的效应（百分点 / 摄氏度），95% 区间")
    ax.set_title("平均效应盖住了穷国与富国的差别")
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig06_poor_vs_rich")


def fig07_trend_control() -> None:
    """加入国家特定趋势前后的穷国效应。"""
    base = SUMMARY["interaction"]["poor_net"]
    trend = SUMMARY["country_trend"]["poor_net"]
    fig, ax = plt.subplots(figsize=(8.4, 2.4))
    rows = [("基准（国家 + 地区×年份 + 穷国×年份）", base, SLATE),
            ("再加国家特定线性趋势", trend, BRICK)]
    ys = [1, 0]
    for y, (name, step, colour) in zip(ys, rows):
        lo, hi = ci(step)
        ax.plot([lo, hi], [y, y], color=colour, linewidth=2.6, solid_capstyle="round")
        ax.scatter([step["b"]], [y], s=70, color=colour, zorder=5)
        ax.text(hi + 0.08, y, lab(step), va="center", fontsize=11, color=colour, fontweight="bold")
        ax.text(-2.9, y + 0.30, name, va="center", fontsize=10, color=SLATE)
    ax.axvline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(-3.0, 0.2)
    ax.set_xlabel("穷国气温效应（百分点 / 摄氏度），95% 区间")
    ax.set_title("趋势控制不是免费的：数值更大，也更依赖短期波动")
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig07_trend_control")


def fig08_dynamic_path() -> None:
    """滞后系数路径：当期与各期滞后。"""
    lg = SUMMARY["lags"]["lag10"]
    coefs = [lg["poor_contemp"]] + lg["poor_lags"]
    xs = np.arange(len(coefs))

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    bs = [c["b"] for c in coefs]
    ses = [c["se"] for c in coefs]
    ax.errorbar(xs, bs, yerr=[1.96 * s for s in ses], fmt="o", color=BRICK,
                ecolor=BRICK, elinewidth=1.6, capsize=3, markersize=6, zorder=5,
                label="单期系数及95%近似区间")
    ax.plot(xs, np.cumsum(bs), color=NAVY, linewidth=2.2, marker="s", markersize=5,
            label="截至该期的系数和")
    ax.axhline(0, color=NAVY, linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["当期"] + [f"L{k}" for k in range(1, 11)])
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("穷国气温系数（百分点 / 摄氏度）")
    ax.set_title("冲击不是只作用在当期")
    ax.legend(fontsize=10, loc="lower left")
    clean(ax)
    fig.tight_layout()
    save(fig, "fig08_dynamic_path")


def fig09_cumulative_effect() -> None:
    """当期效应与累积效应：0/1/5/10 阶。"""
    lg = SUMMARY["lags"]
    keys = ["lag0", "lag1", "lag5", "lag10"]
    names = ["无滞后", "1 阶滞后", "5 阶滞后", "10 阶滞后"]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    width = 0.36
    xs = np.arange(len(keys))
    contemp = [lg[k]["poor_contemp"] for k in keys]
    cum = [lg[k]["poor_cum"] for k in keys]
    ax.bar(xs - width / 2, [c["b"] for c in contemp], width, color=SLATE, label="当期效应")
    ax.bar(xs + width / 2, [c["b"] for c in cum], width, color=BRICK, label="累积效应")
    for x, c in zip(xs - width / 2, contemp):
        ax.text(x, -0.13, f"{c['b']:.2f}", ha="center", va="top", fontsize=10, color="white")
    for x, c in zip(xs + width / 2, cum):
        ax.text(x, -0.13, f"{c['b']:.2f}", ha="center", va="top", fontsize=10, color="white")
    ax.axhline(0, color=NAVY, linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(names)
    ax.set_ylim(-2.15, 0.35)
    ax.set_ylabel("穷国气温效应（百分点 / 摄氏度）")
    ax.set_title("各滞后规格：当期系数与全部温度系数之和")
    ax.legend(fontsize=10)
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig09_cumulative_effect")


def fig10_level_vs_growth() -> None:
    """一次性天气冲击：水平效应恢复，增长效应留下产出水平缺口。"""
    years = np.arange(0, 21)
    base = np.full_like(years, 100.0, dtype=float)
    shock_year = 5
    level = base.copy()
    growth = base.copy()
    for i, t in enumerate(years):
        if t < shock_year:
            level[t] = 100.0
            growth[t] = 100.0
        elif t == shock_year:
            level[t] = 96.0
            growth[t] = 96.0
        else:
            level[t] = 100.0
            growth[t] = 96.0

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0), sharey=True)
    axes[0].plot(years, base, color=NAVY, linewidth=1.1, linestyle="--", label="无冲击基准")
    axes[0].plot(years, level, color=SLATE, linewidth=2.4)
    axes[0].axvline(shock_year, color=BRICK, linewidth=1.2, linestyle="--")
    axes[0].set_title("水平效应：掉下去还能弹回来")
    axes[0].set_xlabel("年份")
    axes[0].set_ylabel("产出指数（概念示意）")
    axes[0].set_ylim(94.5, 101.5)
    axes[0].legend(loc="lower left", fontsize=10)
    axes[0].text(shock_year + 0.4, 97.4, "冲击", color=BRICK, fontsize=10)
    clean(axes[0])

    axes[1].plot(years, base, color=NAVY, linewidth=1.1, linestyle="--")
    axes[1].plot(years, growth, color=BRICK, linewidth=2.4)
    axes[1].axvline(shock_year, color=BRICK, linewidth=1.2, linestyle="--")
    axes[1].set_title("增长效应：留下产出水平缺口")
    axes[1].set_xlabel("年份")
    axes[1].text(shock_year + 0.4, 97.4, "冲击", color=BRICK, fontsize=10)
    axes[1].text(10, 98.1, "增长率恢复，水平缺口仍在", color=BRICK, fontsize=11)
    clean(axes[1])
    axes[1].grid(axis="y", color=SLATEPALE, linewidth=0.7, alpha=0.8)

    fig.suptitle("一次性天气冲击；无冲击基准增长率设为0（示意）", fontsize=15,
                 fontweight="bold", color=NAVY)
    fig.tight_layout()
    save(fig, "fig10_level_vs_growth")


def fig11_residual_autocorrelation() -> None:
    """残差在国家内部的跨期相关。"""
    panel = PANEL.dropna(subset=["resid_m6"]).sort_values(["cc_num", "year"])
    lags = range(1, 8)
    corrs: list[float] = []
    for k in lags:
        lagged = panel[["cc_num", "year", "resid_m6"]].copy()
        lagged["year"] += k
        lagged = lagged.rename(columns={"resid_m6": "lag_resid"})
        pairs = panel.merge(lagged, on=["cc_num", "year"], how="inner", validate="one_to_one")
        x = pairs["lag_resid"].to_numpy()
        y = pairs["resid_m6"].to_numpy()
        corrs.append(float(np.corrcoef(x, y)[0, 1]))

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(list(lags), corrs, color=SLATE, width=0.6)
    ax.axhline(0, color=NAVY, linewidth=1)
    for k, c in zip(lags, corrs):
        ax.text(k, c + 0.008 if c >= 0 else c - 0.008, f"{c:.3f}",
                ha="center", va="bottom" if c >= 0 else "top", fontsize=10, color=NAVY)
    ax.set_xlabel("滞后阶数")
    ax.set_ylabel("同一国家内部残差的相关系数")
    ax.set_title("同一个国家的误差，隔了几年还相关")
    ax.set_ylim(min(min(corrs), 0) - 0.045, max(corrs) * 1.30)
    clean(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig11_residual_autocorrelation")


FIGURES = {
    1: fig01_cross_section_vs_within,
    2: fig02_income_temperature,
    3: fig03_ladder_coefficients,
    4: fig04_region_year_shocks,
    5: fig05_se_comparison,
    6: fig06_poor_vs_rich,
    7: fig07_trend_control,
    8: fig08_dynamic_path,
    9: fig09_cumulative_effect,
    10: fig10_level_vs_growth,
    11: fig11_residual_autocorrelation,
}


def main() -> None:
    wanted = [int(a) for a in sys.argv[1:]] or sorted(FIGURES)
    for n in wanted:
        FIGURES[n]()
        print(f"  fig{n:02d} 完成")
    if GLYPH_WARNINGS:
        raise SystemExit("缺字形警告（必须为 0）：\n" + "\n".join(GLYPH_WARNINGS))
    print("全部图形完成，缺字形警告 = 0")


if __name__ == "__main__":
    main()
