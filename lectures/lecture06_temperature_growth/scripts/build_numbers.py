#!/usr/bin/env python3
r"""把 build_panel.do 产出的回归结果汇总成幻灯片可用的数字。

输入：data/regression_results.csv、data/descriptives.csv
输出：data/model_summary.json     全量结构，供 generate_figures.py 与人工核对
      data/model_numbers.tex     幻灯片里所有数字的唯一权威（宏定义）

数字纪律：幻灯片正文不写字面数字，一律 \MacroName；本脚本是这些宏的唯一产地。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "output" / "data"


def load_results() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with (DATA / "regression_results.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[(row["spec"], row["term"])] = {
                "b": float(row["b"]),
                "se": float(row["se"]),
                "t": float(row["t"]),
                "p": float(row["p"]),
                "n": int(row["n"]),
            }
    return rows


def load_descriptives() -> dict[str, float]:
    out: dict[str, float] = {}
    with (DATA / "descriptives.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["item"]] = float(row["value"])
    return out


def regression_sample() -> dict[str, int]:
    """resid_m6 仅在基准模型 e(sample) 中非缺失；据此统计实际样本。"""
    with (DATA / "analysis_panel.csv").open(encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row["resid_m6"].strip() not in {"", "."}]
    years = {int(row["year"]) for row in rows}
    return {
        "n_obs": len(rows),
        "n_country": len({row["cc_num"] for row in rows}),
        "n_year": len(years),
        "year_min": min(years),
        "year_max": max(years),
        "n_region_year": len({row["rynum"] for row in rows}),
        "n_parent": len({row["parent_id"] for row in rows}),
    }


def stars(p: float) -> str:
    if p <= 0.01:
        return "***"
    if p <= 0.05:
        return "**"
    if p <= 0.10:
        return "*"
    return ""


def fmt(b: float, p: float, dp: int = 3) -> str:
    return f"{b:.{dp}f}{stars(p)}"


def signed(b: float, p: float, dp: int = 2) -> str:
    """带正负号、用于叙述性文字的写法。"""
    return f"{b:+.{dp}f}{stars(p)}"


def main() -> None:
    res = load_results()
    desc = load_descriptives()
    sample = regression_sample()
    assert sample["n_obs"] == res[("m6_interaction", "wtem")]["n"], "基准样本与回归记录不一致"

    ladder = ["m1_pooled", "m2_countryfe", "m3_country_year", "m4_region_year", "m5_poor_year"]
    ladder_labels = [
        "无固定效应",
        "国家固定效应",
        "+ 年份固定效应",
        "+ 地区×年份固定效应",
        "+ 穷国×年份固定效应（论文基准）",
    ]

    summary = {
        "source": {
            "paper": "Dell, Jones & Olken (2012), AEJ:Macro 4(3), 66-95",
            "replication_package": "AEJ-Mac2010-0092-reproduction",
            "data_license": "CC-BY 4.0 (American Economic Association)",
            "estimator": "教学规格：Stata reghdfe；论文原包：cgmreg",
            "note": "核心点估计与包内论文稿按3位小数吻合；单例处理、协方差修正与推断口径不同，部分显著性不同",
        },
        "sample": {
            "n_obs": desc["n_obs"],
            "n_obs_regression": res[("m5_poor_year", "wtem")]["n"],
            "n_country": sample["n_country"],
            "n_year": sample["n_year"],
            "year_min": sample["year_min"],
            "year_max": sample["year_max"],
            "n_region_year": sample["n_region_year"],
            "n_parent": sample["n_parent"],
            "panel_n_country": desc["n_country"],
            "panel_n_year": desc["n_year"],
            "descriptive_scope": "温度描述统计使用构造后完整面板，而非基准回归样本",
            "wtem_mean": desc["wtem_mean"],
            "wtem_sd": desc["wtem_sd"],
            "wtem_mean_poor": desc["wtem_mean_poor"],
            "wtem_mean_rich": desc["wtem_mean_rich"],
        },
        "ladder": [
            {
                "spec": s,
                "label": lab,
                "b": res[(s, "wtem")]["b"],
                "se": res[(s, "wtem")]["se"],
                "t": res[(s, "wtem")]["t"],
                "p": res[(s, "wtem")]["p"],
                "n": res[(s, "wtem")]["n"],
            }
            for s, lab in zip(ladder, ladder_labels)
        ],
        "interaction": {
            "rich": res[("m6_interaction", "wtem")],
            "poor_extra": res[("m6_interaction", "wtem_poor")],
            "poor_net": res[("m6_interaction", "poor_net")],
        },
        "country_trend": {
            "rich": res[("m7_country_trend", "wtem")],
            "poor_extra": res[("m7_country_trend", "wtem_poor")],
            "poor_net": res[("m7_country_trend", "poor_net")],
        },
        "std_errors": {
            key: {
                "label": lab,
                "poor": res[(key, "wtem_poor")],
                "rich": res[(key, "wtem")],
            }
            for key, lab in [
                ("se_ols", "常规标准误"),
                ("se_robust", "异方差稳健"),
                ("se_cl_country", "按国家聚类"),
                ("se_cl_twoway", "双向聚类（国家 & 地区×年）"),
            ]
        },
        "lags": {
            "lag0": {
                "poor_contemp": res[("lag0", "poor_contemp")],
                "poor_cum": res[("lag0", "poor_cum")],
                "rich_cum": res[("lag0", "rich_cum")],
                "n": res[("lag0", "poor_cum")]["n"],
            },
            "lag1": {
                "poor_contemp": res[("lag1", "poor_contemp")],
                "poor_cum": res[("lag1", "poor_cum")],
                "rich_cum": res[("lag1", "rich_cum")],
                "n": res[("lag1", "poor_cum")]["n"],
            },
            "lag5": {
                "poor_contemp": res[("lag5", "poor_contemp")],
                "poor_lags": [res[("lag5", f"poor_L{k}")] for k in range(1, 6)],
                "poor_cum": res[("lag5", "poor_cum")],
                "rich_cum": res[("lag5", "rich_cum")],
                "n": res[("lag5", "poor_cum")]["n"],
            },
            "lag10": {
                "poor_contemp": res[("lag10", "poor_contemp")],
                "poor_lags": [res[("lag10", f"poor_L{k}")] for k in range(1, 11)],
                "poor_cum": res[("lag10", "poor_cum")],
                "rich_cum": res[("lag10", "rich_cum")],
                "n": res[("lag10", "poor_cum")]["n"],
            },
        },
    }

    (DATA / "model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ---------------------------------------------------------------- 宏定义
    it = summary["interaction"]
    ct = summary["country_trend"]
    se = summary["std_errors"]
    lg = summary["lags"]
    sm = summary["sample"]

    macros: list[tuple[str, str]] = [
        # 样本
        ("SampNObs", f"{int(sm['n_obs_regression']):,}".replace(",", "{,}")),
        ("SampNCountry", str(int(sm["n_country"]))),
        ("SampNYears", str(int(sm["n_year"]))),
        ("SampYearFirst", str(sm["year_min"])),
        ("SampYearLast", str(sm["year_max"])),
        ("SampNParent", str(sm["n_parent"])),
        ("PanelNObs", f"{int(sm['n_obs']):,}".replace(",", "{,}")),
        ("PanelNCountry", str(int(sm["panel_n_country"]))),
        ("PanelNYears", str(int(sm["panel_n_year"]))),
        ("SampNRegionYear", str(int(sm["n_region_year"]))),
        ("SampTempMean", f"{sm['wtem_mean']:.1f}"),
        ("SampTempSDPoor", f"{sm['wtem_sd']:.1f}"),
        ("SampTempPoor", f"{sm['wtem_mean_poor']:.1f}"),
        ("SampTempRich", f"{sm['wtem_mean_rich']:.1f}"),
    ]

    # 阶梯（带星号，供表格）
    # ⚠️ LaTeX 的 \newcommand 只接受纯字母名字：\LadderCoef1 会让 \LadderCoef + 字符 1
    # 被当成两个 token，报 “Missing \begin{document}”。故一律用英文数词后缀。
    suffixes = ["One", "Two", "Three", "Four", "Five"]
    for suffix, step in zip(suffixes, summary["ladder"]):
        macros.append((f"LadderCoef{suffix}", fmt(step["b"], step["p"])))
        macros.append((f"LadderSE{suffix}", f"({step['se']:.3f})"))
        macros.append((f"LadderN{suffix}", f"{step['n']:,}".replace(",", "{,}")))

    # 交互模型
    macros += [
        ("RichTemp", fmt(it["rich"]["b"], it["rich"]["p"])),
        ("RichTempSE", f"({it['rich']['se']:.3f})"),
        ("PoorExtra", fmt(it["poor_extra"]["b"], it["poor_extra"]["p"])),
        ("PoorExtraSE", f"({it['poor_extra']['se']:.3f})"),
        ("PoorNet", fmt(it["poor_net"]["b"], it["poor_net"]["p"])),
        ("PoorNetSE", f"({it['poor_net']['se']:.3f})"),
        ("PoorNetT", f"{it['poor_net']['t']:.2f}"),
    ]

    # 国家趋势
    macros += [
        ("TrendPoorExtra", fmt(ct["poor_extra"]["b"], ct["poor_extra"]["p"])),
        ("TrendPoorNet", fmt(ct["poor_net"]["b"], ct["poor_net"]["p"])),
        ("TrendPoorNetSE", f"({ct['poor_net']['se']:.3f})"),
        ("TrendRichTemp", fmt(ct["rich"]["b"], ct["rich"]["p"])),
    ]

    # 标准误对比
    se_names = {
        "se_ols": "Ols",
        "se_robust": "Robust",
        "se_cl_country": "Cl",
        "se_cl_twoway": "Two",
    }
    for key, tag in se_names.items():
        macros.append((f"SEPoor{tag}", f"{se[key]['poor']['se']:.3f}"))
        macros.append((f"TPoor{tag}", f"{se[key]['poor']['t']:.2f}"))
        macros.append((f"SERich{tag}", f"{se[key]['rich']['se']:.3f}"))
    macros.append(("SEPoorCoef", fmt(se["se_cl_twoway"]["poor"]["b"], se["se_cl_twoway"]["poor"]["p"])))

    # 滞后
    macros += [
        ("LagZeroCum", fmt(lg["lag0"]["poor_cum"]["b"], lg["lag0"]["poor_cum"]["p"])),
        ("LagOneContemp", fmt(lg["lag1"]["poor_contemp"]["b"], lg["lag1"]["poor_contemp"]["p"])),
        ("LagOneCum", fmt(lg["lag1"]["poor_cum"]["b"], lg["lag1"]["poor_cum"]["p"])),
        ("LagOneCumSE", f"({lg['lag1']['poor_cum']['se']:.3f})"),
        ("LagFiveContemp", fmt(lg["lag5"]["poor_contemp"]["b"], lg["lag5"]["poor_contemp"]["p"])),
        ("LagFiveCum", fmt(lg["lag5"]["poor_cum"]["b"], lg["lag5"]["poor_cum"]["p"])),
        ("LagTenContemp", fmt(lg["lag10"]["poor_contemp"]["b"], lg["lag10"]["poor_contemp"]["p"])),
        ("LagTenCum", fmt(lg["lag10"]["poor_cum"]["b"], lg["lag10"]["poor_cum"]["p"])),
        ("LagTenCumSE", f"({lg['lag10']['poor_cum']['se']:.3f})"),
        ("LagTenCumP", f"{lg['lag10']['poor_cum']['p']:.4f}"),
        ("RichLagZero", fmt(lg["lag0"]["rich_cum"]["b"], lg["lag0"]["rich_cum"]["p"])),
        ("RichLagTenCum", fmt(lg["lag10"]["rich_cum"]["b"], lg["lag10"]["rich_cum"]["p"])),
    ]

    lines = [
        "% Auto-generated by scripts/build_numbers.py; do not edit manually.",
        "% 数字来源：data/regression_results.csv（scripts/build_panel.do 产出）。",
        "% 系数复现 Dell, Jones & Olken (2012)；数据授权 CC-BY 4.0。",
        "",
    ]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros]
    (DATA / "model_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"已写出 {DATA / 'model_summary.json'}")
    print(f"已写出 {DATA / 'model_numbers.tex'}（{len(macros)} 个宏）")
    print()
    print("复现核对：")
    print(f"  穷国净效应 = {PoorNetValue(lg, it)}  包内论文稿 = -1.394***")
    print(f"  温度主效应 = {it['rich']['b']:.3f}          包内论文稿 = 0.261")


def PoorNetValue(lg, it) -> str:
    return f"{it['poor_net']['b']:.4f}"


if __name__ == "__main__":
    main()
