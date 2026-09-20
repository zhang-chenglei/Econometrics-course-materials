from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


CHAPTER_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = CHAPTER_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR / "共享代码"))

from common import (  # noqa: E402
    BLUE,
    GOLD,
    GREY,
    LIGHT_GREY,
    MAIN_CONTROLS,
    CITY_CONTROLS,
    chapter_dir,
    export_table,
    load_data,
    save_figure,
    setup_plot_style,
    standardized_difference,
)


VARIABLE_LABELS = {
    "rd_intensity_w": "研发投入强度",
    "ln_patent_invention": "发明专利申请对数",
    "patent_app_invention": "发明专利申请数",
    "size": "企业规模",
    "lev": "资产负债率",
    "roa": "总资产收益率",
    "growth": "成长性",
    "fixed": "有形资产比重",
    "cashflow": "经营现金流",
    "firm_age": "企业年龄",
    "soe": "国有企业",
    "ln_gov_subsidy": "政府补助对数",
    "kz_index": "KZ融资约束",
    "sa_index": "SA融资约束",
    "ln_rd_staff": "研发人员对数",
    "c_gdp": "城市GDP",
    "c_gdp_pc": "人均GDP",
    "c_gdp_growth": "GDP增速",
    "c_gdp_share_tertiary": "第三产业比重",
    "c_sci_tech_exp": "科技支出",
    "c_telecom_revenue": "电信业务收入",
    "c_mobile_subscribers": "移动电话用户",
}


def main() -> None:
    setup_plot_style()
    chapter = chapter_dir(13)
    table_dir = chapter / "表格"
    figure_dir = chapter / "图形"
    note_dir = chapter / "说明"

    data = load_data()
    # 样本形成过程，口径与教材第13章表13-2 一致。
    # 第 1 行给出的是上游教学底板的固定规模（2,832 家／32,064 观测），
    # 不是本样本的规模，故按字面值列出；第 2 行起为本书实际执行的步骤。
    sample_flow = pd.DataFrame(
        [
            [
                1,
                "控制变量完整的匿名教学底板",
                2832,
                32064,
                "企业控制变量与专利结果完整",
            ],
            [
                2,
                "按试点范围与对照组分层抽取约50%企业",
                data["firm_id"].nunique(),
                len(data),
                "企业层面抽样，随机种子20260720",
            ],
            [
                3,
                "重新匿名并清理异常值",
                data["firm_id"].nunique(),
                len(data),
                "年度1%—99%缩尾；修正明显量纲错误",
            ],
            [
                4,
                "生成半合成结果与机制变量",
                data["firm_id"].nunique(),
                len(data),
                "随机种子20260721；主要分析变量无缺失",
            ],
            [
                5,
                "最终回归样本",
                data["firm_id"].nunique(),
                len(data),
                "2011—2024年企业—年份面板",
            ],
        ],
        columns=["order", "step", "firms", "observations", "note"],
    )
    export_table(sample_flow, table_dir / "表13-1_样本筛选流程")

    descriptive_variables = [
        "rd_intensity_w",
        "ln_patent_invention",
        "patent_app_invention",
        *MAIN_CONTROLS,
        "soe",
        "ln_gov_subsidy",
        "kz_index",
        "sa_index",
        "ln_rd_staff",
    ]
    descriptive_rows = []
    for variable in descriptive_variables:
        series = data[variable]
        descriptive_rows.append(
            {
                "variable": variable,
                "变量": VARIABLE_LABELS[variable],
                "观测数": int(series.notna().sum()),
                "均值": series.mean(),
                "标准差": series.std(),
                "最小值": series.min(),
                "中位数": series.median(),
                "最大值": series.max(),
            }
        )
    descriptive = pd.DataFrame(descriptive_rows)
    export_table(descriptive, table_dir / "表13-2_描述性统计")

    missing_variables = [
        "firm_id",
        "year",
        "analysis_city_id",
        "did_post_full",
        "gvar",
        *descriptive_variables,
        *CITY_CONTROLS,
        "pre_size",
        "pre_soe",
        "pre_patent_mean",
        "tech_industry",
    ]
    missingness = pd.DataFrame(
        {
            "variable": missing_variables,
            "变量": [VARIABLE_LABELS.get(value, value) for value in missing_variables],
            "missing_n": [int(data[value].isna().sum()) for value in missing_variables],
            "missing_pct": [
                float(data[value].isna().mean() * 100) for value in missing_variables
            ],
        }
    )
    export_table(missingness, table_dir / "表13-3_主要分析变量缺失率")

    # Firm-level pre-policy balance, avoiding repeated weighting by panel length.
    pre = data.loc[data["year"].le(2018)].copy()
    balance_variables = [
        "size",
        "lev",
        "roa",
        "growth",
        "fixed",
        "cashflow",
        "firm_age",
        "soe",
        "c_gdp",
        "c_gdp_pc",
        "c_gdp_growth",
        "c_gdp_share_tertiary",
        "c_sci_tech_exp",
        "c_telecom_revenue",
        "c_mobile_subscribers",
    ]
    pre_firm = (
        pre.groupby("firm_id")
        .agg(
            did_treated=("did_treated", "first"),
            **{variable: (variable, "mean") for variable in balance_variables},
        )
        .reset_index()
    )
    balance_rows = []
    for variable in balance_variables:
        treated = pre_firm.loc[pre_firm["did_treated"].eq(1), variable]
        control = pre_firm.loc[pre_firm["did_treated"].eq(0), variable]
        test = ttest_ind(treated, control, equal_var=False, nan_policy="omit")
        balance_rows.append(
            {
                "variable": variable,
                "变量": VARIABLE_LABELS[variable],
                "处理组均值": treated.mean(),
                "对照组均值": control.mean(),
                "均值差": treated.mean() - control.mean(),
                "标准化差异": standardized_difference(treated, control),
                "均值差检验p值": test.pvalue,
                "处理组企业数": int(treated.notna().sum()),
                "对照组企业数": int(control.notna().sum()),
            }
        )
    balance = pd.DataFrame(balance_rows)
    export_table(balance, table_dir / "表13-4_政策前特征平衡")

    batch_support = (
        data.loc[data["did_treated"].eq(1)]
        .groupby(["gvar", "year"])
        .agg(
            firms=("firm_id", "nunique"),
            observations=("firm_id", "size"),
            rd_mean=("rd_intensity_w", "mean"),
            patent_mean=("ln_patent_invention", "mean"),
        )
        .reset_index()
    )
    export_table(batch_support, table_dir / "表13-5_年度处理组样本覆盖")

    # Figure 12-1: outcome distributions.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].hist(
        data["rd_intensity_w"],
        bins=35,
        color=BLUE,
        edgecolor="white",
        alpha=0.9,
    )
    axes[0].set_title("研发投入强度分布")
    axes[0].set_xlabel("研发投入强度（比例）")
    axes[0].set_ylabel("企业—年份观测数")
    axes[0].grid(axis="y")
    axes[0].grid(axis="x", visible=False)

    axes[1].hist(
        data["ln_patent_invention"],
        bins=35,
        color=GOLD,
        edgecolor="white",
        alpha=0.9,
    )
    axes[1].set_title("发明专利申请分布")
    axes[1].set_xlabel("ln(1+发明专利申请数)")
    axes[1].set_ylabel("企业—年份观测数")
    axes[1].grid(axis="y")
    axes[1].grid(axis="x", visible=False)
    fig.suptitle("主要结果变量的分布", y=1.02, fontsize=15)
    fig.text(
        0.5,
        -0.01,
        f"半合成教学样本，2011—2024年，N={len(data):,}",
        ha="center",
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图13-1_主要结果变量分布")

    # Figure 12-2: raw trends by ever-treated status.
    trend = (
        data.groupby(["year", "did_treated"])
        .agg(
            rd=("rd_intensity_w", "mean"),
            patent=("ln_patent_invention", "mean"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for treated_value, label, color, style in [
        (0, "从未处理企业", GREY, "--"),
        (1, "试点范围企业", BLUE, "-"),
    ]:
        subset = trend.loc[trend["did_treated"].eq(treated_value)]
        axes[0].plot(
            subset["year"],
            subset["rd"],
            label=label,
            color=color,
            linestyle=style,
            marker="o",
            markersize=4,
        )
        axes[1].plot(
            subset["year"],
            subset["patent"],
            label=label,
            color=color,
            linestyle=style,
            marker="o",
            markersize=4,
        )
    axes[0].set_title("研发投入强度的原始均值")
    axes[0].set_ylabel("平均研发投入强度")
    axes[1].set_title("发明专利申请的原始均值")
    axes[1].set_ylabel("平均ln(1+申请数)")
    for axis in axes:
        axis.set_xlabel("年份")
        axis.axvline(2019.5, color=GOLD, linestyle=":", linewidth=1.2)
        axis.legend(frameon=False, loc="upper left")
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    fig.suptitle("处理组与对照组的原始时间趋势", y=1.02, fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "竖线表示首批试点获批附近；原始均值不等于因果效应",
        ha="center",
        color=GREY,
        fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图13-2_处理组与对照组原始趋势")

    # Figure 12-3: Love plot.
    love = balance.sort_values("标准化差异")
    positions = np.arange(len(love))
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.scatter(
        love["标准化差异"],
        positions,
        color=BLUE,
        s=45,
        zorder=3,
    )
    axis.axvline(0, color=INK if "INK" in globals() else "#24313A", linewidth=1)
    axis.axvline(-0.1, color=GOLD, linestyle="--", linewidth=1)
    axis.axvline(0.1, color=GOLD, linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(love["变量"])
    axis.set_xlabel("政策前标准化均值差异")
    axis.set_title("处理组与对照组的政策前特征差异")
    axis.text(
        0,
        1.015,
        "虚线为|标准化差异|=0.10参考线；平衡不等于随机分配",
        transform=axis.transAxes,
        color=GREY,
        fontsize=10,
    )
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    save_figure(fig, figure_dir / "图13-3_政策前特征平衡图")

    # Figure 12-4: cohort support heatmap.
    support_matrix = (
        batch_support.pivot(index="gvar", columns="year", values="firms")
        .fillna(0)
        .sort_index()
    )
    fig, axis = plt.subplots(figsize=(12, 3.6))
    image = axis.imshow(support_matrix.values, aspect="auto", cmap="Blues")
    axis.set_yticks(np.arange(len(support_matrix.index)))
    axis.set_yticklabels([f"{int(value)}年度处理组" for value in support_matrix.index])
    axis.set_xticks(np.arange(len(support_matrix.columns)))
    axis.set_xticklabels(support_matrix.columns.astype(int), rotation=45)
    for row in range(support_matrix.shape[0]):
        for column in range(support_matrix.shape[1]):
            value = int(support_matrix.iloc[row, column])
            axis.text(
                column,
                row,
                f"{value}",
                ha="center",
                va="center",
                color="white" if value > support_matrix.values.max() * 0.55 else INK
                if "INK" in globals()
                else "#24313A",
                fontsize=8,
            )
    axis.set_xlabel("年份")
    axis.set_title("各年度处理组的企业样本支持")
    fig.colorbar(image, ax=axis, label="企业数", fraction=0.025, pad=0.03)
    fig.tight_layout()
    save_figure(fig, figure_dir / "图13-4_处理组支持热图")

    max_abs_smd = balance["标准化差异"].abs().max()
    summary = f"""# 第13章案例结果说明

> 本章使用半合成教学样本。异常值清理、抽样和结果变量生成规则已经记录在数据README和清理日志中。

最终样本包含{data['firm_id'].nunique():,}家企业和{len(data):,}个企业—年份观测，年份覆盖{data['year'].min()}—{data['year'].max()}。研发、专利、企业控制变量、城市特征、机制变量和预设异质性变量在最终分析样本中均无缺失；政策名称、批次和事件时间在从未处理企业中保留结构性空值。

研发投入强度均值为{data['rd_intensity_w'].mean():.4f}，发明专利申请对数均值为{data['ln_patent_invention'].mean():.4f}。原始专利申请数呈明显右偏，因此正式估计使用`ln(1+x)`，并把反双曲正弦变换留作稳健性检验。

政策前平衡表显示，处理组和对照组的最大绝对标准化差异为{max_abs_smd:.3f}。这些差异用于说明试点并非随机分配，不能把描述性平衡直接当作平行趋势成立。第14章将通过交错DID动态估计和政策前联合检验进一步检查识别条件。
"""
    (note_dir / "第13章案例结果说明.md").write_text(summary, encoding="utf-8")

    print(chapter)


if __name__ == "__main__":
    main()
