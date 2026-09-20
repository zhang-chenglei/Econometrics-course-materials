from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.special import expit


CHAPTER_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = CHAPTER_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR / "共享代码"))

from common import (  # noqa: E402
    BLUE,
    GOLD,
    GREY,
    INK,
    LIGHT_GREY,
    MAIN_CONTROLS,
    OUTCOME_LABELS,
    chapter_dir,
    export_table,
    fit_interaction,
    fit_twfe,
    load_data,
    save_figure,
    setup_plot_style,
    significance_stars,
    staggered_did,
)


def propensity_weights(data: pd.DataFrame) -> pd.Series:
    pre = data.loc[data["year"].le(2018)].copy()
    firm_pre = (
        pre.groupby("firm_id")
        .agg(
            did_treated=("did_treated", "first"),
            pre_size=("pre_size", "first"),
            pre_soe=("pre_soe", "first"),
            pre_patent_mean=("pre_patent_mean", "first"),
            c_gdp=("c_gdp", "mean"),
            c_gdp_pc=("c_gdp_pc", "mean"),
            c_gdp_share_tertiary=("c_gdp_share_tertiary", "mean"),
            c_sci_tech_exp=("c_sci_tech_exp", "mean"),
        )
        .reset_index()
    )
    features = [
        "pre_size",
        "pre_soe",
        "pre_patent_mean",
        "c_gdp",
        "c_gdp_pc",
        "c_gdp_share_tertiary",
        "c_sci_tech_exp",
    ]
    matrix = firm_pre[features].copy()
    matrix["pre_patent_mean"] = np.log1p(matrix["pre_patent_mean"])
    matrix["c_gdp"] = np.log1p(matrix["c_gdp"])
    matrix["c_gdp_pc"] = np.log1p(matrix["c_gdp_pc"])
    matrix["c_sci_tech_exp"] = np.log1p(matrix["c_sci_tech_exp"])
    matrix = (matrix - matrix.mean()) / matrix.std(ddof=0)
    matrix = sm.add_constant(matrix)
    model = sm.GLM(
        firm_pre["did_treated"],
        matrix,
        family=sm.families.Binomial(),
    )
    fitted = model.fit_regularized(alpha=0.02, L1_wt=0.0)
    propensity = pd.Series(
        expit(np.asarray(matrix) @ np.asarray(fitted.params)),
        index=firm_pre["firm_id"],
    ).clip(0.05, 0.95)
    treated = firm_pre.set_index("firm_id")["did_treated"]
    firm_weights = pd.Series(
        np.where(treated.eq(1), 1.0, propensity / (1 - propensity)),
        index=propensity.index,
    )
    return data["firm_id"].map(firm_weights)


def main() -> None:
    setup_plot_style()
    chapter = chapter_dir(14)
    table_dir = chapter / "表格"
    figure_dir = chapter / "图形"
    note_dir = chapter / "说明"

    data = load_data()

    # ------------------------------------------------------------------
    # 13.5a TWFE teaching benchmark.
    # ------------------------------------------------------------------
    twfe_specs = []
    for outcome, label in [
        ("rd_intensity_w", "研发投入强度"),
        ("ln_patent_invention", "发明专利申请"),
    ]:
        row, _ = fit_twfe(
            data,
            outcome,
            label=f"{label}：企业与年份固定效应",
        )
        twfe_specs.append(row)
        row, _ = fit_twfe(
            data,
            outcome,
            controls=MAIN_CONTROLS,
            label=f"{label}：加入企业控制变量",
        )
        twfe_specs.append(row)
        row, _ = fit_twfe(
            data,
            outcome,
            controls=MAIN_CONTROLS,
            industry_year_effects=True,
            label=f"{label}：行业×年份固定效应",
        )
        twfe_specs.append(row)
    twfe_table = pd.DataFrame(twfe_specs)
    twfe_table["stars"] = twfe_table["p_value"].map(significance_stars)
    export_table(twfe_table, table_dir / "表14-1_TWFE教学基准")

    # ------------------------------------------------------------------
    # 13.5a-b Staggered DID and event studies.
    # ------------------------------------------------------------------
    event_tables = []
    cohort_tables = []
    overall_rows = []
    for outcome in ["rd_intensity_w", "ln_patent_invention"]:
        for control_mode in ["never", "not_yet"]:
            event, cohorts, overall = staggered_did(
                data,
                outcome,
                control_mode=control_mode,
            )
            event_tables.append(event)
            cohort_tables.append(cohorts)
            overall_rows.append(overall)
    event_table = pd.concat(event_tables, ignore_index=True)
    cohort_table = pd.concat(cohort_tables, ignore_index=True)
    overall_table = pd.DataFrame(overall_rows)
    overall_table["stars"] = overall_table["p_value"].map(significance_stars)
    export_table(overall_table, table_dir / "表14-2_交错DID总体ATT")
    export_table(event_table, table_dir / "表14-3_动态事件研究")
    export_table(cohort_table, table_dir / "表14-4_年度处理组效应")

    pretrend = overall_table[
        [
            "outcome",
            "control_mode",
            "pretrend_chi2",
            "pretrend_df",
            "pretrend_p_value",
            "all_individual_pre_p_gt_005",
        ]
    ].copy()
    export_table(pretrend, table_dir / "表14-5_平行趋势联合检验")

    # ------------------------------------------------------------------
    # 13.5c Robustness checks.
    # ------------------------------------------------------------------
    robustness_rows = []

    def append_spec(
        frame: pd.DataFrame,
        outcome: str,
        label: str,
        treatment: str = "did_post_full",
        controls: list[str] | None = None,
        industry_year: bool = False,
        weights: pd.Series | None = None,
    ) -> None:
        row, _ = fit_twfe(
            frame,
            outcome,
            treatment=treatment,
            controls=controls,
            label=label,
            industry_year_effects=industry_year,
            weights=weights,
        )
        robustness_rows.append(row)

    ipw = propensity_weights(data)
    for outcome, outcome_label in [
        ("rd_intensity_w", "研发投入强度"),
        ("ln_patent_invention", "发明专利申请"),
    ]:
        append_spec(data, outcome, f"{outcome_label}：主设定")
        append_spec(
            data,
            outcome,
            f"{outcome_label}：加入企业控制变量",
            controls=MAIN_CONTROLS,
        )
        append_spec(
            data,
            outcome,
            f"{outcome_label}：批复当年作为处理起点",
            treatment="did_post_approval",
        )
        append_spec(
            data,
            outcome,
            f"{outcome_label}：行业×年份固定效应",
            controls=MAIN_CONTROLS,
            industry_year=True,
        )
        append_spec(
            data.loc[data["year"].ne(2020)],
            outcome,
            f"{outcome_label}：排除2020年",
        )
        append_spec(
            data.loc[data["assign_city"].ne("武汉市")],
            outcome,
            f"{outcome_label}：排除武汉",
        )
        append_spec(
            data.loc[
                ~data["assign_city"].isin(["北京市", "上海市", "深圳市"])
            ],
            outcome,
            f"{outcome_label}：排除北京上海深圳",
        )
        append_spec(
            data,
            outcome,
            f"{outcome_label}：控制同期政策教学代理",
            controls=["teach_overlap_policy"],
        )
        append_spec(
            data.loc[data["teach_overlap_policy"].eq(0)],
            outcome,
            f"{outcome_label}：剔除同期政策教学代理",
        )
        append_spec(
            data.loc[data["teach_neighbor_proxy"].eq(0)],
            outcome,
            f"{outcome_label}：剔除同省近邻教学代理",
        )
        append_spec(
            data,
            outcome,
            f"{outcome_label}：办公地址教学替代口径",
            treatment="teach_did_post_office",
        )
        append_spec(
            data,
            outcome,
            f"{outcome_label}：政策前特征ATT加权",
            weights=ipw,
        )

    append_spec(
        data,
        "patent_asinh",
        "发明专利申请：反双曲正弦变换",
    )
    append_spec(
        data,
        "rd_sales_pct",
        "研发投入强度：以营业收入为分母（百分比）",
    )
    append_spec(
        data,
        "patent_app_invention",
        "发明专利申请：原始申请件数（线性固定效应）",
    )
    robustness_table = pd.DataFrame(robustness_rows)
    robustness_table["stars"] = robustness_table["p_value"].map(
        significance_stars
    )
    export_table(robustness_table, table_dir / "表14-6_稳健性检验")

    leave_one_rows = []
    pilots = sorted(
        value for value in data["policy_pilot_id"].dropna().unique()
    )
    for outcome, outcome_label in [
        ("rd_intensity_w", "研发投入强度"),
        ("ln_patent_invention", "发明专利申请"),
    ]:
        for pilot in pilots:
            row, _ = fit_twfe(
                data.loc[data["policy_pilot_id"].ne(pilot) | data["policy_pilot_id"].isna()],
                outcome,
                label=f"{outcome_label}：剔除{pilot}",
            )
            row["excluded_pilot"] = pilot
            leave_one_rows.append(row)
    leave_one = pd.DataFrame(leave_one_rows)
    export_table(leave_one, table_dir / "表14-7_留一试点影响力诊断")

    # ------------------------------------------------------------------
    # 13.5d Heterogeneity with direct interaction tests.
    # ------------------------------------------------------------------
    firm_unique = data.drop_duplicates("firm_id").copy()
    size_cut = firm_unique["pre_size"].median()
    patent_cut = firm_unique["pre_patent_mean"].median()
    data["high_pre_size"] = data["pre_size"].gt(size_cut).astype(int)
    data["high_pre_patent"] = data["pre_patent_mean"].gt(patent_cut).astype(int)

    dimensions = [
        ("pre_soe", "所有制（1=国有）", "非国有", "国有"),
        ("high_pre_size", "政策前规模（1=较大）", "较小", "较大"),
        (
            "high_pre_patent",
            "政策前创新基础（1=较高）",
            "较低",
            "较高",
        ),
        ("tech_industry", "技术相关行业（1=是）", "其他行业", "技术相关行业"),
    ]
    interaction_rows = []
    group_effect_rows = []
    for outcome in ["rd_intensity_w", "ln_patent_invention"]:
        for variable, label, group0, group1 in dimensions:
            interaction = fit_interaction(
                data,
                outcome,
                variable,
                label=label,
            )
            interaction["group0_label"] = group0
            interaction["group1_label"] = group1
            interaction_rows.append(interaction)
            for value, group_label in [(0, group0), (1, group1)]:
                row, _ = fit_twfe(
                    data.loc[data[variable].eq(value)],
                    outcome,
                    label=f"{label}：{group_label}",
                )
                row["dimension"] = label
                row["group"] = group_label
                group_effect_rows.append(row)
    interaction_table = pd.DataFrame(interaction_rows)
    interaction_table["difference_stars"] = interaction_table[
        "difference_p_value"
    ].map(significance_stars)
    group_effects = pd.DataFrame(group_effect_rows)
    group_effects["stars"] = group_effects["p_value"].map(significance_stars)
    export_table(interaction_table, table_dir / "表14-8_异质性直接差异检验")
    export_table(group_effects, table_dir / "表14-9_异质性分组效应")

    # ------------------------------------------------------------------
    # 13.5e Mechanisms.
    # ------------------------------------------------------------------
    mechanism_rows = []
    for outcome in ["ln_gov_subsidy", "kz_index", "sa_index", "ln_rd_staff"]:
        row, _ = fit_twfe(
            data,
            outcome,
            label=OUTCOME_LABELS[outcome],
        )
        row["outcome_sd"] = data[outcome].std()
        row["standardized_effect"] = row["coefficient"] / row["outcome_sd"]
        row["standardized_se"] = row["std_error"] / row["outcome_sd"]
        row["standardized_ci_low"] = (
            row["standardized_effect"] - 1.96 * row["standardized_se"]
        )
        row["standardized_ci_high"] = (
            row["standardized_effect"] + 1.96 * row["standardized_se"]
        )
        mechanism_rows.append(row)
    mechanism_table = pd.DataFrame(mechanism_rows)
    mechanism_table["stars"] = mechanism_table["p_value"].map(
        significance_stars
    )
    export_table(mechanism_table, table_dir / "表14-10_机制分析")

    mechanism_staggered_rows = []
    for outcome in ["ln_gov_subsidy", "kz_index", "sa_index", "ln_rd_staff"]:
        _, _, overall = staggered_did(
            data,
            outcome,
            control_mode="never",
            event_window=(-5, 4),
        )
        overall["label"] = OUTCOME_LABELS[outcome]
        overall["outcome_sd"] = data[outcome].std()
        overall["standardized_effect"] = overall["att"] / overall["outcome_sd"]
        overall["standardized_se"] = overall["std_error"] / overall["outcome_sd"]
        overall["standardized_ci_low"] = (
            overall["standardized_effect"] - 1.96 * overall["standardized_se"]
        )
        overall["standardized_ci_high"] = (
            overall["standardized_effect"] + 1.96 * overall["standardized_se"]
        )
        overall["stars"] = significance_stars(overall["p_value"])
        mechanism_staggered_rows.append(overall)
    mechanism_staggered_table = pd.DataFrame(mechanism_staggered_rows)
    export_table(
        mechanism_staggered_table,
        table_dir / "表14-10b_机制交错DID",
    )

    evidence_map = pd.DataFrame(
        [
            [
                "是否提高研发投入强度",
                "交错DID总体ATT、事件研究、稳健性",
                "显著为正；政策后逐步上升",
            ],
            [
                "是否提高发明专利申请",
                "交错DID总体ATT、事件研究、替代变换",
                "显著为正；存在动态增强",
            ],
            [
                "政策前趋势是否可接受",
                "政策前各期与联合检验",
                "政策前各期不显著，联合检验不拒绝",
            ],
            [
                "结果是否由单一试点驱动",
                "18次留一试点估计",
                "方向和显著性保持稳定",
            ],
            [
                "哪些企业反应更强",
                "四个预设维度及直接差异检验",
                "以直接组间差异检验为准",
            ],
            [
                "哪些渠道得到支持",
                "补助、融资约束、研发人员",
                "补助和研发人员得到支持；融资约束未得到支持",
            ],
        ],
        columns=["问题", "主要证据", "教学样本判断"],
    )
    export_table(evidence_map, table_dir / "表14-11_证据地图")

    # ------------------------------------------------------------------
    # Figures.
    # ------------------------------------------------------------------
    main_event = event_table.loc[event_table["control_mode"].eq("never")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for axis, outcome, title in [
        (axes[0], "rd_intensity_w", "研发投入强度"),
        (axes[1], "ln_patent_invention", "发明专利申请"),
    ]:
        plot_data = main_event.loc[main_event["outcome"].eq(outcome)]
        axis.errorbar(
            plot_data["event_time"],
            plot_data["att"],
            yerr=1.96 * plot_data["std_error"],
            fmt="o-",
            color=BLUE,
            ecolor=BLUE,
            capsize=3,
            linewidth=1.8,
            markersize=5,
        )
        axis.axhline(0, color=GREY, linestyle="--", linewidth=1)
        axis.axvline(-0.5, color=GOLD, linestyle=":", linewidth=1.4)
        axis.set_title(title)
        axis.set_xlabel("相对首个完整处理年度")
        axis.set_ylabel("组别—时期ATT")
        axis.set_xticks(plot_data["event_time"])
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    fig.suptitle("人工智能试验区与企业创新：动态事件研究", y=1.02, fontsize=15)
    fig.text(
        0.5,
        -0.02,
        "对照组为从未处理企业；点为事件时间聚合ATT，误差线为城市聚类95%置信区间；基准期为-1",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-1_动态事件研究")

    main_cohort = cohort_table.loc[cohort_table["control_mode"].eq("never")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for axis, outcome, title in [
        (axes[0], "rd_intensity_w", "研发投入强度"),
        (axes[1], "ln_patent_invention", "发明专利申请"),
    ]:
        plot_data = main_cohort.loc[main_cohort["outcome"].eq(outcome)]
        axis.errorbar(
            plot_data["cohort"].astype(str),
            plot_data["att"],
            yerr=1.96 * plot_data["std_error"],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=4,
            markersize=7,
        )
        axis.axhline(0, color=GREY, linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("年度处理组")
        axis.set_ylabel("批次平均ATT")
        axis.grid(axis="y")
        axis.grid(axis="x", visible=False)
    fig.suptitle("不同年度处理组的平均效应", y=1.02, fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "每个点按该年度处理组可观察的政策后时期聚合；误差线为95%置信区间",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-2_年度处理组效应")

    # Robustness coefficient plot, excluding the asinh-only row from patent panel.
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
    for axis, outcome, title in [
        (axes[0], "rd_intensity_w", "研发投入强度"),
        (axes[1], "ln_patent_invention", "发明专利申请"),
    ]:
        plot_data = robustness_table.loc[
            robustness_table["outcome"].eq(outcome)
        ].copy()
        plot_data = plot_data.iloc[::-1]
        positions = np.arange(len(plot_data))
        axis.errorbar(
            plot_data["coefficient"],
            positions,
            xerr=1.96 * plot_data["std_error"],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=3,
        )
        axis.axvline(0, color=GREY, linestyle="--", linewidth=1)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            plot_data["label"].str.replace(f"{title}：", "", regex=False),
            fontsize=8.5,
        )
        axis.set_xlabel("TWFE处理系数及95%置信区间")
        axis.set_title(title)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    fig.suptitle("主要结果的稳健性检验", y=1.01, fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "同期政策、办公地址和近邻检验使用明确标注的教学代理变量",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-3_稳健性系数图")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7))
    for axis, outcome, title in [
        (axes[0], "rd_intensity_w", "研发投入强度"),
        (axes[1], "ln_patent_invention", "发明专利申请"),
    ]:
        plot_data = leave_one.loc[leave_one["outcome"].eq(outcome)].copy()
        positions = np.arange(len(plot_data))
        axis.errorbar(
            plot_data["coefficient"],
            positions,
            xerr=1.96 * plot_data["std_error"],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=2,
            markersize=4,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(plot_data["excluded_pilot"], fontsize=8)
        axis.axvline(0, color=GREY, linestyle="--", linewidth=1)
        axis.set_xlabel("剔除单个试点后的处理系数")
        axis.set_title(title)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    fig.suptitle("留一试点影响力诊断", y=1.01, fontsize=15)
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-4_留一试点影响力诊断")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for axis, outcome, title in [
        (axes[0], "rd_intensity_w", "研发投入强度"),
        (axes[1], "ln_patent_invention", "发明专利申请"),
    ]:
        plot_data = group_effects.loc[group_effects["outcome"].eq(outcome)].copy()
        plot_data["plot_label"] = (
            plot_data["dimension"].str.replace(r"（1=.*?）", "", regex=True)
            + "："
            + plot_data["group"]
        )
        plot_data = plot_data.iloc[::-1]
        positions = np.arange(len(plot_data))
        axis.errorbar(
            plot_data["coefficient"],
            positions,
            xerr=1.96 * plot_data["std_error"],
            fmt="o",
            color=BLUE,
            ecolor=BLUE,
            capsize=3,
        )
        axis.axvline(0, color=GREY, linestyle="--", linewidth=1)
        axis.set_yticks(positions)
        axis.set_yticklabels(plot_data["plot_label"], fontsize=8.5)
        axis.set_xlabel("分组TWFE处理系数")
        axis.set_title(title)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
    fig.suptitle("预设维度的异质性估计", y=1.01, fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "是否存在组间差异以表14-8中的直接交互项检验为准",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-5_异质性系数图")

    mechanism_plot = mechanism_staggered_table.copy().iloc[::-1]
    mechanism_plot["t_stat"] = (
        mechanism_plot["att"] / mechanism_plot["std_error"]
    )
    colors = np.where(mechanism_plot["p_value"].lt(0.05), GOLD, GREY)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    positions = np.arange(len(mechanism_plot))
    axis.barh(
        positions,
        mechanism_plot["t_stat"],
        color=colors,
        edgecolor="white",
    )
    axis.axvline(0, color=INK, linewidth=1)
    axis.axvline(-1.96, color=GREY, linestyle="--", linewidth=1)
    axis.axvline(1.96, color=GREY, linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(mechanism_plot["label"])
    axis.set_xlabel("处理系数的t统计量")
    axis.set_title(
        "机制变量的估计精度",
        loc="left",
        pad=30,
        fontweight="bold",
    )
    axis.text(
        0,
        1.01,
        "虚线为|t|=1.96；金色表示5%水平显著；正式机制结果采用从未处理组的交错DID，见表14-10b",
        transform=axis.transAxes,
        color=GREY,
        fontsize=9.5,
        va="bottom",
    )
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "图14-6_机制结果")

    main_overall = overall_table.loc[
        overall_table["control_mode"].eq("never")
    ].set_index("outcome")
    rd = main_overall.loc["rd_intensity_w"]
    patent = main_overall.loc["ln_patent_invention"]
    summary = f"""# 第14章案例结果说明

> 所有数值均来自半合成教学数据，只用于演示估计、检验和写作流程，不能解释为真实人工智能试验区的政策效果。

传统TWFE教学基准显示，试验区处理与研发投入强度和发明专利申请均呈显著正相关。正式的交错DID主估计使用从未处理企业作为对照：研发投入强度总体ATT为{rd['att']:.4f}，95%置信区间为[{rd['ci_low']:.4f}, {rd['ci_high']:.4f}]；发明专利申请对数总体ATT为{patent['att']:.4f}，95%置信区间为[{patent['ci_low']:.4f}, {patent['ci_high']:.4f}]。

研发政策前动态的联合检验p值为{rd['pretrend_p_value']:.3f}，专利为{patent['pretrend_p_value']:.3f}，政策前各期也均未在5%水平显著偏离0。政策后估计随事件时间上升，符合半合成数据生成过程中预设的渐进效应。

处理起点、固定效应、疫情年份、特殊城市、政策前特征加权和教学代理检验没有改变主要结果方向。留一试点估计用于确认结果并非由单个试点范围驱动。同期政策、办公地址和空间近邻没有真实观测，相关结果只能演示代码路径，不应写成现实稳健性证据。

异质性分析分别报告组内效应和直接组间差异检验；不能依据“一组显著、另一组不显著”判断组间差异。机制部分同时保留TWFE教学基准（表14-10）与从未处理组交错DID（表14-10b）；正式解释以后者为准。政府补助和研发人员显著增加，KZ与SA融资约束指标不显著。这里应写成“证据与补助和研发人才渠道一致”，不能宣称完整中介机制已经得到识别。
"""
    (note_dir / "第14章案例结果说明.md").write_text(summary, encoding="utf-8")

    print(chapter)


if __name__ == "__main__":
    main()
