from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import chi2, norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


PACKAGE_DIR = Path(__file__).resolve().parents[1]

_DATA_REL = Path("data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv")
_POLICY_REL = Path("data/policy/case_policy_approvals.csv")


def _resolve_materials_dir() -> Path:
    """定位 data/ 所在的资源根目录。

    仓库内布局：教材根/GitHub课程网站/materials/{code,data,...}
    学生包布局：包根/{code,data,...}
    """
    candidates = [
        candidate
        for base in (PACKAGE_DIR, *PACKAGE_DIR.parents)
        for candidate in (base, base / "GitHub课程网站/materials")
    ]
    for candidate in candidates:
        if (candidate / _DATA_REL).is_file():
            return candidate
    raise FileNotFoundError(
        f"找不到半合成教学数据 {_DATA_REL}；"
        "请确认 data/ 与 code/ 的相对位置未被改动。"
    )


MATERIALS_DIR = _resolve_materials_dir()
DATA_PATH = MATERIALS_DIR / _DATA_REL
POLICY_PATH = MATERIALS_DIR / _POLICY_REL

MAIN_CONTROLS = ["size", "lev", "roa", "growth", "fixed", "cashflow", "firm_age"]
CITY_CONTROLS = [
    "c_gdp",
    "c_gdp_pc",
    "c_gdp_growth",
    "c_gdp_share_tertiary",
    "c_sci_tech_exp",
    "c_telecom_revenue",
    "c_mobile_subscribers",
]
OUTCOME_LABELS = {
    "rd_intensity_w": "研发投入强度",
    "ln_patent_invention": "发明专利申请",
    "patent_asinh": "发明专利申请（反双曲正弦）",
    "ln_gov_subsidy": "政府补助",
    "kz_index": "KZ融资约束",
    "sa_index": "SA融资约束",
    "ln_rd_staff": "研发人员",
}

BLUE = "#245B84"
GOLD = "#C69214"
ORANGE = "#D97732"
INK = "#24313A"
GREY = "#747F87"
LIGHT_GREY = "#D7DDE2"
OPEN_BLUE = "#DCEAF4"


def setup_plot_style() -> None:
    font_path = Path("/Library/Fonts/Arial Unicode.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.titleweight": "normal",
            "axes.grid": True,
            "grid.color": LIGHT_GREY,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "font.size": 11,
        }
    )


def chapter_dir(chapter: int) -> Path:
    path = PACKAGE_DIR / f"第{chapter}章"
    for subdir in ["代码", "表格", "图形", "说明"]:
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path


def load_data(add_teaching_extensions: bool = True) -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH, low_memory=False)
    assert data["is_semisynthetic"].eq(1).all()
    assert data.duplicated(["firm_id", "year"]).sum() == 0
    if add_teaching_extensions:
        data = add_robustness_extensions(data)
    data["patent_asinh"] = np.arcsinh(data["patent_app_invention"])
    return data


def load_policy() -> pd.DataFrame:
    return pd.read_csv(POLICY_PATH, low_memory=False)


def add_robustness_extensions(data: pd.DataFrame) -> pd.DataFrame:
    """Create transparent teaching-only proxies for unavailable extensions.

    These fields are not factual overlapping-policy, distance, or office-address
    data. They allow the code path described in Chapter 13 to be demonstrated and
    are always labelled with the ``teach_`` prefix.
    """
    result = data.copy()
    city_numeric = pd.to_numeric(result["analysis_city_id"], errors="coerce")
    result["teach_overlap_policy"] = (
        city_numeric.mod(5).eq(0) & result["year"].ge(2018)
    ).astype(int)

    province_has_pilot = result.groupby("assign_province")["did_treated"].transform(
        "max"
    )
    result["teach_neighbor_proxy"] = (
        province_has_pilot.eq(1) & result["did_treated"].eq(0)
    ).astype(int)

    firm_number = result["firm_id"].str[1:].astype(int)
    office_shift = firm_number.mod(20).eq(0) & result["did_treated"].eq(1)
    alternative_start = result["policy_first_full_year"] + office_shift.astype(int)
    result["teach_did_post_office"] = np.where(
        result["did_treated"].eq(1),
        (result["year"] >= alternative_start).astype(int),
        0,
    )
    return result


def export_table(
    frame: pd.DataFrame,
    base_path: Path,
    index: bool = False,
    markdown: bool = True,
) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(base_path.with_suffix(".csv"), index=index, encoding="utf-8-sig")
    if markdown:
        base_path.with_suffix(".md").write_text(
            frame.to_markdown(index=index),
            encoding="utf-8",
        )


def write_json(payload: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fit_twfe(
    data: pd.DataFrame,
    outcome: str,
    treatment: str = "did_post_full",
    controls: list[str] | None = None,
    label: str | None = None,
    industry_year_effects: bool = False,
    weights: pd.Series | None = None,
) -> tuple[dict, object]:
    controls = controls or []
    columns = [
        "firm_id",
        "year",
        "analysis_city_id",
        "industry_group",
        outcome,
        treatment,
        *controls,
    ]
    panel = data[columns].dropna().copy()
    if weights is not None:
        panel["_weight"] = weights.reindex(panel.index)
        panel = panel.dropna(subset=["_weight"])
    panel["industry_year"] = (
        panel["industry_group"].astype(str) + "_" + panel["year"].astype(str)
    )
    panel = panel.set_index(["firm_id", "year"]).sort_index()
    regressors = [treatment, *controls]
    kwargs = {
        "entity_effects": True,
        "drop_absorbed": True,
        "check_rank": False,
    }
    if industry_year_effects:
        kwargs["other_effects"] = panel[["industry_year"]]
    else:
        kwargs["time_effects"] = True
    model = PanelOLS(
        panel[outcome],
        panel[regressors],
        weights=panel["_weight"] if weights is not None else None,
        **kwargs,
    )
    result = model.fit(
        cov_type="clustered",
        clusters=panel[["analysis_city_id"]],
    )
    coefficient = float(result.params[treatment])
    standard_error = float(result.std_errors[treatment])
    row = {
        "label": label or f"{OUTCOME_LABELS.get(outcome, outcome)}：{treatment}",
        "outcome": outcome,
        "treatment": treatment,
        "coefficient": coefficient,
        "std_error": standard_error,
        "p_value": float(result.pvalues[treatment]),
        "ci_low": coefficient - 1.96 * standard_error,
        "ci_high": coefficient + 1.96 * standard_error,
        "observations": int(result.nobs),
        "firms": int(panel.index.get_level_values("firm_id").nunique()),
        "cities": int(panel["analysis_city_id"].nunique()),
        "controls": "、".join(controls) if controls else "无",
        "firm_fe": "是",
        "time_fe": "行业×年份" if industry_year_effects else "年份",
    }
    return row, result


def fit_interaction(
    data: pd.DataFrame,
    outcome: str,
    group_variable: str,
    treatment: str = "did_post_full",
    controls: list[str] | None = None,
    label: str | None = None,
) -> dict:
    controls = controls or []
    frame = data.copy()
    interaction = f"{treatment}_x_{group_variable}"
    frame[interaction] = frame[treatment] * frame[group_variable]
    columns = [
        "firm_id",
        "year",
        "analysis_city_id",
        outcome,
        treatment,
        interaction,
        *controls,
    ]
    panel = frame[columns].dropna().set_index(["firm_id", "year"]).sort_index()
    regressors = [treatment, interaction, *controls]
    result = PanelOLS(
        panel[outcome],
        panel[regressors],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
        check_rank=False,
    ).fit(
        cov_type="clustered",
        clusters=panel[["analysis_city_id"]],
    )
    base_effect = float(result.params[treatment])
    difference = float(result.params[interaction])
    covariance = result.cov
    group1_variance = (
        covariance.loc[treatment, treatment]
        + covariance.loc[interaction, interaction]
        + 2 * covariance.loc[treatment, interaction]
    )
    group1_se = float(np.sqrt(group1_variance))
    group1_effect = base_effect + difference
    return {
        "dimension": label or group_variable,
        "outcome": outcome,
        "group0_effect": base_effect,
        "group0_se": float(result.std_errors[treatment]),
        "group1_effect": group1_effect,
        "group1_se": group1_se,
        "difference": difference,
        "difference_se": float(result.std_errors[interaction]),
        "difference_p_value": float(result.pvalues[interaction]),
        "observations": int(result.nobs),
        "firms": int(panel.index.get_level_values("firm_id").nunique()),
    }


def group_time_att(
    data: pd.DataFrame,
    outcome: str,
    cohort: int,
    time: int,
    control_mode: str = "never",
) -> dict:
    baseline = int(cohort - 1)
    if control_mode == "never":
        control_mask = data["gvar"].eq(0)
    elif control_mode == "not_yet":
        control_mask = data["gvar"].eq(0) | data["gvar"].gt(time)
    else:
        raise ValueError(f"Unknown control_mode: {control_mode}")
    selection = data["gvar"].eq(cohort) | control_mask
    subset = data.loc[
        selection & data["year"].isin([baseline, time]),
        ["firm_id", "year", "analysis_city_id", "gvar", outcome],
    ].dropna()
    wide = subset.pivot(index="firm_id", columns="year", values=outcome).dropna()
    meta = (
        subset.drop_duplicates("firm_id")
        .set_index("firm_id")[["analysis_city_id", "gvar"]]
        .loc[wide.index]
    )
    delta = wide[time] - wide[baseline]
    treated = meta["gvar"].eq(cohort)
    mean_treated = delta[treated].mean()
    mean_control = delta[~treated].mean()
    att = float(mean_treated - mean_control)
    n_treated = int(treated.sum())
    n_control = int((~treated).sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError("Empty treated or control group")

    influence: dict[int | str, float] = {}
    treated_contribution = (
        (delta[treated] - mean_treated) / n_treated
    ).groupby(meta.loc[treated, "analysis_city_id"]).sum()
    control_contribution = (
        -(delta[~treated] - mean_control) / n_control
    ).groupby(meta.loc[~treated, "analysis_city_id"]).sum()
    for city, value in treated_contribution.items():
        influence[city] = influence.get(city, 0.0) + float(value)
    for city, value in control_contribution.items():
        influence[city] = influence.get(city, 0.0) + float(value)
    return {
        "cohort": int(cohort),
        "time": int(time),
        "event_time": int(time - cohort),
        "att": att,
        "n_treated": n_treated,
        "n_control": n_control,
        "influence": influence,
    }


def aggregate_att(rows: list[dict]) -> tuple[float, float, dict]:
    weights = np.array([row["n_treated"] for row in rows], dtype=float)
    weights /= weights.sum()
    estimate = float(sum(w * row["att"] for w, row in zip(weights, rows)))
    cities = sorted(set().union(*(set(row["influence"]) for row in rows)))
    cluster_if = np.array(
        [
            sum(
                w * row["influence"].get(city, 0.0)
                for w, row in zip(weights, rows)
            )
            for city in cities
        ]
    )
    correction = len(cities) / (len(cities) - 1)
    standard_error = float(np.sqrt(correction * np.sum(cluster_if**2)))
    return estimate, standard_error, dict(zip(cities, cluster_if))


def staggered_did(
    data: pd.DataFrame,
    outcome: str,
    control_mode: str = "never",
    event_window: tuple[int, int] = (-5, 4),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohorts = sorted(int(value) for value in data.loc[data["gvar"].gt(0), "gvar"].unique())
    raw_rows = []
    for cohort in cohorts:
        for event_time in range(event_window[0], event_window[1] + 1):
            if event_time == -1:
                continue
            time = cohort + event_time
            if time < data["year"].min() or time > data["year"].max():
                continue
            raw_rows.append(
                group_time_att(
                    data,
                    outcome,
                    cohort,
                    time,
                    control_mode=control_mode,
                )
            )

    event_rows = []
    event_influences = {}
    for event_time in sorted({row["event_time"] for row in raw_rows}):
        selected = [row for row in raw_rows if row["event_time"] == event_time]
        estimate, standard_error, influence = aggregate_att(selected)
        event_influences[event_time] = influence
        event_rows.append(
            {
                "outcome": outcome,
                "control_mode": control_mode,
                "event_time": event_time,
                "att": estimate,
                "std_error": standard_error,
                "p_value": float(2 * norm.sf(abs(estimate / standard_error))),
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "annual_cohorts": len(selected),
                "treated_pairs": int(sum(row["n_treated"] for row in selected)),
            }
        )
    event_table = pd.DataFrame(event_rows)

    post_rows = [row for row in raw_rows if row["event_time"] >= 0]
    overall_att, overall_se, _ = aggregate_att(post_rows)
    overall = {
        "outcome": outcome,
        "control_mode": control_mode,
        "att": overall_att,
        "std_error": overall_se,
        "p_value": float(2 * norm.sf(abs(overall_att / overall_se))),
        "ci_low": overall_att - 1.96 * overall_se,
        "ci_high": overall_att + 1.96 * overall_se,
        "annual_cohorts": len(cohorts),
        "observations": len(data),
        "firms": data["firm_id"].nunique(),
        "cities": data["analysis_city_id"].nunique(),
    }

    cohort_rows = []
    for cohort in cohorts:
        selected = [
            row
            for row in raw_rows
            if row["cohort"] == cohort and row["event_time"] >= 0
        ]
        estimate, standard_error, _ = aggregate_att(selected)
        cohort_rows.append(
            {
                "outcome": outcome,
                "control_mode": control_mode,
                "cohort": cohort,
                "att": estimate,
                "std_error": standard_error,
                "p_value": float(2 * norm.sf(abs(estimate / standard_error))),
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "post_periods": len(selected),
            }
        )
    cohort_table = pd.DataFrame(cohort_rows)

    pre_events = sorted(
        int(value) for value in event_table.loc[event_table["event_time"].le(-2), "event_time"]
    )
    cities = sorted(
        set().union(*(set(event_influences[event]) for event in pre_events))
    )
    psi = np.array(
        [
            [
                event_influences[event].get(city, 0.0)
                for event in pre_events
            ]
            for city in cities
        ]
    )
    covariance = len(cities) / (len(cities) - 1) * psi.T @ psi
    beta = (
        event_table.set_index("event_time")
        .loc[pre_events, "att"]
        .to_numpy()
    )
    statistic = float(beta @ np.linalg.pinv(covariance) @ beta)
    overall["pretrend_chi2"] = statistic
    overall["pretrend_df"] = len(beta)
    overall["pretrend_p_value"] = float(chi2.sf(statistic, len(beta)))
    overall["all_individual_pre_p_gt_005"] = bool(
        (
            event_table.loc[event_table["event_time"].le(-2), "p_value"]
            > 0.05
        ).all()
    )
    return event_table, cohort_table, overall


def standardized_difference(
    treated: pd.Series,
    control: pd.Series,
) -> float:
    numerator = treated.mean() - control.mean()
    denominator = np.sqrt((treated.var() + control.var()) / 2)
    return float(numerator / denominator) if denominator > 0 else np.nan


def significance_stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
