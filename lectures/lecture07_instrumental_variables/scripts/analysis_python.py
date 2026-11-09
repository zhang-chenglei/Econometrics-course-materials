"""
第7讲工具变量案例：Python 参考分析。

读取 data/firm_iv_case.csv，完成：
1. OLS 与单工具 2SLS 对比；
2. 第一阶段强度与弱工具对比；
3. 双工具 2SLS 与过度识别检验；
4. Durbin-Wu-Hausman 内生性检验与三模型汇总。

输出写到 output/python/。点估计应与 Stata 完全一致；标准误可能因
软件默认的小样本修正不同而有细微差异。
"""

from pathlib import Path
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "firm_iv_case.csv"
OUT = ROOT / "output" / "python"
OUT.mkdir(parents=True, exist_ok=True)


def fit_ols(df: pd.DataFrame, y: str, xs: list[str]):
    X = sm.add_constant(df[xs])
    return sm.OLS(df[y], X).fit(cov_type="HC1")


def fit_iv(df: pd.DataFrame, y: str, endog: str, instruments: list[str],
           controls: list[str]):
    exog = sm.add_constant(df[controls])
    return IV2SLS(df[y], exog, df[endog], df[instruments]).fit(cov_type="robust")


def main() -> None:
    df = pd.read_csv(DATA)

    # ---------- 任务1：OLS vs IV ----------
    m_ols = fit_ols(df, "ROE", ["RD", "size", "age"])
    m_iv = fit_iv(df, "ROE", "RD", ["ResDensity"], ["size", "age"])

    vars_ = ["RD", "size", "age"]
    ols_iv = pd.DataFrame({
        "variable": vars_,
        "OLS_coef": [m_ols.params[v] for v in vars_],
        "OLS_se": [m_ols.bse[v] for v in vars_],
        "IV_coef": [m_iv.params[v] for v in vars_],
        "IV_se": [m_iv.std_errors[v] for v in vars_],
    })
    ols_iv.to_csv(OUT / "ols_vs_iv.csv", index=False, float_format="%.6f")

    # ---------- 任务2：第一阶段与弱工具 ----------
    m_first = fit_ols(df, "RD", ["ResDensity", "size", "age"])
    m_first_no_z = fit_ols(df, "RD", ["size", "age"])

    first = pd.DataFrame({
        "variable": ["const", "ResDensity", "size", "age"],
        "coef": [m_first.params[v] for v in ["const", "ResDensity", "size", "age"]],
        "se": [m_first.bse[v] for v in ["const", "ResDensity", "size", "age"]],
        "t": [m_first.tvalues[v] for v in ["const", "ResDensity", "size", "age"]],
        "p": [m_first.pvalues[v] for v in ["const", "ResDensity", "size", "age"]],
    })
    first.to_csv(OUT / "first_stage.csv", index=False, float_format="%.6f")

    # 单个被排除工具的偏 F：稳健 t 的平方，等价于单约束 Wald 检验。
    partial_f = float(m_first.tvalues["ResDensity"] ** 2)
    partial_r2 = float(m_first.rsquared - m_first_no_z.rsquared)
    pd.DataFrame({
        "partial_F": [partial_f],
        "partial_R2": [partial_r2],
        "first_stage_R2": [m_first.rsquared],
    }).to_csv(OUT / "first_stage_diagnostics.csv", index=False,
              float_format="%.6f")

    # 弱工具对照：数据中固定保存的纯噪声工具。
    m_first_weak = fit_ols(df, "RD", ["noise_iv", "size", "age"])
    weak_f = float(m_first_weak.tvalues["noise_iv"] ** 2)
    m_iv_weak = fit_iv(df, "ROE", "RD", ["noise_iv"], ["size", "age"])

    weak = pd.DataFrame({
        "instrument": ["noise_iv (weak)", "ResDensity (strong)"],
        "RD_coef": [m_iv_weak.params["RD"], m_iv.params["RD"]],
        "RD_se": [m_iv_weak.std_errors["RD"], m_iv.std_errors["RD"]],
        "CI_low": [m_iv_weak.conf_int().loc["RD", "lower"],
                   m_iv.conf_int().loc["RD", "lower"]],
        "CI_high": [m_iv_weak.conf_int().loc["RD", "upper"],
                    m_iv.conf_int().loc["RD", "upper"]],
        "first_stage_F": [weak_f, partial_f],
    })
    weak.to_csv(OUT / "weak_vs_strong.csv", index=False, float_format="%.6f")

    # ---------- 任务3：双工具与过度识别 ----------
    m_iv2 = fit_iv(df, "ROE", "RD", ["ResDensity", "University"],
                   ["size", "age"])
    single_two = pd.DataFrame({
        "variable": vars_,
        "single_IV_coef": [m_iv.params[v] for v in vars_],
        "single_IV_se": [m_iv.std_errors[v] for v in vars_],
        "two_IV_coef": [m_iv2.params[v] for v in vars_],
        "two_IV_se": [m_iv2.std_errors[v] for v in vars_],
    })
    single_two.to_csv(OUT / "single_vs_two_iv.csv", index=False,
                      float_format="%.6f")

    overid = m_iv2.wooldridge_overid
    pd.DataFrame({
        "statistic": [float(overid.stat)],
        "df": [int(overid.df)],
        "p_value": [float(overid.pval)],
    }).to_csv(OUT / "overid.csv", index=False, float_format="%.6f")

    # ---------- 任务4：DWH 与汇总 ----------
    first_resid = df["RD"] - m_first.predict(sm.add_constant(
        df[["ResDensity", "size", "age"]]))
    dwh_df = df[["ROE", "RD", "size", "age"]].copy()
    dwh_df["first_stage_resid"] = first_resid
    m_dwh = fit_ols(dwh_df, "ROE", ["RD", "size", "age", "first_stage_resid"])
    dwh_t = float(m_dwh.tvalues["first_stage_resid"])
    dwh_p = float(m_dwh.pvalues["first_stage_resid"])
    pd.DataFrame({
        "t": [dwh_t],
        "p_value": [dwh_p],
        "F": [dwh_t ** 2],
        "df": [1],
    }).to_csv(OUT / "dwh.csv", index=False, float_format="%.6f")

    summary = pd.DataFrame({
        "variable": vars_,
        "OLS_coef": [m_ols.params[v] for v in vars_],
        "OLS_se": [m_ols.bse[v] for v in vars_],
        "IV1_coef": [m_iv.params[v] for v in vars_],
        "IV1_se": [m_iv.std_errors[v] for v in vars_],
        "IV2_coef": [m_iv2.params[v] for v in vars_],
        "IV2_se": [m_iv2.std_errors[v] for v in vars_],
        "N": [int(m_ols.nobs)] * len(vars_),
    })
    summary.to_csv(OUT / "summary.csv", index=False, float_format="%.6f")

    print("=== OLS vs IV ===")
    print(ols_iv.round(4).to_string(index=False))
    print("\n=== 第一阶段诊断 ===")
    print(f"partial F = {partial_f:.3f}, partial R2 = {partial_r2:.4f}")
    print("\n=== 弱工具 vs 强工具 ===")
    print(weak.round(4).to_string(index=False))
    print("\n=== 单工具 vs 双工具 ===")
    print(single_two.round(4).to_string(index=False))
    print("\n=== 过度识别检验 ===")
    print(overid)
    print("\n=== DWH 内生性检验 ===")
    print(f"t = {dwh_t:.4f}, p = {dwh_p:.6f}")
    print(f"\nwritten to: {OUT}")


if __name__ == "__main__":
    main()
