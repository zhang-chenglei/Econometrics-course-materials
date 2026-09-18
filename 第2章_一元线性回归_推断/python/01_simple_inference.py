"""第2章案例：读懂一元回归表，并观察斜率的样本波动。"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

SEED = 20260914
N = 800
TRUE_BETA = 1.20
SPLIT_REPS = 5

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
ai = np.clip(rng.normal(7.5, 3.0, size=N), 0, 18)
score = 70 + TRUE_BETA * ai + rng.normal(0, 5.5, size=N)
df = pd.DataFrame({"score": score, "ai": ai})


def fit_model(data: pd.DataFrame):
    return sm.OLS(data["score"], sm.add_constant(data[["ai"]])).fit()


def ai_result(data: pd.DataFrame):
    fit = fit_model(data)
    lo, hi = fit.conf_int().loc["ai"]
    return int(fit.nobs), fit.params["ai"], fit.bse["ai"], lo, hi


model = fit_model(df)
ci = model.conf_int(alpha=0.05)
table = pd.DataFrame(
    {
        "coef": model.params,
        "std_err": model.bse,
        "t_value": model.tvalues,
        "p_value": model.pvalues,
        "ci_lower": ci[0],
        "ci_upper": ci[1],
    }
)
table.to_csv(OUTPUT_DIR / "ch02_regression_table.csv", encoding="utf-8-sig")

print("一元回归表（被解释变量：课程成绩）：")
print(table.round(5).to_string())
print(f"\n样本量：{int(model.nobs)}，R²={model.rsquared:.4f}")
print(f"AI斜率手工t值：{model.params['ai'] / model.bse['ai']:.5f}")

rows = []
n, b, se, lo, hi = ai_result(df)
rows.append(("全样本", n, b, se, lo, hi))

split_rng = np.random.default_rng(20261075)
order = split_rng.permutation(N)
for label, idx in [
    ("随机半样本（A）", order[: N // 2]),
    ("随机半样本（B）", order[N // 2 :]),
]:
    n, b, se, lo, hi = ai_result(df.iloc[idx])
    rows.append((label, n, b, se, lo, hi))

half_betas, half_ses, covered = [], [], 0
for _ in range(SPLIT_REPS):
    perm = split_rng.permutation(N)
    for idx in (perm[: N // 2], perm[N // 2 :]):
        _, b, se, lo, hi = ai_result(df.iloc[idx])
        half_betas.append(b)
        half_ses.append(se)
        covered += int(lo <= TRUE_BETA <= hi)

sub = pd.DataFrame(rows, columns=["sample", "n", "ai_coef", "std_err", "ci_lower", "ci_upper"])
sub["covers_truth"] = (sub.ci_lower <= TRUE_BETA) & (TRUE_BETA <= sub.ci_upper)
sub.to_csv(OUTPUT_DIR / "ch02_sample_variation.csv", index=False, encoding="utf-8-sig")

print("\n样本变化：")
print(sub.round(4).to_string(index=False))
print(f"10个半样本斜率范围：{min(half_betas):.4f}–{max(half_betas):.4f}")
print(f"半样本标准误均值：{np.mean(half_ses):.4f}；全样本标准误×√2：{model.bse['ai'] * np.sqrt(2):.4f}")
print(f"半样本置信区间覆盖真实值：{covered}/{len(half_betas)}")

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.2, 3.2))
b = model.params["ai"]
lo, hi = ci.loc["ai"]
ax.errorbar([b], [0], xerr=[[b - lo], [hi - b]], fmt="o", color="#2A6F97", capsize=5, markersize=8)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.axvline(TRUE_BETA, color="#C44536", linestyle=":", linewidth=1.5, label=f"真实斜率={TRUE_BETA}")
ax.set_yticks([0], ["AI学习时间"])
ax.set_xlabel("一元回归斜率及95%置信区间")
ax.set_title("成绩回归表：点估计与不确定性")
ax.legend(frameon=False)
fig.tight_layout()
fig1 = OUTPUT_DIR / "ch02-fig3-regression-table-and-ci.png"
fig.savefig(fig1, dpi=220, bbox_inches="tight")
plt.close(fig)

y = np.arange(len(sub))[::-1]
fig, ax = plt.subplots(figsize=(8.6, 4.0))
ax.errorbar(
    sub["ai_coef"], y,
    xerr=[sub["ai_coef"] - sub["ci_lower"], sub["ci_upper"] - sub["ai_coef"]],
    fmt="o", color="#2A6F97", ecolor="#6C8EAD", capsize=4, markersize=7,
)
ax.axvline(TRUE_BETA, color="#C44536", linestyle="--", linewidth=1.5, label=f"真实斜率={TRUE_BETA}")
ax.set_yticks(y, [f"{s}（n={n}）" for s, n in zip(sub["sample"], sub["n"])])
ax.set_xlabel("一元回归斜率及95%置信区间")
ax.set_title("换一份样本，斜率估计会变化")
ax.legend(frameon=False)
fig.tight_layout()
fig2 = OUTPUT_DIR / "ch02-fig4-sample-variation.png"
fig.savefig(fig2, dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"\n图形已保存：\n  {fig1}\n  {fig2}")
