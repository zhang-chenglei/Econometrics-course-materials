"""案例三：读懂一张成绩回归表，并看清估计值的样本波动。

生成两张图（各系数的点估计与95%置信区间、换样本后 AI 系数如何波动）和两份
结果表，全部保存到本脚本同级的 output/ 文件夹。

数据与第2章完全同源：同一份数据生成过程、同一个随机种子，因此两章的数字可以直接
对照。第2章关心"AI 系数等于多少"，本章关心"这个数字有多确定"。
"""

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
TRUE_BETA_AI = 1.25
SPLIT_REPS = 5          # 重复几次随机平分，用来看样本波动的规律

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_read_regression_table.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)

ability = rng.normal(0, 1, size=N)
age = np.clip(np.round(rng.normal(20, 1.6, size=N)), 17, 26)
female = rng.integers(0, 2, size=N)

ai = 3 + 3.2 * ability + 0.25 * age - 1.2 * female + rng.normal(0, 3.5, size=N)
ai = np.clip(ai, 0, 30)
score = (
    56
    + TRUE_BETA_AI * ai
    + 4.5 * ability
    + 0.6 * age
    - 1.8 * female
    + rng.normal(0, 5.5, size=N)
)

df = pd.DataFrame(
    {"score": score, "ai": ai, "age": age, "female": female, "ability": ability}
)
XS = ["ai", "age", "female", "ability"]


def estimate(d):
    """估计完整模型，返回 AI 系数的点估计、标准误和 95% 置信区间。"""
    fit = sm.OLS(d["score"], sm.add_constant(d[XS])).fit()
    ci = fit.conf_int().loc["ai"]
    return fit.params["ai"], fit.bse["ai"], ci[0], ci[1], int(fit.nobs)


# ------------------------------------------------------------------
# 任务1、2：一张回归表
# ------------------------------------------------------------------
model = sm.OLS(df["score"], sm.add_constant(df[XS])).fit()
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
table.to_csv(OUTPUT_DIR / "ch03_regression_table.csv", encoding="utf-8-sig")

print("回归表（被解释变量：课程成绩）：")
print(table.round(5).to_string())
print(f"\n样本量：{int(model.nobs)}，R² = {model.rsquared:.4f}，"
      f"调整R² = {model.rsquared_adj:.4f}")
print(f"整体F统计量：{model.fvalue:.3f}，p = {model.f_pvalue:.4g}")
print(f"\nAI 系数的手工t值：{model.params['ai'] / model.bse['ai']:.5f}")

# ------------------------------------------------------------------
# 任务3：换一份样本，估计值会变多少
# ------------------------------------------------------------------
print("\n换样本后 AI 系数的波动：")
rows = []

b, se, lo, hi, n = estimate(df)
print(f"  全样本：n={n}, β={b:.4f}, se={se:.4f}, 95%CI=({lo:.3f}, {hi:.3f})")
rows.append(("全样本", n, b, lo, hi))

# 切法种子：随机平分本身也是随机的，不同切法会给出不同的两组。这里固定一个切法
# 以保证结果可复现，并选了一组能清楚展示"两个半样本可以差得不小、但区间都覆盖
# 同一个值"的切法；下面还会重复 5 次，报告真实的范围。
SPLIT_SEED = 20261075
split_rng = np.random.default_rng(SPLIT_SEED)
order = split_rng.permutation(N)
halves = [("随机半样本（A）", df.iloc[order[: N // 2]]),
          ("随机半样本（B）", df.iloc[order[N // 2:]])]
for label, d in halves:
    b, se, lo, hi, n = estimate(d)
    print(f"  {label}：n={n}, β={b:.4f}, se={se:.4f}, 95%CI=({lo:.3f}, {hi:.3f})")
    rows.append((label, n, b, lo, hi))

# 再换 5 组切法，观察样本波动的规律
half_betas, half_ses, covered = [], [], 0
for _ in range(SPLIT_REPS):
    perm = split_rng.permutation(N)
    for part in (perm[: N // 2], perm[N // 2:]):
        b, se, lo, hi, _ = estimate(df.iloc[part])
        half_betas.append(b)
        half_ses.append(se)
        covered += int(lo <= TRUE_BETA_AI <= hi)

half_betas = np.array(half_betas)
half_ses = np.array(half_ses)
print(f"\n  共 {len(half_betas)} 个半样本：")
print(f"    β 的范围 {half_betas.min():.4f}–{half_betas.max():.4f}，"
      f"均值 {half_betas.mean():.4f}（全样本为 {model.params['ai']:.4f}）")
print(f"    标准误范围 {half_ses.min():.4f}–{half_ses.max():.4f}；"
      f"全样本标准误 × √2 = {model.bse['ai'] * np.sqrt(2):.4f}")
print(f"    置信区间覆盖真实值 {TRUE_BETA_AI} 的个数：{covered}/{len(half_betas)}")

for label, d in [
    ("剔除 AI 使用最高 5%", df[df.ai <= df.ai.quantile(0.95)]),
    ("剔除成绩最低 5%", df[df.score >= df.score.quantile(0.05)]),
]:
    b, se, lo, hi, n = estimate(d)
    print(f"  {label}：n={n}, β={b:.4f}, se={se:.4f}, 95%CI=({lo:.3f}, {hi:.3f})")
    rows.append((label, n, b, lo, hi))

sub = pd.DataFrame(rows, columns=["sample", "n", "ai_coef", "ci_lower", "ci_upper"])
sub["covers_truth"] = (sub.ci_lower <= TRUE_BETA_AI) & (TRUE_BETA_AI <= sub.ci_upper)
sub.to_csv(OUTPUT_DIR / "ch03_sample_variation.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# 图3-3：回归表中各系数的点估计与置信区间（截距不绘制）
# ------------------------------------------------------------------
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

plot_vars = ["ai", "age", "female", "ability"]
labels = ["AI 使用时间", "年龄", "性别（女性=1）", "认知能力"]
tbl = table.loc[plot_vars]
y = np.arange(len(plot_vars))

fig, ax = plt.subplots(figsize=(8.6, 4.8))
ax.errorbar(
    tbl["coef"], y,
    xerr=[tbl["coef"] - tbl["ci_lower"], tbl["ci_upper"] - tbl["coef"]],
    fmt="o", color="#2A6F97", ecolor="#6C8EAD", capsize=4, markersize=7,
)
ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlabel("回归系数及 95% 置信区间")
ax.set_title("成绩回归表：点估计与不确定性")
fig.tight_layout()
output = OUTPUT_DIR / "ch03-fig3-regression-table-and-ci.png"
fig.savefig(output, dpi=220, bbox_inches="tight")

# ------------------------------------------------------------------
# 图3-4：换样本后 AI 系数如何波动
# ------------------------------------------------------------------
y2 = np.arange(len(sub))[::-1]
fig, ax = plt.subplots(figsize=(8.8, 4.6))
ax.errorbar(
    sub["ai_coef"], y2,
    xerr=[sub["ai_coef"] - sub["ci_lower"], sub["ci_upper"] - sub["ai_coef"]],
    fmt="o", color="#2A6F97", ecolor="#6C8EAD", capsize=4, markersize=7,
)
ax.axvspan(half_betas.min(), half_betas.max(), color="#E8EEF3", zorder=0)
ax.axvline(TRUE_BETA_AI, color="#C44536", linestyle="--", linewidth=1.6,
           label=f"真实系数 = {TRUE_BETA_AI}")
ax.set_yticks(y2, [f"{s}（n={n}）" for s, n in zip(sub["sample"], sub["n"])])
ax.set_xlabel("AI 使用时间的系数及 95% 置信区间")
ax.set_title("换一份样本，系数会变；但每次都落在可预期的范围内")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
output2 = OUTPUT_DIR / "ch03-fig4-sample-variation.png"
fig.savefig(output2, dpi=220, bbox_inches="tight")

print(f"\n图形已保存：\n  {output}\n  {output2}")
