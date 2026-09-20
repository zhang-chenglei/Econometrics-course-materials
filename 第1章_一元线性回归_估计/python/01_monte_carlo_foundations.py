"""案例一：从样本到总体——大数定律、中心极限定理与OLS的重复抽样。

生成三张图（大数定律、中心极限定理、OLS斜率的抽样分布）和一份斜率汇总表，
全部保存到本脚本同级的 output/ 文件夹。

三个任务共用一条逻辑：样本量越大，样本信息越接近总体；把抽样这个过程重复
多次，又能看出估计量本身的分布。全部使用教学用模拟数据，总体与真实参数由
数据生成过程人为设定并已知。
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm

SEED = 20260914
REPS = 3000

LLN_SIZES = [10, 100, 1000, 10000]
CLT_SIZES = [1, 5, 30, 100, 500]
POP_LOW, POP_HIGH = 0.0, 10.0

OLS_SIZES = [30, 100, 500]
BETA0, BETA1 = 2.0, 0.5

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
# 需要把生成的图汇入教材插图库时，运行时指定：
#   CASE_OUTPUT_DIR=<仓库>/30_教学/09_计量教材/图片/教材插图 python 01_monte_carlo_foundations.py
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------
# 任务1：大数定律——样本量越大，样本分布越接近总体
# 总体为标准正态分布 N(0,1)，总体均值 0。
# ---------------------------------------------------------------
lln_samples = {}
print("任务1 大数定律（总体 N(0,1)，总体均值 0）：")
for n in LLN_SIZES:
    z = rng.normal(0, 1, size=n)
    lln_samples[n] = z
    print(f"  n={n:6d}：样本均值 {z.mean():+.4f}，样本标准差 {z.std(ddof=1):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, n in zip(axes.ravel(), LLN_SIZES):
    ax.hist(lln_samples[n], bins=30, density=True, color="#9FB8CC", edgecolor="white")
    grid = np.linspace(-4, 4, 300)
    ax.plot(grid, norm.pdf(grid), color="#2A6F97", linewidth=1.8)
    ax.axvline(0, color="#C44536", linestyle="--", linewidth=1.2)
    ax.set(title=f"n = {n}", xlabel="观测值", ylabel="密度", xlim=(-4, 4))
fig.suptitle("大数定律：样本量越大，样本分布越接近总体分布", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig6-large-numbers.png", dpi=220, bbox_inches="tight")

# ---------------------------------------------------------------
# 任务2：中心极限定理——样本均值的分布随样本量增加趋近正态
# 总体为均匀分布 U(0,10)，明显不是正态分布。
# ---------------------------------------------------------------
pop_mean = (POP_LOW + POP_HIGH) / 2
pop_sd = (POP_HIGH - POP_LOW) / np.sqrt(12)

clt_means = {}
print(f"\n任务2 中心极限定理（总体 U(0,10)，总体均值 {pop_mean:.1f}）：")
for n in CLT_SIZES:
    means = rng.uniform(POP_LOW, POP_HIGH, size=(REPS, n)).mean(axis=1)
    clt_means[n] = means
    print(
        f"  n={n:4d}：样本均值的均值 {means.mean():.4f}，"
        f"标准差 {means.std(ddof=1):.4f}，理论标准差 {pop_sd / np.sqrt(n):.4f}"
    )

fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
axes = axes.ravel()

population = rng.uniform(POP_LOW, POP_HIGH, size=20000)
axes[0].hist(population, bins=40, density=True, color="#B9B9B9", edgecolor="white")
axes[0].set(title="总体分布 U(0,10)", xlabel="数值", ylabel="密度", xlim=(0, 10))

grid_clt = np.linspace(0, 10, 400)
for ax, n in zip(axes[1:], CLT_SIZES):
    values = clt_means[n]
    ax.hist(values, bins=40, density=True, color="#9FB8CC", edgecolor="white")
    ax.plot(grid_clt, norm.pdf(grid_clt, pop_mean, pop_sd / np.sqrt(n)),
            color="#C44536", linewidth=1.8)
    ax.axvline(pop_mean, color="#2A6F97", linestyle="--", linewidth=1.2)
    ax.set(title=f"样本均值分布 n = {n}", xlabel="样本均值", ylabel="密度",
           xlim=(0, 10), ylim=(0, None))
fig.suptitle("中心极限定理：总体不是正态，样本均值的分布却随样本量增加趋近正态", fontsize=14)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig7-central-limit.png", dpi=220, bbox_inches="tight")

# ---------------------------------------------------------------
# 任务3：把重复抽样思想用到OLS——斜率的抽样分布
# 真实模型 y = 2 + 0.5x + u，x ~ U(0,10)，u ~ N(0,1)。
# ---------------------------------------------------------------
print(f"\n任务3 OLS斜率的重复抽样（真实斜率 {BETA1}）：")
ols_records = []
for n in OLS_SIZES:
    for _ in range(REPS):
        x = rng.uniform(0, 10, size=n)
        u = rng.normal(0, 1, size=n)
        y = BETA0 + BETA1 * x + u
        slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
        ols_records.append({"n": n, "beta1_hat": slope})

ols_results = pd.DataFrame(ols_records)
ols_summary = (
    ols_results.groupby("n")["beta1_hat"]
    .agg(mean="mean", std="std")
    .assign(bias=lambda d: d["mean"] - BETA1)
)
print(ols_summary.round(5).to_string())
ols_summary.to_csv(OUTPUT_DIR / "ch01_summary.csv", encoding="utf-8-sig")

colors = {30: "#E07A5F", 100: "#3D85C6", 500: "#2A9D8F"}
beta_grid = np.linspace(
    ols_results.beta1_hat.quantile(0.002), ols_results.beta1_hat.quantile(0.998), 500
)
fig, ax = plt.subplots(figsize=(8, 4.8))
for n in OLS_SIZES:
    values = ols_results.loc[ols_results.n == n, "beta1_hat"]
    ax.plot(beta_grid, gaussian_kde(values)(beta_grid), label=f"n = {n}", color=colors[n])
ax.axvline(BETA1, color="black", linestyle="--", label=f"真实斜率 = {BETA1}")
ax.set(title="OLS斜率的抽样分布：一次估计会波动，样本越大越集中",
       xlabel="斜率估计值", ylabel="密度")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "ch01-fig8-ols-sampling-distribution.png", dpi=220, bbox_inches="tight")

print(f"\n图形已保存至：{OUTPUT_DIR}")
