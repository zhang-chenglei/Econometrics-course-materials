"""
第十讲：DML 与因果森林｜课堂演示完整代码
================================================
用途：
    在课堂上按步骤演示 OLS、DML 和教学版因果森林。
    所有数据都是教学模拟，不是论文原始数据，也不是赵晶等/周迪等论文的复现。

运行：
    python3 完整代码.py --step all
    python3 完整代码.py --step data
    python3 完整代码.py --step dml
    python3 完整代码.py --step forest

输出：
    data/    三份模拟数据
    outputs/ CSV、两张图、课堂结果摘要.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import norm, gaussian_kde

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
FOREST_OUT = OUT / "森林_后续"

DML_SEED = 20261109
FOREST_SEED = 20261114
XCOLS = ["x1", "x2", "x3"]
FEATS = ["large", "high_rd", "skill", "age"]


# ============================================================
# 一、生成教学模拟数据
# ============================================================
def make_dml(n=2000, seed=DML_SEED, nonlinear=True):
    """DML 模拟：真值 theta=2；nonlinear=True 时 OLS 的线性控制会遗漏非线性混杂。"""
    r = np.random.default_rng(seed)
    x1 = r.normal(size=n)
    x2 = r.normal(size=n)
    x3 = r.normal(size=n)
    sq = x1 * x1 - 1
    s = 1.0 if nonlinear else 0.0
    m = 0.65 * x1 + 0.35 * x2 + s * (0.85 * sq + 0.40 * np.sin(2 * x2))
    g0 = 1 + 0.50 * x1 - 0.20 * x3 + s * (0.85 * sq + 0.38 * np.sin(2 * x2))
    d = m + r.normal(scale=1.0, size=n)
    y = 2.0 * d + g0 + r.normal(scale=1.0, size=n)
    return pd.DataFrame({
        "id": np.arange(n) + 1,
        "x1": x1, "x2": x2, "x3": x3,
        "digitalization_D": d,
        "productivity_Y": y,
        "true_effect": 2.0,
    })


def make_forest(n=4000, seed=FOREST_SEED):
    """因果森林模拟：随机试点 W；四类企业理论均值对应 PPT 的 10/11、10/15、20/21、20/25。"""
    r = np.random.default_rng(seed)
    large = r.binomial(1, 0.5, size=n)
    high_rd = r.binomial(1, 0.5, size=n)
    skill = np.clip(r.normal(0.5 + 0.12 * high_rd, 0.15, size=n), 0, 1)
    age = np.maximum(1, r.normal(12 + 4 * large, 4, size=n))
    w = r.binomial(1, 0.5, size=n)  # 随机试点：由设计保证可比性
    y0_mean = 10 + 10 * large
    tau = 1 + 4 * high_rd
    y = y0_mean + w * tau + r.normal(0, 2.0, size=n)
    return pd.DataFrame({
        "id": np.arange(n) + 1,
        "large": large, "high_rd": high_rd,
        "skill": skill, "age": age,
        "trial_W": w,
        "productivity_Y": y,
        "true_Y0_mean": y0_mean,
        "true_CATE": tau,
    })


def generate_data(n_dml=2000, n_forest=4000, seed_dml=DML_SEED, seed_forest=FOREST_SEED):
    """生成三份 CSV，供学生先看数据。"""
    make_dml(n_dml, seed_dml, nonlinear=False).to_csv(DATA / "dml_linear.csv", index=False)
    make_dml(n_dml, seed_dml, nonlinear=True).to_csv(DATA / "dml_nonlinear.csv", index=False)
    make_forest(n_forest, seed_forest).to_csv(DATA / "causal_forest_trial.csv", index=False)
    print(f"[数据] 已生成 {DATA}/dml_linear.csv, dml_nonlinear.csv, causal_forest_trial.csv")
    print("[数据] DML 真值=2；因果森林四类企业理论效应=1,5,1,5（均值 10/11、10/15、20/21、20/25）")


# ============================================================
# 二、OLS 与 DML
# ============================================================
def ols_est(df):
    """普通 OLS：只线性控制 x1,x2,x3（设定不完全，非线性情景会偏高）。"""
    fit = sm.OLS(
        df["productivity_Y"],
        sm.add_constant(df[["digitalization_D"] + XCOLS]),
    ).fit(cov_type="HC1")
    return float(fit.params["digitalization_D"]), float(fit.bse["digitalization_D"])


def ols_correct_est(df):
    """正确设定的 OLS：显式加入 x1² 与 sin(2x2)，作为“知道真实函数形式”的对照。"""
    X = df[["digitalization_D", "x1", "x2", "x3"]].copy()
    X["x1_sq"] = df["x1"] ** 2
    X["sin_2x2"] = np.sin(2 * df["x2"])
    fit = sm.OLS(df["productivity_Y"], sm.add_constant(X)).fit(cov_type="HC1")
    return float(fit.params["digitalization_D"]), float(fit.bse["digitalization_D"])


def make_learner(name, seed, trees=60):
    """返回一个辅助学习器；DML 的 m(X) 与 g(X) 各用一份。"""
    if name == "lasso":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=5, random_state=seed, max_iter=5000)),
        ])
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=trees, min_samples_leaf=8, max_features=0.9,
            n_jobs=-1, random_state=seed,
        )
    if name == "gbm":
        return GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2,
            random_state=seed,
        )
    if name == "nn":
        return MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=400,
            early_stopping=True, random_state=seed,
        )
    raise ValueError(f"未知学习器: {name}")


def dml_est(df, learner="rf", seed=87, trees=60, folds=5):
    """部分线性 DML：5 折交叉拟合 + 两个辅助学习器 + 残差回归。

    learner 可选：lasso / rf / gbm / nn。
    """
    X = df[XCOLS].to_numpy()
    D = df["digitalization_D"].to_numpy()
    Y = df["productivity_Y"].to_numpy()
    md = np.empty(len(df))
    my = np.empty(len(df))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (train, test) in enumerate(kf.split(X)):
        m = make_learner(learner, seed + k, trees=trees)
        g = make_learner(learner, seed + k, trees=trees)
        m.fit(X[train], D[train])
        g.fit(X[train], Y[train])
        md[test] = m.predict(X[test])
        my[test] = g.predict(X[test])
    dr = D - md
    yr = Y - my
    fit = sm.OLS(yr, dr.reshape(-1, 1)).fit(cov_type="HC1")
    res = pd.DataFrame({
        "D": D, "D_hat_oof": md, "D_resid": dr,
        "Y": Y, "Y_hat_oof": my, "Y_resid": yr,
    })
    return float(fit.params[0]), float(fit.bse[0]), res


def draw_three_line_table(table):
    """把六行对比表画成规范三线表（只有顶线、表头线、底线）。"""
    plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.98, "DML 与 OLS 对比（非线性情景，真值 = 2）",
            ha="center", va="top", fontsize=14, fontweight="bold")
    x_m, x_c, x_ci = 0.03, 0.66, 0.73
    y_h = 0.82
    ax.text(x_m, y_h, "方法", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.text(x_c, y_h, "系数估计", ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(x_ci, y_h, "95% 置信区间", ha="left", va="center", fontsize=11, fontweight="bold")
    # 三线：顶线、表头线、底线
    ax.hlines([0.88, 0.76, 0.10], 0.02, 0.98,
              colors="black", linewidths=[1.3, 0.8, 1.3])
    y = 0.68
    for _, r in table.iterrows():
        ax.text(x_m, y, r["method"], ha="left", va="center", fontsize=10.5)
        ax.text(x_c, y, f"{r['coef']:.3f}", ha="right", va="center", fontsize=10.5)
        ax.text(x_ci, y, f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]",
                ha="left", va="center", fontsize=10.5)
        y -= 0.095
    ax.text(0.02, 0.03,
            "注：教学模拟；HC1 稳健标准误；DML 辅助学习器为 Lasso、随机森林、梯度提升树、神经网络。",
            ha="left", va="bottom", fontsize=8.2, color="#444444")
    fig.savefig(OUT / "dml_three_line_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_dml(n=2000, seed=DML_SEED, trees=60, folds=5, reps=50):
    """只输出两张结果：密度分布图 + 规范化三线表。"""
    # ---------- A. 一张样本上的多学习器对比表 ----------
    df_nl = make_dml(n=n, seed=seed, nonlinear=True)
    df_nl.to_csv(DATA / "dml_nonlinear.csv", index=False)

    rows = []
    o, ose = ols_est(df_nl)
    rows.append({"method": "OLS（线性设定）", "coef": o, "se_HC1": ose})
    oc, ocse = ols_correct_est(df_nl)
    rows.append({"method": "OLS（正确设定）", "coef": oc, "se_HC1": ocse})

    learner_labels = [
        ("lasso", "DML-Lasso（线性）"),
        ("rf", "DML-随机森林"),
        ("gbm", "DML-梯度提升树"),
        ("nn", "DML-神经网络"),
    ]
    for name, label in learner_labels:
        c, s, _ = dml_est(df_nl, learner=name, seed=seed, trees=trees, folds=folds)
        rows.append({"method": label, "coef": c, "se_HC1": s})
    table = pd.DataFrame(rows)
    table["CI_low"] = table["coef"] - 1.96 * table["se_HC1"]
    table["CI_high"] = table["coef"] + 1.96 * table["se_HC1"]
    draw_three_line_table(table)

    print("\n[DML] 非线性情景：不同 OLS/DML 设定对比（真值=2）")
    print(f"{'方法':<20}{'系数':>8}{'95% 置信区间':>22}")
    for _, r in table.iterrows():
        ci = f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]"
        print(f"{r['method']:<20}{r['coef']:>8.3f}{ci:>22}")
    print("[DML] OLS 线性与 DML-Lasso 仍偏高；正确设定 OLS 与灵活 DML 接近 2。")

    df_lin = make_dml(n=n, seed=seed, nonlinear=False)
    df_lin.to_csv(DATA / "dml_linear.csv", index=False)

    # ---------- B. 重复模拟：密度分布图 + 覆盖率 ----------
    method_order = ["OLS（线性设定）", "OLS（正确设定）", "DML-Lasso", "DML-随机森林",
                    "DML-梯度提升树", "DML-神经网络"]
    records = []
    for scenario in ["linear", "nonlinear"]:
        for j in range(reps):
            d = make_dml(n=n, seed=seed + j * 17, nonlinear=(scenario == "nonlinear"))
            o, ose = ols_est(d)
            oc, ocse = ols_correct_est(d)
            l, lse, _ = dml_est(d, learner="lasso", seed=seed + j, trees=trees, folds=folds)
            r, rse, _ = dml_est(d, learner="rf", seed=seed + j, trees=trees, folds=folds)
            vals = {
                "OLS（线性设定）": (o, ose),
                "OLS（正确设定）": (oc, ocse),
                "DML-Lasso": (l, lse),
                "DML-随机森林": (r, rse),
            }
            # GBM / NN 只跑非线性情景，避免左图六条线挤在一起，也控制运行时间。
            if scenario == "nonlinear":
                g, gse, _ = dml_est(d, learner="gbm", seed=seed + j, trees=trees, folds=folds)
                nnet, nse, _ = dml_est(d, learner="nn", seed=seed + j, trees=trees, folds=folds)
                vals["DML-梯度提升树"] = (g, gse)
                vals["DML-神经网络"] = (nnet, nse)
            for method, (c, s) in vals.items():
                records.append({
                    "scenario": scenario, "repeat": j + 1, "method": method,
                    "estimate": c, "se_HC1": s,
                    "CI_low": c - 1.96 * s, "CI_high": c + 1.96 * s,
                    "covers_true_2": (c - 1.96 * s <= 2 <= c + 1.96 * s),
                })
    est = pd.DataFrame(records)
    summary = est.groupby(["scenario", "method"], as_index=False).agg(
        mean_estimate=("estimate", "mean"),
        sd_estimate=("estimate", "std"),
        coverage=("covers_true_2", "mean"),
    )

    colors = {
        "OLS（线性设定）": "#5A5A5A",   # 深灰
        "OLS（正确设定）": "#1F77B4",   # 蓝
        "DML-Lasso": "#2CA02C",         # 绿
        "DML-随机森林": "#D62728",      # 红
        "DML-梯度提升树": "#9467BD",    # 紫
        "DML-神经网络": "#FF7F0E",      # 橙
    }
    linestyles = {
        "OLS（线性设定）": "--",
        "OLS（正确设定）": "-",
        "DML-Lasso": "-.",
        "DML-随机森林": "-",
        "DML-梯度提升树": "-",
        "DML-神经网络": "-",
    }
    plot_labels = {
        "OLS（线性设定）": "OLS (linear)",
        "OLS（正确设定）": "OLS (correct)",
        "DML-Lasso": "DML-Lasso",
        "DML-随机森林": "DML-Random Forest",
        "DML-梯度提升树": "DML-GBM",
        "DML-神经网络": "DML-Neural Net",
    }
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    handles = {}
    for ax, scenario in zip(axs, ["linear", "nonlinear"]):
        sub = est[est.scenario == scenario]
        for method in method_order:
            # 线性情景下 OLS 线性设定已经正确，不再重复画“OLS 正确设定”。
            if scenario == "linear" and method == "OLS（正确设定）":
                continue
            vals = sub[sub.method == method]["estimate"].to_numpy()
            if len(vals) >= 3:
                kde = gaussian_kde(vals, bw_method=.5)
                xs = np.linspace(1.65, 2.95, 300)
                line, = ax.plot(xs, kde(xs), color=colors[method], linestyle=linestyles[method],
                                linewidth=2.4, label=plot_labels[method])
                handles[method] = line
        ax.axvline(2, linestyle=":", color="black", linewidth=1.2)
        ax.set_xlim(1.7, 2.9)
        ax.set_title(scenario + " confounding")
        ax.set_xlabel("Estimated treatment effect")
    axs[0].text(
        0.98, 0.05,
        "Left: 3 methods. OLS (linear) is already correct;\n"
        "'OLS (correct)' and GBM/NN only in the right panel.",
        transform=axs[0].transAxes, fontsize=8, ha="right", va="bottom",
        color="#444444",
    )
    axs[0].set_ylabel("Density")
    order = [m for m in method_order if m in handles]
    fig.legend([handles[m] for m in order], [plot_labels[m] for m in order],
               loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle(f"True effect=2 | n={n} | repetitions={reps}")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT / "dml_density_comparison.png", dpi=180)
    plt.close(fig)

    print("\n[DML] 重复模拟摘要：")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("[DML] 覆盖率是模拟评价量；非线性下 OLS 线性/DML-Lasso 偏离真值，OLS 正确设定/DML-随机森林更接近 2。")
    return table, summary


# ============================================================
# 三、教学版诚实因果森林
# ============================================================
def leaf_tau(df, idx):
    a = df.iloc[idx]
    t = a[a.trial_W == 1].productivity_Y
    c = a[a.trial_W == 0].productivity_Y
    if min(len(t), len(c)) < 12:
        return np.nan
    return float(t.mean() - c.mean())


class CausalTreeTeaching:
    """教学版诚实因果树：一半样本选切分，另一半估计叶内效应。"""

    def __init__(self, depth=2, minleaf=80, rng=None):
        self.depth = depth
        self.minleaf = minleaf
        self.rng = rng

    def fit(self, df):
        perm = self.rng.permutation(len(df))
        h = len(perm) // 2
        self.split_df = df.iloc[perm[:h]].reset_index(drop=True)
        self.est_df = df.iloc[perm[h:]].reset_index(drop=True)
        self.root = self._build(np.arange(len(self.split_df)), 0)
        return self

    def _build(self, ix, depth):
        split = self.split_df
        if depth >= self.depth or len(ix) < 2 * self.minleaf:
            return {"leaf": True, "split_ix": ix}
        best = None
        top = -1
        for f in FEATS:
            vals = split.iloc[ix][f].to_numpy()
            thresholds = ([.5] if f in ("large", "high_rd")
                          else np.unique(np.quantile(vals, [.2, .35, .5, .65, .8])))
            for v in thresholds:
                l = ix[vals <= v]
                rr = ix[vals > v]
                if min(len(l), len(rr)) < self.minleaf:
                    continue
                tl = leaf_tau(split, l)
                tr = leaf_tau(split, rr)
                if not (np.isfinite(tl) and np.isfinite(tr)):
                    continue
                score = (len(l) * len(rr) / len(ix)) * (tl - tr) ** 2
                if score > top:
                    top = score
                    best = (f, float(v), l, rr)
        if best is None:
            return {"leaf": True, "split_ix": ix}
        f, v, l, rr = best
        return {"leaf": False, "f": f, "v": v,
                "left": self._build(l, depth + 1),
                "right": self._build(rr, depth + 1)}

    def _follow(self, x):
        node = self.root
        while not node["leaf"]:
            node = node["left"] if x[node["f"]] <= node["v"] else node["right"]
        return node

    def predict(self, X):
        keys = {}
        for j in range(len(self.est_df)):
            node = self._follow(self.est_df.iloc[j])
            key = id(node)
            keys.setdefault(key, []).append(j)
        overall = leaf_tau(self.est_df, np.arange(len(self.est_df)))
        estimates = {key: leaf_tau(self.est_df, np.asarray(ids)) for key, ids in keys.items()}
        out = []
        for _, x in X.iterrows():
            z = estimates.get(id(self._follow(x)), np.nan)
            out.append(z if np.isfinite(z) else overall)
        return np.asarray(out)

    def root_split(self):
        return ("none", None) if self.root["leaf"] else (self.root["f"], self.root["v"])


def forest_predict(train, test, trees=60, depth=2, minleaf=70, seed=FOREST_SEED):
    rng = np.random.default_rng(seed)
    estimates = []
    splits = []
    for _ in range(trees):
        ix = rng.choice(len(train), size=int(.75 * len(train)), replace=False)
        t = CausalTreeTeaching(depth, minleaf, rng).fit(train.iloc[ix].reset_index(drop=True))
        estimates.append(t.predict(test[FEATS]))
        splits.append(t.root_split()[0])
    return np.nanmean(np.vstack(estimates), axis=0), pd.Series(splits).value_counts()


def draw_forest_figure(group_df, gates, diff, root_counts):
    """因果森林直观图：左图四组估计 vs 理论；右图独立评估高低组。"""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = ["Small/Low RD", "Small/High RD", "Large/Low RD", "Large/High RD"]
    x = np.arange(4)
    est = group_df["effect"].to_numpy()
    lo = group_df["CI_low"].to_numpy()
    hi = group_df["CI_high"].to_numpy()
    colors = ["#6BAED6", "#D62728", "#6BAED6", "#D62728"]
    axs[0].bar(x, est, color=colors, alpha=.85, width=.62, edgecolor="white")
    axs[0].errorbar(x, est, yerr=[est - lo, hi - est], fmt="none",
                    ecolor="black", capsize=4, linewidth=1.2)
    theory = group_df["ppt_theory"].to_numpy()
    axs[0].scatter(x, theory, marker="D", s=55, color="black", zorder=3, label="Theory")
    axs[0].set_xticks(x, labels, fontsize=8.5)
    axs[0].set_ylabel("Estimated treatment effect")
    n_high = int(root_counts.get("high_rd", 0))
    n_all = int(root_counts.sum())
    axs[0].set_title(f"Four groups (first split: {n_high}/{n_all} on High R&D)", fontsize=10)
    axs[0].axhline(0, color="black", linewidth=.8)
    axs[0].legend(frameon=False, fontsize=8, loc="upper left")

    labels2 = ["Low predicted CATE", "High predicted CATE"]
    x2 = np.arange(2)
    est2 = gates["ATE"].to_numpy()
    lo2 = gates["CI_low"].to_numpy()
    hi2 = gates["CI_high"].to_numpy()
    axs[1].bar(x2, est2, color=["#6BAED6", "#D62728"], alpha=.85, width=.55, edgecolor="white")
    axs[1].errorbar(x2, est2, yerr=[est2 - lo2, hi2 - est2], fmt="none",
                    ecolor="black", capsize=5, linewidth=1.2)
    axs[1].set_xticks(x2, labels2, fontsize=9)
    axs[1].set_ylabel("Group ATE (95% CI)")
    axs[1].set_title("Independent holdout evaluation", fontsize=10)
    axs[1].axhline(0, color="black", linewidth=.8)
    d = diff.iloc[0]
    axs[1].text(0.5, 0.92,
                f"High - Low = {d['High_minus_Low']:.2f}\np = {d['p_value']:.1e}",
                transform=axs[1].transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))
    fig.suptitle("Causal forest: what did it learn?")
    fig.tight_layout()
    fig.savefig(FOREST_OUT / "forest_demo.png", dpi=180)
    plt.close(fig)


def draw_forest_three_line_table(group_df, gates, diff, root_counts):
    """因果森林结果的标准三线表。"""
    plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = []
    for _, r in group_df.iterrows():
        size = "小企业" if int(r["large"]) == 0 else "大企业"
        rd = "低研发" if int(r["high_rd"]) == 0 else "高研发"
        rows.append([
            f"{size}、{rd}",
            f"{r['effect']:.3f}",
            f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]",
            f"{int(r['ppt_theory'])}",
        ])
    label_map = {"Low predicted": "低预测 CATE 组", "High predicted": "高预测 CATE 组"}
    for _, r in gates.iterrows():
        rows.append([
            label_map[r["group"]],
            f"{r['ATE']:.3f}",
            f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]",
            "—",
        ])
    d = diff.iloc[0]
    rows.append([
        "高减低",
        f"{d['High_minus_Low']:.3f}",
        f"se = {d['se']:.3f}",
        f"p = {d['p_value']:.1e}",
    ])

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.98, "因果森林结果汇总（教学模拟）",
            ha="center", va="top", fontsize=14, fontweight="bold")
    x1, x2, x3, x4 = 0.03, 0.50, 0.62, 0.90
    yh = 0.82
    ax.text(x1, yh, "组别", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.text(x2, yh, "估计效应", ha="right", va="center", fontsize=11, fontweight="bold")
    ax.text(x3, yh, "95% 置信区间", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.text(x4, yh, "理论值/说明", ha="left", va="center", fontsize=11, fontweight="bold")
    ax.hlines([0.88, 0.76, 0.10], 0.02, 0.98,
              colors="black", linewidths=[1.3, 0.8, 1.3])
    y = 0.68
    for row in rows:
        ax.text(x1, y, row[0], ha="left", va="center", fontsize=10)
        ax.text(x2, y, row[1], ha="right", va="center", fontsize=10)
        ax.text(x3, y, row[2], ha="left", va="center", fontsize=10)
        ax.text(x4, y, row[3], ha="left", va="center", fontsize=10)
        y -= 0.082
    n_high = int(root_counts.get("high_rd", 0))
    n_all = int(root_counts.sum())
    ax.text(0.02, 0.03,
            f"注：四组估计与高低组检验均来自未参与森林训练的评估样本；"
            f"理论值为 1/5/1/5；第一刀 {n_high}/{n_all} 棵树按高研发切。",
            ha="left", va="bottom", fontsize=8.2, color="#444444")
    fig.savefig(FOREST_OUT / "forest_three_line_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_forest(n=4000, trees=60, seed=FOREST_SEED):
    FOREST_OUT.mkdir(exist_ok=True)
    df = make_forest(n, seed)
    df.to_csv(DATA / "causal_forest_trial.csv", index=False)

    rng = np.random.default_rng(seed + 2)
    perm = rng.permutation(len(df))
    tr = perm[:int(.6 * n)]
    ev = perm[int(.6 * n):]
    train = df.iloc[tr].copy().reset_index(drop=True)
    test = df.iloc[ev].copy().reset_index(drop=True)

    ols = smf.ols("productivity_Y ~ trial_W * high_rd + large + skill + age",
                  data=train).fit(cov_type="HC1")
    pred, root_counts = forest_predict(train, test, trees=trees, seed=seed)
    test["predicted_CATE"] = pred

    group = []
    for (large, rd), g in test.groupby(["large", "high_rd"]):
        a = g[g.trial_W == 1].productivity_Y.to_numpy()
        b = g[g.trial_W == 0].productivity_Y.to_numpy()
        effect = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        group.append({
            "large": large, "high_rd": rd, "n": len(g),
            "effect": effect, "se": se,
            "CI_low": effect - 1.96 * se,
            "CI_high": effect + 1.96 * se,
            "ppt_theory": 1 + 4 * rd,
        })
    group_df = pd.DataFrame(group).sort_values(["large", "high_rd"]).reset_index(drop=True)

    med = float(np.median(pred))
    test["predicted_high_group"] = (pred >= med).astype(int)
    ga = []
    for label, g in test.groupby("predicted_high_group"):
        y1 = g.loc[g.trial_W == 1, "productivity_Y"].to_numpy()
        y0 = g.loc[g.trial_W == 0, "productivity_Y"].to_numpy()
        eff = y1.mean() - y0.mean()
        se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
        ga.append({
            "code": int(label),
            "group": "High predicted" if label else "Low predicted",
            "n": len(g), "ATE": eff, "se": se,
            "CI_low": eff - 1.96 * se,
            "CI_high": eff + 1.96 * se,
        })
    gates = pd.DataFrame(ga).sort_values("code").drop(columns="code").reset_index(drop=True)

    e = float(gates.iloc[1].ATE - gates.iloc[0].ATE)
    se = float(np.hypot(gates.iloc[0].se, gates.iloc[1].se))
    p = float(2 * norm.sf(abs(e / se)))
    diff = pd.DataFrame([{"High_minus_Low": e, "se": se, "p_value": p}])

    draw_forest_figure(group_df, gates, diff, root_counts)
    draw_forest_three_line_table(group_df, gates, diff, root_counts)

    print("\n[FOREST] 四类企业理论效应：1, 5, 1, 5")
    print("第一刀变量分布：", root_counts.to_dict())
    print("交互项 OLS（W × HighRD）：", round(float(ols.params["trial_W:high_rd"]), 3))
    print("四组评估样本估计：\n", group_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("高低预测组：\n", gates.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"高减低 = {e:.3f}，se = {se:.3f}，p = {p:.2e}")
    print("[FOREST] CATE 是条件平均效应，不是个体真实效应；本算法是教学版，不是官方 GRF。")
    return group_df, gates, diff, root_counts


# ============================================================
# 四、课堂结果摘要
# ============================================================
def write_summary(dml_table, dml_summary, forest_groups, gates, diff, root_counts, reps):
    lines = []
    lines.append("第十讲：DML 与因果森林｜课堂演示结果摘要")
    lines.append("=" * 52)
    lines.append("说明：全部为教学模拟，不是论文原始数据，也不是论文复现。\n")
    lines.append("[一、DML] 非线性情景：不同 OLS/DML 设定对比（真值 = 2）")
    lt = dml_table.copy()
    lt["95% CI"] = lt.apply(
        lambda r: f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]", axis=1
    )
    lines.append(lt[["method", "coef", "95% CI"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"
    ))
    lines.append("")
    lines.append(f"[一、DML] {reps} 次重复模拟摘要（均值、标准差、覆盖率）")
    lines.append(dml_summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append(f"重复次数：{reps}。覆盖率是模拟评价量，不是论文的 79%。\n")
    lines.append("[二、因果森林] 四类企业理论效应 = 1, 5, 1, 5")
    lines.append(forest_groups.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")
    lines.append("高低预测组（独立评估样本）：")
    lines.append(gates.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    lines.append("")
    lines.append("高减低差异检验：")
    d = diff.iloc[0]
    lines.append(f"高减低 = {d['High_minus_Low']:.3f}，se = {d['se']:.3f}，p = {d['p_value']:.2e}")
    lines.append("")
    lines.append("第一刀变量分布：")
    lines.append(root_counts.to_string())
    lines.append("")
    lines.append("提醒：CATE 预测不等于企业真实个体处理效应；")
    lines.append("本演示用教学版诚实因果树集成，不是官方 GRF/EconML 实现。")
    (OUT / "课堂结果摘要.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[摘要] 已写入 {OUT/'课堂结果摘要.txt'}")


# ============================================================
# 五、命令行入口
# ============================================================
def main():
    p = argparse.ArgumentParser(description="第十讲 DML 与因果森林课堂演示")
    p.add_argument("--step", choices=["all", "data", "dml", "forest"], default="all")
    p.add_argument("--n_dml", type=int, default=2000)
    p.add_argument("--n_forest", type=int, default=4000)
    p.add_argument("--trees", type=int, default=60)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--seed_dml", type=int, default=DML_SEED)
    p.add_argument("--seed_forest", type=int, default=FOREST_SEED)
    a = p.parse_args()

    need_data = ((not (DATA / "dml_linear.csv").exists())
                 or (not (DATA / "causal_forest_trial.csv").exists()))
    if a.step in ("all", "data") or need_data:
        generate_data(a.n_dml, a.n_forest, a.seed_dml, a.seed_forest)

    dml_table = dml_summary = None
    forest_groups = gates = diff = root_counts = None
    if a.step in ("all", "dml"):
        dml_table, dml_summary = run_dml(n=a.n_dml, seed=a.seed_dml, trees=a.trees,
                                         folds=a.folds, reps=a.reps)
    if a.step in ("all", "forest"):
        forest_groups, gates, diff, root_counts = run_forest(
            n=a.n_forest, trees=a.trees, seed=a.seed_forest
        )
    if a.step == "all":
        write_summary(dml_table, dml_summary, forest_groups, gates, diff, root_counts, a.reps)
    print("\nDONE. 结果在 outputs/ 文件夹。\n")


if __name__ == "__main__":
    main()
