"""
第9章 现代因果推断与准实验方法 — 案例实践（B部分：断点回归）
配套教材：《计量经济学：理论与实践》

内容：生成虚构退休资格与消费数据，依次完成两个RDD任务——
      任务3：资格规则对结果的简化式跳跃
      任务4：模糊断点回归与稳健性检验（Fuzzy RDD）
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据生成：虚构的60岁退休资格规则与消费
# 2000个体，年龄按连续值记录
# ============================================================
np.random.seed(20250915)
n_rdd = 2000

age = np.random.uniform(55, 70, n_rdd)
running = age - 60
D = (age >= 60).astype(int)

# 模糊断点：退休概率随年龄平滑上升，并在60岁额外跳跃
# 不设置65岁的第二个断点，避免污染较宽带宽下的60岁估计
retired_prob = np.clip(0.1 + 0.01*running + 0.7*D, 0.02, 0.98)
retired = np.random.binomial(1, retired_prob)

# 控制变量
# 处理前收入：在断点处平滑，且不由退休状态决定
income = np.random.normal(80000, 20000, n_rdd) - 3000*running
education = np.random.normal(12, 3, n_rdd)
health = np.clip(80 - 0.5*(age-55) + np.random.normal(0, 5, n_rdd), 10, 100)
family_sz = np.clip(np.round(np.random.normal(3, 1, n_rdd)), 1, 6)
urban = np.random.binomial(1, 0.6, n_rdd)

# DGP：年消费额（退休导致消费下降约4000）
consumption = (50000 + 500*(age-55) + 0.3*income + 200*education
               + 50*health - 500*family_sz + 2000*urban
               - 4000*retired + np.random.normal(0, 5000, n_rdd))

df_rdd = pd.DataFrame({
    'consumption': consumption, 'age': age, 'running': running,
    'D': D, 'retired': retired, 'income': income, 'education': education,
    'health': health, 'family_sz': family_sz, 'urban': urban
})

print('========== B部分：断点回归（RDD）==========')
print(f'个体数: {n_rdd}, 年龄范围: 55-70, 断点: 60岁')
print(f'60+比例: {D.mean():.2f}, 退休比例: {retired.mean():.2f}')

# ――――――――――――――――――――――――――――――――――――――――――――
# 任务3：资格规则对结果的简化式跳跃
# ――――――――――――――――――――――――――――――――――――――――――――
print('\n--- 任务3：资格规则对结果的简化式跳跃 ---')

# 不同带宽下的简化式RDD
reduced_rows = []
for h in [3, 5, 7]:
    df_bw = df_rdd[abs(df_rdd['running']) <= h].copy()
    df_bw['running_D'] = df_bw['running'] * df_bw['D']
    X_s = sm.add_constant(df_bw[['running', 'D', 'running_D']])
    m_sharp = sm.OLS(df_bw['consumption'], X_s).fit(cov_type='HC1')
    print(f'  带宽 h={h}: D系数={m_sharp.params["D"]:.2f} (se={m_sharp.bse["D"]:.2f})')
    reduced_rows.append({'带宽': f'h={h}', 'D系数(简化式)': m_sharp.params['D'],
                         '标准误': m_sharp.bse['D'], 'p值': m_sharp.pvalues['D'],
                         '样本量': int(m_sharp.nobs)})
pd.DataFrame(reduced_rows).round(4).to_csv(
    OUTPUT_DIR / "ch09_rdd_reduced_form.csv", index=False, encoding="utf-8-sig")

print('\n  注意：D系数是资格规则对结果的简化式跳跃')
print('  因为退休并不完全由门槛决定，它不是退休本身的处理效应')

# ――――――――――――――――――――――――――――――――――――――――――――
# 任务4：Fuzzy RDD
# ――――――――――――――――――――――――――――――――――――――――――――
print('\n--- 任务4：Fuzzy RDD ---')

fuzzy_rows = []
for h in [3, 5, 7]:
    df_bw = df_rdd[abs(df_rdd['running']) <= h].copy()
    df_bw['running_D'] = df_bw['running'] * df_bw['D']

    # 第一阶段
    X_s1 = sm.add_constant(df_bw[['running', 'D', 'running_D']])
    m_s1 = sm.OLS(df_bw['retired'], X_s1).fit()
    D_jump = m_s1.params['D']

    # 第二阶段（2SLS）
    exog = sm.add_constant(df_bw[['running', 'running_D']])
    m_fuzzy = IV2SLS(
        df_bw['consumption'],
        exog,
        df_bw['retired'],
        df_bw['D']
    ).fit(cov_type='robust')

    l_retired = m_fuzzy.params.get('retired', np.nan)
    print(f'  带宽 h={h}: 第一阶段D跳跃={D_jump:.3f}, LATE(退休)={l_retired:.1f}')
    fuzzy_rows.append({'带宽': f'h={h}', '第一阶段D跳跃': D_jump,
                       'LATE(退休)': l_retired,
                       'LATE标准误': m_fuzzy.std_errors.get('retired', np.nan)})
pd.DataFrame(fuzzy_rows).round(4).to_csv(
    OUTPUT_DIR / "ch09_rdd_fuzzy_late.csv", index=False, encoding="utf-8-sig")

# 加入控制变量
df_bw5 = df_rdd[abs(df_rdd['running']) <= 5].copy()
df_bw5['running_D'] = df_bw5['running'] * df_bw5['D']
exog_ctrl = sm.add_constant(
    df_bw5[['running', 'running_D', 'income', 'education',
            'health', 'family_sz', 'urban']])
m_fuzzy_ctrl = IV2SLS(
    df_bw5['consumption'],
    exog_ctrl,
    df_bw5['retired'],
    df_bw5['D']
).fit(cov_type='robust')
print(f'\n  Fuzzy RDD + 控制变量 (h=5): LATE(退休)={m_fuzzy_ctrl.params.get("retired", np.nan):.1f}')

# 协变量平滑性检验
print('\n--- 协变量平滑性检验（断点处不应有跳跃）---')
covariates = ['income', 'education', 'health', 'family_sz', 'urban']
smooth_rows = []
for var in covariates:
    X_cov = sm.add_constant(df_bw5[['running', 'D', 'running_D']])
    m_cov = sm.OLS(df_bw5[var], X_cov).fit(cov_type='HC1')
    pval = m_cov.pvalues['D']
    flag = ' ⚠ 有跳跃!' if pval < 0.05 else ''
    print(f'  {var}: p(D)={pval:.4f}{flag}')
    smooth_rows.append({'协变量': var, '断点跳跃': m_cov.params['D'], 'p值': pval})
pd.DataFrame(smooth_rows).round(4).to_csv(
    OUTPUT_DIR / "ch09_rdd_covariate_smoothness.csv", index=False, encoding="utf-8-sig")

# 安慰剂检验（假断点）
df_rdd['running58'] = df_rdd['age'] - 58
df_rdd['fake58'] = (df_rdd['age'] >= 58).astype(int)
df_rdd['running58_D58'] = df_rdd['running58'] * df_rdd['fake58']
df_placebo58 = df_rdd[(df_rdd['age'] < 60) & (abs(df_rdd['running58']) <= 5)]
X_p58 = sm.add_constant(df_placebo58[['running58', 'fake58', 'running58_D58']])
m_p58 = sm.OLS(df_placebo58['consumption'], X_p58).fit(cov_type='HC1')
print(f'\n  假断点58岁: D系数={m_p58.params["fake58"]:.1f} (p={m_p58.pvalues["fake58"]:.4f})')

df_rdd['running62'] = df_rdd['age'] - 62
df_rdd['fake62'] = (df_rdd['age'] >= 62).astype(int)
df_rdd['running62_D62'] = df_rdd['running62'] * df_rdd['fake62']
df_placebo62 = df_rdd[(df_rdd['age'] > 60) & (abs(df_rdd['running62']) <= 5)]
X_p62 = sm.add_constant(df_placebo62[['running62', 'fake62', 'running62_D62']])
m_p62 = sm.OLS(df_placebo62['consumption'], X_p62).fit(cov_type='HC1')
print(f'  假断点62岁: D系数={m_p62.params["fake62"]:.1f} (p={m_p62.pvalues["fake62"]:.4f})')

pd.DataFrame([
    {'假断点': '58岁', 'D系数': m_p58.params['fake58'], 'p值': m_p58.pvalues['fake58']},
    {'假断点': '62岁', 'D系数': m_p62.params['fake62'], 'p值': m_p62.pvalues['fake62']},
]).round(4).to_csv(OUTPUT_DIR / "ch09_rdd_placebo.csv", index=False, encoding="utf-8-sig")
print(f'\n结果已保存至：{OUTPUT_DIR}')

print('\n=== 提示 ===')
print('  资格指示D的结果跳跃 = 简化式资格效应（常称ITT）')
print('  Fuzzy RDD = LATE（局部平均处理效应）= 简化式跳跃 / 第一阶段跳跃')
print('  协变量平滑性检验：若未出现明显跳跃，与断点附近连续性相容；')
print('  但较大的p值不能证明个体必然可比。')
print('  安慰剂检验：假断点若未出现明显跳跃，可作为设计有效性的辅助证据；')
print('  它不能单独证明政策效应真实存在。')
print('  以上判断以你自己的实际输出为准：本次运行的逐项数值见 output/ 下的 CSV；')
print('  出现显著跳跃时，应把它当作对设计的质疑，而不是可以忽略的噪声。')

print('\n=== RDD 标准流程 ===')
print('  步骤1: 散点+拟合图 → 视觉确认断点跳跃')
print('  步骤2: 简化式RDD多带宽 → 资格效应')
print('  步骤3: 第一阶段 → 验证断点影响处理概率')
print('  步骤4: Fuzzy RDD → LATE估计')
print('  步骤5: 协变量平滑性 → 检查连续性假设是否受到明显质疑')
print('  步骤6: 假断点安慰剂 → 检查是否存在其他异常跳跃')
