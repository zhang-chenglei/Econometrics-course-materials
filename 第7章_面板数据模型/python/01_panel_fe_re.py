"""
第7章 面板数据模型 — 案例实践
配套教材：《计量经济学：理论与实践》

内容：生成模拟上市公司面板数据，依次完成四个任务——
1. 混合OLS与固定效应对比
2. 三种FE方法验证（组内去心 / LSDV / 一阶差分）
3. 随机效应与豪斯曼检验
4. 双向固定效应
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects, FirstDifferenceOLS
from scipy import stats

# ============================================================
# 数据生成（200家公司 × 5年 = 1000观测值）
# ============================================================
np.random.seed(20250715)
n_company, n_year = 200, 5

# 公司固定效应 α_i（与R&D正相关）
company_fe = pd.DataFrame({
    'company_id': np.arange(1, n_company + 1),
    'alpha_i': np.random.normal(0, 1.5, n_company)
})

# 面板结构
panels = []
for t in range(1, n_year + 1):
    panel_t = company_fe.copy()
    panel_t['year'] = t
    panels.append(panel_t)
df = pd.concat(panels, ignore_index=True)

# 年份固定效应 γ_t
gamma = {1: -0.30, 2: -0.15, 3: 0.00, 4: 0.15, 5: 0.30}
df['gamma_t'] = df['year'].map(gamma)

# 时变解释变量
# R&D：与公司FE正相关（创新文化→高R&D）
df['RD'] = np.clip(
    np.random.normal(5, 2, len(df)) + 0.3 * df['alpha_i'],
    0.5, None
)

# 总资产：与公司FE弱正相关
df['assets'] = np.clip(
    np.random.normal(100, 30, len(df)) + 0.15 * df['alpha_i'],
    10, None
)

# 负债率：独立时变变量
df['debt'] = np.clip(np.random.normal(30, 10, len(df)), 0, None)

# 因变量：利润
# DGP: profit = 2 + 0.18*RD + 0.04*assets - 0.03*debt + alpha_i + gamma_t + u
u = np.random.normal(0, 2, len(df))
df['profit'] = (2.0 + 0.18*df['RD'] + 0.04*df['assets']
                - 0.03*df['debt'] + df['alpha_i'] + df['gamma_t'] + u)

# 设置 MultiIndex（面板数据结构）
df = df.set_index(['company_id', 'year'])

print(f'样本量: {len(df)}')
print(f'公司数: {n_company}, 年份: {n_year}')
print(f'面板结构: 平衡面板 ({n_company} × {n_year})')
print(df[['profit', 'RD', 'assets', 'debt']].describe())

# ============================================================
# 任务1：混合OLS与固定效应对比
# ============================================================
print('\n' + '='*60)
print('任务1：混合OLS vs FE')
print('='*60)

# 混合OLS（忽略面板结构）
y_pooled = df['profit']
X_pooled = sm.add_constant(df[['RD', 'assets', 'debt']])
m_pooled = sm.OLS(y_pooled, X_pooled).fit(cov_type='cluster',
    cov_kwds={'groups': df.index.get_level_values('company_id')})

# 固定效应（组内估计）
m_fe = PanelOLS(
    df['profit'], df[['RD', 'assets', 'debt']],
    entity_effects=True
).fit(cov_type='clustered', cluster_entity=True)

print('\n--- 混合OLS vs FE 系数对比 ---')
comparison_1 = pd.DataFrame({
    'Pooled OLS': m_pooled.params,
    'FE': m_fe.params,
}).round(5)
print(comparison_1)

rd_pooled = m_pooled.params['RD']
rd_fe = m_fe.params['RD']
print(f'\nR&D系数: Pooled OLS={rd_pooled:.4f}, FE={rd_fe:.4f}')
print(f'系数变化: {(rd_pooled - rd_fe):.4f}')

print('\n=== 提示 ===')
print('  如果混合OLS的RD系数 > FE的RD系数：')
print('  → 公司FE与RD正相关，混合OLS高估了R&D的真实效应')
print('  → FE通过组内去心消除了不变的公司特质')
print('  → FE的估计更接近R&D的"净效应"')

# ============================================================
# 任务2：三种FE方法验证
# ============================================================
print('\n' + '='*60)
print('任务2：三种FE方法对比')
print('='*60)

# 方法1：组内去心法
m_within = PanelOLS(
    df['profit'], df[['RD', 'assets', 'debt']],
    entity_effects=True
).fit()

# 方法2：LSDV（加公司虚拟变量）
# 用 PanelOLS 的 entity_effects=True 本质就是 LSDV
# 此处显式用 statsmodels OLS + 公司虚拟变量来展示
df_lsdv = df.reset_index()
df_lsdv = pd.get_dummies(df_lsdv, columns=['company_id'],
    drop_first=True, dtype=float)
company_dummies = [c for c in df_lsdv.columns if c.startswith('company_id_')]
X_lsdv = sm.add_constant(df_lsdv[['RD', 'assets', 'debt'] + company_dummies])
m_lsdv = sm.OLS(df_lsdv['profit'], X_lsdv).fit()

# 方法3：一阶差分法
m_fd = FirstDifferenceOLS(
    df['profit'], df[['RD', 'assets', 'debt']]
).fit()

rd_within = m_within.params['RD']
rd_lsdv = m_lsdv.params['RD']
rd_fd = m_fd.params['RD']

print('\n--- 三种FE方法 R&D系数对比 ---')
print(f'  组内去心法: {rd_within:.5f}')
print(f'  LSDV法:     {rd_lsdv:.5f}')
print(f'  一阶差分法: {rd_fd:.5f}')

print('\n=== 提示 ===')
print('  组内去心法 & LSDV → 系数应完全一致（等价变换）')
print('  一阶差分法 → 系数通常接近但不完全一致')
print('    → 差分法丢失第一期数据（N减少）')
print('    → 差分法对测量误差更敏感')

# ============================================================
# 任务3：随机效应与豪斯曼检验
# ============================================================
print('\n' + '='*60)
print('任务3：随机效应与豪斯曼检验')
print('='*60)

# 随机效应模型
m_re = RandomEffects(
    df['profit'], sm.add_constant(df[['RD', 'assets', 'debt']])
).fit(cov_type='clustered', cluster_entity=True)

print('\n--- FE vs RE 系数对比 ---')
comparison_3 = pd.DataFrame({
    'FE': m_fe.params,
    'RE': m_re.params,
}).round(5)
print(comparison_3)

# 豪斯曼检验
print('\n--- 豪斯曼检验 ---')
# 聚类稳健模型用于主结果；传统协方差版本用于Hausman演示。
m_fe_haus = PanelOLS(
    df['profit'], df[['RD', 'assets', 'debt']], entity_effects=True
).fit()
m_re_haus = RandomEffects(
    df['profit'], sm.add_constant(df[['RD', 'assets', 'debt']])
).fit()
common = ['RD', 'assets', 'debt']
diff = m_fe_haus.params[common] - m_re_haus.params[common]
cov_diff = (
    m_fe_haus.cov.loc[common, common]
    - m_re_haus.cov.loc[common, common]
)
hausman_stat = float(diff.T @ np.linalg.pinv(cov_diff.to_numpy()) @ diff)
hausman_df = len(common)
hausman_p = stats.chi2.sf(max(hausman_stat, 0), hausman_df)
print(f'Hausman χ²({hausman_df}) = {hausman_stat:.4f}, p = {hausman_p:.4f}')

print('\n=== 提示 ===')
print('  H0: 随机效应一致（RE假设成立，α_i与解释变量不相关）')
print('  H1: 固定效应一致，随机效应不一致')
print('  如果 p < 0.05 → 拒绝RE → 使用FE')
print('  本DGP中α_i与RD正相关 → 预期拒绝RE')

# ============================================================
# 任务4：双向固定效应
# ============================================================
print('\n' + '='*60)
print('任务4：双向固定效应')
print('='*60)

# 创建年份虚拟变量
df_reset = df.reset_index()
year_dummies = pd.get_dummies(df_reset['year'], prefix='yr',
    drop_first=True, dtype=float)
df_reset = pd.concat([df_reset, year_dummies], axis=1)
df_reset = df_reset.set_index(['company_id', 'year'])
yr_cols = [c for c in year_dummies.columns]

# 双向固定效应
m_twoway = PanelOLS(
    df_reset['profit'],
    df_reset[['RD', 'assets', 'debt'] + yr_cols],
    entity_effects=True
).fit(cov_type='clustered', cluster_entity=True)

# 三模型对比
comparison_4 = pd.DataFrame({
    'Pooled OLS': m_pooled.params[['RD', 'assets', 'debt']],
    '单向FE': m_fe.params[['RD', 'assets', 'debt']],
    '双向FE': m_twoway.params[['RD', 'assets', 'debt']],
}).round(5)
print('\n--- 混合OLS vs 单向FE vs 双向FE ---')
print(comparison_4)

print('\n=== 提示 ===')
print('  1. 比较单向FE和双向FE的R&D系数：')
print('     → 如果接近 → 年份冲击与R&D相关性不大')
print('     → 如果差异明显 → 省略年份FE会导致额外偏误')
print('  2. 年份FE的系数捕捉了各年的全行业共性冲击')
print('  3. 双向FE是面板论文中最常见的基准设定')

# ============================================================
# 最终汇总
# ============================================================
print('\n' + '='*60)
print('最终汇总：四任务对比表')
print('='*60)

summary = pd.DataFrame({
    'Pooled OLS': m_pooled.params[['RD', 'assets', 'debt']],
    'FE': m_fe.params[['RD', 'assets', 'debt']],
    'RE': m_re.params[['RD', 'assets', 'debt']],
    '双向FE': m_twoway.params[['RD', 'assets', 'debt']],
}).round(5)
print(summary)
# 不报告 _cons/截距行，因为 FE 模型中截距的释义与 Pooled OLS 不同

print('\n=== 面板数据分析标准流程 ===')
print('  步骤1: 声明面板结构（set_index/xtset）—— 不是可选的！')
print('  步骤2: 比较Pooled OLS、FE与RE的假设和估计对象')
print('  步骤3: Hausman检验 —— 提供关于FE/RE差异的统计证据')
print('  步骤4: 加入时间FE —— 双向固定效应')
print()
print('  这个顺序不是随意的——每一步都有明确的逻辑目的。')
print('  记不住命令没关系，但必须记住这个顺序和意义。')
