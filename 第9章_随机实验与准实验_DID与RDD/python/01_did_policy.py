"""
第9章 现代因果推断与准实验方法 — 案例实践（A部分：双重差分）
配套教材：《计量经济学：理论与实践》

内容：生成虚构“智能生产支持计划”面板数据，依次完成两个DiD任务——
      任务1：DiD估计与平行趋势检验
      任务2：事件研究法与安慰剂检验
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from linearmodels.panel import PanelOLS

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
# 输出目录：默认写到本脚本同级的 output/，下载到任何位置都能直接运行，不依赖仓库结构。
OUTPUT_DIR = Path(os.environ.get("CASE_OUTPUT_DIR", HERE / "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据生成：虚构“智能生产支持计划”对企业数字创新指标的影响
# 60个虚构城市 × 10家企业 × 7年（2018-2024）
# 城市1-30 = 计划组（2021年起），城市31-60 = 对照组
# ============================================================
np.random.seed(20250901)

n_cities = 60
n_firms_per_city = 10
n_firms = n_cities * n_firms_per_city
n_years = 7

# 城市层级
city_treated = np.repeat(np.arange(1, n_cities+1) <= 30, n_firms_per_city * n_years)
city_id = np.repeat(np.arange(1, n_cities+1), n_firms_per_city * n_years)
city_fe_raw = np.random.normal(0, 0.5, n_cities)
city_fe = np.repeat(city_fe_raw, n_firms_per_city * n_years)

# 企业层级
firm_id = np.tile(np.repeat(np.arange(1, n_firms+1), n_years), 1)
firm_size_raw = np.random.normal(5, 1, n_firms)
firm_size = np.tile(np.repeat(firm_size_raw, n_years), 1)
firm_size = np.clip(firm_size, 2, None)
firm_age_raw = np.random.uniform(1, 30, n_firms)
firm_age = np.tile(np.repeat(firm_age_raw, n_years), 1)
firm_fe_raw = np.random.normal(0, 0.3, n_firms)
firm_fe = np.tile(np.repeat(firm_fe_raw, n_years), 1)

# 时间
year = np.tile(np.arange(2018, 2025), n_firms)
year_fe = 0.1 * (year - 2018)

# 政策变量
post = (year >= 2021).astype(int)
treated_raw = np.repeat(np.arange(1, n_cities+1) <= 30, n_firms_per_city)
treated = np.tile(np.repeat(treated_raw, n_years), 1)
did = treated * post

# DGP：创新专利数
u = np.random.normal(0, 1, len(year))
innovation = (3.0 + 0.5*firm_size + 0.02*firm_age + city_fe + firm_fe
              + year_fe + 1.5*did + u)
innovation = np.clip(innovation, 0, None)

df_did = pd.DataFrame({
    'innovation': innovation, 'treated': treated, 'post': post,
    'did': did, 'firm_size': firm_size, 'firm_age': firm_age,
    'year': year, 'firm_id': np.tile(np.repeat(np.arange(1, n_firms+1), n_years), 1),
    'city_id': np.repeat(np.repeat(np.arange(1, n_cities+1), n_firms_per_city), n_years)
})

print('========== A部分：双重差分（DiD）==========')
print(f'面板结构: {n_firms}企业 × {n_years}年')
print(f'处理组城市: 1-30, 对照组城市: 31-60')
print(f'政策实施年份: 2021')

# ――――――――――――――――――――――――――――――――――――――――――――
# 任务1：DiD估计与平行趋势检验
# ――――――――――――――――――――――――――――――――――――――――――――
print('\n--- 任务1：DiD估计 ---')

# 基准DiD（OLS + 交互项）
m_did_base = ols('innovation ~ treated + post + did', data=df_did).fit(
    cov_type='cluster', cov_kwds={'groups': df_did['city_id']})
did_coef = m_did_base.params['did']
print(f'基准DiD: Treat×Post = {did_coef:.4f}')

# 加入控制变量
m_did_ctrl = ols('innovation ~ treated + post + did + firm_size + firm_age',
                 data=df_did).fit(
                     cov_type='cluster', cov_kwds={'groups': df_did['city_id']})
print(f'+控制变量: Treat×Post = {m_did_ctrl.params["did"]:.4f}')

# 固定效应DiD
df_did_indexed = df_did.set_index(['firm_id', 'year'])
m_did_fe = PanelOLS(
    df_did_indexed['innovation'],
    df_did_indexed[['did']],
    entity_effects=True, time_effects=True
).fit(cov_type='clustered', clusters=df_did_indexed[['city_id']])
print(f'+固定效应: did = {m_did_fe.params["did"]:.4f}')

# 三种设定的 DiD 系数并排落盘，便于与正文表格逐项对照
pd.DataFrame({
    'DiD系数': [did_coef, m_did_ctrl.params['did'], m_did_fe.params['did']],
    '标准误': [m_did_base.bse['did'], m_did_ctrl.bse['did'], m_did_fe.std_errors['did']],
}, index=['基准交互项', '加控制变量', '企业FE+年份FE']).round(4).to_csv(
    OUTPUT_DIR / "ch09_did_three_specs.csv", encoding="utf-8-sig")

# 平行趋势图数据
trend = df_did.groupby(['year', 'treated'])['innovation'].mean().unstack()
print('\n各组创新均值（检查政策前趋势）:')
print(trend.round(2))
trend.round(4).to_csv(OUTPUT_DIR / "ch09_pretrend_means.csv", encoding="utf-8-sig")

print('\n=== 提示 ===')
print('  基准DiD的Treat×Post应接近1.5（数据生成过程设定的政策效应）')
print('  三种设定下系数应接近且显著')

# ――――――――――――――――――――――――――――――――――――――――――――
# 任务2：事件研究法与安慰剂检验
# ――――――――――――――――――――――――――――――――――――――――――――
print('\n--- 任务2：事件研究法 ---')

df_did['rel_year'] = df_did['year'] - 2021

# 事件研究回归（以rel_year=-1为基准）
for yr in range(-3, 4):
    if yr != -1:
        label = f'did_{"pre" if yr<0 else "post"}{abs(yr)}'
        df_did[label] = df_did['treated'] * (df_did['rel_year'] == yr).astype(int)

event_dummies = [c for c in df_did.columns if c.startswith('did_')]
formula_event = 'innovation ~ ' + ' + '.join(event_dummies) + ' + C(firm_id) + C(year)'
m_event = ols(formula_event, data=df_did).fit(
    cov_type='cluster', cov_kwds={'groups': df_did['city_id']})
print('\n事件研究系数:')
event_rows = []
for c in sorted(event_dummies):
    coef = m_event.params.get(c, np.nan)
    pval = m_event.pvalues.get(c, np.nan)
    sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.1 else ''))
    print(f'  {c}: {coef:.4f}{sig} (p={pval:.4f})')
    event_rows.append({'项': c, '系数': coef, '标准误': m_event.bse.get(c, np.nan),
                       'p值': pval})
pd.DataFrame(event_rows).round(4).to_csv(
    OUTPUT_DIR / "ch09_event_study.csv", index=False, encoding="utf-8-sig")

# 安慰剂检验（假政策时点=2019）
df_pre = df_did[df_did['year'] <= 2020].copy()
df_pre['fake_post'] = (df_pre['year'] >= 2019).astype(int)
df_pre['fake_did'] = df_pre['treated'] * df_pre['fake_post']
m_placebo_time = ols('innovation ~ treated + fake_post + fake_did + firm_size + firm_age',
                     data=df_pre).fit(
                         cov_type='cluster', cov_kwds={'groups': df_pre['city_id']})
fake_did = m_placebo_time.params.get('fake_did', np.nan)
print(f'\n假政策时点(2019)安慰剂检验: Treat×Post = {fake_did:.4f}')
print('  本DGP下通常应接近0；明显偏离0时需排查政策前差异趋势或模型设定')

# 空间安慰剂
np.random.seed(42)
random_treated_cities = set(np.random.choice(np.arange(1, n_cities + 1),
                                             size=n_cities // 2, replace=False))
df_did['random_treat'] = df_did['city_id'].isin(random_treated_cities).astype(int)
df_did['random_did'] = df_did['random_treat'] * df_did['post']
m_placebo_space = ols('innovation ~ random_treat + post + random_did + firm_size + firm_age',
                      data=df_did).fit(
                          cov_type='cluster', cov_kwds={'groups': df_did['city_id']})
rand_did = m_placebo_space.params.get('random_did', np.nan)
print(f'随机假处理组安慰剂: Treat×Post = {rand_did:.4f}')
print('  本DGP下通常应接近0；一次随机结果只能提供辅助诊断')

pd.DataFrame({
    'DiD系数': [fake_did, rand_did],
    'p值': [m_placebo_time.pvalues.get('fake_did', np.nan),
           m_placebo_space.pvalues.get('random_did', np.nan)],
}, index=['时间安慰剂(假政策时点2019)', '空间安慰剂(随机假处理组)']).round(4).to_csv(
    OUTPUT_DIR / "ch09_placebo_tests.csv", encoding="utf-8-sig")
print(f'\n结果已保存至：{OUTPUT_DIR}')

print('\n=== DiD 标准流程 ===')
print('  步骤1: 平行趋势图 → 视觉检查')
print('  步骤2: 基准回归（多种设定）→ 系数稳定性')
print('  步骤3: 事件研究 → 政策前系数检查')
print('  步骤4: 安慰剂检验 → 提供时间和空间上的辅助证据')
print('  提醒：安慰剂结果不显著不能证明识别假设必然成立。')
