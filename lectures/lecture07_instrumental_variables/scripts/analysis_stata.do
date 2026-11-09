/*==============================================================================
  第7讲工具变量案例：Stata 参考分析

  读取 data/firm_iv_case.csv，完成：
  1. OLS 与单工具 2SLS 对比；
  2. 第一阶段强度与弱工具对比；
  3. 双工具 2SLS 与过度识别检验；
  4. Durbin-Wu-Hausman 内生性检验与三模型汇总。

  在案例目录运行：
      stata-mp -b do scripts/analysis_stata.do
  输出写到 output/stata/。
==============================================================================*/

clear all
set more off
version 17

capture mkdir output
capture mkdir output/stata
capture log close
log using "output/stata/analysis_stata.log", replace text

import delimited "data/firm_iv_case.csv", clear varnames(1) asdouble
rename roe ROE
rename rd RD
rename resdensity ResDensity
rename university University

* ---------------------------------------------------------------------------
* 任务1：OLS 与单工具 2SLS
* ---------------------------------------------------------------------------
reg ROE RD size age, robust
estimates store m_ols

ivregress 2sls ROE (RD = ResDensity) size age, robust
estimates store m_iv

tempname fh
file open `fh' using "output/stata/ols_vs_iv.csv", write replace
file write `fh' "variable,OLS_coef,OLS_se,IV_coef,IV_se" _n
estimates restore m_ols
local ols_rd = _b[RD]
local ols_rd_se = _se[RD]
local ols_size = _b[size]
local ols_size_se = _se[size]
local ols_age = _b[age]
local ols_age_se = _se[age]
estimates restore m_iv
file write `fh' "RD," %12.6f (`ols_rd') "," %12.6f (`ols_rd_se') "," ///
    %12.6f (_b[RD]) "," %12.6f (_se[RD]) _n
file write `fh' "size," %12.6f (`ols_size') "," %12.6f (`ols_size_se') "," ///
    %12.6f (_b[size]) "," %12.6f (_se[size]) _n
file write `fh' "age," %12.6f (`ols_age') "," %12.6f (`ols_age_se') "," ///
    %12.6f (_b[age]) "," %12.6f (_se[age]) _n
file close `fh'

* ---------------------------------------------------------------------------
* 任务2：第一阶段与弱工具
* ---------------------------------------------------------------------------
reg RD ResDensity size age, robust
estimates store m_first
local b_z  = _b[ResDensity]
local se_z = _se[ResDensity]
local t_z  = _b[ResDensity] / _se[ResDensity]
local p_z  = 2 * ttail(e(df_r), abs(`t_z'))
local r2_full = e(r2)

test ResDensity
local partial_f = r(F)

tempname fh1
file open `fh1' using "output/stata/first_stage.csv", write replace
file write `fh1' "variable,coef,se,t,p" _n
file write `fh1' "const," %12.6f (_b[_cons]) "," %12.6f (_se[_cons]) "," ///
    %12.6f (_b[_cons]/_se[_cons]) "," %12.6f (2*ttail(e(df_r),abs(_b[_cons]/_se[_cons]))) _n
file write `fh1' "ResDensity," %12.6f (`b_z') "," %12.6f (`se_z') "," ///
    %12.6f (`t_z') "," %12.6f (`p_z') _n
file write `fh1' "size," %12.6f (_b[size]) "," %12.6f (_se[size]) "," ///
    %12.6f (_b[size]/_se[size]) "," %12.6f (2*ttail(e(df_r),abs(_b[size]/_se[size]))) _n
file write `fh1' "age," %12.6f (_b[age]) "," %12.6f (_se[age]) "," ///
    %12.6f (_b[age]/_se[age]) "," %12.6f (2*ttail(e(df_r),abs(_b[age]/_se[age]))) _n
file close `fh1'

reg RD size age, robust
local r2_noz = e(r2)
local partial_r2 = `r2_full' - `r2_noz'

tempname fh2
file open `fh2' using "output/stata/first_stage_diagnostics.csv", write replace
file write `fh2' "partial_F,partial_R2,first_stage_R2" _n
file write `fh2' %12.6f (`partial_f') "," %12.6f (`partial_r2') "," ///
    %12.6f (`r2_full') _n
file close `fh2'

* 弱工具对照：数据中固定保存的纯噪声工具
reg RD noise_iv size age, robust
test noise_iv
local weak_f = r(F)
ivregress 2sls ROE (RD = noise_iv) size age, robust
local weak_b  = _b[RD]
local weak_se = _se[RD]
local weak_lo = _b[RD] - invnormal(0.975) * _se[RD]
local weak_hi = _b[RD] + invnormal(0.975) * _se[RD]

estimates restore m_iv
local strong_b  = _b[RD]
local strong_se = _se[RD]
local strong_lo = _b[RD] - invnormal(0.975) * _se[RD]
local strong_hi = _b[RD] + invnormal(0.975) * _se[RD]

tempname fh3
file open `fh3' using "output/stata/weak_vs_strong.csv", write replace
file write `fh3' "instrument,RD_coef,RD_se,CI_low,CI_high,first_stage_F" _n
file write `fh3' "noise_iv (weak)," %12.6f (`weak_b') "," %12.6f (`weak_se') "," ///
    %12.6f (`weak_lo') "," %12.6f (`weak_hi') "," %12.6f (`weak_f') _n
file write `fh3' "ResDensity (strong)," %12.6f (`strong_b') "," %12.6f (`strong_se') "," ///
    %12.6f (`strong_lo') "," %12.6f (`strong_hi') "," %12.6f (`partial_f') _n
file close `fh3'

* ---------------------------------------------------------------------------
* 任务3：双工具与过度识别
* ---------------------------------------------------------------------------
ivregress 2sls ROE (RD = ResDensity University) size age, robust
estimates store m_iv2

tempname fh4
file open `fh4' using "output/stata/single_vs_two_iv.csv", write replace
file write `fh4' "variable,single_IV_coef,single_IV_se,two_IV_coef,two_IV_se" _n
estimates restore m_iv
local s_rd = _b[RD]
local s_rd_se = _se[RD]
local s_size = _b[size]
local s_size_se = _se[size]
local s_age = _b[age]
local s_age_se = _se[age]
estimates restore m_iv2
file write `fh4' "RD," %12.6f (`s_rd') "," %12.6f (`s_rd_se') "," ///
    %12.6f (_b[RD]) "," %12.6f (_se[RD]) _n
file write `fh4' "size," %12.6f (`s_size') "," %12.6f (`s_size_se') "," ///
    %12.6f (_b[size]) "," %12.6f (_se[size]) _n
file write `fh4' "age," %12.6f (`s_age') "," %12.6f (`s_age_se') "," ///
    %12.6f (_b[age]) "," %12.6f (_se[age]) _n
file close `fh4'

estat overid
local oid_stat = r(score)
local oid_df   = r(df)
local oid_p    = r(p_score)

tempname fh5
file open `fh5' using "output/stata/overid.csv", write replace
file write `fh5' "statistic,df,p_value" _n
file write `fh5' %12.6f (`oid_stat') "," %9.0f (`oid_df') "," ///
    %12.6f (`oid_p') _n
file close `fh5'

* ---------------------------------------------------------------------------
* 任务4：DWH 与三模型汇总
* ---------------------------------------------------------------------------
ivregress 2sls ROE (RD = ResDensity) size age, robust
estat endogenous
local dwh_F  = r(regF)
local dwh_df = r(regFdf_n)
local dwh_p  = r(p_regF)

tempname fh6
file open `fh6' using "output/stata/dwh.csv", write replace
file write `fh6' "F,df,p_value" _n
file write `fh6' %12.6f (`dwh_F') "," %9.0f (`dwh_df') "," ///
    %12.6f (`dwh_p') _n
file close `fh6'

tempname fh7
file open `fh7' using "output/stata/summary.csv", write replace
file write `fh7' "variable,OLS_coef,OLS_se,IV1_coef,IV1_se,IV2_coef,IV2_se,N" _n
estimates restore m_ols
local ols_rd = _b[RD]
local ols_rd_se = _se[RD]
local ols_size = _b[size]
local ols_size_se = _se[size]
local ols_age = _b[age]
local ols_age_se = _se[age]
local n_obs = e(N)
estimates restore m_iv
local iv_rd = _b[RD]
local iv_rd_se = _se[RD]
local iv_size = _b[size]
local iv_size_se = _se[size]
local iv_age = _b[age]
local iv_age_se = _se[age]
estimates restore m_iv2
file write `fh7' "RD," %12.6f (`ols_rd') "," %12.6f (`ols_rd_se') "," ///
    %12.6f (`iv_rd') "," %12.6f (`iv_rd_se') "," ///
    %12.6f (_b[RD]) "," %12.6f (_se[RD]) "," %9.0f (`n_obs') _n
file write `fh7' "size," %12.6f (`ols_size') "," %12.6f (`ols_size_se') "," ///
    %12.6f (`iv_size') "," %12.6f (`iv_size_se') "," ///
    %12.6f (_b[size]) "," %12.6f (_se[size]) "," %9.0f (`n_obs') _n
file write `fh7' "age," %12.6f (`ols_age') "," %12.6f (`ols_age_se') "," ///
    %12.6f (`iv_age') "," %12.6f (`iv_age_se') "," ///
    %12.6f (_b[age]) "," %12.6f (_se[age]) "," %9.0f (`n_obs') _n
file close `fh7'

display _n "=== 结果已写入 output/stata/ ==="
log close
