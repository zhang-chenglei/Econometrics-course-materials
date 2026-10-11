/*******************************************************************************
 第六讲《面板数据模型（下）》· 复现数据构建脚本
 -------------------------------------------------------------------------------
 数据源：data/climate_panel.dta
         Dell, Jones & Olken (2012), "Temperature Shocks and Economic Growth:
         Evidence from the Last Half Century", AEJ:Macro 4(3), 66-95.
         复现包：AEJ-Mac2010-0092-reproduction，数据授权 CC-BY 4.0。
 口径来源：scripts/original_maketable2.do、original_maketable3.do（论文原代码）
 差异说明：论文原用 cgmreg，此教学脚本采用 reghdfe。
           核心点估计与包内论文稿按3位小数吻合；单例处理、协方差修正
           及推断口径不同，标准误与部分显著性有差异。原包另行完整运行。
 输出：data/regression_results.csv（长表：spec, term, b, se, t, p, n）
       由 scripts/build_numbers.py 汇总为 model_summary.json + model_numbers.tex
 运行：cd 课件/第六讲 && stata-mp -b do scripts/build_panel.do
*******************************************************************************/

clear all
set more off
set matsize 2000

* 输出路径（脚本假定当前工作目录为 课件/第六讲）
capture file close res
file open res using "output/data/regression_results.csv", write replace text
file write res "spec,term,b,se,t,p,n" _n

capture program drop rec
program define rec
    * 把上一条 lincom 的结果写入 CSV：rec <spec> <term> <n>
    args spec term n
    local b  : display %14.8f r(estimate)
    local se : display %14.8f r(se)
    local t  : display %10.4f r(estimate)/r(se)
    local p  : display %12.6f r(p)
    file write res "`spec',`term',`b',`se',`t',`p',`n'" _n
end


* ============================================================================
* 1. 构建分析样本（严格照论文 maketable2.do 的口径）
* ============================================================================
use "original/climate_panel.dta", clear

keep if year <= 2003

encode parent,    g(parent_num)      // 聚类：parent-country
encode fips60_06, g(cc_num)          // 固定效应：country

sort country_code year
tsset cc_num year

gen lngdpwdi = ln(gdpLCU)
gen g = (lngdpwdi - l.lngdpwdi) * 100   // 年增长率，百分点

* 至少 20 年增长数据
gen tempnonmis = 1 if g != .
replace tempnonmis = 0 if g == .
bys fips60_06: egen tempsumnonmis = sum(tempnonmis)
drop if tempsumnonmis < 20

* 农业与工业子成分同期非缺失（论文口径）
gen misdum = 0
replace misdum = 1 if gdpWDIGDPAGR == .
replace misdum = 1 if gdpWDIGDPIND == .

* 穷国哑变量：初始人均 GDP（对数）低于中位数
preserve
    keep if lnrgdpl_t0 < .
    bys fips60_06: keep if _n == 1
    xtile initgdpbin = ln(lnrgdpl_t0), nq(2)
    keep fips60_06 initgdpbin
    tempfile poorbin
    save `poorbin', replace
restore
merge m:1 fips60_06 using `poorbin', nogen keep(master match)

tab initgdpbin, g(initxtilegdp)       // 生成 initxtilegdp1（穷）/ initxtilegdp2（富）

gen wtem_poor = wtem * initxtilegdp1
gen wpre_poor = wpre * initxtilegdp1
gen wtem_rich = wtem * initxtilegdp2

* 地区 × 年份、穷国 × 年份
gen str6 region = ""
foreach X in _MENA _SSAF _LAC _WEOFF _EECA _SEAS {
    replace region = "`X'" if `X' == 1
}
gen regionyear = region + string(year)
encode regionyear, g(rynum)
egen poorid = group(initxtilegdp1 year)

sort cc_num year
tsset cc_num year

* 滞后的分组温度（Table 3 用完全交互形式）
forvalues k = 1/10 {
    gen L`k'wtem_poor = L`k'.wtem * initxtilegdp1
    gen L`k'wtem_rich = L`k'.wtem * initxtilegdp2
}

di as txt "分析样本：N = " _N


* ============================================================================
* 2. Q1 固定效应阶梯：固定效应加到什么程度才够？
*    （所有规格统一用双向聚类，以便逐步只看固定效应带来的变化）
* ============================================================================
di as res _n "=== Q1 固定效应阶梯 ==="

di as txt "--- m1_pooled：无固定效应 ---"
reghdfe g wtem, noabsorb vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m1_pooled wtem `n'

di as txt "--- m2_countryfe：+ 国家固定效应 ---"
reghdfe g wtem, absorb(cc_num) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m2_countryfe wtem `n'

di as txt "--- m3_country_year：+ 国家 + 年份固定效应 ---"
reghdfe g wtem, absorb(cc_num year) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m3_country_year wtem `n'

di as txt "--- m4_region_year：+ 国家 + 地区x年份固定效应 ---"
reghdfe g wtem, absorb(cc_num rynum) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m4_region_year wtem `n'

di as txt "--- m5_poor_year：+ 国家 + 地区x年份 + 穷国x年份（论文基准）---"
reghdfe g wtem, absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m5_poor_year wtem `n'

di as txt "--- m6_interaction：基准 + 温度x穷国（论文 Table 2 第2列）---"
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m6_interaction wtem `n'
lincom wtem_poor
rec m6_interaction wtem_poor `n'
lincom wtem + wtem_poor
rec m6_interaction poor_net `n'

di as txt "--- m7_country_trend：+ 国家特定线性趋势 ---"
reghdfe g wtem wtem_poor, absorb(cc_num cc_num#c.year rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem
rec m7_country_trend wtem `n'
lincom wtem_poor
rec m7_country_trend wtem_poor `n'
lincom wtem + wtem_poor
rec m7_country_trend poor_net `n'


* ============================================================================
* 3. Q2 标准误：同一个回归式，四种标准误
* ============================================================================
di as res _n "=== Q2 标准误对比（同一回归式 m6）==="

di as txt "--- se_ols：常规标准误 ---"
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid)
local n = e(N)
lincom wtem_poor
rec se_ols wtem_poor `n'

di as txt "--- se_robust：异方差稳健 ---"
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(robust)
local n = e(N)
lincom wtem_poor
rec se_robust wtem_poor `n'

di as txt "--- se_cl_country：按国家聚类 ---"
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num)
local n = e(N)
lincom wtem_poor
rec se_cl_country wtem_poor `n'

di as txt "--- se_cl_twoway：双向聚类（国家 & 地区x年份）---"
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem_poor
rec se_cl_twoway wtem_poor `n'

* 富国侧同做一遍，作为对照
foreach v in ols robust cl_country cl_twoway {
    if "`v'" == "ols"        reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid)
    if "`v'" == "robust"     reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(robust)
    if "`v'" == "cl_country" reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num)
    if "`v'" == "cl_twoway"  reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
    local n = e(N)
    lincom wtem
    rec se_`v' wtem `n'
}


* ============================================================================
* 4. Q3 动态效应：当期 vs 累积（论文 Table 3）
* ============================================================================
di as res _n "=== Q3 滞后结构 ==="

* 0 阶
di as txt "--- lag0 ---"
reghdfe g wtem_poor wtem_rich, absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem_poor
rec lag0 poor_contemp `n'
lincom wtem_poor
rec lag0 poor_cum `n'
lincom wtem_rich
rec lag0 rich_contemp `n'
lincom wtem_rich
rec lag0 rich_cum `n'

* 1 阶
di as txt "--- lag1 ---"
reghdfe g wtem_poor L1wtem_poor wtem_rich L1wtem_rich, ///
    absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem_poor
rec lag1 poor_contemp `n'
lincom wtem_poor + L1wtem_poor
rec lag1 poor_cum `n'
lincom wtem_rich
rec lag1 rich_contemp `n'
lincom wtem_rich + L1wtem_rich
rec lag1 rich_cum `n'

* 5 阶
di as txt "--- lag5 ---"
reghdfe g wtem_poor L1wtem_poor L2wtem_poor L3wtem_poor L4wtem_poor L5wtem_poor ///
          wtem_rich L1wtem_rich L2wtem_rich L3wtem_rich L4wtem_rich L5wtem_rich, ///
    absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem_poor
rec lag5 poor_contemp `n'
forvalues k = 1/5 {
    lincom L`k'wtem_poor
    rec lag5 poor_L`k' `n'
}
lincom wtem_poor + L1wtem_poor + L2wtem_poor + L3wtem_poor + L4wtem_poor + L5wtem_poor
rec lag5 poor_cum `n'
lincom wtem_rich + L1wtem_rich + L2wtem_rich + L3wtem_rich + L4wtem_rich + L5wtem_rich
rec lag5 rich_cum `n'

* 10 阶
di as txt "--- lag10 ---"
reghdfe g wtem_poor L1wtem_poor L2wtem_poor L3wtem_poor L4wtem_poor L5wtem_poor ///
          L6wtem_poor L7wtem_poor L8wtem_poor L9wtem_poor L10wtem_poor ///
          wtem_rich L1wtem_rich L2wtem_rich L3wtem_rich L4wtem_rich L5wtem_rich ///
          L6wtem_rich L7wtem_rich L8wtem_rich L9wtem_rich L10wtem_rich, ///
    absorb(cc_num rynum poorid) vce(cluster parent_num rynum)
local n = e(N)
lincom wtem_poor
rec lag10 poor_contemp `n'
forvalues k = 1/10 {
    lincom L`k'wtem_poor
    rec lag10 poor_L`k' `n'
}
lincom wtem_poor + L1wtem_poor + L2wtem_poor + L3wtem_poor + L4wtem_poor + L5wtem_poor ///
     + L6wtem_poor + L7wtem_poor + L8wtem_poor + L9wtem_poor + L10wtem_poor
rec lag10 poor_cum `n'
lincom wtem_rich + L1wtem_rich + L2wtem_rich + L3wtem_rich + L4wtem_rich + L5wtem_rich ///
     + L6wtem_rich + L7wtem_rich + L8wtem_rich + L9wtem_rich + L10wtem_rich
rec lag10 rich_cum `n'


* ============================================================================
* 5. 描述统计（供图 1 与正文用）
* ============================================================================
di as res _n "=== 描述统计 ==="
capture file close desc
file open desc using "output/data/descriptives.csv", write replace text
file write desc "item,value" _n
quietly summarize wtem
file write desc "wtem_mean," (string(r(mean), "%12.6f")) _n
file write desc "wtem_sd,"   (string(r(sd),   "%12.6f")) _n
quietly summarize wtem if initxtilegdp1 == 1
file write desc "wtem_mean_poor," (string(r(mean), "%12.6f")) _n
quietly summarize wtem if initxtilegdp1 == 0
file write desc "wtem_mean_rich," (string(r(mean), "%12.6f")) _n
count
file write desc "n_obs," (string(r(N), "%9.0f")) _n
quietly levelsof cc_num
file write desc "n_country," (string(r(r), "%9.0f")) _n
quietly levelsof year
file write desc "n_year," (string(r(r), "%9.0f")) _n
quietly levelsof rynum
file write desc "n_regionyear," (string(r(r), "%9.0f")) _n
file close desc

file close res


* ============================================================================
* 6. 导出出图用的分析样本
* ============================================================================
* 残差取自基准交互模型 m6，供「残差自相关」图使用
reghdfe g wtem wtem_poor, absorb(cc_num rynum poorid) vce(cluster parent_num rynum) resid
predict double resid_m6, resid

preserve
    gen byte poor = initxtilegdp1
    gen long parent_id = parent_num
    rename region region_lab
    keep cc_num country year parent_id poor region_lab g wtem wpre resid_m6 rynum
    order cc_num country year parent_id poor region_lab g wtem wpre resid_m6 rynum
    export delimited using "output/data/analysis_panel.csv", replace
restore

* 国家层面汇总（横截面图用）
preserve
    collapse (mean) mean_g=g mean_wtem=wtem (first) lnrgdpl_t0 poor=initxtilegdp1 ///
             region_lab=region, by(cc_num country)
    export delimited using "output/data/country_summary.csv", replace
restore

di as res _n "完成：data/regression_results.csv、analysis_panel.csv、country_summary.csv、descriptives.csv"
