version 18.0
clear all
set more off

* 第15章：一次估计、统一存储、规范出表
* 运行前请将Stata工作目录切换到含 code/ 的教材根目录（仓库内为 30_教学/09_计量教材/）。
global PKG "`c(pwd)'/code/comprehensive_case"
global MAT "`c(pwd)'"
capture confirm file "`c(pwd)'/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"
if _rc global MAT "`c(pwd)'/GitHub课程网站/materials"
global DATA "$MAT/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"

capture log close
log using "$PKG/第15章/说明/ch15_stata.log", text replace
import delimited using "$DATA", clear varnames(1) encoding(UTF-8)
encode firm_id, gen(fid)
xtset fid year
local controls size lev roa growth fixed cashflow firm_age

foreach pkg in ftools reghdfe estout coefplot {
    capture which `pkg'
    if _rc ssc install `pkg', replace
}

* 最终四列表：数字来自同一数据版本和同一代码文件
eststo clear
reghdfe rd_intensity_w did_post_full, absorb(fid year) vce(cluster analysis_city_id)
estadd local controls "否"
estadd local firmfe "是"
estadd local yearfe "是"
eststo m1
reghdfe rd_intensity_w did_post_full `controls', absorb(fid year) vce(cluster analysis_city_id)
estadd local controls "是"
estadd local firmfe "是"
estadd local yearfe "是"
eststo m2
reghdfe ln_patent_invention did_post_full, absorb(fid year) vce(cluster analysis_city_id)
estadd local controls "否"
estadd local firmfe "是"
estadd local yearfe "是"
eststo m3
reghdfe ln_patent_invention did_post_full `controls', absorb(fid year) vce(cluster analysis_city_id)
estadd local controls "是"
estadd local firmfe "是"
estadd local yearfe "是"
eststo m4

esttab m1 m2 m3 m4 using "$PKG/第15章/表格/stata_表15-1_TWFE最终版.rtf", ///
    keep(did_post_full) b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    stats(controls firmfe yearfe N, ///
    labels("企业特征控制" "企业固定效应" "年份固定效应" "观测数")) ///
    mtitles("研发投入" "研发投入" "发明专利" "发明专利") ///
    title("教学基准：双向固定效应估计") replace

coefplot (m1, label("研发投入")) (m3, label("发明专利")), ///
    keep(did_post_full) xline(0) ciopts(recast(rcap)) ///
    title("TWFE教学基准：点估计与95%置信区间")
graph export "$PKG/第15章/图形/stata_图15-1_TWFE主结果.png", width(2200) replace

* 注意：正式交错DID表和事件研究图以第14章csdid输出或Python输出为准。
* 提交前核对摘要、正文、表格、图注和结论中的样本数与系数完全一致。
log close
