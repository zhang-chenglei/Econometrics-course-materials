version 18.0
clear all
set more off

* 第12章：数据检查、变量构造与描述性统计
* 运行前请将Stata工作目录切换到仓库的 materials 文件夹。
global PKG  "`c(pwd)'/code/comprehensive_case"
global DATA "`c(pwd)'/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"

capture log close
log using "$PKG/第12章/说明/ch12_stata.log", text replace
import delimited using "$DATA", clear varnames(1) encoding(UTF-8)
encode firm_id, gen(fid)
xtset fid year

* 唯一性、范围和缺失检查
isid fid year
misstable summarize rd_intensity_w ln_patent_invention size lev roa growth ///
    fixed cashflow firm_age ln_gov_subsidy kz_index sa_index ln_rd_staff
summarize rd_intensity_w ln_patent_invention size lev roa growth fixed ///
    cashflow firm_age ln_gov_subsidy kz_index sa_index ln_rd_staff, detail

* 论文式描述性统计
capture which estpost
if _rc ssc install estout, replace
estpost summarize rd_intensity_w ln_patent_invention size lev roa growth ///
    fixed cashflow firm_age ln_gov_subsidy kz_index sa_index ln_rd_staff
esttab using "$PKG/第12章/表格/stata_描述性统计.csv", ///
    cells("count mean(fmt(4)) sd(fmt(4)) min(fmt(4)) max(fmt(4))") ///
    nonumber nomtitle replace csv

* 处理组政策前与从未处理组的特征平衡
gen pre_or_never = never_treated == 1 | (did_treated == 1 & year < policy_first_full_year)
foreach x in size lev roa growth fixed cashflow firm_age {
    quietly ttest `x' if pre_or_never, by(did_treated) unequal
    display "`x'  mean(control)=" %9.4f r(mu_1) ///
        " mean(treated)=" %9.4f r(mu_2) " p=" %7.4f r(p)
}

* 原始趋势图
preserve
collapse (mean) rd_intensity_w ln_patent_invention, by(year did_treated)
twoway ///
    (connected rd_intensity_w year if did_treated == 0, msymbol(O)) ///
    (connected rd_intensity_w year if did_treated == 1, msymbol(T)), ///
    legend(order(1 "从未处理组" 2 "最终处理组")) ///
    ytitle("研发投入强度") xtitle("年份") title("处理组与对照组的原始趋势")
graph export "$PKG/第12章/图形/stata_图12-2a_研发投入原始趋势.png", width(2200) replace

twoway ///
    (connected ln_patent_invention year if did_treated == 0, msymbol(O)) ///
    (connected ln_patent_invention year if did_treated == 1, msymbol(T)), ///
    legend(order(1 "从未处理组" 2 "最终处理组")) ///
    ytitle("ln(1+发明专利申请)") xtitle("年份") title("处理组与对照组的原始趋势")
graph export "$PKG/第12章/图形/stata_图12-2b_专利原始趋势.png", width(2200) replace
restore

log close
