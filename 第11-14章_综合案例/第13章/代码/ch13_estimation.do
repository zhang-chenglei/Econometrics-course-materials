version 18.0
clear all
set more off

* 第13章：主回归、交错DID、稳健性、异质性与机制
* 教学样本含明确的半合成处理，所有数值仅用于课堂演示。
* 运行前请将Stata工作目录切换到仓库的 materials 文件夹。
global PKG  "`c(pwd)'/code/comprehensive_case"
global DATA "`c(pwd)'/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"

capture log close
log using "$PKG/第13章/说明/ch13_stata.log", text replace
import delimited using "$DATA", clear varnames(1) encoding(UTF-8)
encode firm_id, gen(fid)
egen industry_year = group(industry_group year)
xtset fid year

local controls size lev roa growth fixed cashflow firm_age

foreach pkg in ftools reghdfe estout coefplot csdid drdid {
    capture which `pkg'
    if _rc ssc install `pkg', replace
}

* 13.1 TWFE教学基准
eststo clear
reghdfe rd_intensity_w did_post_full, absorb(fid year) vce(cluster analysis_city_id)
eststo rd_base
reghdfe rd_intensity_w did_post_full `controls', absorb(fid year) vce(cluster analysis_city_id)
eststo rd_ctrl
reghdfe ln_patent_invention did_post_full, absorb(fid year) vce(cluster analysis_city_id)
eststo pat_base
reghdfe ln_patent_invention did_post_full `controls', absorb(fid year) vce(cluster analysis_city_id)
eststo pat_ctrl
esttab rd_base rd_ctrl pat_base pat_ctrl using "$PKG/第13章/表格/stata_TWFE教学基准.csv", ///
    keep(did_post_full) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    stats(N, labels("观测数")) replace csv

* 13.2 交错DID：从未处理组；never treated的gvar必须编码为0
gen gvar = policy_first_full_year
replace gvar = 0 if missing(gvar)

csdid rd_intensity_w, ivar(fid) time(year) gvar(gvar) ///
    method(reg) cluster(analysis_city_id)
estat simple
estat pretrend
estat event, window(-5 2) estore(rd_event)
csdid_plot, title("研发投入强度：动态效应")
graph export "$PKG/第13章/图形/stata_图13-1a_研发投入动态效应.png", width(2200) replace

csdid ln_patent_invention, ivar(fid) time(year) gvar(gvar) ///
    method(reg) cluster(analysis_city_id)
estat simple
estat pretrend
estat event, window(-5 2) estore(pat_event)
csdid_plot, title("发明专利申请：动态效应")
graph export "$PKG/第13章/图形/stata_图13-1b_专利动态效应.png", width(2200) replace

* 13.3 关键稳健性
eststo clear
foreach y in rd_intensity_w ln_patent_invention {
    reghdfe `y' did_post_full, absorb(fid year) vce(cluster analysis_city_id)
    eststo `y'_main
    reghdfe `y' did_post_approval, absorb(fid year) vce(cluster analysis_city_id)
    eststo `y'_approval
    reghdfe `y' did_post_full if year != 2020, absorb(fid year) vce(cluster analysis_city_id)
    eststo `y'_no2020
    reghdfe `y' did_post_full if !inlist(assign_city, "北京市", "上海市", "深圳市"), ///
        absorb(fid year) vce(cluster analysis_city_id)
    eststo `y'_nohub
}
esttab using "$PKG/第13章/表格/stata_关键稳健性.csv", ///
    keep(did_post_full did_post_approval) b(4) se(4) ///
    star(* 0.10 ** 0.05 *** 0.01) stats(N) replace csv

* 13.4 异质性：必须检验交互项，而不是只比较分组显著性
egen tag_firm = tag(fid)
quietly summarize pre_size if tag_firm, detail
gen high_pre_size = pre_size > r(p50)
quietly summarize pre_patent_mean if tag_firm, detail
gen high_pre_patent = pre_patent_mean > r(p50)
foreach y in rd_intensity_w ln_patent_invention {
    foreach h in pre_soe high_pre_size high_pre_patent tech_industry {
        reghdfe `y' c.did_post_full##i.`h', absorb(fid year) vce(cluster analysis_city_id)
        lincom did_post_full + 1.`h'#c.did_post_full
        test 1.`h'#c.did_post_full
    }
}

* 13.5 机制相关结果：只称“与渠道一致”，不称完整中介效应
eststo clear
foreach y in ln_gov_subsidy kz_index sa_index ln_rd_staff {
    reghdfe `y' did_post_full, absorb(fid year) vce(cluster analysis_city_id)
    eststo mech_`y'
}
esttab using "$PKG/第13章/表格/stata_机制相关结果.csv", ///
    keep(did_post_full) b(4) se(4) star(* 0.10 ** 0.05 *** 0.01) ///
    stats(N) replace csv

* 正式机制口径：与主结果一致的交错DID，总体ATT写入日志
foreach y in ln_gov_subsidy kz_index sa_index ln_rd_staff {
    csdid `y', ivar(fid) time(year) gvar(gvar) ///
        method(reg) cluster(analysis_city_id)
    estat simple
}

* 替代函数形式：营业收入分母、专利原始件数与反双曲正弦
gen patent_asinh = asinh(patent_app_invention)
foreach y in rd_sales_pct patent_app_invention patent_asinh {
    reghdfe `y' did_post_full, absorb(fid year) vce(cluster analysis_city_id)
}

log close
