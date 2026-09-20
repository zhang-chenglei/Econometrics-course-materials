version 18.0
clear all
set more off

* 第12章：研究问题、政策事实与研究设计诊断
* 数据为半合成教学样本，结果不得用于评价真实政策。
* 运行前请将Stata工作目录切换到含 code/ 的教材根目录（仓库内为 30_教学/09_计量教材/）。
global PKG "`c(pwd)'/code/comprehensive_case"
global MAT "`c(pwd)'"
capture confirm file "`c(pwd)'/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"
if _rc global MAT "`c(pwd)'/GitHub课程网站/materials"
global DATA "$MAT/data/semisynthetic/ai_pilot_semisynthetic_2011_2024.csv"

capture log close
log using "$PKG/第12章/说明/ch12_stata.log", text replace
import delimited using "$DATA", clear varnames(1) encoding(UTF-8)

isid firm_id year
assert inrange(year, 2011, 2024)
assert is_semisynthetic == 1

encode firm_id, gen(fid)
xtset fid year

* 样本与处理批次
count
egen tag_firm = tag(fid)
count if tag_firm
egen tag_city = tag(analysis_city_id)
count if tag_city
tab policy_first_full_year if tag_firm & did_treated == 1, missing

* 各处理批次的政策前后支持
preserve
keep if did_treated == 1
gen one = 1
egen tag_batch_firm = tag(policy_first_full_year year fid)
collapse (sum) observations=one firms=tag_batch_firm, by(policy_first_full_year year)
export delimited using "$PKG/第12章/表格/stata_处理批次年度支持.csv", replace
restore

* 各试点覆盖企业数
preserve
keep if tag_firm & did_treated == 1
contract policy_scope_name policy_first_full_year
rename _freq firms
gsort policy_first_full_year -firms
export delimited using "$PKG/第12章/表格/stata_各试点覆盖企业数.csv", replace
graph hbar firms, over(policy_scope_name, sort(1) descending label(labsize(vsmall))) ///
    ytitle("企业数") title("各试点在教学样本中的企业覆盖")
graph export "$PKG/第12章/图形/stata_图12-2_各试点企业覆盖.png", width(2200) replace
restore

log close
