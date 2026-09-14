* 第3章案例：读懂一张成绩回归表，并看清估计值的样本波动
* 生成正文图3-3（回归表中各系数的点估计与95%置信区间）和图3-4（换样本后 AI
* 系数如何波动）。
*
* 数据与第2章完全同源：同一份数据生成过程、同一个随机种子，因此两章的数字可以直接
* 对照。第2章关心"AI 系数等于多少"，本章关心"这个数字有多确定"。

clear all
set more off
set seed 20260430
capture mkdir output

* ------------------------------------------------------------
* 1. 与第2章相同的数据生成过程
* ------------------------------------------------------------
set obs 800
gen ability = rnormal(0, 1)
gen age     = round(rnormal(20, 1.6))
replace age = 17 if age < 17
replace age = 26 if age > 26
gen female  = runiform() < 0.5

gen ai = 8 + 3.2*ability + 0.25*(age-20) - 1.2*female + rnormal(0, 3.5)
replace ai = 0  if ai < 0
replace ai = 30 if ai > 30

gen score = 68 + 1.25*ai + 4.5*ability + 0.6*(age-20) - 1.8*female + rnormal(0, 5.5)

tempfile master
save `master'

* ------------------------------------------------------------
* 2. 任务1、2：一张回归表
* ------------------------------------------------------------
regress score ai age female ability

display _newline "回归表（被解释变量：课程成绩）："
matrix b  = e(b)
local names : colnames b
display _newline %-16s "变量" %12s "系数" %12s "标准误" %10s "t值" %12s "95%CI下" %12s "95%CI上"
foreach v of local names {
    local coef = _b[`v']
    local se   = _se[`v']
    local t    = `coef'/`se'
    local lo   = `coef' - invttail(e(df_r), 0.025)*`se'
    local hi   = `coef' + invttail(e(df_r), 0.025)*`se'
    display %-16s "`v'" %12.5f `coef' %12.5f `se' %10.3f `t' %12.5f `lo' %12.5f `hi'
}
display _newline "样本量：" e(N) "，R² = " %6.4f e(r2) "，调整R² = " %6.4f e(r2_a)
display "整体F统计量：" %9.3f e(F) "，p = " %10.4g e(p)
display _newline "AI 系数的手工t值：" %9.5f _b[ai]/_se[ai]

* 导出回归表
tempname mem
tempfile tab
postfile `mem' str12 variable double coef double std_err double t_value ///
    double p_value double ci_lower double ci_upper using `tab', replace
foreach v in ai age female ability {
    local coef = _b[`v']
    local se   = _se[`v']
    local p    = 2*ttail(e(df_r), abs(`coef'/`se'))
    local lo   = `coef' - invttail(e(df_r), 0.025)*`se'
    local hi   = `coef' + invttail(e(df_r), 0.025)*`se'
    post `mem' ("`v'") (`coef') (`se') (`coef'/`se') (`p') (`lo') (`hi')
}
postclose `mem'
use `tab', clear
export delimited using "ch03_regression_table.csv", replace

* ------------------------------------------------------------
* 3. 任务3：换一份样本，估计值会变多少（样本波动）
* ------------------------------------------------------------
display _newline "换样本后 AI 系数的波动："

* 收集用的局部宏
local nrep   = 0
local ncover = 0
local bsum   = 0
local bmin   = 999
local bmax   = -999
local semin  = 999
local semax  = -999

* (1) 全样本
use `master', clear
quietly regress score ai age female ability
local b_full  = _b[ai]
local se_full = _se[ai]
local lo_full = _b[ai] - invttail(e(df_r),0.025)*_se[ai]
local hi_full = _b[ai] + invttail(e(df_r),0.025)*_se[ai]
display "  全样本：n=" %4.0f e(N) "，β=" %6.4f `b_full' ///
    "，se=" %6.4f _se[ai] "，95%CI=(" %6.3f `lo_full' "," %6.3f `hi_full' ")"

* (2) 随机平分两半
set seed 20261075
gen double rndu = runiform()
sort rndu
gen byte grp = cond(_n <= 400, 1, 2)

forvalues g = 1/2 {
    preserve
    quietly keep if grp == `g'
    quietly regress score ai age female ability
    local b_`g'  = _b[ai]
    local lo_`g' = _b[ai] - invttail(e(df_r),0.025)*_se[ai]
    local hi_`g' = _b[ai] + invttail(e(df_r),0.025)*_se[ai]
    local glab = cond(`g' == 1, "A", "B")
    display "  随机半样本（`glab'）：n=" %4.0f e(N) "，β=" %6.4f _b[ai] ///
        "，se=" %6.4f _se[ai] "，95%CI=(" %6.3f `lo_`g'' "," %6.3f `hi_`g'' ")"
    restore
}

* (3) 再换 5 组切法，看波动的规律
forvalues k = 1/5 {
    use `master', clear
    set seed `= 20261075 + `k''
    quietly gen double rndu = runiform()
    quietly sort rndu
    quietly gen byte grp = cond(_n <= 400, 1, 2)
    forvalues g = 1/2 {
        preserve
        quietly keep if grp == `g'
        quietly regress score ai age female ability
        local b  = _b[ai]
        local se = _se[ai]
        local lo = `b' - invttail(e(df_r),0.025)*`se'
        local hi = `b' + invttail(e(df_r),0.025)*`se'
        local ++nrep
        local bsum = `bsum' + `b'
        if `b'  < `bmin'  local bmin  = `b'
        if `b'  > `bmax'  local bmax  = `b'
        if `se' < `semin' local semin = `se'
        if `se' > `semax' local semax = `se'
        if (`lo' <= 1.25 & 1.25 <= `hi') local ++ncover
        restore
    }
}
local bmean = `bsum'/`nrep'
display _newline "  共 `nrep' 个半样本："
display "    β 的范围 " %6.4f `bmin' "–" %6.4f `bmax' "，均值 " %6.4f `bmean'
display "    标准误范围 " %6.5f `semin' "–" %6.5f `semax' ///
    "；全样本标准误 × √2 = " %6.4f `se_full'*1.41421
display "    置信区间覆盖真实值 1.25 的个数：`ncover'/`nrep'"

* (4) 两种"有选择的"换样本方式
use `master', clear
quietly summarize ai, detail
quietly gen byte keep1 = ai <= r(p95)
preserve
    quietly keep if keep1 == 1
    quietly regress score ai age female ability
    local b_t1  = _b[ai]
    local lo_t1 = _b[ai] - invttail(e(df_r),0.025)*_se[ai]
    local hi_t1 = _b[ai] + invttail(e(df_r),0.025)*_se[ai]
    display "  剔除 AI 使用最高 5%：n=" %4.0f e(N) "，β=" %6.4f _b[ai] ///
        "，se=" %6.4f _se[ai] "，95%CI=(" %6.3f `lo_t1' "," %6.3f `hi_t1' ")"
restore

quietly summarize score, detail
quietly gen byte keep2 = score >= r(p5)
preserve
    quietly keep if keep2 == 1
    quietly regress score ai age female ability
    local b_t2  = _b[ai]
    local lo_t2 = _b[ai] - invttail(e(df_r),0.025)*_se[ai]
    local hi_t2 = _b[ai] + invttail(e(df_r),0.025)*_se[ai]
    display "  剔除成绩最低 5%：n=" %4.0f e(N) "，β=" %6.4f _b[ai] ///
        "，se=" %6.4f _se[ai] "，95%CI=(" %6.3f `lo_t2' "," %6.3f `hi_t2' ")"
restore

* 汇总成表
clear
set obs 5
gen byte id = _n
gen str24 sample = ""
gen double n = .
gen double ai_coef = .
gen double ci_lower = .
gen double ci_upper = .

replace sample = "全样本"              in 1
replace n = 800                        in 1
replace ai_coef = `b_full'             in 1
replace ci_lower = `lo_full'           in 1
replace ci_upper = `hi_full'           in 1

replace sample = "随机半样本（A）"      in 2
replace n = 400                        in 2
replace ai_coef = `b_1'                in 2
replace ci_lower = `lo_1'              in 2
replace ci_upper = `hi_1'              in 2

replace sample = "随机半样本（B）"      in 3
replace n = 400                        in 3
replace ai_coef = `b_2'                in 3
replace ci_lower = `lo_2'              in 3
replace ci_upper = `hi_2'              in 3

replace sample = "剔除 AI 使用最高 5%"  in 4
replace n = 760                        in 4
replace ai_coef = `b_t1'               in 4
replace ci_lower = `lo_t1'             in 4
replace ci_upper = `hi_t1'             in 4

replace sample = "剔除成绩最低 5%"      in 5
replace n = 760                        in 5
replace ai_coef = `b_t2'               in 5
replace ci_lower = `lo_t2'             in 5
replace ci_upper = `hi_t2'             in 5

export delimited sample n ai_coef ci_lower ci_upper using "ch03_sample_variation.csv", replace

* ------------------------------------------------------------
* 4. 图3-3：回归表中各系数的点估计与置信区间（截距不绘制）
* ------------------------------------------------------------
use `tab', clear
gen byte ord = 5 - _n
label define vlab 1 "认知能力" 2 "性别（女性=1）" 3 "年龄" 4 "AI 使用时间"
label values ord vlab

twoway (rcap ci_lower ci_upper ord, horizontal lcolor(gs10) lwidth(med)) ///
       (scatter ord coef, msymbol(O) mcolor(navy) msize(medlarge)), ///
    ylabel(1 2 3 4, valuelabel angle(0) labsize(small) noticks) ///
    xline(0, lcolor(black) lpattern(dash) lwidth(med)) ///
    ytitle("") xtitle("回归系数及 95% 置信区间") ///
    title("成绩回归表：点估计与不确定性") ///
    legend(off)
graph save output/ch03_regression_table.gph, replace
graph export output/ch03-fig3-regression-table-and-ci.png, width(2400) replace

* ------------------------------------------------------------
* 5. 图3-4：换样本后 AI 系数如何波动
* ------------------------------------------------------------
import delimited "ch03_sample_variation.csv", clear varnames(1)
gen byte ord2 = 6 - _n
label define slab 1 "剔除成绩最低 5%" 2 "剔除 AI 使用最高 5%" ///
    3 "随机半样本（B）" 4 "随机半样本（A）" 5 "全样本"
label values ord2 slab

twoway (rcap ci_lower ci_upper ord2, horizontal lcolor(gs10) lwidth(med)) ///
       (scatter ord2 ai_coef, msymbol(O) mcolor(navy) msize(medlarge)), ///
    ylabel(1 2 3 4 5, valuelabel angle(0) labsize(small) noticks) yscale(reverse) ///
    xline(1.25, lcolor(maroon) lpattern(dash) lwidth(med)) ///
    ytitle("") xtitle("AI 使用时间的系数及 95% 置信区间") ///
    title("换一份样本，系数会变；但每次都落在可预期的范围内") ///
    legend(off)
graph save output/ch03_sample_variation.gph, replace
graph export output/ch03-fig4-sample-variation.png, width(2400) replace

display _newline "图形已保存至 output/"
