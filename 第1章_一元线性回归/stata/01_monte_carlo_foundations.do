* 第1章案例：从样本到总体——大数定律、中心极限定理与OLS的重复抽样
* 生成正文图1-6（大数定律）、图1-7（中心极限定理）和图1-8（OLS斜率的抽样分布）。
* 三个任务共用一条逻辑：样本量越大，样本信息越接近总体；把抽样这个过程重复多次，
* 又能看出估计量本身的分布。全部使用教学用模拟数据，总体与真实参数由数据生成
* 过程人为设定并已知。

clear all
set more off
set seed 20260914
capture mkdir output

* ------------------------------------------------------------
* 任务1：大数定律——样本量越大，样本分布越接近总体
* 总体为标准正态分布 N(0,1)，总体均值 0。
* ------------------------------------------------------------
display _newline "任务1 大数定律（总体 N(0,1)，总体均值 0）："
foreach n in 10 100 1000 10000 {
    clear
    set obs `n'
    gen z = rnormal(0, 1)
    quietly summarize z
    display "  n=" %6.0f `n' "：样本均值 " %9.4f r(mean) ///
        "，样本标准差 " %9.4f r(sd)

    histogram z, bin(30) color(navy%40) ///
        addplot(function y=normalden(x,0,1), range(-4 4) lcolor(maroon) lwidth(*2)) ///
        xline(0, lcolor(maroon) lpattern(dash)) ///
        xscale(range(-4 4)) xlabel(-4(1)4) ///
        xtitle("观测值") ytitle("密度") title("n = `n'") legend(off)
    graph save "output/lln_`n'.gph", replace
}
graph combine output/lln_10.gph output/lln_100.gph output/lln_1000.gph output/lln_10000.gph, ///
    cols(2) title("大数定律：样本量越大，样本分布越接近总体分布")
graph export output/ch01-fig6-large-numbers.png, width(2200) replace

* ------------------------------------------------------------
* 任务2：中心极限定理——样本均值的分布随样本量增加趋近正态
* 总体为均匀分布 U(0,10)，明显不是正态分布。
* ------------------------------------------------------------
display _newline "任务2 中心极限定理（总体 U(0,10)，总体均值 5.0）："

capture program drop draw_mean
program define draw_mean, rclass
    syntax, N(integer)
    drop _all
    set obs `n'
    gen z = runiform(0, 10)
    quietly summarize z
    return scalar zmean = r(mean)
end

* 总体分布图
clear
set obs 20000
gen population = runiform(0, 10)
histogram population, bin(40) color(gs12) ///
    xscale(range(0 10)) xlabel(0(2)10) ///
    xtitle("数值") ytitle("密度") title("总体分布 U(0,10)") legend(off)
graph save output/clt_pop.gph, replace

* 各样本量下的样本均值分布
tempfile clt_all part
foreach size in 1 5 30 100 500 {
    capture confirm file `clt_all'
    simulate zmean=r(zmean), reps(3000) nodots: draw_mean, n(`size')
    gen n = `size'
    quietly summarize zmean
    local sd = 10/sqrt(12)/sqrt(`size')
    display "  n=" %4.0f `size' "：样本均值的均值 " %6.4f r(mean) ///
        "，标准差 " %6.4f r(sd) "，理论标准差 " %6.4f `sd'

    histogram zmean, bin(40) color(navy%40) ///
        addplot(function y=normalden(x,5,`sd'), range(0 10) lcolor(maroon) lwidth(*2)) ///
        xline(5, lcolor(navy) lpattern(dash)) ///
        xscale(range(0 10)) xlabel(0(2)10) ///
        xtitle("样本均值") ytitle("密度") title("样本均值分布 n = `size'") legend(off)
    graph save "output/clt_`size'.gph", replace
    save `part', replace
    capture confirm file `clt_all'
    if _rc == 0 {
        use `clt_all', clear
        append using `part'
        save `clt_all', replace
    }
    else {
        use `part', clear
        save `clt_all', replace
    }
}

graph combine output/clt_pop.gph output/clt_1.gph output/clt_5.gph ///
              output/clt_30.gph output/clt_100.gph output/clt_500.gph, ///
    cols(3) title("中心极限定理：总体不是正态，样本均值的分布却随样本量增加趋近正态")
graph export output/ch01-fig7-central-limit.png, width(3000) replace

* ------------------------------------------------------------
* 任务3：把重复抽样思想用到OLS——斜率的抽样分布
* 真实模型 y = 2 + 0.5x + u，x ~ U(0,10)，u ~ N(0,1)。
* ------------------------------------------------------------
display _newline "任务3 OLS斜率的重复抽样（真实斜率 0.5）："

capture program drop draw_ols
program define draw_ols, rclass
    syntax, N(integer)
    drop _all
    set obs `n'
    gen x = 10*runiform()
    gen y = 2 + 0.5*x + rnormal()
    quietly regress y x
    return scalar b1 = _b[x]
end

tempfile ols_all
foreach size in 30 100 500 {
    simulate b1=r(b1), reps(3000) nodots: draw_ols, n(`size')
    gen n = `size'
    save `part', replace
    capture confirm file `ols_all'
    if _rc == 0 {
        use `ols_all', clear
        append using `part'
    }
    else {
        use `part', clear
    }
    save `ols_all', replace
    use `part', clear
    table n, statistic(mean b1) statistic(sd b1)
}

use `ols_all', clear
twoway (kdensity b1 if n==30, lcolor(orange) lwidth(*2)) ///
       (kdensity b1 if n==100, lcolor(blue) lwidth(*2)) ///
       (kdensity b1 if n==500, lcolor(green) lwidth(*2)), ///
       xline(0.5, lcolor(black) lpattern(dash)) ///
       title("OLS斜率的抽样分布：一次估计会波动，样本越大越集中") ///
       xtitle("斜率估计值") ytitle("密度") ///
       legend(order(1 "n = 30" 2 "n = 100" 3 "n = 500"))
graph save output/ch01_ols.gph, replace
graph export output/ch01-fig8-ols-sampling-distribution.png, width(2200) replace

* 汇总各样本量下斜率估计的均值与标准差，保存为与Python版对照的表
preserve
collapse (mean) mean=b1 (sd) std=b1, by(n)
gen bias = mean - 0.5
export delimited using "ch01_summary.csv", replace
list, noobs
restore

display _newline "图形已保存至 output/"
