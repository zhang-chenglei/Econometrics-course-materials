/*══════════════════════════════════════════════════════════════
  第9章 因果推断前沿方法 — 案例实践（A部分：双重差分）
  配套教材：《计量经济学：理论与实践》
  功能：生成虚构“智能生产支持计划”面板数据，依次完成两个DiD任务：
        任务1：DiD估计与平行趋势检验
        任务2：事件研究法与安慰剂检验
  ══════════════════════════════════════════════════════════════*/

clear
set more off
set seed 20250901

*============================================================
* 数据生成：虚构“智能生产支持计划”对企业数字创新指标的影响
* 60个虚构城市 × 10家企业 × 7年（2018-2024）
* 城市1-30 = 计划组（2021年起），城市31-60 = 对照
*============================================================

set obs 60
gen city_id = _n
gen treated = (city_id <= 30)           // 城市1-30 = 处理组
gen city_fe = rnormal(0, 0.5)           // 城市固定特质（创新基础）

expand 10  // 每城市10家企业
sort city_id
gen firm_id = _n

* 企业特征
gen firm_size = rnormal(5, 1)           // 规模（对数）
replace firm_size = max(2, firm_size)
gen firm_age  = runiform(1, 30)         // 企业年龄
gen firm_fe   = rnormal(0, 0.3)         // 企业固定特质

* 展开为面板（7年）
expand 7
bysort firm_id: gen year = _n + 2017    // 2018-2024
xtset firm_id year

* 时间固定效应（宏观经济趋势）
gen year_fe = 0.1*(year - 2018)

* 政策变量
gen post = (year >= 2021)
gen did  = treated * post

* 因变量：数字创新专利数
* 数据生成过程设定的政策效应 = +1.5（从2021年开始）
gen u = rnormal(0, 1)
gen innovation = 3.0 + 0.5*firm_size + 0.02*firm_age + city_fe + firm_fe + year_fe ///
                 + 1.5*did + u
replace innovation = max(0, innovation)

label variable innovation "数字创新专利数"
label variable treated    "试点城市=1"
label variable post       "2021年及以后=1"
label variable did        "Treat × Post"
label variable firm_size  "企业规模（对数）"
label variable firm_age   "企业年龄"

sum innovation treated post did

* ――――――――――――――――――――――――――――――――――――――――――――
* 任务1：DiD估计与平行趋势检验
* ――――――――――――――――――――――――――――――――――――――――――――
display _n "========== 任务1：DiD估计与平行趋势检验 =========="

* 1a：绘制平行趋势图
preserve
collapse (mean) innovation, by(year treated)
separate innovation, by(treated) veryshortlabel
twoway (line innovation1 year, lcolor(blue) lwidth(medthick)) ///
       (line innovation0 year, lcolor(red) lwidth(medthick)),  ///
       legend(label(1 "试点城市") label(2 "非试点城市"))      ///
       xline(2021, lpattern(dash) lcolor(black))              ///
       title("政策前后试点组与对照组创新趋势")                ///
       ytitle("平均创新专利数") xtitle("年份")
restore
display "  看图要点：2021年前两条线是否大致平行？2021年后是否分化？"

* 1b：基准DiD回归（无控制变量）
reg innovation i.treated##i.post, vce(cluster city_id)
estimates store m_did_base

* 1c：加入控制变量
reg innovation i.treated##i.post firm_size firm_age, vce(cluster city_id)
estimates store m_did_ctrl

* 1d：固定效应DiD（更标准的做法）
xtreg innovation did i.year, fe vce(cluster city_id)
estimates store m_did_fe

* 对照表
display _n "--- DiD系数对比 ---"
estimates table m_did_base m_did_ctrl m_did_fe, ///
    stats(N) b(%9.4f) se(%9.4f)

display _n "=== 提示 ==="
display "  基准DiD: 1.treat#1.post ≈ 1.5（真实效应）"
display "  加入控制变量 → 系数应接近1.5"
display "  FE中treated主效应被吸收（不随时间变化），post被年份FE吸收"
display "  重点是1.treat#1.post的符号和显著性"

* ――――――――――――――――――――――――――――――――――――――――――――
* 任务2：事件研究法与安慰剂检验
* ――――――――――――――――――――――――――――――――――――――――――――
display _n "========== 任务2：事件研究法与安慰剂检验 =========="

* 2a：事件研究法
gen rel_year = year - 2021
tab rel_year, gen(yr_)

* 生成DID交互（各年 × treated）
forvalues r = -3/3 {
    if `r' != -1 {
        gen did_`=cond(`r'<0,"pre","post")'`=abs(`r')' = treated * yr_`=string(`r'+4)'
    }
}

* 事件研究回归（以rel_year=-1为基准）
xtreg innovation did_pre3 did_pre2 did_post0 did_post1 did_post2 did_post3 ///
    i.year, fe vce(cluster city_id)

display _n "--- 事件研究系数 ---"
display "  政策前（did_pre3, did_pre2）→ 应不显著（平行趋势成立）"
display "  政策后（did_post0~did_post3）→ 应显著为正（政策效应）"
display "  如果政策前系数显著 → 平行趋势不成立 → DiD不可信"

* 2b：时间安慰剂检验（假政策时点=2019）
preserve
keep if year <= 2020
gen fake_post = (year >= 2019)
gen fake_did  = treated * fake_post
xtreg innovation fake_did i.year, fe vce(cluster city_id)
display _n "--- 假政策时点（2019）安慰剂检验 ---"
display "  本DGP下假DID系数通常应接近0"
display "  明显偏离0时需排查政策前差异趋势或模型设定"
restore

* 2c：空间安慰剂检验（随机假处理组）
preserve
bysort city_id: gen random_treat = runiform() < 0.5 if _n == 1
bysort city_id: replace random_treat = random_treat[1]
gen random_did   = random_treat * post
xtreg innovation random_did i.year, fe vce(cluster city_id)
display _n "--- 随机假处理组安慰剂检验 ---"
display "  本DGP下通常应接近0；一次随机结果只能提供辅助诊断"
restore

display _n "=== 提示 ==="
display "  安慰剂结果可提供辅助证据，但不能证明识别假设必然成立"
display "  但这只是间接证据——最终的信服力依赖于制度论证"
display ""
display "  DiD标准流程："
display "  步骤1: 平行趋势图 → 视觉检查"
display "  步骤2: 基准回归（多种设定）→ 系数稳定性"
display "  步骤3: 事件研究 → 政策前系数检查"
display "  步骤4: 安慰剂 → 提供时间和空间上的辅助证据"
