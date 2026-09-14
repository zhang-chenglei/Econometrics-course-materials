/*══════════════════════════════════════════════════════════════
  第8章 工具变量模型 — 案例实践
  配套教材：《计量经济学：理论与实践》
  功能：生成模拟企业数据，依次完成四个任务：
        任务1：OLS vs IV（内生性偏误诊断）
        任务2：第一阶段诊断（弱工具变量检测）
        任务3：多工具变量与过度识别检验
        任务4：内生性检验与稳健性汇总
  ══════════════════════════════════════════════════════════════*/

clear
set more off
set seed 20250801
set obs 500

* ============================================================
* 数据生成（500家企业，单一截面）
* ============================================================

* 工具变量
gen ResDensity = rnormal(0, 1)               // Z1: 科研资源密度（外生）
gen University  = rnormal(0, 1) + 0.3*ResDensity  // Z2: 高校数量（与Z1相关）

* 不可观测的企业能力（内生性来源——同时影响R&D和ROE）
gen ability = rnormal(0, 1)

* 控制变量
gen size = rnormal(5, 1)           // 企业规模（对数）
replace size = max(2, size)
gen age = runiform(1, 40)          // 企业年龄

* 内生解释变量：研发投入
* RD = f(ResDensity, University, size, age, ability) + noise
gen u_rd = rnormal(0, 1)
gen RD = 2.0 + 0.7*ResDensity + 0.1*University + 0.3*size - 0.02*age + 0.6*ability + u_rd
replace RD = max(0.1, RD)

* 因变量：净资产收益率
* ROE = f(RD, size, age, ability) + noise
* 真实RD效应 = 1.2; ability同时影响RD和ROE → OLS高估
gen v_roe = rnormal(0, 1.5)
gen ROE = 5.0 + 1.2*RD + 0.5*size - 0.01*age + 0.8*ability + v_roe

label variable ROE          "净资产收益率（%）"
label variable RD           "研发投入（百万元）"
label variable ResDensity   "科研资源密度（Z1）"
label variable University   "高校数量（Z2）"
label variable size         "企业规模（对数）"
label variable age          "企业年龄"

sum ROE RD ResDensity University size age

* ============================================================
* 任务1：OLS vs IV（内生性偏误诊断）
* ============================================================
display _n "========== 任务1：OLS vs IV =========="

* OLS（有偏——能力遗漏导致RD系数高估）
reg ROE RD size age, robust
estimates store m_ols
display "  OLS 的 RD 系数包含了能力偏误（高估）"

* 2SLS（用科研资源密度作为工具变量）
ivregress 2sls ROE (RD = ResDensity) size age, robust
estimates store m_iv

* 对照表
display _n "--- OLS vs IV 系数对比 ---"
estimates table m_ols m_iv, ///
    stats(N) b(%9.4f) se(%9.4f) keep(RD size age)

display _n "=== 提示 ==="
display "  OLS的RD系数 > IV的RD系数 → 能力偏误为正"
display "  ability同时与RD正相关、与ROE正相关 → OLS高估"
display "  在本DGP中，IV利用ResDensity推动的RD变动 → 应更接近设定值1.2"
display ""
display "  如果OLS系数 < IV系数 → 偏误方向为负（思考：什么情况下会发生？）"

* ============================================================
* 任务2：第一阶段诊断（弱工具变量检测）
* ============================================================
display _n "========== 任务2：第一阶段诊断 =========="

* 第一阶段回归
display _n "--- 第一阶段：RD = f(ResDensity, controls) ---"
reg RD ResDensity size age, robust

* 正式的第一阶段诊断
ivregress 2sls ROE (RD = ResDensity) size age, robust
estat firststage

display _n "=== 提示 ==="
display "  第一阶段F统计量可用于提示弱识别风险"
display "  常见的F>10只是简单同方差设定下的经验预警线，不是强工具证书"
display "  正式研究应使用适合模型与误差结构的弱识别诊断"
display ""
display "  本DGP中ResDensity系数约0.7，因此第一阶段应当较强"

* 弱工具变量模拟（仅用于教学对比）
display _n "--- 弱工具对比：用纯噪声作为工具 ---"
gen noise_iv = rnormal(0, 1)
ivregress 2sls ROE (RD = noise_iv) size age, robust
estat firststage
drop noise_iv

display _n "  上面的F值应接近0——演示什么是'弱工具'"

* ============================================================
* 任务3：多工具变量与过度识别检验
* ============================================================
display _n "========== 任务3：多工具与过度识别 =========="

* 使用两个工具变量（ResDensity + University）
ivregress 2sls ROE (RD = ResDensity University) size age, robust
estimates store m_iv2

* 第一阶段诊断
estat firststage

* 过度识别检验（Sargan检验）
estat overid

display _n "=== 提示 ==="
display "  过度识别检验（Sargan/Hansen J）："
display "    H0: 过度识别约束成立（解释依赖至少有足够有效工具等前提）"
display "    p > 0.05 → 未发现明显冲突，但不能证明所有工具都外生"
display "    p < 0.05 → 至少某些工具或模型约束与数据不相容"
display ""
display "  注意：恰好识别时（1个IV解决1个内生变量）无法做此检验"
display "  过度识别检验的必要前提：至少有一个工具是真正外生的"

* 单IV vs 双IV 对比
display _n "--- 单IV vs 双IV 系数对比 ---"
estimates table m_iv m_iv2, ///
    stats(N) b(%9.4f) se(%9.4f) keep(RD size age)

display _n "  通常双IV比单IV的标准误更小（效率更高）"
display "  但前提是两个IV都是有效的"

* ============================================================
* 任务4：内生性检验与稳健性汇总
* ============================================================
display _n "========== 任务4：内生性检验与汇总 =========="

* Durbin-Wu-Hausman 内生性检验
ivregress 2sls ROE (RD = ResDensity) size age, robust
estat endogenous

display _n "=== 提示 ==="
display "  Durbin-Wu-Hausman检验："
display "    H0: RD是外生的（OLS和IV都一致）"
display "    p < 0.05 → 数据反对RD外生；是否采用IV还取决于工具是否可信"
display "    p > 0.05 → 未发现明确内生性证据，不等于证明RD外生"
display "    → 主结果选择应依据研究设计，而不是机械服从一次检验"

* 最终汇总
display _n "========== 最终汇总：四模型对比 =========="
estimates table m_ols m_iv m_iv2, ///
    stats(N) b(%9.4f) se(%9.4f) keep(RD size age)

display _n "=== IV分析标准流程 ==="
display "  步骤1: OLS基准 → 确认存在内生性偏误的可能"
display "  步骤2: 第一阶段诊断 → 评估弱识别风险"
display "  步骤3: 2SLS估计 → 在识别假设成立时解释局部因果效应"
display "  步骤4: 过度识别/DWH检验 → 提供诊断证据，不替代制度论证"
display ""
display "  工具变量的核心不在于命令，而在于论证："
display "    为什么Z会影响X？（相关性机制）"
display "    为什么Z不会通过其他渠道影响Y？（外生性/排除限制）"
