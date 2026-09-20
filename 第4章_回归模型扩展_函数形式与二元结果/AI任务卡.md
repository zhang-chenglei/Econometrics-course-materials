# 第4章 AI Agent任务卡：AI使用与成绩的回归模型扩展——函数形式与二元结果

## 目标

在同一个AI与成绩案例中依次理解对数、AI使用时间的平方项、“AI×性别”交互项，以及连续成绩与二元达标结果回答的问题有何不同。

## 工作顺序

1. 只读列出各步模型的因变量、新增项、基准组、系数单位和目标解释量。
2. 经确认后运行现有形式扩展代码，保存模型比较表、AI使用曲线和分组预测曲线。
3. 核对平方项极值点是否位于样本AI使用范围内。
4. 最终模型同时含`ai²`和`ai×female`；在代表性AI取值处计算两组边际效应及置信区间，并报告组别AME。
5. 核对`ch04_representative_marginal_effects.csv`是否使用一次项、平方项和交互项的完整公式；旧的`ch04_group_slopes.csv`口径不得作为提交结果。
6. 按75分的事先达标线运行精简二元结果脚本，分别报告LPM的AI系数、Logit/Probit的AI导数型AME和三种模型的预测概率范围。
7. 将模型形式选择、结果类型取舍和因果解释边界记录到`ai_log.md`。

## 必须核验

- 水平模型与对数模型的 AI 系数单位不同；
- 二次项必须与一次项共同解释，倒 U 形不是二次项的必然结论；
- 含平方项和交互项时，完整边际效应为$\beta_{ai}+2\beta_{ai^2}ai+\beta_{ai\times female}female$；
- $\beta_{ai}+\beta_{ai\times female}$只对应女生在`ai=0`处的局部斜率，不能称为女生在所有AI水平上的恒定斜率；
- 加入交互项后 AI 主效应的含义会改变，不能直接与前一模型比较数值；
- LPM系数、Logit/Probit原始系数、胜算比、预测概率和AME不是同一个量；
- 连续结果二分必须有事先确定的实质理由，并说明信息损失；
- 换成Logit或Probit不会自动解决遗漏变量、反向因果或选择偏误；
- 模型不是越复杂越好，形式应服务于研究问题。

## 主要文件

- Python：`code/python/ch04/01_ai_score_form_specification.py`
- Python（二元结果）：`code/python/ch04/02_binary_outcome_models.py`
- Stata：`code/stata/ch04/01_ai_score_form_specification.do`
- Stata（二元结果）：`code/stata/ch04/02_binary_outcome_models.do`

连续成绩脚本应生成代表性边际效应和组别AME；二元结果脚本只保留LPM、Logit和Probit，不再承担有序选择模型。
