# 第十讲：DML 与因果森林｜课堂演示包

> 版本：2026-11-09  
> 配套课件：《第十讲_机器学习在计量经济学中的应用_阅读版》75 页  
> 定位：**教学模拟**，不是论文原始数据，也不是赵晶等、周迪等论文的严格复现。

## 1. 这个文件夹是做什么的

这份演示按“提出问题 → 跑一段代码 → 看一张表/图 → 用两三句话解释”的节奏展开：

1. **OLS 与 DML**：同样真值 2，线性控制和非线性控制下会发生什么。
2. **因果森林**：多棵树怎样形成局部参考权重，最后输出 CATE、CATE 分布和高低效应组差异检验。

代码默认参数与随机种子已经固定，结果可复现。运行后会生成 `data/` 和 `outputs/`；初始文件夹里只有代码和文档，没有预生成结果。

## 2. 文件夹结构

```
lecture10_ml_causal_forest/
├── README.md              # 本文件：总览与快速开始
├── 项目指南.md            # 给你的 AI 助手的逐步骤操作指南（学生视角）
├── 完整代码.py            # 单文件完整代码，支持分步运行
├── requirements.txt       # Python 依赖
├── data/                  # 运行后生成：三份模拟数据
└── outputs/               # 运行后生成：DML 两张图；森林在 森林_后续/
```

## 3. 一键运行

```bash
cd "lecture10_ml_causal_forest"
python3 完整代码.py --step all
```

首次运行约 **2.5 分钟**（DML 50 次，右图另含 GBM/NN 各 50 次）。运行后会生成两张 DML 结果图。

如果本机还没装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python3 完整代码.py --step all
```

## 4. 分步运行

```bash
python3 完整代码.py --step data     # 只生成三份数据
python3 完整代码.py --step dml      # 只跑 OLS/DML
python3 完整代码.py --step forest   # 只跑因果森林
```

每一步都会把结果写入 `data/` 或 `outputs/`，不会删除已有结果。

## 5. 怎么开始

**先看 DML**：第 1—2 步，只看两张图——《`outputs/dml_three_line_table.png`》三线表和《`outputs/dml_density_comparison.png`》密度分布图。因果森林（第 3—5 步）后面再看。

把整个文件夹和 `项目指南.md` 交给你的 AI 助手（Claude Code、Cursor、Codex、Windsurf、GitHub Copilot 等都可以），对它说：

> 请读取 `项目指南.md`，按第 1—5 步执行；每一步只运行指南里的命令，不要修改随机种子；运行后先展示指定图片，再用“怎么读”向我解释。

如果你的 AI 助手不能运行终端，就把指南里的命令复制到终端自己运行，再打开 `outputs/` 里的图片看。每一节都写了“你要做 / 你会看到 / 怎么读 / 别误会”。

## 6. 重要提醒

- 所有数据都是**教学模拟**；四类企业 1/5/1/5 是模拟设定，不是论文估计结果。
- DML 真值是 2；非线性情景下，**OLS 线性与 DML-Lasso 约 2.6**，**OLS 正确设定与 DML-随机森林/梯度提升/神经网络约 2.0**。DML 不是自动更好，取决于辅助学习器。
- **正确设定的 OLS**（显式加入 `x1²`、`sin(2x2)`）也能估对；这是本演示特意加入的对照。
- 覆盖率是**模拟评价量**；本演示不承诺重现 PPT 第 39 页的 79%。
- 因果森林输出的是 **CATE（条件平均处理效应）**，不是企业真实个体处理效应（ITE）。
- `完整代码.py` 里的森林是**教学版诚实树集成**，用于解释思想，不是官方 GRF/EconML 实现，不能作为论文方法引用。
- 随机试点 `trial_W` 由模拟设计保证可比；真实观察性数据不能直接照搬组均值差。

## 7. 结果速查（默认种子）

| 项目 | 预期结果 |
|---|---|
| DML 对比表（非线性，真值 2） | OLS 线性 2.605；OLS 正确 1.984；DML-Lasso 2.607；DML-随机森林 2.033；DML-梯度提升树 2.012；DML-神经网络 1.982 |
| DML 密度图 | 线性情景三种主方法≈2.0；非线性 OLS 线性/DML-Lasso≈2.6，OLS 正确/DML-随机森林/GBM/NN≈2.0 |
| DML 覆盖率（50 次） | 线性各方法≈0.90—0.92；非线性 OLS 正确≈0.92，DML-神经网络≈0.78、随机森林≈0.48、梯度提升≈0.42，OLS 线性/DML-Lasso≈0.00 |
| DML 结果图 | `outputs/dml_density_comparison.png`、`outputs/dml_three_line_table.png` |
| 森林四组（小/低、小/高、大/低、大/高） | ≈1.07 / 4.99 / 1.08 / 5.10（理论 1/5/1/5） |
| 高低预测组 | 低≈1.60，高≈4.51；高减低≈2.90，p≈7.5e-8 |
| 第一刀 | 60 棵树全部先按 `high_rd` 切 |
| 森林结果图 | `outputs/森林_后续/forest_demo.png`、`outputs/森林_后续/forest_three_line_table.png` |

详细解读见 `项目指南.md`。
