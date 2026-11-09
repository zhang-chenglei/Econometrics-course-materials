"""
第7讲工具变量案例：生成固定教学样本。

500 家虚构企业的单一截面数据。数据中人为设定：
- ability 同时影响 RD 与 ROE，因此 RD 内生；
- ResDensity 与 University 在 DGP 中为外生工具；
- RD 对 ROE 的真实结构系数为 1.2。

ability 只在生成时使用，不写入 CSV，学生在数据中看不到它。
"""

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 20250801
N = 500
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "firm_iv_case.csv"


def build_data(seed: int = SEED, n: int = N) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    res_density = rng.normal(0, 1, n)
    university = rng.normal(0, 1, n) + 0.3 * res_density
    ability = rng.normal(0, 1, n)

    size = np.clip(rng.normal(5, 1, n), 2, None)
    age = rng.uniform(1, 40, n)

    u_rd = rng.normal(0, 1, n)
    rd = (2.0 + 0.7 * res_density + 0.1 * university
          + 0.3 * size - 0.02 * age + 0.6 * ability + u_rd)
    rd = np.clip(rd, 0.1, None)

    v_roe = rng.normal(0, 1.5, n)
    roe = 5.0 + 1.2 * rd + 0.5 * size - 0.01 * age + 0.8 * ability + v_roe

    # 仅用于任务2的弱工具对照：与 RD 无关的随机变量。
    noise_iv = rng.normal(0, 1, n)

    return pd.DataFrame({
        "firm_id": np.arange(1, n + 1),
        "ROE": roe,
        "RD": rd,
        "ResDensity": res_density,
        "University": university,
        "size": size,
        "age": age,
        "noise_iv": noise_iv,
    })


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = build_data()
    df.to_csv(OUT, index=False, float_format="%.10f")
    print(f"written: {OUT} ({len(df)} rows)")
    print(df[["ROE", "RD", "ResDensity", "University", "size", "age"]].describe().round(3))


if __name__ == "__main__":
    main()
