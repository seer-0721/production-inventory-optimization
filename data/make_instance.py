"""生成虚拟工厂实例 data/instance.json（固定 seed=20260415）。

场景：6 个产品族 × 12 个月的多品种多周期生产计划与库存联合优化。
需求 = 基准需求 × 季节因子 × 对数正态扰动（seed 固定，可复现）；
产能 = 名义需求总工时 / 目标利用率（目标利用率 < 1，保证可行但整体紧张）。

运行：
    python data/make_instance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 20260415
PERIODS = 12
HOLDING_RATE = 0.03  # 月度持有成本率（占单位材料成本），约合 36%/年
TARGET_UTILIZATION = 0.89  # 名义需求总工时 / 总产能（越低越宽松）

# 季节因子：Q4 旺季（10/11/12 月），年初淡季
SEASON = [0.85, 0.90, 1.00, 1.00, 0.95, 1.00, 1.05, 1.00, 1.05, 1.15, 1.30, 1.20]

# id, 名称, 单位工时(h/件), 单位材料成本, 换型成本, 缺货惩罚, 基准月需求, 需求变异系数
PRODUCTS = [
    ("P1", "液压阀体", 0.9, 38.0, 420.0, 95.0, 320, 0.12),
    ("P2", "精密齿轮组", 1.2, 45.0, 520.0, 110.0, 260, 0.10),
    ("P3", "伺服支架", 1.6, 30.0, 380.0, 78.0, 180, 0.14),
    ("P4", "主轴套筒", 2.0, 55.0, 640.0, 140.0, 140, 0.11),
    ("P5", "密封组件", 0.8, 42.0, 460.0, 100.0, 220, 0.13),
    ("P6", "连杆毛坯", 1.4, 26.0, 300.0, 68.0, 160, 0.10),
]

SAFETY_FACTOR = 0.35  # 安全库存 = 0.35 个月基准需求
INIT_FACTOR = 0.6  # 初始库存 = 0.6 × 安全库存


def build_instance() -> dict:
    rng = np.random.default_rng(SEED)
    n = len(PRODUCTS)
    t_len = PERIODS

    products = []
    for pid, name, hours, mat, setup, penalty, base, cv in PRODUCTS:
        products.append(
            {
                "id": pid,
                "name": name,
                "hours": float(hours),
                "material_cost": float(mat),
                "holding_cost": round(HOLDING_RATE * mat, 2),
                "setup_cost": float(setup),
                "shortage_penalty": float(penalty),
                "base_demand": int(base),
                "demand_cv": float(cv),
                "safety_stock": int(round(SAFETY_FACTOR * base)),
                "initial_inventory": int(round(INIT_FACTOR * SAFETY_FACTOR * base)),
            }
        )

    # 名义需求：基准需求 × 季节因子 × 对数正态扰动（σ≈5%）
    demand = np.zeros((t_len, n), dtype=int)
    demand_std = np.zeros((t_len, n), dtype=float)
    for t in range(t_len):
        for i, p in enumerate(products):
            noise = float(np.exp(rng.normal(0.0, 0.05)))
            mu = p["base_demand"] * SEASON[t]
            demand[t, i] = int(round(mu * noise))
            # 仿真重抽样用的标准差：由基准需求 × 季节因子决定，不随本次抽样波动
            demand_std[t, i] = p["demand_cv"] * mu

    # 产能：名义需求总工时 / 目标利用率，再平均到每个周期（单班制，不加班）
    total_hours = float(
        sum(demand[t, i] * products[i]["hours"] for t in range(t_len) for i in range(n))
    )
    capacity = int(np.ceil(total_hours / (TARGET_UTILIZATION * t_len)))

    inst = {
        "meta": {
            "name": "prod-inventory-opt / 6 族 × 12 月 生产-库存联合优化实例",
            "seed": SEED,
            "periods": t_len,
            "n_products": n,
            "time_unit": "月",
            "currency": "元",
            "generator": "data/make_instance.py",
            "note": "全部数据为合成场景，非真实工厂数据",
        },
        "season": SEASON,
        "products": products,
        "demand": demand.tolist(),
        "demand_std": np.round(demand_std, 2).tolist(),
        "capacity": [capacity] * t_len,
        "config": {
            "production_integer": False,  # x 为连续量（可约定为批量下的连续近似）
            "inventory_cap_factor": 2.0,  # 仓库容量 = 2.0 × 基准月需求
            "enforce_safety_stock": False,  # MILP 不硬性要求安全库存，风险由缺货惩罚定价
            "mip_time_limit": 60.0,
        },
    }
    return inst


def main() -> None:
    inst = build_instance()
    out = Path(__file__).resolve().parent / "instance.json"
    out.write_text(json.dumps(inst, ensure_ascii=False, indent=1), encoding="utf-8")

    T, N = inst["meta"]["periods"], inst["meta"]["n_products"]
    d = np.array(inst["demand"])
    hours = np.array([p["hours"] for p in inst["products"]])
    cap = inst["capacity"][0]
    load = d.sum(axis=1) * 1.0
    period_hours = (d * hours).sum(axis=1)
    print(f"实例已写入 {out}  ({out.stat().st_size/1024:.1f} KB)")
    print(f"周期数={T}  产品族={N}  每期产能={cap} h  总产能={cap*T} h")
    print(f"名义需求总量={d.sum()} 件  总工时需求={period_hours.sum():.0f} h"
          f"  整体产能利用率={period_hours.sum()/(cap*T):.1%}")
    print("分周期产能利用率：")
    for t in range(T):
        print(f"  M{t+1:02d} 需求 {load[t]:5.0f} 件  工时 {period_hours[t]:7.0f} h"
              f"  利用率 {period_hours[t]/cap:6.1%}")


if __name__ == "__main__":
    main()
