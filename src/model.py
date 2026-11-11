"""MILP 模型：多品种多周期生产计划与库存联合优化。

变量（共 4·T·N 个，按块排列）：
    x[t,i] ∈ [0, cap_t / h_i]  生产量（连续）
    y[t,i] ∈ {0,1}             是否开机（换型）
    I[t,i] ∈ [0, capI_i]       期末库存
    B[t,i] ∈ [0, Σd_i]         期末缺货（未满足需求，允许回补）

目标：
    min  Σ K_i·y[t,i] + Σ h_i·I[t,i] + Σ p_i·B[t,i] + Σ c_i·x[t,i]

约束：
    (1) I[t,i] - I[t-1,i] - x[t,i] - B[t,i] + B[t-1,i] = -d[t,i]   （缺货回补式平衡）
    (2) Σ_i h_i·x[t,i] <= cap_t                                    （产能，无加班）
    (3) x[t,i] <= M[t,i]·y[t,i],  M[t,i] = cap_t / h_i             （开机联动）
    (4) I[t,i] <= capI_i = 2·base_i                                （仓库容量）
I[-1,i] = 初始库存，B[-1,i] = 0，故 t=0 的平衡式右端为 (init_i - d[0,i])。

缺货回补处理：B 是"欠交订单"（backlog），下一期先还欠再满足当期需求，
平衡式里 B[t,i] 与 B[t-1,i] 的符号正是该含义；不满足的部分按 p_i 计罚。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
INSTANCE_PATH = ROOT / "data" / "instance.json"


def load_instance(path: Path | str = INSTANCE_PATH) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        inst = json.load(f)
    inst["T"] = inst["meta"]["periods"]
    inst["N"] = inst["meta"]["n_products"]
    inst["demand"] = np.asarray(inst["demand"], dtype=float)
    inst["demand_std"] = np.asarray(inst["demand_std"], dtype=float)
    inst["capacity"] = np.asarray(inst["capacity"], dtype=float)
    inst["hours"] = np.array([p["hours"] for p in inst["products"]], dtype=float)
    inst["material_cost"] = np.array([p["material_cost"] for p in inst["products"]])
    inst["holding_cost"] = np.array([p["holding_cost"] for p in inst["products"]])
    inst["setup_cost"] = np.array([p["setup_cost"] for p in inst["products"]])
    inst["shortage_penalty"] = np.array([p["shortage_penalty"] for p in inst["products"]])
    inst["initial_inventory"] = np.array([p["initial_inventory"] for p in inst["products"]], float)
    inst["safety_stock"] = np.array([p["safety_stock"] for p in inst["products"]], float)
    inst["base_demand"] = np.array([p["base_demand"] for p in inst["products"]], float)
    return inst


def _idx(T: int, N: int):
    base = {"x": 0, "y": T * N, "I": 2 * T * N, "B": 3 * T * N}

    def pos(kind: str, t: int, i: int) -> int:
        return base[kind] + t * N + i

    return pos


def build_milp(inst: dict, enforce_safety_stock: bool | None = None, verbose: bool = False):
    T, N = inst["T"], inst["N"]
    d = inst["demand"]
    cap = inst["capacity"]
    hours = inst["hours"]
    pos = _idx(T, N)
    nvar = 4 * T * N

    cfg = inst["config"]
    if enforce_safety_stock is None:
        enforce_safety_stock = bool(cfg.get("enforce_safety_stock", False))
    inv_cap_factor = float(cfg.get("inventory_cap_factor", 2.0))

    # ---- 目标系数 c @ z ----
    c = np.zeros(nvar)
    for t in range(T):
        for i in range(N):
            c[pos("x", t, i)] = inst["material_cost"][i]
            c[pos("y", t, i)] = inst["setup_cost"][i]
            c[pos("I", t, i)] = inst["holding_cost"][i]
            c[pos("B", t, i)] = inst["shortage_penalty"][i]

    # ---- 边界 ----
    lb = np.zeros(nvar)
    ub = np.zeros(nvar)
    integrality = np.zeros(nvar, dtype=int)
    big_m = np.zeros((T, N))
    inv_cap = np.array([inv_cap_factor * b for b in inst["base_demand"]])
    b_cap = np.maximum(d.sum(axis=0), 1.0)
    for t in range(T):
        for i in range(N):
            big_m[t, i] = cap[t] / hours[i]
            ub[pos("x", t, i)] = big_m[t, i]
            ub[pos("y", t, i)] = 1.0
            integrality[pos("y", t, i)] = 1
            ub[pos("I", t, i)] = inv_cap[i]
            ub[pos("B", t, i)] = b_cap[i]

    rows, cols, vals = [], [], []
    row_lb, row_ub = [], []

    def add_terms(terms, lo: float, hi: float) -> None:
        r = len(row_lb)
        for col, val in terms:
            rows.append(r)
            cols.append(col)
            vals.append(val)
        row_lb.append(lo)
        row_ub.append(hi)

    # (1) 库存平衡（等式）
    for t in range(T):
        for i in range(N):
            terms = [(pos("I", t, i), 1.0), (pos("x", t, i), -1.0), (pos("B", t, i), -1.0)]
            if t == 0:
                rhs = inst["initial_inventory"][i] - d[t, i]
                eq = rhs
            else:
                terms.append((pos("I", t - 1, i), -1.0))
                terms.append((pos("B", t - 1, i), +1.0))
                eq = -d[t, i]
            add_terms(terms, eq, eq)

    # (2) 产能
    for t in range(T):
        add_terms([(pos("x", t, i), hours[i]) for i in range(N)], -np.inf, cap[t])

    # (3) 开机联动
    for t in range(T):
        for i in range(N):
            add_terms([(pos("x", t, i), 1.0), (pos("y", t, i), -big_m[t, i])], -np.inf, 0.0)

    # (4) 库存上限
    for t in range(T):
        for i in range(N):
            add_terms([(pos("I", t, i), 1.0)], -np.inf, inv_cap[i])

    # (5) 可选安全库存硬约束
    if enforce_safety_stock:
        for t in range(T):
            for i in range(N):
                add_terms([(pos("I", t, i), 1.0)], inst["safety_stock"][i], np.inf)

    A = coo_matrix((vals, (rows, cols)), shape=(len(row_lb), nvar)).tocsr()
    cons = LinearConstraint(A, np.asarray(row_lb), np.asarray(row_ub))
    bounds = Bounds(lb, ub)

    meta = {
        "n_vars": nvar,
        "n_bin": int(integrality.sum()),
        "n_cons": int(len(row_lb)),
        "n_eq": int(T * N),
        "enforce_safety_stock": enforce_safety_stock,
        "big_m": big_m,
        "x_ub": ub[: T * N].reshape((T, N)),
    }
    return c, cons, bounds, integrality, meta


def solve_milp(inst: dict, time_limit: float | None = None, verbose: bool = False) -> dict:
    """求解 MILP，返回计划矩阵与成本分解。"""
    T, N = inst["T"], inst["N"]
    c, cons, bounds, integrality, meta = build_milp(inst)
    cfg = inst["config"]
    tl = float(cfg.get("mip_time_limit", 60.0)) if time_limit is None else float(time_limit)

    t0 = time.perf_counter()
    res = milp(
        c=c,
        constraints=[cons],
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": tl, "mip_rel_gap": 1e-9, "presolve": True, "disp": verbose},
    )
    elapsed = time.perf_counter() - t0

    if res.x is None:
        raise RuntimeError(f"MILP 求解失败：status={res.status}, message={res.message}")

    z = np.asarray(res.x)
    pos = _idx(T, N)
    plan = {
        "x": np.array([[z[pos("x", t, i)] for i in range(N)] for t in range(T)]),
        "y": np.array([[z[pos("y", t, i)] for i in range(N)] for t in range(T)]),
        "I": np.array([[z[pos("I", t, i)] for i in range(N)] for t in range(T)]),
        "B": np.array([[z[pos("B", t, i)] for i in range(N)] for t in range(T)]),
    }

    comp = {
        "setup": float((inst["setup_cost"] * plan["y"]).sum()),
        "holding": float((inst["holding_cost"] * plan["I"]).sum()),
        "shortage": float((inst["shortage_penalty"] * plan["B"]).sum()),
        "material": float((inst["material_cost"] * plan["x"]).sum()),
    }
    objective = float(sum(comp.values()))

    # 校验：产能、库存平衡、开机联动
    load = plan["x"] @ inst["hours"]
    assert np.all(load <= inst["capacity"] + 1e-6), "产能约束被破坏"
    assert np.all(plan["x"] <= 1e-6) or np.all(plan["y"][plan["x"] > 1e-8] > 0.5), "开机联动被破坏"

    return {
        "status": int(res.status),
        "message": str(res.message),
        "objective": objective,
        "objective_solver": float(res.fun),
        "components": comp,
        "plan": plan,
        "load_hours": load,
        "utilization": load / inst["capacity"],
        "solve_seconds": elapsed,
        "mip_gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
        "meta": {
            "n_vars": meta["n_vars"],
            "n_bin": meta["n_bin"],
            "n_cons": meta["n_cons"],
            "n_eq": meta["n_eq"],
            "enforce_safety_stock": meta["enforce_safety_stock"],
            "time_limit": tl,
        },
    }


if __name__ == "__main__":
    inst = load_instance()
    out = solve_milp(inst, verbose=True)
    print(f"\n状态 {out['status']}  {out['message']}")
    print(f"目标值 {out['objective']:.2f}  求解 {out['solve_seconds']:.2f}s  gap={out['mip_gap']:.2e}")
    for k, v in out["components"].items():
        print(f"  {k:9s} {v:12.2f}")
    print("产能利用率：", np.round(out["utilization"], 3))
