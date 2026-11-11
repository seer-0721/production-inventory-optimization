"""周期盘点仿真 + 蒙特卡洛评估。

仿真协议（对所有策略完全一致）
------------------------------
* 每期先决策产量 x_t，再实现需求 d_t，最后结算库存/缺货。
* 可服务量 = 期初库存 + 本期产量；需求 = 本期需求 + 上期欠交 backlog。
* served = min(可服务量, 需求)；I_t = 可服务量 - served；B_t = 需求 - served。
* 成本口径（四个策略完全一致）：
    setup    = Σ K_i · 1[x_{t,i} > 0]                （每期每族开机一次计一次换型）
    holding  = Σ h_i · I_{t,i}                        （期末库存 × 月持有成本）
    shortage = Σ p_i · B_{t,i}                        （期末欠交 × 单位缺货惩罚）
    material = Σ c_i · x_{t,i}                        （投产材料成本）
* MIP 计划在仿真中**固定**（不随实现需求重新优化），基线则是按实现库存状态反应的策略。
* 产能截断规则：当 Σ h_i·req_i > cap_t 时，按"单位工时缺货惩罚 p_i/h_i"降序分配产能，
  高价值密度族优先，分不到的产量作废（不跨期结转，对应"产能不足时丢单"）。

随机需求：d_sim = max(0, round(nominal + σ·z))，σ 见 instance["demand_std"]，
seed = 0..29，每 seed 一个独立 rng(seed)。
"""
from __future__ import annotations

import numpy as np

from policies import build_policies

SEEDS = list(range(30))


def sample_demand(inst: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(inst["demand"].shape)
    d = inst["demand"] + inst["demand_std"] * z
    return np.maximum(0.0, np.round(d))


def priority_order(inst: dict) -> np.ndarray:
    """按单位工时缺货惩罚（价值密度）降序的产能优先级。"""
    return np.argsort(-(inst["shortage_penalty"] / inst["hours"]))


def allocate_capacity(req: np.ndarray, cap_t: float, inst: dict, prio: np.ndarray,
                      mode: str = "priority") -> np.ndarray:
    """在单期产能 cap_t 下分配产量。

    mode="priority"     按 p_i/h_i 降序装满，高价值密度族优先（默认，见模块说明）；
    mode="proportional" 各族按同一比例削减（鲁棒性检查用）。
    """
    hours = inst["hours"]
    need = float(req @ hours)
    if need <= cap_t + 1e-9:
        return req.copy()
    if mode == "proportional":
        return req * (cap_t / need)
    alloc = np.zeros_like(req)
    left = cap_t
    for i in prio:
        if left <= 1e-9:
            break
        max_i = min(req[i], left / hours[i])
        if max_i > 0:
            alloc[i] = max_i
            left -= max_i * hours[i]
    return alloc


def simulate_path(inst: dict, demand: np.ndarray, prod_fn, prio: np.ndarray | None = None,
                  truncation: str = "priority") -> dict:
    """跑一条需求路径，返回成本、服务水平与逐期轨迹。"""
    T, N = inst["T"], inst["N"]
    hours = inst["hours"]
    K, h, p, c = inst["setup_cost"], inst["holding_cost"], inst["shortage_penalty"], inst["material_cost"]
    if prio is None:
        prio = priority_order(inst)

    inv = inst["initial_inventory"].astype(float).copy()
    backlog = np.zeros(N)
    comp = {"setup": 0.0, "holding": 0.0, "shortage": 0.0, "material": 0.0}
    tr_inv, tr_backlog, tr_x, tr_served, tr_load, tr_trunc = [], [], [], [], [], []
    served_total = 0.0
    trunc_total = 0.0
    setups = 0

    for t in range(T):
        req = np.asarray(prod_fn(t, inv, backlog), dtype=float)
        req = np.maximum(req, 0.0)
        cap_t = inst["capacity"][t]
        if req @ hours <= cap_t + 1e-9:
            alloc = req
        else:
            alloc = allocate_capacity(req, cap_t, inst, prio, mode=truncation)
            trunc_total += float((req - alloc).sum())

        avail = inv + alloc
        need = demand[t] + backlog
        served = np.minimum(avail, need)
        inv = avail - served
        backlog = need - served

        on = alloc > 1e-9
        setups += int(on.sum())
        comp["setup"] += float((K * on).sum())
        comp["holding"] += float((h * inv).sum())
        comp["shortage"] += float((p * backlog).sum())
        comp["material"] += float((c * alloc).sum())

        served_total += float(served.sum())
        tr_inv.append(inv.copy())
        tr_backlog.append(backlog.copy())
        tr_x.append(alloc.copy())
        tr_served.append(served.copy())
        tr_load.append(float(alloc @ hours))
        tr_trunc.append(float((req - alloc).sum()))

    inv = np.asarray(tr_inv)
    backlog_arr = np.asarray(tr_backlog)
    x_arr = np.asarray(tr_x)
    demand_total = float(demand.sum())
    total_cost = float(sum(comp.values()))
    inv_total = inv.sum(axis=1)
    avg_inv = float(inv_total.mean())
    res = {
        "total_cost": total_cost,
        "components": {k: float(v) for k, v in comp.items()},
        "fill_rate": served_total / demand_total if demand_total > 0 else 1.0,
        "demand_total": demand_total,
        "served_total": served_total,
        "shortage_units": float(backlog_arr.sum()),  # 件·月（期末欠交的累计）
        "unserved_units": demand_total - served_total,  # 未满足需求件数（唯一计一次）
        "avg_inventory": avg_inv,
        "avg_inventory_value": float((inv * c).sum(axis=1).mean()),
        "turnover": served_total / avg_inv if avg_inv > 0 else float("inf"),
        "stockout_product_periods": int((backlog_arr > 1e-6).sum()),
        "production_total": float(x_arr.sum()),
        "setups": setups,
        "truncated_units": trunc_total,
        "max_inventory": float(inv.max()),
        "inv_traj": inv_total,
        "backlog_traj": backlog_arr.sum(axis=1),
        "load_traj": np.asarray(tr_load),
        "x_traj": x_arr,
        "demand_used": np.asarray(demand, dtype=float),
    }
    return res


def make_mip_prod_fn(plan: dict):
    x = plan["x"]

    def prod_fn(t, inv, backlog):
        return x[t]

    return prod_fn


def evaluate_all(inst: dict, mip_plan: dict, demand: np.ndarray,
                 truncation: str = "priority") -> dict:
    """在同一条需求路径上评估 MIP 计划与三个基线。"""
    prio = priority_order(inst)
    out = {"mip": simulate_path(inst, demand, make_mip_prod_fn(mip_plan), prio, truncation)}
    for name, pol in build_policies(inst).items():
        out[name] = simulate_path(inst, demand, lambda t, inv, bl, _p=pol: _p.requests(t, inv, bl),
                                  prio, truncation)
    return out


def monte_carlo(inst: dict, mip_plan: dict, seeds: list[int] | None = None,
                truncation: str = "priority") -> dict:
    """30 个种子的蒙特卡洛，返回每策略的逐 seed 结果与聚合统计。"""
    seeds = SEEDS if seeds is None else list(seeds)
    per_seed = {k: [] for k in ("mip", "fixed_batch", "eoq", "sS")}
    traj = {k: {"inv": [], "backlog": [], "load": [], "x": []} for k in per_seed}

    for s in seeds:
        demand = sample_demand(inst, s)
        res = evaluate_all(inst, mip_plan, demand, truncation)
        for k, r in res.items():
            per_seed[k].append(
                {kk: v for kk, v in r.items()
                 if kk not in ("inv_traj", "backlog_traj", "load_traj", "x_traj", "demand_used")}
            )
            traj[k]["inv"].append(r["inv_traj"])
            traj[k]["backlog"].append(r["backlog_traj"])
            traj[k]["load"].append(r["load_traj"])
            traj[k]["x"].append(r["x_traj"])

    summary = {}
    for k, rows in per_seed.items():
        fields = ["total_cost", "fill_rate", "avg_inventory", "avg_inventory_value",
                  "shortage_units", "unserved_units", "turnover", "served_total", "production_total",
                  "setups", "truncated_units", "stockout_product_periods", "max_inventory"]
        agg = {}
        for f in fields:
            arr = np.array([r[f] for r in rows], dtype=float)
            agg[f] = {"mean": float(arr.mean()),
                      "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                      "min": float(arr.min()), "max": float(arr.max())}
        for comp in ("setup", "holding", "shortage", "material"):
            arr = np.array([r["components"][comp] for r in rows], dtype=float)
            agg[f"component_{comp}"] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
        agg["cost_share"] = {
            c: float(np.mean([r["components"][c] for r in rows]) / max(np.mean([r["total_cost"] for r in rows]), 1e-9))
            for c in ("setup", "holding", "shortage", "material")
        }
        summary[k] = agg

    trajectory = {
        k: {kk: np.mean(np.asarray(vv), axis=0) for kk, vv in d.items()} for k, d in traj.items()
    }
    return {"seeds": seeds, "per_seed": per_seed, "summary": summary, "trajectory": trajectory}


def evaluate_on_nominal(inst: dict, mip_plan: dict) -> dict:
    out = evaluate_all(inst, mip_plan, inst["demand"])
    return {k: {kk: v for kk, v in r.items()
                if kk not in ("inv_traj", "backlog_traj", "load_traj", "x_traj", "demand_used")}
            for k, r in out.items()}
