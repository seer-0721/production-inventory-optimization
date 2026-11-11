"""一键跑通全流程：生成/载入实例 → 求解 MILP → 蒙特卡洛仿真 → 出图 → 写 metrics.json。

用法：
    cd prod-inventory-opt
    python src/run_all.py
可选：
    --seeds 30        蒙特卡洛种子个数（默认 30，种子编号 0..seeds-1）
    --rebuild         强制重新生成 data/instance.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "data"))

import make_instance  # noqa: E402
import plots  # noqa: E402
from model import load_instance, solve_milp  # noqa: E402
from simulate import SEEDS, evaluate_on_nominal, monte_carlo  # noqa: E402

INSTANCE_PATH = ROOT / "data" / "instance.json"
OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"
ORDER = ["mip", "fixed_batch", "eoq", "sS"]
LABELS = {
    "mip": "MIP 联合优化",
    "fixed_batch": "固定批量",
    "eoq": "EOQ 分别订货",
    "sS": "(s,S) 策略",
}


def rnd(obj, nd: int = 2):
    """递归四舍五入，保证 metrics.json 里都是可读数字。"""
    if isinstance(obj, dict):
        return {k: rnd(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [rnd(v, nd) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, nd)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return rnd(obj.tolist(), nd)
    return obj


def ensure_instance(rebuild: bool = False) -> Path:
    if rebuild or not INSTANCE_PATH.exists():
        make_instance.main()
    return INSTANCE_PATH


def plan_to_csv(mip: dict, inst: dict, path: Path) -> None:
    lines = ["period,product,produce,setup_flag,end_inventory,backlog,demand"]
    d = inst["demand"]
    for t in range(inst["T"]):
        for i, p in enumerate(inst["products"]):
            lines.append(
                f"{t + 1},{p['id']},{mip['plan']['x'][t, i]:.4f},{int(round(mip['plan']['y'][t, i]))},"
                f"{mip['plan']['I'][t, i]:.4f},{mip['plan']['B'][t, i]:.4f},{d[t, i]:.0f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_metrics(inst: dict, mip: dict, det: dict, mc: dict, runtime: float,
                  robust: dict | None = None) -> dict:
    T, N = inst["T"], inst["N"]
    demand_hours = float((inst["demand"] @ inst["hours"]).sum())
    period_hours = inst["demand"] @ inst["hours"]
    summ = mc["summary"]
    mip_cost = summ["mip"]["total_cost"]["mean"]
    mip_fill = summ["mip"]["fill_rate"]["mean"]

    comparison = {
        "cost_reduction_vs_pct": {k: (summ[k]["total_cost"]["mean"] - mip_cost) / mip_cost * 100
                                  for k in ORDER if k != "mip"},
        "fill_rate_gap_pp": {k: (mip_fill - summ[k]["fill_rate"]["mean"]) * 100
                             for k in ORDER if k != "mip"},
        "unserved_ratio_vs_mip": {k: summ[k]["unserved_units"]["mean"] / max(summ["mip"]["unserved_units"]["mean"], 1e-9)
                                  for k in ORDER if k != "mip"},
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(runtime, 1),
        "meta": {
            "project": "多品种多周期生产计划与库存联合优化（MILP + 蒙特卡洛评估）",
            "course": "东华大学 机械工程学院 智能制造工程 《生产系统智能管理与决策》课程设计",
            "textbook": "《生产运作管理》第 7 章 生产计划、第 8 章 库存管理",
            "instance_file": "data/instance.json",
            "instance_seed": inst["meta"]["seed"],
            "simulation_seeds": f"0..{len(mc['seeds']) - 1}",
            "n_simulation_seeds": len(mc["seeds"]),
            "note": "全部数据为合成场景，指标均由 src/run_all.py 实际运行得到",
        },
        "instance": {
            "T": T,
            "N": N,
            "capacity_per_period": inst["capacity"][0],
            "total_capacity_hours": float(inst["capacity"].sum()),
            "nominal_demand_units": float(inst["demand"].sum()),
            "nominal_demand_hours": demand_hours,
            "overall_utilization": demand_hours / float(inst["capacity"].sum()),
            "peak_period": int(np.argmax(period_hours)) + 1,
            "peak_period_utilization": float(period_hours.max() / inst["capacity"].max()),
            "period_utilization": (period_hours / inst["capacity"]).tolist(),
            "period_demand_hours": period_hours.tolist(),
            "products": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "hours_per_unit": p["hours"],
                    "material_cost": p["material_cost"],
                    "setup_cost": p["setup_cost"],
                    "holding_cost": p["holding_cost"],
                    "shortage_penalty": p["shortage_penalty"],
                    "initial_inventory": p["initial_inventory"],
                    "safety_stock": p["safety_stock"],
                    "base_demand": p["base_demand"],
                }
                for p in inst["products"]
            ],
        },
        "milp": {
            "status": mip["status"],
            "message": mip["message"],
            "objective": mip["objective"],
            "components": mip["components"],
            "solve_seconds": mip["solve_seconds"],
            "mip_gap": mip["mip_gap"],
            "n_vars": mip["meta"]["n_vars"],
            "n_binary": mip["meta"]["n_bin"],
            "n_constraints": mip["meta"]["n_cons"],
            "n_equalties": mip["meta"]["n_eq"],
            "enforce_safety_stock": mip["meta"]["enforce_safety_stock"],
            "production_integer": inst["config"]["production_integer"],
            "load_hours": mip["load_hours"],
            "utilization": mip["utilization"],
            "setups": int((mip["plan"]["x"] > 1e-9).sum()),
            "plan_x": mip["plan"]["x"],
            "plan_y": mip["plan"]["y"],
            "plan_I": mip["plan"]["I"],
            "plan_B": mip["plan"]["B"],
        },
        "deterministic_nominal": det,
        "monte_carlo": {
            "seeds": mc["seeds"],
            "summary": summ,
            "trajectory": mc["trajectory"],
        },
        "comparison": comparison,
        "robustness": robust or {},
        "figures": sorted(p.name for p in FIG_DIR.glob("*.png")),
    }


def print_report(metrics: dict) -> None:
    inst = metrics["instance"]
    milp = metrics["milp"]
    summ = metrics["monte_carlo"]["summary"]
    print("\n" + "=" * 86)
    print("实例概况")
    print("=" * 86)
    print(f"  {inst['N']} 个产品族 × {inst['T']} 个计划周期；每期产能 {inst['capacity_per_period']:.0f} h，"
          f"总产能 {inst['total_capacity_hours']:.0f} h")
    print(f"  名义需求 {inst['nominal_demand_units']:.0f} 件 / {inst['nominal_demand_hours']:.0f} h；"
          f"整体产能利用率 {inst['overall_utilization']:.1%}，"
          f"峰值月 M{inst['peak_period']} 达 {inst['peak_period_utilization']:.1%}")

    print("\n" + "=" * 86)
    print("MILP 求解（scipy.optimize.milp / HiGHS）")
    print("=" * 86)
    print(f"  变量 {milp['n_vars']}（0/1 变量 {milp['n_binary']}），约束 {milp['n_constraints']}，"
          f"求解 {milp['solve_seconds']:.2f} s，gap {milp['mip_gap']:.2e}，状态 {milp['status']}")
    c = milp["components"]
    print(f"  目标值 {milp['objective']:,.2f} 元 = 材料 {c['material']:,.0f} + 换型 {c['setup']:,.0f}"
          f" + 持有 {c['holding']:,.0f} + 缺货 {c['shortage']:,.0f}")

    print("\n" + "=" * 86)
    print(f"蒙特卡洛仿真（{metrics['meta']['n_simulation_seeds']} 个种子，需求重抽样）")
    print("=" * 86)
    header = f"{'策略':<14}{'总成本(元)':>16}{'标准差':>11}{'满足率':>9}{'平均库存(件)':>13}{'未满足(件)':>11}{'周转率':>8}"
    print(header)
    print("-" * 86)
    for k in ORDER:
        s = summ[k]
        print(f"{LABELS[k]:<14}{s['total_cost']['mean']:>16,.0f}{s['total_cost']['std']:>11,.0f}"
              f"{s['fill_rate']['mean'] * 100:>8.2f}%{s['avg_inventory']['mean']:>13.1f}"
              f"{s['unserved_units']['mean']:>11.0f}{s['turnover']['mean']:>8.2f}")

    comp = metrics["comparison"]
    print("\n相对 MIP 的成本差 / 满足率差：")
    for k in ORDER:
        if k == "mip":
            continue
        print(f"  {LABELS[k]:<14} 成本 +{comp['cost_reduction_vs_pct'][k]:6.1f}%   "
              f"满足率 {comp['fill_rate_gap_pp'][k]:+.2f} pp   "
              f"未满足需求为 MIP 的 {comp['unserved_ratio_vs_mip'][k]:.1f} 倍")
    print("=" * 86 + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生产计划与库存联合优化全流程")
    ap.add_argument("--seeds", type=int, default=len(SEEDS), help="蒙特卡洛种子个数（默认 30）")
    ap.add_argument("--rebuild", action="store_true", help="重新生成 data/instance.json")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ensure_instance(rebuild=args.rebuild)
    inst = load_instance(INSTANCE_PATH)
    print(f"[1/4] 实例载入：{INSTANCE_PATH.relative_to(ROOT)} "
          f"({inst['T']}×{inst['N']}，seed={inst['meta']['seed']})")

    mip = solve_milp(inst)
    print(f"[2/4] MILP 求解完成：目标值 {mip['objective']:,.2f} 元，"
          f"{mip['solve_seconds']:.2f}s，gap {mip['mip_gap']:.2e}")

    det = evaluate_on_nominal(inst, mip["plan"])
    gap = abs(det["mip"]["total_cost"] - mip["objective"]) / mip["objective"]
    assert gap < 1e-6, f"仿真评估与 MILP 目标值不一致（相对差 {gap:.2e}），检查成本口径"
    print(f"[3/4] 名义需求复核：仿真评估 {det['mip']['total_cost']:,.2f} 元，"
          f"与模型目标值相对差 {gap:.2e}")

    seeds = list(range(args.seeds))
    mc = monte_carlo(inst, mip["plan"], seeds=seeds)
    print(f"[4/4] 蒙特卡洛完成：{len(seeds)} 个种子 × 4 个策略")

    # 鲁棒性检查：把产能截断规则从"按价值密度优先"换成"等比例削减"，看结论是否稳定
    mc_prop = monte_carlo(inst, mip["plan"], seeds=seeds, truncation="proportional")
    robust = {
        "rule": "proportional_truncation",
        "summary": mc_prop["summary"],
        "cost_reduction_vs_pct": {
            k: (mc_prop["summary"][k]["total_cost"]["mean"]
                - mc_prop["summary"]["mip"]["total_cost"]["mean"])
            / mc_prop["summary"]["mip"]["total_cost"]["mean"] * 100
            for k in ORDER if k != "mip"
        },
    }
    print(f"      鲁棒性（等比例截断）：" + "，".join(
        f"{LABELS[k]} {v:+.1f}%" for k, v in robust["cost_reduction_vs_pct"].items()))

    runtime = time.perf_counter() - t_start
    metrics = rnd(build_metrics(inst, mip, det, mc, runtime, robust), 4)
    metrics["figures"] = []
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    plan_to_csv(mip, inst, OUT_DIR / "plan_mip.csv")

    figs = plots.plot_all(inst, mip, mc)
    print("出图：" + "、".join(p.name for p in figs))

    # 更新 figures 列表后重写一次（图名在 metrics 里可查）
    metrics["figures"] = [p.name for p in figs]
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    if not args.quiet:
        print_report(metrics)
    print(f"完成。总耗时 {runtime:.1f}s；结果见 outputs/metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
