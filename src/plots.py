"""出图到 figures/（中文标注，Noto Sans CJK）。

图：
  fig1_demand_capacity.png  需求工时 vs 产能、MIP 计划生产工时 vs 产能
  fig2_inventory.png        四策略平均期末库存 / 欠交轨迹（蒙特卡洛均值）
  fig3_cost.png             成本构成堆叠对比 + 相对 MIP 的成本差
  fig4_service.png          满足率与未满足需求件数对比
  fig5_mip_plan.png         MIP 生产计划热力图（6 族 × 12 月）
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

COLORS = {
    "mip": "#1f4e79",
    "fixed_batch": "#e07b39",
    "eoq": "#8c6d31",
    "sS": "#3a7d44",
}
LABELS = {
    "mip": "MIP 联合优化",
    "fixed_batch": "固定批量",
    "eoq": "EOQ 分别订货",
    "sS": "(s,S) 策略",
}
ORDER = ["mip", "fixed_batch", "eoq", "sS"]
COMP_COLORS = {
    "material": "#9aa5b1",
    "setup": "#f2b134",
    "holding": "#4c9f70",
    "shortage": "#c0392b",
}
COMP_LABELS = {
    "material": "材料成本",
    "setup": "换型成本",
    "holding": "持有成本",
    "shortage": "缺货惩罚",
}


def setup_mpl() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.family": "sans-serif",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def _periods(T: int) -> list[str]:
    return [f"M{t + 1}" for t in range(T)]


def fig_demand_capacity(inst: dict, mip: dict, path: Path) -> Path:
    T = inst["T"]
    x = np.arange(T)
    w = 0.65
    demand_hours = inst["demand"] @ inst["hours"]
    cap = inst["capacity"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, (vals, title, color) in zip(
        axes,
        [
            (demand_hours, "(a) 名义需求工时 vs 产能", "#7fa8c9"),
            (mip["load_hours"], "(b) MIP 计划生产工时 vs 产能", "#1f4e79"),
        ],
    ):
        ax.bar(x, vals, width=w, color=color, label=title.split(") ")[1].split(" vs")[0])
        ax.axhline(cap[0], color="#c0392b", ls="--", lw=1.6, label="每期可用产能")
        ax.set_xticks(x, _periods(T))
        ax.set_xlabel("计划周期（月）")
        ax.set_ylabel("工时（h）")
        ax.set_title(title)
        for t in range(T):
            r = vals[t] / cap[t]
            if r > 1.0:
                ax.text(t, vals[t] + 25, f"{r:.0%}", ha="center", va="bottom",
                        fontsize=8, color="#c0392b", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("产能约束：旺季单月需求超出可用产能，需靠提前生产平滑", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_inventory(mc: dict, path: Path) -> Path:
    traj = mc["trajectory"]
    T = len(traj["mip"]["inv"])
    x = np.arange(T)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for k in ORDER:
        axes[0].plot(x, traj[k]["inv"], marker="o", ms=3.5, lw=1.6,
                     color=COLORS[k], label=LABELS[k])
        axes[1].plot(x, traj[k]["backlog"], marker="s", ms=3.5, lw=1.6,
                     color=COLORS[k], label=LABELS[k])
    axes[0].set_title("(a) 平均期末库存（6 族合计，30 个种子均值）")
    axes[0].set_ylabel("库存（件）")
    axes[1].set_title("(b) 平均期末欠交量（缺货深度）")
    axes[1].set_ylabel("欠交（件）")
    for ax in axes:
        ax.set_xticks(x, _periods(T))
        ax.set_xlabel("计划周期（月）")
        ax.legend(fontsize=8)
    axes[1].axvspan(9, 11, color="#c0392b", alpha=0.06)
    axes[1].text(9.1, axes[1].get_ylim()[1] * 0.92, "Q4 旺季", color="#c0392b", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_cost(mc: dict, path: Path) -> Path:
    summ = mc["summary"]
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    bottom = np.zeros(len(ORDER))
    for comp in ("material", "setup", "holding", "shortage"):
        vals = np.array([summ[k][f"component_{comp}"]["mean"] for k in ORDER]) / 1e4
        axes[0].bar(x, vals, bottom=bottom, width=0.6, color=COMP_COLORS[comp],
                    label=COMP_LABELS[comp], edgecolor="white", linewidth=0.6)
        bottom += vals
    totals = np.array([summ[k]["total_cost"]["mean"] for k in ORDER]) / 1e4
    errs = np.array([summ[k]["total_cost"]["std"] for k in ORDER]) / 1e4
    axes[0].errorbar(x, totals, yerr=errs, fmt="none", ecolor="#333333", capsize=4, lw=1.2)
    for xi, tot, err in zip(x, totals, errs):
        axes[0].text(xi, tot + err + 5.0, f"{tot:.1f}", ha="center", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x, [LABELS[k] for k in ORDER], fontsize=9)
    axes[0].set_ylabel("总成本（万元）")
    axes[0].set_title("(a) 总成本构成（30 种子均值 ± 标准差）")
    axes[0].legend(fontsize=8, loc="upper left")

    base = summ["mip"]["total_cost"]["mean"]
    deltas = np.array([(summ[k]["total_cost"]["mean"] - base) / base * 100 for k in ORDER])
    bars = axes[1].bar(x, deltas, width=0.6, color=[COLORS[k] for k in ORDER])
    for b, d in zip(bars, deltas):
        axes[1].text(b.get_x() + b.get_width() / 2, d + (1.5 if d >= 0 else -3.5),
                     f"{d:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].axhline(0, color="#333333", lw=1)
    axes[1].set_xticks(x, [LABELS[k] for k in ORDER], fontsize=9)
    axes[1].set_ylabel("相对 MIP 的成本差异（%）")
    axes[1].set_title("(b) 各策略相对 MIP 联合优化的成本差异")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_service(mc: dict, path: Path) -> Path:
    summ = mc["summary"]
    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fr = np.array([summ[k]["fill_rate"]["mean"] for k in ORDER]) * 100
    fr_std = np.array([summ[k]["fill_rate"]["std"] for k in ORDER]) * 100
    bars = axes[0].bar(x, fr, width=0.6, color=[COLORS[k] for k in ORDER], yerr=fr_std,
                       capsize=4, error_kw={"ecolor": "#333333", "lw": 1.1})
    for b, v, sd in zip(bars, fr, fr_std):
        axes[0].text(b.get_x() + b.get_width() / 2, v + sd + 0.30, f"{v:.2f}%",
                     ha="center", fontsize=9, fontweight="bold")
    axes[0].set_ylim(min(fr) - 3, 100.6)
    axes[0].set_xticks(x, [LABELS[k] for k in ORDER], fontsize=9)
    axes[0].set_ylabel("满足率 fill rate（%）")
    axes[0].set_title("(a) 需求满足率（按件数加权）")

    uns = np.array([summ[k]["unserved_units"]["mean"] for k in ORDER])
    bars = axes[1].bar(x, uns, width=0.6, color=[COLORS[k] for k in ORDER])
    for b, v in zip(bars, uns):
        axes[1].text(b.get_x() + b.get_width() / 2, v + uns.max() * 0.02, f"{v:.0f}",
                     ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(x, [LABELS[k] for k in ORDER], fontsize=9)
    axes[1].set_ylabel("未满足需求（件）")
    axes[1].set_title("(b) 累计未满足需求件数（12 个月合计）")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_plan_heatmap(inst: dict, mip: dict, path: Path) -> Path:
    T, N = inst["T"], inst["N"]
    x = mip["plan"]["x"]
    names = [p["id"] + " " + p["name"] for p in inst["products"]]
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    im = ax.imshow(x.T, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(T), _periods(T))
    ax.set_yticks(np.arange(N), names, fontsize=9)
    ax.set_xlabel("计划周期（月）")
    for t in range(T):
        for i in range(N):
            v = x[t, i]
            ax.text(t, i, f"{v:.0f}" if v > 0.5 else "–", ha="center", va="center",
                    fontsize=8, color="white" if v > x.max() * 0.55 else "#222222")
    ax.set_title("MIP 生产计划（件）：换型经济性形成两月一批轮换，淡季富余产能提前备货")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="产量（件）", fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(inst: dict, mip: dict, mc: dict) -> list[Path]:
    setup_mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_demand_capacity(inst, mip, FIG_DIR / "fig1_demand_capacity.png"),
        fig_inventory(mc, FIG_DIR / "fig2_inventory.png"),
        fig_cost(mc, FIG_DIR / "fig3_cost.png"),
        fig_service(mc, FIG_DIR / "fig4_service.png"),
        fig_plan_heatmap(inst, mip, FIG_DIR / "fig5_mip_plan.png"),
    ]
    return paths


if __name__ == "__main__":
    print("plots.py 由 src/run_all.py 调用；单独运行请先跑 run_all.py")
