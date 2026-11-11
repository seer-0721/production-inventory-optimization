"""三个经典基线策略（均忽略产能共享，只按本族库存状态下生产指令）。

约定
----
* 期初决策：第 t 期开始时，依据 t-1 期末的库存 inv 与欠交 backlog 决定本期产量，
  本期需求在本期结束时才实现（先生产、后到货）。
* 可承诺库存（库存位置）IP_i = inv_i - backlog_i。
* 三个策略都**不看**当期与未来的总产能，因此在旺季会互相挤占产能，
  由 simulate.allocate_capacity 按统一规则截断。

策略定义
--------
1. FixedBatchPolicy：每族固定批量 Q_i = 1.2 × 基准月需求，IP_i < 安全库存时下 Q_i。
2. EoqPolicy：Q_i = sqrt(2·K_i·D_i / h_i)（D_i 为月均需求、h_i 为月持有成本，
   量纲一致），订货间隔 n_i = max(1, round(Q_i / D_i)) 期，按产品序号错开相位，
   每到自己的订货期就下一次 Q_i（"按周期摊平下生产指令"）。
3. SSPolicy：(s_i, S_i) = (安全库存, 安全库存 + Q_i)，IP_i < s_i 时补到 S_i。

说明：EOQ 公式中的 D 与 h 必须同量纲。这里 h 是"元/件·月"，
故 D 取月均需求，得到的 Q 是"每批件数"，n_i 是"订货间隔月数"。
"""
from __future__ import annotations

import numpy as np


class BasePolicy:
    name = "base"
    label = "基线"

    def __init__(self, inst: dict):
        self.inst = inst
        self.N = inst["N"]
        self.mean_demand = inst["demand"].mean(axis=0)  # 历史平均需求（件/月）
        self.ss = inst["safety_stock"].copy()
        self.Q = np.zeros(self.N)

    def requests(self, t: int, inv: np.ndarray, backlog: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "label": self.label,
                "Q": np.round(self.Q, 1).tolist(),
                "safety_stock": self.ss.tolist()}


class FixedBatchPolicy(BasePolicy):
    """固定批量：Q = 1.2 × 平均月需求，库存位置低于安全库存才补一批。"""

    name = "fixed_batch"
    label = "固定批量"

    def __init__(self, inst: dict, batch_factor: float = 1.2):
        super().__init__(inst)
        self.batch_factor = batch_factor
        self.Q = batch_factor * self.mean_demand

    def requests(self, t, inv, backlog):
        ip = inv - backlog
        order = np.where(ip < self.ss, self.Q, 0.0)
        return np.maximum(order, 0.0)


class EoqPolicy(BasePolicy):
    """EOQ 分别订货：各族独立算 EOQ，按固定间隔摊平下单。"""

    name = "eoq"
    label = "EOQ 分别订货"

    def __init__(self, inst: dict):
        super().__init__(inst)
        K = inst["setup_cost"]
        h = inst["holding_cost"]
        D = np.maximum(self.mean_demand, 1.0)
        self.Q = np.sqrt(2.0 * K * D / h)
        self.cycle = np.maximum(1, np.round(self.Q / D)).astype(int)
        self.phase = np.arange(self.N) % self.cycle  # 各族订货相位错开

    def requests(self, t, inv, backlog):
        order = np.where((t - self.phase) % self.cycle == 0, self.Q, 0.0)
        return np.maximum(order, 0.0)


class SSPolicy(BasePolicy):
    """(s, S)：s = 安全库存，S = s + EOQ，库存跌破 s 就补到 S。"""

    name = "sS"
    label = "(s,S) 策略"

    def __init__(self, inst: dict):
        super().__init__(inst)
        K = inst["setup_cost"]
        h = inst["holding_cost"]
        D = np.maximum(self.mean_demand, 1.0)
        self.Q = np.sqrt(2.0 * K * D / h)
        self.S = self.ss + self.Q

    def requests(self, t, inv, backlog):
        ip = inv - backlog
        order = np.where(ip < self.ss, self.S - ip, 0.0)
        return np.maximum(order, 0.0)


def build_policies(inst: dict) -> dict[str, BasePolicy]:
    pols = [FixedBatchPolicy(inst), EoqPolicy(inst), SSPolicy(inst)]
    return {p.name: p for p in pols}
