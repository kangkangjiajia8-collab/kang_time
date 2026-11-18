"""历史 prior 库：跨窗口持久化，供 τ_{j,m} 结构相似检索。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Sequence, Tuple

from prior_enhance.schemas import StructuredPrior


@dataclass
class StoredPriorRecord:
    """单条历史 prior 及其观测后的持久 τ。"""

    store_key: str
    window_index: int
    decision_time: str
    prior: StructuredPrior
    tau: float
    scenario: str = ""

    def to_dict(self) -> dict:
        return {
            "store_key": self.store_key,
            "window_index": self.window_index,
            "decision_time": self.decision_time,
            "scenario": self.scenario,
            "tau": self.tau,
            "prior": self.prior.model_dump(mode="json"),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "StoredPriorRecord":
        return cls(
            store_key=str(raw["store_key"]),
            window_index=int(raw.get("window_index", 0)),
            decision_time=str(raw.get("decision_time", "")),
            scenario=str(raw.get("scenario", "")),
            tau=float(raw.get("tau", 0.5)),
            prior=StructuredPrior.model_validate(raw["prior"]),
        )


@dataclass
class PriorStore:
    """JSONL 持久化的历史先验库。"""

    path: str
    records: List[StoredPriorRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "PriorStore":
        store = cls(path=path)
        if not path or not os.path.isfile(path):
            return store
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    store.records.append(StoredPriorRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        return store

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    def append_window(
        self,
        *,
        window_index: int,
        decision_time: datetime,
        priors: Sequence[StructuredPrior],
        tau_map: dict[str, float],
        scenario: str = "",
    ) -> None:
        """窗口 observation 后将有效 prior 写入库。"""
        ts = decision_time.isoformat() if hasattr(decision_time, "isoformat") else str(decision_time)
        for p in priors:
            if not p.valid:
                continue
            key = f"win{window_index:04d}_{p.prior_id}"
            self.records.append(
                StoredPriorRecord(
                    store_key=key,
                    window_index=window_index,
                    decision_time=ts,
                    scenario=scenario,
                    prior=p,
                    tau=float(tau_map.get(p.prior_id, 0.5)),
                )
            )

    def candidate_pairs(self) -> List[Tuple[StructuredPrior, float]]:
        return [(r.prior, r.tau) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)
