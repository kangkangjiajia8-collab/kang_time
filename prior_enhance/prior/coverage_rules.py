"""先验数量等轻量规则。"""

from __future__ import annotations

from typing import List

from prior_enhance.schemas import StructuredPrior


def coverage_issues(valid: List[StructuredPrior], min_priors: int, max_priors: int) -> List[str]:
    issues: List[str] = []
    if len(valid) < min_priors:
        issues.append(f"有效 prior 仅 {len(valid)} 条，需要 ≥{min_priors} 条")
    if len(valid) > max_priors:
        issues.append(f"有效 prior {len(valid)} 条，超过上限 {max_priors}")
    return issues
