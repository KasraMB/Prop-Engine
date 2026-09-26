"""Single-attempt scalar objectives (ARCHITECTURE §14.3; BUILD_SPEC Step 10).

An **objective** is any function ``raw_outcomes → scalar`` the deferred optimizer
(§16) maximizes. Because both the distribution and time axes are available
(:mod:`propfirm_engine.statistics`), an objective can be well-posed for a convex,
fee-per-attempt instrument — which a bare mean cannot express: maximizing
``E[payout]`` alone is blind to convexity (accepts a lottery-ticket account) and
to velocity (accepts one that ties the fee up for years).
"""

from __future__ import annotations

import numpy as np

from .statistics import (
    attributable_fee,
    prob_profitable,
    return_on_fee_per_year,
)


def expected_net_payoff(o) -> float:
    """``E[net_payout − attributable_fee]`` — expected net profit per attempt."""
    return float(np.mean(o.net_payout - attributable_fee(o)))


def expected_payout_st_profitable(o, floor: float = 0.5) -> float:
    """Maximize expected net payoff **subject to** ``P(profitable) ≥ floor`` (§14.3).
    Returns ``-inf`` when the constraint is violated, so a search cannot buy raw
    expectation with an unacceptably low chance of profit."""
    if prob_profitable(o) < floor:
        return float("-inf")
    return expected_net_payoff(o)


def annualized_return_on_fee(o) -> float:
    """The time-aware objective: annualized return on the attributable fee (§14.2).
    Penalizes duration the mean ignores — a fast account beats a slow one at equal
    total payout."""
    return return_on_fee_per_year(o)


__all__ = [
    "expected_net_payoff",
    "expected_payout_st_profitable",
    "annualized_return_on_fee",
]
