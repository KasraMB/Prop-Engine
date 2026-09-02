"""The Results object — raw outcomes wrapped with lazy statistics (ARCHITECTURE
§14.5; BUILD_SPEC Step 10).

``Results`` wraps a batch of raw :class:`~propfirm_engine.engine.Outcomes` and
exposes both axes lazily. The mean (``E[payout]``) remains available but is one
number among these, not the summary — reporting it alone would hide exactly the
convexity and velocity the engine exists to measure (§0). The optimizer entry
point ``objective(fn, **kw)`` lets any §14/§15 function be evaluated over the
outcomes without the caller reaching inside.
"""

from __future__ import annotations

import numpy as np

from . import statistics as stats


class Results:
    """Lazy statistics over a batch of raw outcomes (§14.5)."""

    def __init__(self, outcomes):
        self._o = outcomes

    @property
    def outcomes(self):
        return self._o

    @property
    def fingerprint(self) -> str:
        return self._o.fingerprint

    @property
    def provenance(self):
        return self._o.provenance

    @property
    def n_attempts(self) -> int:
        return self._o.n_attempts

    # --- distribution axis ------------------------------------------------- #

    @property
    def prob_profitable(self) -> float:
        return stats.prob_profitable(self._o)

    @property
    def payout_count_dist(self) -> np.ndarray:
        return stats.payout_count_dist(self._o)

    @property
    def payoff_quantiles(self) -> np.ndarray:
        return stats.payoff_quantiles(self._o)

    @property
    def return_on_fee(self) -> np.ndarray:
        return stats.return_on_fee(self._o)

    @property
    def mean_payout(self) -> float:
        return stats.mean_payout(self._o)

    # --- time axis --------------------------------------------------------- #

    @property
    def payout_velocity(self) -> float:
        return stats.payout_velocity(self._o)

    @property
    def time_to_first_payout(self) -> np.ndarray:
        return stats.time_to_first_payout(self._o)

    @property
    def roi_per_year(self) -> float:
        return stats.return_on_fee_per_year(self._o)

    # --- eval-phase metric (never a funded success measure, §H3) ----------- #

    @property
    def pass_rate(self) -> float:
        return stats.pass_rate(self._o.code)

    # --- optimizer entry point --------------------------------------------- #

    def objective(self, fn, **kw) -> float:
        """Evaluate any objective ``fn(outcomes, **kw)`` over this batch (§14.5)."""
        return fn(self._o, **kw)


__all__ = ["Results"]
