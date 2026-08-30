"""The integer vocabulary shared by the DSL and the kernel (ARCHITECTURE §3).

This module is the contract between the expressive object layer and the
numerical hot loop. Every value here is a plain integer with fixed semantics:

- ``ExitCode``   — how a simulation ends (alive = 0, any failure >= 10).
- ``StateField`` — an index into the per-simulation state vector.
- ``Action``     — what a rule's predicate does when it triggers.
- ``Severity``   — hard vs soft, meaningful only for ``Action.FAIL``.
- ``Timing``     — the two orthogonal per-rule timing axes (§6a).
- ``Stage``      — independent bit positions in ``StateField.STAGE_MASK``.

The concrete integer values are an implementation detail (BUILD_SPEC Step 1
"Freedom"); only the ordering semantics are contractual: ``ALIVE == 0``, every
failure code ``>= FAILURE_THRESHOLD``, and all failure codes mutually distinct.
"""

from enum import IntEnum

#: A simulation has failed iff its terminal code is >= this threshold (§3).
FAILURE_THRESHOLD = 10


class ExitCode(IntEnum):
    """How a single simulation terminates."""

    ALIVE = 0  # still running
    PASSED = 1  # cleared the phase (all pass-predicates satisfied)

    # --- failure codes start at 10; each rule owns one for attribution mode ---
    FAIL_GENERIC = 10  # pass/fail mode uses only this for any failure
    FAIL_TRAILING_DD = 11
    FAIL_STATIC_DD = 12
    FAIL_DAILY_LOSS = 13
    FAIL_MIN_DAYS = 14  # RESERVED — MinimumTradingDaysRule is a PASS gate, so a
    #                     shortfall surfaces as TIMED_OUT, not this code. Kept for
    #                     firms that actively FAIL on min-days.
    FAIL_CONSISTENCY = 15  # RESERVED / unused — consistency is an eligibility GATE
    #                        (ConsistencyGateRule), not a failure. Kept for a
    #                        hypothetical fail-on-consistency firm (MODEL_RISKS §C8).

    # --- non-failure terminals ---
    TIMED_OUT = 20  # ran out of trades without passing
    CAPPED_OUT = 21  # DEFERRED with the sizing policy (MODEL_RISKS C2/A2); reserved
    #                  but never emitted under a constant-size policy.
    MAXED_OUT = 22  # funded: reached max_payouts — an economic SUCCESS, distinct
    #                 from eval PASSED (MODEL_RISKS H3).

    @property
    def is_failure(self) -> bool:
        """True for a terminal *failure* code (``FAIL_*``): >= ``FAILURE_THRESHOLD``
        but below the non-failure terminals (``TIMED_OUT`` and above)."""
        return FAILURE_THRESHOLD <= self.value < ExitCode.TIMED_OUT.value


class StateField(IntEnum):
    """Index into the per-sim state vector. The requirements resolver (§8) decides
    which fields are live for a given phase; unused fields are never computed."""

    EQUITY = 0  # equity at trade close (realized)
    PEAK_EQUITY = 1  # running max of closing equity (reference for trailing DD)
    DD_FLOOR = 2  # the trailing-DD line itself (peak - amount, or locked value)
    DD_LOCKED = 3  # 0/1: has the floor locked and stopped trailing?
    DAY_PNL = 4
    TOTAL_PNL = 5
    DAY_INDEX = 6
    N_TRADING_DAYS = 7
    MAX_DAY_PNL = 8  # for consistency
    N_QUALIFYING_DAYS = 9  # winning days meeting a profit threshold
    PAYOUTS_TAKEN = 10  # increments on each payout
    N_SOFT_BREACHES = 11  # for soft-breach escalation
    STAGE_MASK = 12  # bitmask of active stage predicates (see Stage)
    PROFIT_TARGET = 13  # live, mutable target (seeded from rule p0; raised by ADJUST)
    DAY_LOW = 14  # worst floating equity reached so far this day (intraday checks)
    CYCLE_START_EQUITY = 15  # equity at the start of the current payout cycle
    CUMULATIVE_PAID = 16  # total gross dollars paid out so far (drives tiered split)


class Action(IntEnum):
    """What a predicate does when it triggers."""

    FAIL = 0  # end the simulation in failure (see Severity for hard vs soft)
    PASS = 1  # terminal success — the phase is cleared
    PAYOUT = 2  # repeatable success — record a payout, reset counters, continue
    ADJUST = 3  # mutate a target StateField (e.g. raise PROFIT_TARGET)


class Severity(IntEnum):
    """Only meaningful for ``Action.FAIL`` predicates."""

    HARD = 0  # account terminated
    SOFT = 1  # current day truncated; remaining trades skipped; resume next day


class Timing(IntEnum):
    """Two independent per-rule axes (§6a): when the reference point advances, and
    when the predicate is checked. Orthogonal — 'trails EOD, detected intraday on
    floating balance' is ``update_timing=EOD, check_timing=CONTINUOUS``."""

    CONTINUOUS = 0  # every trade; checks read the day's intraday low-water mark
    EOD = 1  # only at day close; updates read the day's closing equity


class Stage(IntEnum):
    """Independent bit positions in ``StateField.STAGE_MASK``. Multiple may be set
    at once, so membership is a bitmask, never a single current-stage integer."""

    IN_PROFIT = 0
    CONSISTENCY_SATISFIED = 1
    PRE_FIRST_PAYOUT = 2
    PAYOUT_ELIGIBLE = 3
    # add bits freely; each is an independent predicate, never a state-machine phase


__all__ = [
    "FAILURE_THRESHOLD",
    "ExitCode",
    "StateField",
    "Action",
    "Severity",
    "Timing",
    "Stage",
]
