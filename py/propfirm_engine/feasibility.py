"""Feasibility projection — desired risk vs executable position (ARCHITECTURE
§16.4b; BUILD_SPEC Step 14; MODEL_RISKS §C2/§A2).

The sizing policy (§16.1) outputs a **desired** position in continuous size-space;
that is not necessarily executable. This module is the deterministic layer between
policy and kernel that turns a desired position into an *executable* one against
the remaining trailing-drawdown buffer:

    desired size  ─▶  project_position(...)  ─▶  executable size q   (account trades on)
                              │
                              └─ no position expressible  ─▶  CAPPED_OUT (non-tradable)

**Two outcomes that must never be conflated (§16.4b):**

* **Clipping** — the buffer permits a smaller position than requested, so ``q`` is
  reduced and the account *keeps trading*. Normal operation, a health signal
  (``reduced``), **never** terminal. A policy that sizes down near the barrier is
  behaving correctly.
* **Non-tradability** — the budget cannot express even one minimum position
  (``α·B < L_min``): **no** size is executable, and the attempt terminates as
  ``CAPPED_OUT`` — a *distinct* code from an actual drawdown breach
  (``FAIL_TRAILING_DD``). This is what closes the ``r → 0`` exploit: a microscopic-
  risk policy still risks ``L_min`` each trade and eventually withers, so the
  renewal objective cannot hide behind indefinite survival.

The binding constraint is **pre-trade planned loss**, not simulated MAE (MAE still
drives breach detection *during* the trade, §6a): with a position of ``q`` units
and per-unit worst-case loss ``unit_loss``, planned loss is ``q · unit_loss`` and
must satisfy ``q · unit_loss ≤ α · B`` where ``B`` is the buffer to the trailing
floor and ``α ∈ (0, 1]`` a safety factor. Positions are integer multiples of the
minimum executable quantity ``q_min`` (you cannot trade 0.8 of a contract).

The projection is a **pure, deterministic function** shared verbatim by the fast
kernel and the reference oracle, so activating it keeps the Level-1 bitwise gate
(§G6) intact: both call :func:`project_position` with the same float64 operations
in the same order. It never enters the policy state (§16.3) — "don't risk 200 when
the buffer is 173" is mechanical, not learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from numba import njit


@dataclass(frozen=True)
class FeasibilitySpec:
    """Per-instrument feasibility config (the contract layer, §16.4b/§7).

    * ``q_min`` — the minimum executable quantity (one contract); positions are
      integer multiples of it. ``> 0``.
    * ``unit_loss`` — the per-unit worst-case loss (stop distance × point value),
      in the same $ units as equity. ``L_min = q_min · unit_loss`` is the smallest
      expressible loss. ``> 0``.
    * ``alpha`` — safety factor in ``(0, 1]``: the fraction of the buffer a single
      planned loss may consume. ``1.0`` = spend the whole buffer.
    * ``min_buffer`` — a hard drawdown floor ($): the attempt is non-tradable
      (``CAPPED_OUT``) once the buffer drops below this, *independent* of the
      contract size. ``0.0`` = no extra floor (the only cap is then "can't fit one
      contract"). Lets a caller say "stop when < $100 of drawdown remains" while
      keeping fine sizing above it.

    Authored/verified per firm+instrument. When a phase has no trailing restriction
    the projection is inactive (firm-agnostic), so this is only consulted where a
    buffer ``B`` exists."""

    q_min: float
    unit_loss: float
    alpha: float = 1.0
    min_buffer: float = 0.0

    def __post_init__(self) -> None:
        if self.q_min <= 0.0:
            raise ValueError(f"q_min must be > 0, got {self.q_min}")
        if self.unit_loss <= 0.0:
            raise ValueError(f"unit_loss must be > 0, got {self.unit_loss}")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")
        if self.min_buffer < 0.0:
            raise ValueError(f"min_buffer must be >= 0, got {self.min_buffer}")

    @property
    def l_min(self) -> float:
        """The smallest expressible loss ``q_min · unit_loss`` (the non-tradability
        threshold: once ``α·B`` drops below this, no position is executable)."""
        return self.q_min * self.unit_loss


@njit(cache=True)
def project_position(desired, buffer, q_min, unit_loss, alpha, min_buffer=0.0):
    """Project a *desired* position to an *executable* one against ``buffer``.

    Returns ``(q, capped_out, reduced, at_cap)``:

    * ``capped_out`` — ``True`` iff ``α·buffer < L_min`` (non-tradable): no position,
      not even ``q_min``, fits the buffer. The caller terminates the attempt as
      ``CAPPED_OUT``. ``q`` is ``0.0`` in this case.
    * otherwise ``q`` is the executable size: ``desired`` rounded **down** to a
      ``q_min`` multiple, floored **up** to ``q_min`` (a policy that wants less than
      one contract still trades one — this is what keeps ``r → 0`` from surviving),
      then capped by the largest ``q_min`` multiple whose planned loss ``q·unit_loss``
      stays within ``α·buffer``.
    * ``reduced`` — ``True`` iff the executed ``q`` is strictly less than ``desired``
      (a clip: the size was cut below what the policy wanted): a health/aggressiveness
      signal, **never** failure.
    * ``at_cap`` — ``True`` iff the **buffer cap** was the binding limit (``q ==
      cap_units``), i.e. the MLL buffer — not the desired size or the contract grid —
      set the position. This is the §16.9 *mll_constrained* signal, and it is
      *distinct* from ``reduced``: a request that lands exactly on the cap is at_cap
      but not reduced, and a request cut only by grid-rounding is reduced but not
      at_cap.

    This is the single source of the projection arithmetic; the kernel and the
    reference both call it, so the executed path is bit-identical between them
    (§G6). All arithmetic is plain float64 in a fixed order for that reason.
    """
    budget = alpha * buffer
    l_min = q_min * unit_loss
    if buffer < min_buffer or budget < l_min:
        # Non-tradable: either the buffer fell below the caller's hard drawdown floor
        # (``min_buffer``), or not even the minimum position's planned loss fits it.
        # Note the terminal condition is measured against the *safety-scaled* budget:
        # α·B < L_min (equivalently B < L_min/α). MODEL_RISKS §C2 writes the bare
        # `B < L_min`; the α form is the same condition under the §16.4b binding
        # constraint L(q) ≤ α·B (the minimum position q_min itself must satisfy it),
        # and it is what keeps the cap arithmetic below from yielding a 0 position.
        return 0.0, True, False, False
    # Largest q_min-multiple whose planned loss stays within the budget. budget >=
    # l_min > 0 guarantees cap_mult >= 1, so cap_units >= q_min.
    cap_mult = int(budget / (unit_loss * q_min))
    cap_units = cap_mult * q_min
    # The size the policy actually wants: desired rounded DOWN to the contract grid,
    # then floored UP to one contract (a sub-q_min desire still trades q_min).
    grid_mult = int(desired / q_min)
    q_wanted = grid_mult * q_min
    if q_wanted < q_min:
        q_wanted = q_min
    if q_wanted < cap_units:
        q = q_wanted  # the desire/grid bound the size; the buffer had room
        at_cap = False
    else:
        q = cap_units
        # mll_constrained iff the buffer cap *strictly* cut below what the policy
        # wanted. At an exact tie (cap == wanted, incl. the sub-q_min-floored case)
        # the buffer did not force a smaller size, so it is NOT flagged — this keeps
        # the §16.9 signal a clean "buffer actively limited the position" (⊆ reduced).
        at_cap = cap_units < q_wanted
    reduced = q < desired
    return q, False, reduced, at_cap


@dataclass
class FeasibilityDiag:
    """Per-attempt feasibility diagnostics (§16.9) — cheap post-processing that
    separates *why* a policy failed and *how* it used its buffer. Recorded only
    when a caller asks for it; never influences the outcome (so parity holds).

    ``D_trade = B / L_min`` (the tradability ratio) is a **derived** feature, not a
    policy-state variable (§16.9): it is tracked here for diagnostics only."""

    trades: int = 0  # trades actually executed (sizing reached)
    reduced: int = 0  # trades whose executed size was clipped below desired (any reason)
    constrained: int = 0  # trades where the MLL buffer was the binding limit (at_cap);
    #                       §16.9 mll_constrained — distinct from `reduced` (grid clips)
    sum_desired: float = 0.0
    sum_executed: float = 0.0
    min_buffer: float = field(default=float("inf"))
    min_tradability_ratio: float = field(default=float("inf"))  # min B / L_min
    capped_out: bool = False  # the attempt ended non-tradable (B < L_min)
    breached: bool = False  # the attempt ended on an actual drawdown breach
    time_to_nontradable: int = -1  # trading-day index of CAPPED_OUT, else -1
    time_to_breach: int = -1  # trading-day index of the breach, else -1

    def record_trade(self, desired, executed, buffer, l_min, reduced, at_cap) -> None:
        self.trades += 1
        self.sum_desired += desired
        self.sum_executed += executed
        if reduced:  # size clipped below desired, for ANY reason (grid or buffer)
            self.reduced += 1
        if at_cap:  # the MLL buffer was the binding limit (§16.9 mll_constrained) —
            self.constrained += 1  # distinct from `reduced`: a grid-only clip is not it
        self._fold_buffer(buffer, l_min)

    def record_cap(self, buffer, l_min) -> None:
        """Fold the *withering* trade's buffer into the min statistics (§16.9).

        The non-tradable trade never executes, so ``record_trade`` is not called for
        it — but its sub-threshold buffer is exactly the closest the account ever
        came to non-tradability, so it MUST reach ``min_buffer`` /
        ``min_tradability_ratio`` (otherwise a withered attempt would report a
        min-ratio ≥ 1, backwards from the metric's purpose)."""
        self._fold_buffer(buffer, l_min)

    def _fold_buffer(self, buffer, l_min) -> None:
        if buffer < self.min_buffer:
            self.min_buffer = buffer
        ratio = buffer / l_min if l_min > 0 else float("inf")
        if ratio < self.min_tradability_ratio:
            self.min_tradability_ratio = ratio

    # --- derived rates (safe on an empty attempt) --------------------------- #

    @property
    def fraction_size_reduced(self) -> float:
        return self.reduced / self.trades if self.trades else 0.0

    @property
    def fraction_constrained(self) -> float:
        return self.constrained / self.trades if self.trades else 0.0

    @property
    def average_requested_risk(self) -> float:
        return self.sum_desired / self.trades if self.trades else 0.0

    @property
    def average_executed_risk(self) -> float:
        return self.sum_executed / self.trades if self.trades else 0.0


@dataclass
class FeasibilityAgg:
    """Batch-level feasibility diagnostics (§16.9), accumulated across attempts.

    Cheap running aggregate so the optimizer can separate an *aggressive* policy
    (often buffer-constrained → breaches) from an *over-conservative* one (rarely
    constrained → too slow) from a good one, without retaining a per-attempt diag
    object for every path."""

    attempts: int = 0
    nontradable_failures: int = 0  # attempts that ended CAPPED_OUT (withered)
    actual_breach_failures: int = 0  # attempts that ended on a real rule breach
    _sum_frac_reduced: float = 0.0
    _sum_frac_constrained: float = 0.0
    _sum_avg_requested: float = 0.0
    _sum_avg_executed: float = 0.0

    def add(self, diag: "FeasibilityDiag") -> None:
        self.attempts += 1
        if diag.capped_out:
            self.nontradable_failures += 1
        if diag.breached:
            self.actual_breach_failures += 1
        self._sum_frac_reduced += diag.fraction_size_reduced
        self._sum_frac_constrained += diag.fraction_constrained
        self._sum_avg_requested += diag.average_requested_risk
        self._sum_avg_executed += diag.average_executed_risk

    def _mean(self, total: float) -> float:
        return total / self.attempts if self.attempts else 0.0

    def as_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "nontradable_failures": self.nontradable_failures,
            "actual_breach_failures": self.actual_breach_failures,
            "nontradable_rate": self._mean(self.nontradable_failures),
            "breach_rate": self._mean(self.actual_breach_failures),
            "mean_fraction_size_reduced": self._mean(self._sum_frac_reduced),
            "mean_fraction_constrained": self._mean(self._sum_frac_constrained),
            "mean_requested_risk": self._mean(self._sum_avg_requested),
            "mean_executed_risk": self._mean(self._sum_avg_executed),
        }


__all__ = ["FeasibilitySpec", "FeasibilityDiag", "FeasibilityAgg", "project_position"]
