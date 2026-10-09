"""The reference implementation — a slow one-attempt simulator
(ARCHITECTURE §12; BUILD_SPEC Step 6; MODEL_RISKS §G6, Level 1).

This provides a parity cross-check, not independent proof of fidelity. It implements the
same predicate/action semantics as :mod:`propfirm_engine.kernels`, but as a plain
stateful class — state is instance attributes, control flow is explicit, and an
optional per-trade ``trace`` records state for debugging ("why did this account
fail on trade 4,217?"). It is never used in the Monte Carlo loop.

The semantics it pins (each cites where it is decided):

* **Trailing drawdown** lives from trade 1 (``dd_floor = start − amount``, §C3);
  its floor ratchets under ``update_timing`` (CONTINUOUS intraday / EOD at close)
  and locks at ``lock_at`` (§6a). A CONTINUOUS breach check reads the day's
  current trade's low against the established floor; an EOD check reads closing equity.
  CONTINUOUS updates currently use closing peaks, not unobserved open profits.
* **Fail predicates are disjunctive, hard before soft**, then rule order. A hard
  breach terminates; a soft breach truncates the day (skip its remaining trades),
  the partial loss stands, the day still counts as a trading day but not a winning
  day, and the sim resumes next day (§C5).
* **Pass predicates are conjunctive** (§6); **payouts** fire only through the full
  fire gate (qualifying conjunction AND ``gross_request ≥ min_request`` AND the
  post-withdrawal balance stays ≥ ``buffer_floor``), never for a zero/blocked
  amount (§6b). Firing records the trader's *net* share, advances
  ``cumulative_paid`` by *gross*, applies the post-payout transition, and — at
  ``max_payouts`` — returns ``MAXED_OUT`` (§6b.2).
* **Every day-end runs one ``_close_day``** — natural rollover, soft-breach
  truncation, and end-of-path (§B1) — with one fold-then-evaluate order (fold the
  day into ``max_day_pnl``/winning-day counter first, then evaluate EOD
  fail→adjust→pass→payout, §C9); the day's closing equity is the equity after its
  last *executed* trade (§C5).
* **Consistency is an eligibility gate, not a failure** (§C8): a payout/pass is
  withheld while ``max_day_pnl > threshold × cycle_profit (+cushion)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import isfinite

import numpy as np

from .enums import ExitCode, Severity, Stage, StateField, Timing
from .feasibility import project_position
from .rules import RuleKind

_CONTINUOUS = int(Timing.CONTINUOUS)
_EOD = int(Timing.EOD)
_HARD = int(Severity.HARD)

_ALIVE = int(ExitCode.ALIVE)
_PASSED = int(ExitCode.PASSED)
_TIMED_OUT = int(ExitCode.TIMED_OUT)
_MAXED_OUT = int(ExitCode.MAXED_OUT)
_CAPPED_OUT = int(ExitCode.CAPPED_OUT)


@dataclass
class SimResult:
    """The outcome of one attempt: terminal code + the payouts it released."""

    code: int
    payout_amounts: list[float] = field(default_factory=list)
    payout_days: list[int] = field(default_factory=list)
    total_trading_days: int = 0  # trading days this attempt actually ran (§14.4)
    trace: list[dict] = field(default_factory=list)
    feasibility: object = None  # a FeasibilityDiag when the projection was active (§16.9)

    @property
    def payouts_taken(self) -> int:
        return len(self.payout_amounts)


def size_for(stage_mask: int, policy_params: np.ndarray, size_base: float) -> float:
    """Position size for a trade (§16.1). A length-1 ``policy_params`` reproduces a
    single uniform size (``size_base × policy_params[0]``); a longer array is a
    per-stage-mask multiplier table (clamped), so size reacts to the stage the
    account is *entering* the trade in (a one-trade lag, §12)."""
    n = policy_params.shape[0]
    if n == 1:
        return size_base * float(policy_params[0])
    idx = stage_mask if stage_mask < n else n - 1
    return size_base * float(policy_params[idx])


class _ReferenceSim:
    """One attempt, run as a mutable object so ``_close_day`` shares state plainly."""

    def __init__(self, cp, size_base, policy_params, start_equity, trace,
                 feasibility=None, diag=None, trade_cost=0.0, external_payouts=False,
                 exact_money=False):
        self.cp = cp
        self.schema = cp.payout
        self.external_payouts = external_payouts
        self.exact_money = exact_money
        start_equity = self._cash(start_equity)
        self.size_base = size_base
        self.policy = np.asarray(policy_params, dtype=np.float64)
        self.start_equity = start_equity
        self.trade_cost = float(trade_cost)
        self.want_trace = trace
        self.feasibility = feasibility
        self.diag = diag  # a FeasibilityDiag to fill, or None

        self.equity = start_equity
        self.peak = start_equity
        self.dd_floor = start_equity - self._cash(cp.dd_amount)  # live from trade 1 (§C3)
        if not self._has_trailing():
            static = [self._cash(cp.p0[i]) for i in cp.fail_idx if cp.kind[i] == RuleKind.STATIC_DD]
            if static:
                self.dd_floor = start_equity - min(static)
        self.dd_locked = False
        self.day_pnl = self._cash(0)
        self.total_pnl = self._cash(0)
        self.max_day_pnl = self._cash(0)
        self.day_low = start_equity
        self.profit_target = self._cash(cp.profit_target0)
        self.cycle_start_equity = start_equity
        self.cumulative_paid = 0.0
        self.n_days = 0
        self.n_qual_days = 0
        self.payouts_taken = 0
        self.is_funded = cp.role == "funded"
        # sizing regime index (§16.4): 0 = eval; funded = 1..4 over in-profit ×
        # pre/post-first-payout. First funded trade is flat & pre-payout (index 1).
        self.stage_mask = 1 if self.is_funded else 0
        self.cur_day = -1

        self.res = SimResult(code=_ALIVE)

    # --- helpers ---------------------------------------------------------- #

    def _cash(self, value):
        if self.exact_money and isfinite(value):
            return value if isinstance(value, Fraction) else Fraction(str(value))
        return float(value)

    def _has_trailing(self) -> bool:
        return np.isfinite(self.cp.dd_amount)

    def _mark_breach(self, code) -> None:
        # Side record only (§16.9): a rule-breach terminal, for the optimizer's
        # "blew up vs withered" split. Never influences control flow (parity-safe).
        if self.diag is not None and 10 <= code < _TIMED_OUT:
            self.diag.breached = True
            self.diag.time_to_breach = self.n_days

    def _stage_mask(self) -> int:
        # Phase-aware sizing regime index (matches the kernel bit-for-bit): eval is a
        # single regime (0); funded splits 1..4 over in-profit × pre/post-first-payout.
        if not self.is_funded:
            return 0
        ip = 1 if self.equity > self.start_equity else 0
        post = 0 if self.payouts_taken == 0 else 1
        return 1 + post * 2 + ip

    def _first_fail(self, phase, test_equity, *, daily_pnl=None):
        """Hard failure precedes soft at the same observation, then rule order;
        check_timing matches ``phase``. Returns (hit, severity, fail_code)."""
        cp = self.cp
        soft = (False, 0, 0)
        for i in cp.fail_idx:
            if int(cp.check_timing[i]) != phase:
                continue
            kind = int(cp.kind[i])
            breached = False
            if kind == int(RuleKind.TRAILING_DD):
                breached = test_equity <= self.dd_floor
            elif kind == int(RuleKind.STATIC_DD):
                floor = self.start_equity - self._cash(cp.p0[i])
                if self.dd_locked and not self._has_trailing():
                    floor = max(floor, self.dd_floor)
                breached = test_equity <= floor
            elif kind == int(RuleKind.DAILY_LOSS):
                breached = (self.day_pnl if daily_pnl is None else daily_pnl) <= -self._cash(cp.p0[i])
            if breached:
                result = (True, int(cp.severity[i]), int(cp.fail_code[i]))
                if int(cp.severity[i]) == _HARD:
                    return result
                if not soft[0]:
                    soft = result
        return soft

    def _apply_adjusts(self, phase):
        cp = self.cp
        for i in cp.adjust_idx:
            if int(cp.check_timing[i]) != phase:
                continue
            if int(cp.kind[i]) == int(RuleKind.CONSISTENCY_ADJUST):
                threshold = float(cp.p0[i])
                raise_to = float(cp.p1[i])
                if self.max_day_pnl > threshold * self.total_pnl:
                    if raise_to > self.profit_target:
                        self.profit_target = raise_to

    def _consistency_gate_ok(self, i) -> bool:
        threshold = self._cash(self.cp.p0[i])
        cushion = self._cash(self.cp.p1[i])
        cycle_profit = self.equity - self.cycle_start_equity
        # The biggest single day includes the CURRENT (in-progress) day: when the
        # gate is read intraday inside a pass/payout conjunction, the running
        # day_pnl can already be the largest day, and it must count — otherwise a
        # dominant day is ignored until it folds at close and the gate is bypassed
        # (§C8). At a day close the day is already folded, so this is a no-op there.
        biggest_day = self.max_day_pnl if self.max_day_pnl >= self.day_pnl else self.day_pnl
        return biggest_day <= threshold * cycle_profit + cushion

    def _all_pass(self, equity) -> bool:
        cp = self.cp
        if cp.pass_idx.shape[0] == 0:
            return False
        for i in cp.pass_idx:
            kind = int(cp.kind[i])
            if kind == int(RuleKind.PROFIT_TARGET):
                if not (equity - self.start_equity >= self.profit_target):
                    return False
            elif kind == int(RuleKind.MIN_DAYS):
                if not (self.n_days >= float(cp.p0[i])):
                    return False
            elif kind == int(RuleKind.CONSISTENCY_GATE):
                if not self._consistency_gate_ok(i):
                    return False
        return True

    def _try_payout(self, equity):
        """Full payout fire gate; returns (fired, net, gross). Uses ``equity`` as
        the balance being tested (current equity intraday, closing equity at EOD)."""
        cp, schema = self.cp, self.schema
        if self.external_payouts or schema is None or cp.payout_idx.shape[0] == 0:
            return False, 0.0, 0.0
        for i in cp.payout_idx:
            kind = int(cp.kind[i])
            if kind == int(RuleKind.MIN_WINNING_DAYS):
                if not (self.n_qual_days >= float(cp.p0[i])):
                    return False, 0.0, 0.0
            elif kind == int(RuleKind.CONSISTENCY_GATE):
                if not self._consistency_gate_ok(i):
                    return False, 0.0, 0.0
        cycle_profit = equity - self.cycle_start_equity
        if cycle_profit < schema.min_cycle_profit:
            return False, 0.0, 0.0
        reference = (self.start_equity if schema.profit_reference_balance is None
                     else schema.profit_reference_balance)
        fraction_profit = equity - reference if schema.fraction_basis == 1 else cycle_profit
        gross = min(schema.dollar_cap_at(self.payouts_taken),
                    schema.cap_fraction * fraction_profit)
        # never fire a zero/blocked amount (§6b): a $0 release would still record a
        # payout, burn a max_payouts slot, and reset the cycle counters. min_request
        # cannot supply this (neutral default 0.0), so guard gross explicitly.
        if gross <= 0.0 or gross < schema.min_request:
            return False, 0.0, 0.0
        if equity - gross < schema.buffer_floor:
            return False, 0.0, 0.0
        first_gross = min(gross, max(0.0, schema.tier_cap - self.cumulative_paid))
        net = first_gross * schema.first_tier_split + (gross - first_gross) * schema.split
        return True, net, gross

    def _fire_payout(self, net, gross, day_index):
        schema = self.schema
        self.res.payout_amounts.append(net)
        self.res.payout_days.append(day_index)
        self.payouts_taken += 1
        self.cumulative_paid += gross
        if schema.withdraw_reduces_equity:
            self.equity -= gross
        if schema.recompute_floor_on_payout and not self.dd_locked and self._has_trailing():
            self.peak = self.equity
            self.dd_floor = self.equity - self.cp.dd_amount
        if schema.resets_qualifying_days:  # precomputed at compile time (§6b)
            self.n_qual_days = 0
        if int(StateField.MAX_DAY_PNL) in schema.reset_fields:
            self.max_day_pnl = 0.0
        self.cycle_start_equity = self.equity
        return self.payouts_taken >= schema.max_payouts

    def _close_day(self, closing_equity, winning_allowed) -> int:
        """One day-end (§C9): fold first, then EOD fail→adjust→(floor)→pass→payout.
        Returns a terminal ExitCode or ALIVE."""
        cp = self.cp
        # (1) fold the just-closed day into the day-scoped counters
        if self.day_pnl > self.max_day_pnl:
            self.max_day_pnl = self.day_pnl
        if winning_allowed and self.day_pnl >= self._cash(cp.winning_day_threshold):
            self.n_qual_days += 1

        # (2a) EOD FAIL against the established floor / closing equity
        hit, severity, fail_code = self._first_fail(_EOD, closing_equity)
        if hit and severity == _HARD:
            return fail_code

        # (2b) EOD ADJUST
        self._apply_adjusts(_EOD)

        # (2c) EOD floor ratchet (advance off closing equity, then lock)
        if (not self.dd_locked) and cp.dd_update_timing == _EOD and self._has_trailing():
            if closing_equity > self.peak:
                self.peak = closing_equity
            self.dd_floor = self.peak - self._cash(cp.dd_amount)
            if self.dd_floor >= cp.lock_at:
                self.dd_floor = self._cash(cp.lock_at)
                self.dd_locked = True

        # (2d) EOD PASS (conjunctive) against closing equity
        if self._all_pass(closing_equity):
            return _PASSED

        # (2e) EOD PAYOUT (fire gate) against closing equity
        fired, net, gross = self._try_payout(closing_equity)
        if fired and self._fire_payout(net, gross, self.cur_day):
            return _MAXED_OUT
        return _ALIVE

    def observe(self, balance, equity, day, *, traded=False, allow_pass=False):
        """Apply an ordered portfolio observation, keeping balance and equity distinct."""
        if getattr(self, "_observation_day", None) != day:
            self._observation_day = day
            self.day_pnl = self._cash(0)
            self.day_low = equity
        if traded and day != self.cur_day:
            if self.cur_day != -1:
                raise ValueError("close the previous session before observing a new one")
            self.day_pnl = self._cash(0)
            self.day_low = equity
            self.cur_day = day
            self.n_days += 1
        balance, equity = self._cash(balance), self._cash(equity)
        delta = balance - self.equity
        self.equity = balance
        self.day_pnl += delta
        self.total_pnl += delta
        self.day_low = min(self.day_low, equity)
        self.observation_soft = False
        daily_pnl = self.day_pnl + equity - balance
        hit, severity, code = self._first_fail(_CONTINUOUS, equity, daily_pnl=daily_pnl)
        if hit:
            self.observation_soft = severity != _HARD
            return code
        if not self.dd_locked and self.cp.dd_update_timing == _CONTINUOUS and self._has_trailing():
            self.peak = max(self.peak, equity)
            self.dd_floor = self.peak - self._cash(self.cp.dd_amount)
            if self.dd_floor >= self.cp.lock_at:
                self.dd_floor = self._cash(self.cp.lock_at)
                self.dd_locked = True
            hit, severity, code = self._first_fail(_CONTINUOUS, equity, daily_pnl=daily_pnl)
            if hit:
                self.observation_soft = severity != _HARD
                return code
        self._apply_adjusts(_CONTINUOUS)
        self.stage_mask = self._stage_mask()
        return _PASSED if allow_pass and self._all_pass(balance) else _ALIVE

    # --- the main loop ---------------------------------------------------- #

    def run(self, ret, day, trade_low, *, finalize=True) -> SimResult:
        cp = self.cp
        # Feasibility binds only with a spec AND a trailing buffer (§16.4b).
        feas = self.feasibility
        feas_active = feas is not None and self._has_trailing()
        if feas_active:
            feas_lmin = feas.q_min * feas.unit_loss + self.trade_cost
        if self.diag is not None:
            self.res.feasibility = self.diag
        N = ret.shape[0]
        t = 0
        while t < N:
            d = int(day[t])
            if d != self.cur_day:  # ---- day boundary ----
                if self.cur_day != -1:
                    code = self._close_day(self.equity, winning_allowed=True)
                    if code != _ALIVE:
                        self._mark_breach(code)
                        self.res.code = code
                        self.res.total_trading_days = self.n_days
                        return self.res
                self.day_pnl = self._cash(0)
                self.day_low = self.equity
                self.cur_day = d
                self.n_days += 1

            self.stage_mask = self._stage_mask()  # includes intervening day-close events
            size = size_for(self.stage_mask, self.policy, self.size_base)
            if feas_active:
                # Project desired -> executable against the pre-trade buffer (§16.4b).
                desired = size
                buffer = self.equity - self.dd_floor
                size, capped, reduced, at_cap = project_position(
                    desired, buffer, feas.q_min, feas.unit_loss, feas.alpha,
                    feas.min_buffer, self.trade_cost
                )
                if capped:
                    if self.diag is not None:
                        self.diag.capped_out = True
                        self.diag.time_to_nontradable = self.n_days
                        self.diag.record_cap(buffer, feas_lmin)
                    self.res.code = _CAPPED_OUT
                    self.res.total_trading_days = self.n_days
                    return self.res
                if self.diag is not None:
                    self.diag.record_trade(desired, size, buffer, feas_lmin,
                                           reduced, at_cap)
            entry_eq = self.equity
            r = float(ret[t])
            p = self._cash(size) * self._cash(r) - self._cash(self.trade_cost)
            self.equity += p
            self.day_pnl += p
            self.total_pnl += p
            # True intra-trade floating low FROM ENTRY: the lower of the close and the
            # MAE excursion (trade_low = -mae), plus net equity at cost settlement.
            tl = float(trade_low[t])
            exc = r if r < tl else tl        # min(ret, trade_low)
            trade_floor = min(entry_eq + self._cash(size) * self._cash(exc), self.equity)
            if trade_floor < self.day_low:
                self.day_low = trade_floor

            # Check this trade against its established floor before a closing ratchet.
            hit, severity, fail_code = self._first_fail(_CONTINUOUS, trade_floor)
            # trailing reference-point update (closing peaks only in summary mode)
            if not hit and (not self.dd_locked) and cp.dd_update_timing == _CONTINUOUS and self._has_trailing():
                if self.equity > self.peak:
                    self.peak = self.equity
                self.dd_floor = self.peak - cp.dd_amount
                if self.dd_floor >= cp.lock_at:
                    self.dd_floor = cp.lock_at
                    self.dd_locked = True

            # A new floor applies to closing equity, never an earlier excursion.
            if not hit:
                hit, severity, fail_code = self._first_fail(_CONTINUOUS, self.equity)
            if hit:
                if severity == _HARD:
                    self._mark_breach(fail_code)
                    self.res.code = fail_code
                    self.res.total_trading_days = self.n_days
                    return self.res
                # SOFT: truncate the day; closing equity is the equity after this
                # (the last executed) trade; not a winning day (§C5).
                code = self._close_day(self.equity, winning_allowed=False)
                if code != _ALIVE:
                    self._mark_breach(code)
                    self.res.code = code
                    self.res.total_trading_days = self.n_days
                    return self.res
                t = self._advance_to_next_day(day, t)
                self.cur_day = -1
                continue

            # ADJUST (CONTINUOUS)
            self._apply_adjusts(_CONTINUOUS)

            # stage bits (recomputed after the checks; used by the next trade)
            self.stage_mask = self._stage_mask()

            # PASS (conjunctive, intraday against current equity). Passing an eval
            # is a genuine intraday equity event (hitting the target mid-session
            # clears it). PAYOUTS, by contrast, fire only at day close (in
            # _close_day): they hinge on whole-day properties (the winning-day count
            # and consistency ratio fold at close, §C8/§C9), so an intraday payout
            # would read a stale max_day_pnl and could fire on a day its close blocks.
            if self._all_pass(self.equity):
                self.res.code = _PASSED
                self.res.total_trading_days = self.n_days
                return self.res

            if self.want_trace:
                self.res.trace.append(
                    {"t": t, "equity": self.equity, "day_low": self.day_low,
                     "dd_floor": self.dd_floor, "day_pnl": self.day_pnl,
                     "total_pnl": self.total_pnl, "n_days": self.n_days,
                     "n_qual_days": self.n_qual_days, "payouts_taken": self.payouts_taken}
                )
            t += 1

        if not finalize:
            self.res.code = _ALIVE
            self.res.total_trading_days = self.n_days
            return self.res

        # end-of-path close (§B1)
        if self.cur_day != -1:
            code = self._close_day(self.equity, winning_allowed=True)
            if code != _ALIVE:
                self._mark_breach(code)
                self.res.code = code
                self.res.total_trading_days = self.n_days
                return self.res

        self.res.code = _TIMED_OUT
        self.res.total_trading_days = self.n_days
        return self.res

    @staticmethod
    def _advance_to_next_day(day, t):
        d = int(day[t])
        j = t + 1
        while j < day.shape[0] and int(day[j]) == d:
            j += 1
        return j


def simulate_reference(
    cp,
    ret: np.ndarray,
    day: np.ndarray,
    trade_low: np.ndarray,
    size_base: float,
    policy_params: np.ndarray,
    start_equity: float,
    *,
    trace: bool = False,
    feasibility=None,
    diag=None,
    trade_cost: float = 0.0,
) -> SimResult:
    """Simulate one attempt of compiled phase ``cp`` over a fixed trade path.

    ``feasibility`` (a :class:`~propfirm_engine.feasibility.FeasibilitySpec` or
    ``None``) activates the executable-position projection (§16.4b); ``None``
    reproduces the constant-position path **bit-for-bit** (the projection never
    runs). ``diag`` (a :class:`~propfirm_engine.feasibility.FeasibilityDiag`) is
    filled with per-attempt feasibility diagnostics when supplied — a side record
    that never changes the outcome (so the Level-1 gate holds with the projection
    active)."""
    sim = _ReferenceSim(cp, size_base, policy_params, start_equity, trace,
                        feasibility=feasibility, diag=diag, trade_cost=trade_cost)
    return sim.run(ret, day, trade_low)


__all__ = ["SimResult", "simulate_reference", "size_for"]
