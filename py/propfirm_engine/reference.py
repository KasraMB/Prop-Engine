"""The reference oracle — a slow, obviously-correct one-attempt simulator
(ARCHITECTURE §12; BUILD_SPEC Step 6; MODEL_RISKS §G6, Level 1).

This is the *fixed point* the fast kernel is proven against. It implements the
same predicate/action semantics as :mod:`propfirm_engine.kernels`, but as a plain
stateful class — state is instance attributes, control flow is explicit, and an
optional per-trade ``trace`` records state for debugging ("why did this account
fail on trade 4,217?"). It is never used in the Monte Carlo loop.

The semantics it pins (each cites where it is decided):

* **Trailing drawdown** lives from trade 1 (``dd_floor = start − amount``, §C3);
  its floor ratchets under ``update_timing`` (CONTINUOUS intraday / EOD at close)
  and locks at ``lock_at`` (§6a). A CONTINUOUS breach check reads the day's
  intraday low-water mark; an EOD one reads the day's closing equity.
* **Fail predicates are disjunctive, first-in-rule-order wins** (§6, §C4). A hard
  breach terminates; a soft breach truncates the day (skip its remaining trades),
  the partial loss stands, the day still counts as a trading day but not a winning
  day, and the sim resumes next day (§C5).
* **Pass predicates are conjunctive** (§6); **payouts** fire only through the full
  fire gate (qualifying conjunction AND ``cycle_profit ≥ min_request`` AND the
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

import numpy as np

from .enums import ExitCode, Severity, Stage, StateField, Timing
from .rules import RuleKind

_CONTINUOUS = int(Timing.CONTINUOUS)
_EOD = int(Timing.EOD)
_HARD = int(Severity.HARD)

_ALIVE = int(ExitCode.ALIVE)
_PASSED = int(ExitCode.PASSED)
_TIMED_OUT = int(ExitCode.TIMED_OUT)
_MAXED_OUT = int(ExitCode.MAXED_OUT)


@dataclass
class SimResult:
    """The outcome of one attempt: terminal code + the payouts it released."""

    code: int
    payout_amounts: list[float] = field(default_factory=list)
    payout_days: list[int] = field(default_factory=list)
    total_trading_days: int = 0  # trading days this attempt actually ran (§14.4)
    trace: list[dict] = field(default_factory=list)

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

    def __init__(self, cp, size_base, policy_params, start_equity, trace):
        self.cp = cp
        self.schema = cp.payout
        self.size_base = size_base
        self.policy = np.asarray(policy_params, dtype=np.float64)
        self.start_equity = start_equity
        self.want_trace = trace

        self.equity = start_equity
        self.peak = start_equity
        self.dd_floor = start_equity - cp.dd_amount  # live from trade 1 (§C3)
        self.dd_locked = False
        self.day_pnl = 0.0
        self.total_pnl = 0.0
        self.max_day_pnl = 0.0
        self.day_low = start_equity
        self.profit_target = cp.profit_target0
        self.cycle_start_equity = start_equity
        self.cumulative_paid = 0.0
        self.n_days = 0
        self.n_qual_days = 0
        self.payouts_taken = 0
        self.stage_mask = 0
        self.cur_day = -1

        self.res = SimResult(code=_ALIVE)

    # --- helpers ---------------------------------------------------------- #

    def _has_trailing(self) -> bool:
        return np.isfinite(self.cp.dd_amount)

    def _stage_mask(self) -> int:
        mask = 0
        if self.equity > self.start_equity:
            mask |= 1 << int(Stage.IN_PROFIT)
        if self.payouts_taken == 0:
            mask |= 1 << int(Stage.PRE_FIRST_PAYOUT)
        return mask

    def _first_fail(self, phase, test_equity):
        """First fail rule (rule order) whose predicate holds among those whose
        check_timing matches ``phase``. Returns (hit, severity, fail_code)."""
        cp = self.cp
        for i in cp.fail_idx:
            if int(cp.check_timing[i]) != phase:
                continue
            kind = int(cp.kind[i])
            breached = False
            if kind == int(RuleKind.TRAILING_DD):
                breached = test_equity <= self.dd_floor
            elif kind == int(RuleKind.STATIC_DD):
                breached = test_equity <= self.start_equity - float(cp.p0[i])
            elif kind == int(RuleKind.DAILY_LOSS):
                breached = self.day_pnl <= -float(cp.p0[i])
            if breached:
                return True, int(cp.severity[i]), int(cp.fail_code[i])
        return False, 0, 0

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
        threshold = float(self.cp.p0[i])
        cushion = float(self.cp.p1[i])
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
        if schema is None or cp.payout_idx.shape[0] == 0:
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
        if cycle_profit < schema.min_request:
            return False, 0.0, 0.0
        gross = min(schema.dollar_cap_at(self.payouts_taken),
                    schema.cap_fraction * cycle_profit)
        # never fire a zero/blocked amount (§6b): a $0 release would still record a
        # payout, burn a max_payouts slot, and reset the cycle counters. min_request
        # cannot supply this (neutral default 0.0), so guard gross explicitly.
        if gross <= 0.0:
            return False, 0.0, 0.0
        if equity - gross < schema.buffer_floor:
            return False, 0.0, 0.0
        split = (schema.first_tier_split
                 if self.cumulative_paid < schema.tier_cap else schema.split)
        return True, split * gross, gross

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
        if int(StateField.N_QUALIFYING_DAYS) in [int(x) for x in schema.reset_fields]:
            self.n_qual_days = 0
        self.cycle_start_equity = self.equity
        return self.payouts_taken >= schema.max_payouts

    def _close_day(self, closing_equity, winning_allowed) -> int:
        """One day-end (§C9): fold first, then EOD fail→adjust→(floor)→pass→payout.
        Returns a terminal ExitCode or ALIVE."""
        cp = self.cp
        # (1) fold the just-closed day into the day-scoped counters
        if self.day_pnl > self.max_day_pnl:
            self.max_day_pnl = self.day_pnl
        if winning_allowed and self.day_pnl >= cp.winning_day_threshold:
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
            self.dd_floor = self.peak - cp.dd_amount
            if self.dd_floor >= cp.lock_at:
                self.dd_floor = cp.lock_at
                self.dd_locked = True

        # (2d) EOD PASS (conjunctive) against closing equity
        if self._all_pass(closing_equity):
            return _PASSED

        # (2e) EOD PAYOUT (fire gate) against closing equity
        fired, net, gross = self._try_payout(closing_equity)
        if fired and self._fire_payout(net, gross, self.cur_day):
            return _MAXED_OUT
        return _ALIVE

    # --- the main loop ---------------------------------------------------- #

    def run(self, ret, day, trade_low) -> SimResult:
        cp = self.cp
        N = ret.shape[0]
        t = 0
        while t < N:
            d = int(day[t])
            if d != self.cur_day:  # ---- day boundary ----
                if self.cur_day != -1:
                    code = self._close_day(self.equity, winning_allowed=True)
                    if code != _ALIVE:
                        self.res.code = code
                        self.res.total_trading_days = self.n_days
                        return self.res
                self.day_pnl = 0.0
                self.day_low = self.equity
                self.cur_day = d
                self.n_days += 1

            size = size_for(self.stage_mask, self.policy, self.size_base)
            p = size * float(ret[t])
            self.equity += p
            self.day_pnl += p
            self.total_pnl += p
            trade_floor = self.equity + size * float(trade_low[t])
            if trade_floor < self.day_low:
                self.day_low = trade_floor

            # trailing reference-point update (CONTINUOUS)
            if (not self.dd_locked) and cp.dd_update_timing == _CONTINUOUS and self._has_trailing():
                if self.equity > self.peak:
                    self.peak = self.equity
                self.dd_floor = self.peak - cp.dd_amount
                if self.dd_floor >= cp.lock_at:
                    self.dd_floor = cp.lock_at
                    self.dd_locked = True

            # FAIL (CONTINUOUS-check, intraday, against the day low)
            hit, severity, fail_code = self._first_fail(_CONTINUOUS, self.day_low)
            if hit:
                if severity == _HARD:
                    self.res.code = fail_code
                    self.res.total_trading_days = self.n_days
                    return self.res
                # SOFT: truncate the day; closing equity is the equity after this
                # (the last executed) trade; not a winning day (§C5).
                code = self._close_day(self.equity, winning_allowed=False)
                if code != _ALIVE:
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

        # end-of-path close (§B1)
        if self.cur_day != -1:
            code = self._close_day(self.equity, winning_allowed=True)
            if code != _ALIVE:
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
) -> SimResult:
    """Simulate one attempt of compiled phase ``cp`` over a fixed trade path."""
    sim = _ReferenceSim(cp, size_base, policy_params, start_equity, trace)
    return sim.run(ret, day, trade_low)


__all__ = ["SimResult", "simulate_reference", "size_for"]
