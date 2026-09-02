"""The hot-path kernel — one-attempt simulation over a fixed trade path
(ARCHITECTURE §12; BUILD_SPEC Step 6).

This is the fast counterpart to :mod:`propfirm_engine.reference`. The simulation
runs as a flat, ``@njit``-compiled procedure over the compiled struct-of-arrays —
no Python rule objects, lists, or closures in the loop, everything keyed by
integer ``kind``/``action`` codes and carried in primitive scalars / small state
arrays — exactly what the SoA layout (§8) and integer vocabulary (§3) exist to
enable. It is deliberately structured differently from the class-based reference so
that a transcription slip shows up as a parity mismatch.

**No ``fastmath``** (MODEL_RISKS §C1/§G6): every simulation is an independent
sequential accumulation with no within-sim reduction, so with the same float64
operations in the same order this kernel and the pure-Python reference agree
**bit-for-bit** — Numba's default (fastmath off) preserves IEEE-754 operation
order, so JIT does not move a bit. That is the Level-1 gate; a non-bitwise
difference is a real bug, never benign reassociation.

``simulate_one_phase`` is a thin Python wrapper that unpacks the compiled account /
schema / feasibility spec into primitives, runs the jitted core, and rebuilds the
list-of-payouts + :class:`~propfirm_engine.feasibility.FeasibilityDiag` return
contract — so downstream callers see exactly the same interface as before.
"""

from __future__ import annotations

import numpy as np
from numba import njit

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

_K_TRAILING = int(RuleKind.TRAILING_DD)
_K_STATIC = int(RuleKind.STATIC_DD)
_K_DAILY = int(RuleKind.DAILY_LOSS)
_K_PROFIT = int(RuleKind.PROFIT_TARGET)
_K_MINDAYS = int(RuleKind.MIN_DAYS)
_K_WINDAYS = int(RuleKind.MIN_WINNING_DAYS)
_K_CONS_GATE = int(RuleKind.CONSISTENCY_GATE)
_K_CONS_ADJ = int(RuleKind.CONSISTENCY_ADJUST)

_IN_PROFIT_BIT = 1 << int(Stage.IN_PROFIT)
_PRE_PAYOUT_BIT = 1 << int(Stage.PRE_FIRST_PAYOUT)

# --- float-state (fs) and int-state (is_) vector indices -------------------- #
_EQ = 0; _PEAK = 1; _FLOOR = 2; _DPNL = 3; _TPNL = 4  # noqa: E702
_MAXD = 5; _DLOW = 6; _PT = 7; _CYC = 8; _CUM = 9  # noqa: E702
_LOCK = 0; _NDAYS = 1; _NQUAL = 2; _PTK = 3; _STAGE = 4; _CURD = 5; _NPAY = 6  # noqa: E702

# --- diagnostics vector indices (§16.9) ------------------------------------- #
_D_CAPPED = 0; _D_BREACHED = 1; _D_TTN = 2; _D_TTB = 3  # noqa: E702
_D_TRADES = 4; _D_REDUCED = 5; _D_CONSTR = 6  # noqa: E702
_DF_SUMDES = 0; _DF_SUMEXE = 1; _DF_MINBUF = 2; _DF_MINRATIO = 3  # noqa: E702


def _size(stage_mask, policy, size_base):
    """Constant/staged size for a trade (kept for direct callers/tests; the jitted
    core inlines the same computation)."""
    n = policy.shape[0]
    if n == 1:
        return size_base * float(policy[0])
    idx = stage_mask if stage_mask < n else n - 1
    return size_base * float(policy[idx])


# --------------------------------------------------------------------------- #
# Jitted pure predicate helpers (reads only; no state mutation)                #
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _cons_gate_ok(thr, cush, bal, cycle_start, max_day, day_pnl):
    cycle_profit = bal - cycle_start
    biggest = max_day if max_day >= day_pnl else day_pnl
    return biggest <= thr * cycle_profit + cush


@njit(cache=True)
def _first_fail(fail_idx, kind, check_timing, p0, severity, fail_code,
                phase, test_equity, dd_floor, start_equity, day_pnl):
    for j in range(fail_idx.shape[0]):
        i = fail_idx[j]
        if check_timing[i] != phase:
            continue
        k = kind[i]
        breached = False
        if k == _K_TRAILING:
            breached = test_equity <= dd_floor
        elif k == _K_STATIC:
            breached = test_equity <= start_equity - p0[i]
        elif k == _K_DAILY:
            breached = day_pnl <= -p0[i]
        if breached:
            return True, np.int64(severity[i]), np.int64(fail_code[i])
    return False, np.int64(0), np.int64(0)


@njit(cache=True)
def _all_pass(pass_idx, kind, p0, p1, bal, start_equity, profit_target, n_days,
              cycle_start, max_day, day_pnl):
    if pass_idx.shape[0] == 0:
        return False
    for j in range(pass_idx.shape[0]):
        i = pass_idx[j]
        k = kind[i]
        if k == _K_PROFIT:
            if not (bal - start_equity >= profit_target):
                return False
        elif k == _K_MINDAYS:
            if not (n_days >= p0[i]):
                return False
        elif k == _K_CONS_GATE:
            if not _cons_gate_ok(p0[i], p1[i], bal, cycle_start, max_day, day_pnl):
                return False
    return True


@njit(cache=True)
def _try_payout(payout_idx, kind, p0, p1, n_qual, bal, cycle_start, s_min_request,
                s_dollar_cap, n_caps, payouts_taken, s_cap_fraction, s_buffer_floor,
                s_first_tier, s_tier_cap, s_split, cumulative_paid, max_day, day_pnl):
    if payout_idx.shape[0] == 0:
        return False, 0.0, 0.0
    for j in range(payout_idx.shape[0]):
        i = payout_idx[j]
        k = kind[i]
        if k == _K_WINDAYS:
            if not (n_qual >= p0[i]):
                return False, 0.0, 0.0
        elif k == _K_CONS_GATE:
            if not _cons_gate_ok(p0[i], p1[i], bal, cycle_start, max_day, day_pnl):
                return False, 0.0, 0.0
    cycle_profit = bal - cycle_start
    if cycle_profit < s_min_request:
        return False, 0.0, 0.0
    jcap = payouts_taken if payouts_taken < n_caps else n_caps - 1
    capv = s_cap_fraction * cycle_profit
    gross = s_dollar_cap[jcap] if s_dollar_cap[jcap] < capv else capv
    if gross <= 0.0:
        return False, 0.0, 0.0
    if bal - gross < s_buffer_floor:
        return False, 0.0, 0.0
    split = s_first_tier if cumulative_paid < s_tier_cap else s_split
    return True, split * gross, gross


@njit(cache=True)
def _apply_adjusts(adjust_idx, kind, check_timing, p0, p1, phase, max_day,
                   total_pnl, profit_target):
    pt = profit_target
    for j in range(adjust_idx.shape[0]):
        i = adjust_idx[j]
        if check_timing[i] != phase:
            continue
        if kind[i] == _K_CONS_ADJ:
            if max_day > p0[i] * total_pnl:
                rt = p1[i]
                if rt > pt:
                    pt = rt
    return pt


@njit(cache=True)
def _close_day(fs, is_, pay_amt, pay_day, closing_equity, winning_allowed,
               dd_amount, lock_at, dd_update_timing, win_threshold, has_trailing,
               start_equity, kind, p0, p1, severity, check_timing, fail_code,
               fail_idx, pass_idx, payout_idx, adjust_idx,
               s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
               s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
               s_withdraw, s_recompute, n_caps):
    """One day-end (§C9), mutating ``fs``/``is_``/payout arrays in place. Returns a
    terminal ExitCode or ``_ALIVE`` — the exact fold-then-evaluate order of the
    reference (`_ReferenceSim._close_day`)."""
    # (1) fold the just-closed day into the day-scoped counters
    if fs[_DPNL] > fs[_MAXD]:
        fs[_MAXD] = fs[_DPNL]
    if winning_allowed and fs[_DPNL] >= win_threshold:
        is_[_NQUAL] += 1
    # (2a) EOD FAIL against the established floor / closing equity
    hit, sev, fc = _first_fail(fail_idx, kind, check_timing, p0, severity, fail_code,
                               _EOD, closing_equity, fs[_FLOOR], start_equity, fs[_DPNL])
    if hit and sev == _HARD:
        return fc
    # (2b) EOD ADJUST
    fs[_PT] = _apply_adjusts(adjust_idx, kind, check_timing, p0, p1, _EOD,
                             fs[_MAXD], fs[_TPNL], fs[_PT])
    # (2c) EOD floor ratchet (advance off closing equity, then lock)
    if is_[_LOCK] == 0 and dd_update_timing == _EOD and has_trailing:
        if closing_equity > fs[_PEAK]:
            fs[_PEAK] = closing_equity
        fs[_FLOOR] = fs[_PEAK] - dd_amount
        if fs[_FLOOR] >= lock_at:
            fs[_FLOOR] = lock_at
            is_[_LOCK] = 1
    # (2d) EOD PASS
    if _all_pass(pass_idx, kind, p0, p1, closing_equity, start_equity, fs[_PT],
                 is_[_NDAYS], fs[_CYC], fs[_MAXD], fs[_DPNL]):
        return _PASSED
    # (2e) EOD PAYOUT (fire gate)
    fired, net, gross = _try_payout(
        payout_idx, kind, p0, p1, is_[_NQUAL], closing_equity, fs[_CYC], s_min_request,
        s_dollar_cap, n_caps, is_[_PTK], s_cap_fraction, s_buffer_floor,
        s_first_tier, s_tier_cap, s_split, fs[_CUM], fs[_MAXD], fs[_DPNL])
    if fired:
        k = is_[_NPAY]
        pay_amt[k] = net
        pay_day[k] = is_[_CURD]
        is_[_NPAY] = k + 1
        is_[_PTK] += 1
        fs[_CUM] += gross
        if s_withdraw:
            fs[_EQ] -= gross
        if s_recompute and is_[_LOCK] == 0 and has_trailing:
            fs[_PEAK] = fs[_EQ]
            fs[_FLOOR] = fs[_EQ] - dd_amount
        if s_reset_qual:
            is_[_NQUAL] = 0
        fs[_CYC] = fs[_EQ]
        if is_[_PTK] >= s_max_payouts:
            return _MAXED_OUT
    return _ALIVE


@njit(cache=True)
def _fold_buffer(diag_f, buffer, feas_lmin):
    if buffer < diag_f[_DF_MINBUF]:
        diag_f[_DF_MINBUF] = buffer
    ratio = buffer / feas_lmin if feas_lmin > 0.0 else np.inf
    if ratio < diag_f[_DF_MINRATIO]:
        diag_f[_DF_MINRATIO] = ratio


@njit(cache=True)
def _simulate_core(ret, day, trade_low, size_base, policy, start_equity, trade_cost,
                   dd_amount, lock_at, dd_update_timing, win_threshold,
                   profit_target0, has_trailing, is_funded,
                   kind, p0, p1, severity, check_timing, fail_code,
                   fail_idx, pass_idx, payout_idx, adjust_idx,
                   s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
                   s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
                   s_withdraw, s_recompute, n_caps,
                   feas_active, feas_qmin, feas_uloss, feas_alpha, feas_lmin,
                   feas_min_buffer,
                   pay_amt, pay_day, want_diag, diag_i, diag_f):
    """The jitted single-attempt core. Fills ``pay_amt``/``pay_day`` (count in
    ``diag``-free ``is_[_NPAY]``) and, when ``want_diag``, ``diag_i``/``diag_f``.
    Returns ``(code, n_pay, n_days)``. Bit-for-bit identical to the reference."""
    n_pol = policy.shape[0]
    fs = np.empty(10, dtype=np.float64)
    is_ = np.zeros(7, dtype=np.int64)
    fs[_EQ] = start_equity
    fs[_PEAK] = start_equity
    fs[_FLOOR] = start_equity - dd_amount
    fs[_DPNL] = 0.0
    fs[_TPNL] = 0.0
    fs[_MAXD] = 0.0
    fs[_DLOW] = start_equity
    fs[_PT] = profit_target0
    fs[_CYC] = start_equity
    fs[_CUM] = 0.0
    is_[_CURD] = -1
    # Sizing regime index: 0 = eval (single regime); funded = 1..4 over
    # in-profit × pre/post-first-payout. The first funded trade is flat & pre-payout
    # (index 1). Eval is always regime 0.
    is_[_STAGE] = 1 if is_funded else 0

    N = ret.shape[0]
    t = 0
    while t < N:
        d = day[t]
        if d != is_[_CURD]:  # ---- day boundary ----
            if is_[_CURD] != -1:
                code = _close_day(
                    fs, is_, pay_amt, pay_day, fs[_EQ], True,
                    dd_amount, lock_at, dd_update_timing, win_threshold, has_trailing,
                    start_equity, kind, p0, p1, severity, check_timing, fail_code,
                    fail_idx, pass_idx, payout_idx, adjust_idx,
                    s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
                    s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
                    s_withdraw, s_recompute, n_caps)
                if code != _ALIVE:
                    if want_diag and 10 <= code < _TIMED_OUT:
                        diag_i[_D_BREACHED] = 1
                        diag_i[_D_TTB] = is_[_NDAYS]
                    return code, is_[_NPAY], is_[_NDAYS]
            fs[_DPNL] = 0.0
            fs[_DLOW] = fs[_EQ]
            is_[_CURD] = d
            is_[_NDAYS] += 1

        # sizing hook (constant/staged) then feasibility projection
        stage = is_[_STAGE]
        idx = 0 if n_pol == 1 else (stage if stage < n_pol else n_pol - 1)
        size = size_base * policy[idx]
        if feas_active:
            desired = size
            buffer = fs[_EQ] - fs[_FLOOR]
            size, capped, reduced, at_cap = project_position(
                desired, buffer, feas_qmin, feas_uloss, feas_alpha, feas_min_buffer)
            if capped:
                if want_diag:
                    diag_i[_D_CAPPED] = 1
                    diag_i[_D_TTN] = is_[_NDAYS]
                    _fold_buffer(diag_f, buffer, feas_lmin)
                return _CAPPED_OUT, is_[_NPAY], is_[_NDAYS]
            if want_diag:
                diag_i[_D_TRADES] += 1
                diag_f[_DF_SUMDES] += desired
                diag_f[_DF_SUMEXE] += size
                if reduced:
                    diag_i[_D_REDUCED] += 1
                if at_cap:
                    diag_i[_D_CONSTR] += 1
                _fold_buffer(diag_f, buffer, feas_lmin)

        entry_eq = fs[_EQ]
        r = ret[t]
        p = size * r - trade_cost           # realized P&L net of the per-trade cost
        fs[_EQ] += p
        fs[_DPNL] += p
        fs[_TPNL] += p
        # True intra-trade floating low, measured FROM ENTRY: the lower of the close
        # and the MAE excursion (trade_low = -mae). Pre-cost — a commission settles
        # at the close, it does not move the intraday low-water mark.
        tl = trade_low[t]
        exc = r if r < tl else tl           # min(ret, trade_low)
        trade_floor = entry_eq + size * exc
        if trade_floor < fs[_DLOW]:
            fs[_DLOW] = trade_floor

        if is_[_LOCK] == 0 and dd_update_timing == _CONTINUOUS and has_trailing:
            if fs[_EQ] > fs[_PEAK]:
                fs[_PEAK] = fs[_EQ]
            fs[_FLOOR] = fs[_PEAK] - dd_amount
            if fs[_FLOOR] >= lock_at:
                fs[_FLOOR] = lock_at
                is_[_LOCK] = 1

        hit, sev, fc = _first_fail(fail_idx, kind, check_timing, p0, severity,
                                   fail_code, _CONTINUOUS, fs[_DLOW], fs[_FLOOR],
                                   start_equity, fs[_DPNL])
        if hit:
            if sev == _HARD:
                if want_diag:
                    diag_i[_D_BREACHED] = 1
                    diag_i[_D_TTB] = is_[_NDAYS]
                return fc, is_[_NPAY], is_[_NDAYS]
            code = _close_day(
                fs, is_, pay_amt, pay_day, fs[_EQ], False,  # soft: not a winning day
                dd_amount, lock_at, dd_update_timing, win_threshold, has_trailing,
                start_equity, kind, p0, p1, severity, check_timing, fail_code,
                fail_idx, pass_idx, payout_idx, adjust_idx,
                s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
                s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
                s_withdraw, s_recompute, n_caps)
            if code != _ALIVE:
                if want_diag and 10 <= code < _TIMED_OUT:
                    diag_i[_D_BREACHED] = 1
                    diag_i[_D_TTB] = is_[_NDAYS]
                return code, is_[_NPAY], is_[_NDAYS]
            dd = day[t]
            j = t + 1
            while j < N and day[j] == dd:
                j += 1
            t = j
            is_[_CURD] = -1
            continue

        fs[_PT] = _apply_adjusts(adjust_idx, kind, check_timing, p0, p1, _CONTINUOUS,
                                 fs[_MAXD], fs[_TPNL], fs[_PT])

        # regime index for the NEXT trade's sizing (one-trade lag, §16.4): eval stays
        # regime 0; funded = 1 + 2*(post-first-payout) + in-profit -> 1..4.
        if not is_funded:
            is_[_STAGE] = 0
        else:
            ip = 1 if fs[_EQ] > start_equity else 0
            post = 0 if is_[_PTK] == 0 else 1
            is_[_STAGE] = 1 + post * 2 + ip

        if _all_pass(pass_idx, kind, p0, p1, fs[_EQ], start_equity, fs[_PT],
                     is_[_NDAYS], fs[_CYC], fs[_MAXD], fs[_DPNL]):
            return _PASSED, is_[_NPAY], is_[_NDAYS]
        t += 1

    if is_[_CURD] != -1:
        code = _close_day(
            fs, is_, pay_amt, pay_day, fs[_EQ], True,
            dd_amount, lock_at, dd_update_timing, win_threshold, has_trailing,
            start_equity, kind, p0, p1, severity, check_timing, fail_code,
            fail_idx, pass_idx, payout_idx, adjust_idx,
            s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
            s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
            s_withdraw, s_recompute, n_caps)
        if code != _ALIVE:
            if want_diag and 10 <= code < _TIMED_OUT:
                diag_i[_D_BREACHED] = 1
                diag_i[_D_TTB] = is_[_NDAYS]
            return code, is_[_NPAY], is_[_NDAYS]

    return _TIMED_OUT, is_[_NPAY], is_[_NDAYS]


@njit(cache=True)
def _simulate_batch_core(day_paths, ds_ret, ds_trade_low, ds_day_first, ds_day_count,
                         size_base, policy, start_equity, trade_cost,
                         dd_amount, lock_at, dd_update_timing, win_threshold,
                         profit_target0, has_trailing, is_funded,
                         kind, p0, p1, severity, check_timing, fail_code,
                         fail_idx, pass_idx, payout_idx, adjust_idx,
                         s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
                         s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
                         s_withdraw, s_recompute, n_caps,
                         feas_active, feas_qmin, feas_uloss, feas_alpha, feas_lmin,
                         feas_min_buffer, pay_cap, max_len, want_diag,
                         out_code, out_ptk, out_net, out_fpd, out_days,
                         agg_i, agg_f):
    """Fused batch: loop attempts entirely in compiled code, gathering each path into
    reused scratch buffers and running :func:`_simulate_core` — no per-attempt Python
    dispatch or allocation. Writes aggregated per-attempt outcomes into the preallocated
    output arrays; identical to running the single-path kernel per attempt (§13)."""
    B, L = day_paths.shape
    scratch_ret = np.empty(max_len, dtype=np.float64)
    scratch_low = np.empty(max_len, dtype=np.float64)
    scratch_day = np.empty(max_len, dtype=np.int32)
    pay_amt = np.empty(pay_cap, dtype=np.float64)
    pay_day = np.empty(pay_cap, dtype=np.int64)
    for b in range(B):
        n = 0
        for k in range(L):
            src = day_paths[b, k]
            start = ds_day_first[src]
            cnt = ds_day_count[src]
            for j in range(cnt):
                scratch_ret[n] = ds_ret[start + j]
                scratch_low[n] = ds_trade_low[start + j]
                scratch_day[n] = k
                n += 1
        diag_i = np.zeros(7, dtype=np.int64)
        diag_i[_D_TTN] = -1
        diag_i[_D_TTB] = -1
        diag_f = np.empty(4, dtype=np.float64)
        diag_f[_DF_SUMDES] = 0.0
        diag_f[_DF_SUMEXE] = 0.0
        diag_f[_DF_MINBUF] = np.inf
        diag_f[_DF_MINRATIO] = np.inf
        code, n_pay, ndays = _simulate_core(
            scratch_ret[:n], scratch_day[:n], scratch_low[:n], size_base, policy,
            start_equity, trade_cost, dd_amount, lock_at, dd_update_timing, win_threshold,
            profit_target0, has_trailing, is_funded,
            kind, p0, p1, severity, check_timing, fail_code,
            fail_idx, pass_idx, payout_idx, adjust_idx,
            s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor,
            s_split, s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual,
            s_withdraw, s_recompute, n_caps,
            feas_active, feas_qmin, feas_uloss, feas_alpha, feas_lmin,
            feas_min_buffer,
            pay_amt, pay_day, want_diag, diag_i, diag_f)
        out_code[b] = code
        out_days[b] = ndays
        out_ptk[b] = n_pay
        s = 0.0
        for i in range(n_pay):
            s += pay_amt[i]
        out_net[b] = s
        out_fpd[b] = pay_day[0] if n_pay > 0 else -1
        if want_diag:
            trades = diag_i[_D_TRADES]
            if trades > 0:
                inv = 1.0 / trades
                agg_f[0] += diag_i[_D_REDUCED] * inv
                agg_f[1] += diag_i[_D_CONSTR] * inv
                agg_f[2] += diag_f[_DF_SUMDES] * inv
                agg_f[3] += diag_f[_DF_SUMEXE] * inv
            agg_i[0] += 1
            if diag_i[_D_CAPPED] != 0:
                agg_i[1] += 1
            if diag_i[_D_BREACHED] != 0:
                agg_i[2] += 1


# --------------------------------------------------------------------------- #
# Python wrappers — same interface as before                                   #
# --------------------------------------------------------------------------- #

_DUMMY_CAP = np.ones(1, dtype=np.float64)


def _unpack(cp, feasibility):
    """Unpack a compiled phase + feasibility spec into the primitive scalars/arrays
    the jitted core/batch take — ONCE per batch instead of once per attempt."""
    dd_amount = cp.dd_amount
    has_trailing = bool(np.isfinite(dd_amount))
    feas_active = feasibility is not None and has_trailing
    if feasibility is not None:
        fq, fu, fa = float(feasibility.q_min), float(feasibility.unit_loss), float(feasibility.alpha)
        fl = fq * fu
        fmb = float(feasibility.min_buffer)
    else:
        fq = fu = fa = fl = 1.0
        fmb = 0.0
    schema = cp.payout
    if schema is not None:
        s = (np.ascontiguousarray(schema.dollar_cap, dtype=np.float64),
             float(schema.cap_fraction), float(schema.min_request),
             float(schema.buffer_floor), float(schema.split),
             float(schema.first_tier_split), float(schema.tier_cap),
             int(schema.max_payouts), bool(schema.resets_qualifying_days),
             bool(schema.withdraw_reduces_equity),
             bool(schema.recompute_floor_on_payout))
        n_caps = s[0].shape[0]
    else:
        s = (_DUMMY_CAP, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, False, False, False)
        n_caps = 1
    dd_amount_f = float(dd_amount) if has_trailing else 0.0
    return (dd_amount_f, float(cp.lock_at), int(cp.dd_update_timing),
            float(cp.winning_day_threshold), float(cp.profit_target0), has_trailing,
            s, n_caps, feas_active, fq, fu, fa, fl, fmb)


def simulate_batch(cp, dataset, day_paths, size_base, policy_params, start_equity,
                   feasibility=None, want_diag=False, trade_cost=0.0):
    """Fused compiled batch over ``day_paths`` (int[B, L]) — the fast path for
    :func:`propfirm_engine.simulate.simulate_phase_batch` when materialization is not
    precomputed. Returns ``(code, payouts_taken, net_payout, first_payout_day,
    total_days, agg_i, agg_f)`` with the arrays aligned to attempts."""
    policy = np.ascontiguousarray(policy_params, dtype=np.float64)
    day_paths = np.ascontiguousarray(day_paths, dtype=np.int64)
    (dd_amount_f, lock_at, dd_update_timing, win_threshold, profit_target0,
     has_trailing, s, n_caps, feas_active, fq, fu, fa, fl, fmb) = _unpack(cp, feasibility)
    (s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor, s_split,
     s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual, s_withdraw,
     s_recompute) = s

    B, L = day_paths.shape
    max_dc = int(dataset.day_count.max()) if dataset.day_count.shape[0] else 0
    max_len = L * max_dc if max_dc > 0 else 1
    pay_cap = s_max_payouts if s_max_payouts > 0 else 1

    out_code = np.empty(B, dtype=np.int64)
    out_ptk = np.zeros(B, dtype=np.int64)
    out_net = np.zeros(B, dtype=np.float64)
    out_fpd = np.full(B, -1, dtype=np.int64)
    out_days = np.zeros(B, dtype=np.int64)
    agg_i = np.zeros(3, dtype=np.int64)
    agg_f = np.zeros(4, dtype=np.float64)

    _simulate_batch_core(
        day_paths, dataset.ret, dataset.trade_low, dataset.day_first, dataset.day_count,
        float(size_base), policy, float(start_equity), float(trade_cost),
        dd_amount_f, lock_at, dd_update_timing, win_threshold, profit_target0,
        has_trailing, cp.role == "funded",
        cp.kind, cp.p0, cp.p1, cp.severity, cp.check_timing, cp.fail_code,
        cp.fail_idx, cp.pass_idx, cp.payout_idx, cp.adjust_idx,
        s_dollar_cap, s_cap_fraction, s_min_request, s_buffer_floor, s_split,
        s_first_tier, s_tier_cap, s_max_payouts, s_reset_qual, s_withdraw, s_recompute,
        n_caps, feas_active, fq, fu, fa, fl, fmb, pay_cap, max_len, want_diag,
        out_code, out_ptk, out_net, out_fpd, out_days, agg_i, agg_f)
    return out_code, out_ptk, out_net, out_fpd, out_days, agg_i, agg_f


def simulate_one_phase(cp, ret, day, trade_low, size_base, policy_params, start_equity,
                       feasibility=None, diag_out=None, trade_cost=0.0):
    """Simulate one attempt; return ``(code, payout_amounts, payout_days, total_days)``.

    ``payout_amounts``/``payout_days`` are lists (net amount, day-index) in fire
    order; ``total_days`` is the number of trading days this attempt actually ran
    (§13/§14.4 — the time axis). Matches
    :func:`propfirm_engine.reference.simulate_reference` bit-for-bit.

    ``feasibility`` (a :class:`~propfirm_engine.feasibility.FeasibilitySpec` or
    ``None``) turns the sizing hook's output into an *executable* position against
    the trailing-drawdown buffer (§16.4b); ``None`` reproduces the constant-position
    path **bit-for-bit** (the projection never runs). ``diag_out``, if a
    :class:`~propfirm_engine.feasibility.FeasibilityDiag`, is filled with per-attempt
    feasibility diagnostics (§16.9) — a pure side record that never influences the
    outcome. This is a thin wrapper over the jitted :func:`_simulate_core`.
    """
    policy = np.ascontiguousarray(policy_params, dtype=np.float64)
    ret = np.ascontiguousarray(ret, dtype=np.float64)
    day = np.ascontiguousarray(day, dtype=np.int64)
    trade_low = np.ascontiguousarray(trade_low, dtype=np.float64)

    dd_amount = cp.dd_amount
    has_trailing = bool(np.isfinite(dd_amount))
    feas_active = feasibility is not None and has_trailing
    if feasibility is not None:
        feas_qmin = float(feasibility.q_min)
        feas_uloss = float(feasibility.unit_loss)
        feas_alpha = float(feasibility.alpha)
        feas_lmin = feas_qmin * feas_uloss
        feas_min_buffer = float(feasibility.min_buffer)
    else:
        feas_qmin = feas_uloss = feas_alpha = feas_lmin = 1.0
        feas_min_buffer = 0.0

    schema = cp.payout
    if schema is not None:
        s_dollar_cap = np.ascontiguousarray(schema.dollar_cap, dtype=np.float64)
        s_cap_fraction = schema.cap_fraction
        s_min_request = schema.min_request
        s_buffer_floor = schema.buffer_floor
        s_split = schema.split
        s_first_tier = schema.first_tier_split
        s_tier_cap = schema.tier_cap
        s_max_payouts = int(schema.max_payouts)
        s_reset_qual = bool(schema.resets_qualifying_days)
        s_withdraw = bool(schema.withdraw_reduces_equity)
        s_recompute = bool(schema.recompute_floor_on_payout)
        n_caps = s_dollar_cap.shape[0]
    else:
        s_dollar_cap = _DUMMY_CAP
        s_cap_fraction = s_min_request = s_buffer_floor = 0.0
        s_split = s_first_tier = s_tier_cap = 0.0
        s_max_payouts = 0
        s_reset_qual = s_withdraw = s_recompute = False
        n_caps = 1

    # lock_at may be +inf; keep as float. dd_amount may be +inf when no trailing —
    # pass a finite dummy so the (never-taken) arithmetic stays defined for numba.
    dd_amount_f = float(dd_amount) if has_trailing else 0.0
    lock_at = float(cp.lock_at)

    pay_cap = s_max_payouts if s_max_payouts > 0 else 1
    pay_amt = np.empty(pay_cap, dtype=np.float64)
    pay_day = np.empty(pay_cap, dtype=np.int64)

    want_diag = diag_out is not None
    diag_i = np.zeros(7, dtype=np.int64)
    diag_i[_D_TTN] = -1
    diag_i[_D_TTB] = -1
    diag_f = np.array([0.0, 0.0, np.inf, np.inf], dtype=np.float64)

    code, n_pay, n_days = _simulate_core(
        ret, day, trade_low, float(size_base), policy, float(start_equity),
        float(trade_cost),
        dd_amount_f, lock_at, int(cp.dd_update_timing), float(cp.winning_day_threshold),
        float(cp.profit_target0), has_trailing, cp.role == "funded",
        cp.kind, cp.p0, cp.p1, cp.severity, cp.check_timing, cp.fail_code,
        cp.fail_idx, cp.pass_idx, cp.payout_idx, cp.adjust_idx,
        s_dollar_cap, float(s_cap_fraction), float(s_min_request), float(s_buffer_floor),
        float(s_split), float(s_first_tier), float(s_tier_cap), s_max_payouts,
        s_reset_qual, s_withdraw, s_recompute, n_caps,
        feas_active, feas_qmin, feas_uloss, feas_alpha, feas_lmin, feas_min_buffer,
        pay_amt, pay_day, want_diag, diag_i, diag_f)

    amounts = [float(pay_amt[i]) for i in range(n_pay)]
    days = [int(pay_day[i]) for i in range(n_pay)]

    if want_diag:
        diag_out.capped_out = bool(diag_i[_D_CAPPED])
        diag_out.breached = bool(diag_i[_D_BREACHED])
        diag_out.time_to_nontradable = int(diag_i[_D_TTN])
        diag_out.time_to_breach = int(diag_i[_D_TTB])
        diag_out.trades = int(diag_i[_D_TRADES])
        diag_out.reduced = int(diag_i[_D_REDUCED])
        diag_out.constrained = int(diag_i[_D_CONSTR])
        diag_out.sum_desired = float(diag_f[_DF_SUMDES])
        diag_out.sum_executed = float(diag_f[_DF_SUMEXE])
        diag_out.min_buffer = float(diag_f[_DF_MINBUF])
        diag_out.min_tradability_ratio = float(diag_f[_DF_MINRATIO])

    return int(code), amounts, days, int(n_days)


__all__ = ["simulate_one_phase", "simulate_batch"]
