"""Monte Carlo explorer — drive the *full* engine pipeline from generator params.

Unlike :mod:`bridge` (which walks a hand-entered day sequence through the
single-path reference for the interactive page), this runs the real batch engine:

    trade-stream generator  →  preprocess  →  Engine.run (Monte Carlo)  →  statistics/renewal

and returns both the headline decision statistics and the raw per-attempt
distributions the UI turns into charts. It uses the **real firm accounts** (which
carry a ``PayoutSchema``, so payouts actually fire) with user-supplied fees so the
fee/renewal metrics are well posed.
"""

from __future__ import annotations

import csv
import io
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np


from propfirm_engine import renewal as rn  # noqa: E402
from propfirm_engine import statistics as st  # noqa: E402
from propfirm_engine.compiler import compile_phase  # noqa: E402
from propfirm_engine.data import InvalidTradeDataError, preprocess  # noqa: E402
from propfirm_engine.engine import Engine, RunConfig  # noqa: E402
from propfirm_engine.enums import ExitCode  # noqa: E402
from propfirm_engine.feasibility import FeasibilitySpec  # noqa: E402
from propfirm_engine.firms import lucidflex  # noqa: E402
from propfirm_engine.optimizer import (  # noqa: E402
    REGIME_LABELS,
    OptConfig,
    PolicySpace,
    RenewalObjective,
    policy_space_for,
    walk_forward,
)
from propfirm_engine.reference import simulate_reference  # noqa: E402
from propfirm_engine.resampling import (  # noqa: E402
    IIDDayBootstrap,
    StationaryDayBootstrap,
    gather_days,
)
from propfirm_engine.synthetic import (  # noqa: E402
    IIDGenerator,
    RegimeSwitchingGenerator,
    StochasticVolGenerator,
)

_ENGINE = Engine()

_SIZE_BY_NAME = {f"{s // 1000}K": s for s in lucidflex.SPECS}
_CODE_NAMES = {int(c): c.name for c in ExitCode}


def registry() -> dict:
    """The accounts the explorer can run (real firm configs, with payout schemas)."""
    return {"Lucid": {"LucidFlex": list(_SIZE_BY_NAME)}}


# --------------------------------------------------------------------------- #
# Building the generator + account from the request                            #
# --------------------------------------------------------------------------- #


def _make_generator(p):
    kind = p.get("generator", "iid")
    common = dict(win_rate=float(p["win_rate"]), rr=float(p["rr"]),
                  trades_per_day=int(p.get("trades_per_day", 4)),
                  intraday_excursion=float(p.get("intraday_excursion", 0.5)))
    if kind == "regime":
        return RegimeSwitchingGenerator(
            persistence=float(p.get("persistence", 0.95)),
            spread=float(p.get("spread", 0.25)), **common)
    if kind == "stochvol":
        return StochasticVolGenerator(
            vol_phi=float(p.get("vol_phi", 0.9)),
            vol_sigma=float(p.get("vol_sigma", 0.5)), **common)
    return IIDGenerator(**common)


def _account(size_name, eval_fee, activation_fee):
    acct = lucidflex.build_account(_SIZE_BY_NAME[size_name])
    return replace(acct, eval_fee=float(eval_fee), activation_fee=float(activation_fee))


def _resampler(p):
    mb = float(p.get("mean_block", 1.0))
    return StationaryDayBootstrap(mb) if mb > 1.0 else IIDDayBootstrap()


def _feasibility(p):
    if not p.get("feasibility"):
        return None
    # The dashboard exposes ONE knob: a hard drawdown floor ($). Sizing is the
    # standard "stop lands at the buffer" rule (unit_loss = 1R stop, alpha = 1, fine
    # 1-unit granularity); the account caps out once less than `hard_floor` of
    # drawdown remains.
    return FeasibilitySpec(q_min=1.0, unit_loss=1.0, alpha=1.0,
                           min_buffer=float(p.get("hard_floor", 0.0)))


def _run_config(p, n_paths, seed):
    return RunConfig(
        n_paths=int(n_paths),
        L_eval=int(p.get("L_eval", 30)),
        L_funded=int(p.get("L_funded", 60)),
        seed=int(seed),
        resampler=_resampler(p),
        size_base=float(p.get("size_base", 100.0)),
        trade_cost=float(p.get("trade_cost", 0.0)),
    )


# --------------------------------------------------------------------------- #
# Chart helpers                                                                 #
# --------------------------------------------------------------------------- #


def _hist(values, bins=32):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"edges": [], "counts": []}
    if np.allclose(v.min(), v.max()):  # a single spike -> a narrow band around it
        c = v.min()
        return {"edges": [c - 0.5, c + 0.5], "counts": [int(v.size)]}
    counts, edges = np.histogram(v, bins=bins)
    return {"edges": [round(float(e), 4) for e in edges],
            "counts": [int(c) for c in counts]}


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 6)


# --------------------------------------------------------------------------- #
# The main entry                                                                #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Uploaded real trades — same pipeline, empirical day pool instead of a generator #
# --------------------------------------------------------------------------- #

# columns preprocess understands (case-insensitive); we keep only these.
_CSV_NUMERIC = ("return", "pnl", "size", "mae")


def _rows_from_csv(text: str) -> list[dict]:
    """Parse an uploaded trades CSV into the raw row-mappings :func:`preprocess`
    consumes. Required: ``timestamp`` + either ``return`` or (``pnl`` and ``size``);
    optional ``mae``. Friendly ``ValueError`` on anything malformed."""
    text = (text or "").lstrip("﻿")  # drop a UTF-8 BOM if present
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("The file is empty or has no header row.")
    headers = {(h or "").strip().lower() for h in reader.fieldnames}
    if "timestamp" not in headers:
        raise ValueError("Missing a 'timestamp' column.")
    if "return" not in headers and not ({"pnl", "size"} <= headers):
        raise ValueError("Provide a 'return' column, or both 'pnl' and 'size'.")
    rows: list[dict] = []
    for i, raw in enumerate(reader, start=1):
        rl = {(k or "").strip().lower(): v for k, v in raw.items()}
        ts = (rl.get("timestamp") or "").strip()
        if not ts:
            raise ValueError(f"Row {i}: missing timestamp.")
        row: dict = {"timestamp": ts}
        for key in _CSV_NUMERIC:
            v = rl.get(key)
            if v is not None and str(v).strip() != "":
                try:
                    row[key] = float(v)
                except ValueError:
                    raise ValueError(f"Row {i}: '{key}' value {v!r} is not a number.")
        rows.append(row)
    if not rows:
        raise ValueError("No trade rows found under the header.")
    return rows


class UploadedTrades:
    """Stands in for a generator so :func:`_build` gets the same ``edge`` /
    ``breakeven`` / provenance interface — but computed empirically from the
    uploaded trades rather than from generator parameters."""

    def __init__(self, ds):
        ret = ds.ret
        wins, losses = ret[ret > 0], ret[ret < 0]
        wr = float(wins.size) / ret.size if ret.size else 0.0
        aw = float(np.mean(wins)) if wins.size else 0.0
        al = float(-np.mean(losses)) if losses.size else 0.0
        self.edge = float(np.mean(ret)) if ret.size else 0.0  # E[R] per trade
        self.breakeven_win_rate = al / (aw + al) if (aw + al) > 0 else float("nan")
        self.provenance = SimpleNamespace(params={
            "source": "your uploaded trades", "trades": int(ret.size),
            "days": int(ds.n_days), "win_rate": round(wr, 3),
            "avg_R_win": round(aw, 3), "avg_R_loss": round(al, 3),
            "days_per_week": round(float(ds.trading_days_per_week), 2)})


def _run_uploaded(params: dict) -> dict:
    """Run the standard (constant-size) pipeline on the user's own trades: their
    realized days become the empirical pool the Monte Carlo bootstraps, so every
    headline stat is computed exactly as in generator mode."""
    try:
        rows = _rows_from_csv(params["uploaded_csv"])
        dataset = preprocess(rows)
    except (ValueError, InvalidTradeDataError) as e:
        return {"error": f"Could not read your trades: {e}"}
    acct = _account(params["size"], params.get("eval_fee", 150.0),
                    params.get("activation_fee", 100.0))
    feas = _feasibility(params)
    seed = int(params.get("seed", 1))
    n_paths = int(params.get("n_paths", 4000))
    cfg = _run_config(params, n_paths, seed)
    src = UploadedTrades(dataset)
    o = _ENGINE.run(acct, dataset, cfg, feasibility=feas)
    result = _build(o, src, feas, dataset, cfg, acct, None)
    result["mode"] = "single"
    result["provenance"] = _provenance(src)
    result["uploaded"] = dict(src.provenance.params)
    return result


def run(params: dict) -> dict:
    """Run the full pipeline for ``params`` and return numbers + chart data.

    Three modes: the default single (constant-size baseline) run; a Tier-1 sizing
    optimization (``optimize``) charted on **held-out** data (see
    :func:`_run_optimized`); and — when ``uploaded_csv`` is supplied — the single
    run driven by the user's own trades (see :func:`_run_uploaded`)."""
    if params.get("uploaded_csv"):
        return _run_uploaded(params)
    gen = _make_generator(params)
    acct = _account(params["size"], params.get("eval_fee", 150.0),
                    params.get("activation_fee", 100.0))
    feas = _feasibility(params)
    n_days = int(params.get("n_days", 160))
    seed = int(params.get("seed", 1))
    n_paths = int(params.get("n_paths", 4000))

    if params.get("optimize"):
        return _run_optimized(params, gen, acct, feas, n_days, seed, n_paths)

    dataset = preprocess(gen.generate(n_days=n_days, seed=seed).rows)
    cfg = _run_config(params, n_paths, seed)
    o = _ENGINE.run(acct, dataset, cfg, feasibility=feas)
    result = _build(o, gen, feas, dataset, cfg, acct, None)
    result["mode"] = "single"
    result["provenance"] = _provenance(gen)
    if params.get("surface"):
        result["surface"] = _surface(params, acct, feas)
    return result


def _provenance(gen):
    prov = gen.provenance if hasattr(gen, "provenance") else None
    return {"generator": type(gen).__name__, "params": prov.params if prov else {}}


# The four reachable stage regimes a Tier-1 policy sizes over (§16.4 / optimizer
# in theta / regime-index order (eval, then the four funded regimes).
_STAGE_LABELS = list(REGIME_LABELS)


def _run_optimized(params, gen, acct, feas, n_days, seed, n_paths):
    """Fit a Tier-1 sizing policy with CMA-ES on a TRAIN stream, then chart the
    baseline and fitted policies on an independent HELD-OUT stream (nested OOS,
    §16.7). Returns both chart blocks plus the optimizer summary."""
    from dataclasses import replace as _replace

    train = preprocess(gen.generate(n_days=n_days, seed=seed).rows)
    test = preprocess(gen.generate(n_days=n_days, seed=seed + 9999).rows)  # held out
    base_cfg = _run_config(params, n_paths, seed)
    # Account-aware multiplier bound (ceiling = risk the whole MLL in one trade), so
    # the reachable effective size is independent of size_base (§16.3).
    space = policy_space_for(acct, base_cfg.size_base, feas)
    obj = RenewalObjective()
    max_gen = int(params.get("opt_generations", 12))
    select = min(int(n_paths), 2500)
    oc = OptConfig(max_gen=max_gen, popsize=8, seed=0,
                   screen_paths=min(int(n_paths), 700), select_paths=select)
    wf = walk_forward(acct, train, test, base_cfg, space=space, objective=obj,
                      opt_config=oc, feasibility=feas, engine=_ENGINE)

    base_policy = space.to_policy(space.x0())
    fitted = wf.policy
    # A FRESH held-out draw both policies are charted on — the honest OOS comparison.
    eval_cfg = _replace(base_cfg, n_paths=select, seed=seed + 4242)
    o_base = _ENGINE.run(acct, test, eval_cfg, policy_params=base_policy, feasibility=feas)
    o_opt = _ENGINE.run(acct, test, eval_cfg, policy_params=fitted, feasibility=feas)

    optimizer = {
        "multipliers": [{"regime": _STAGE_LABELS[i], "mult": round(float(wf.theta[i]), 3)}
                        for i in range(len(wf.theta))],
        "train_score": _f(wf.train_score),
        "baseline_train_score": _f(wf.baseline_train_score),
        "oos_score": _f(wf.oos_score),
        "baseline_oos_score": _f(wf.baseline_oos_score),
        "oos_improvement": _f(wf.oos_improvement),
        "history": [_f(x) for x in wf.history],
        "generations": max_gen,
        "objective": "renewal reward rate  E[R]/E[T] ($/wk), on held-out data",
    }
    return {
        "mode": "optimized",
        "baseline": _build(o_base, gen, feas, test, eval_cfg, acct, base_policy),
        "optimized": _build(o_opt, gen, feas, test, eval_cfg, acct, fitted),
        "optimizer": optimizer,
        "provenance": _provenance(gen),
    }


def _build(o, gen, feas, dataset, cfg, acct, policy):
    """Build the numbers + chart payload from a completed :class:`Outcomes` under a
    given sizing ``policy`` (``None`` = the constant-size baseline)."""
    # --- per-attempt arrays -------------------------------------------------- #
    fee = st.attributable_fee(o)
    net_payoff = o.net_payout - fee
    weeks = st.calendar_weeks(o.total_trading_days, o.trading_days_per_week)
    ret_on_fee = st.return_on_fee(o)
    ttfp = st.time_to_first_payout(o)  # weeks, only attempts that paid

    edge = gen.edge
    breakeven = gen.breakeven_win_rate

    # --- outcome breakdown --------------------------------------------------- #
    codes, counts = np.unique(o.code, return_counts=True)
    outcomes = [{"code": _CODE_NAMES.get(int(c), str(int(c))), "count": int(n)}
                for c, n in zip(codes, counts)]

    # --- reward-vs-time scatter (sampled) ------------------------------------ #
    b = o.n_attempts
    k = min(1200, b)
    rng = np.random.default_rng(int(cfg.seed))
    pick = rng.choice(b, size=k, replace=False) if k < b else np.arange(b)
    scatter = {
        "weeks": [round(float(x), 3) for x in weeks[pick]],
        "net": [round(float(x), 2) for x in net_payoff[pick]],
        "profitable": [bool(x) for x in (net_payoff[pick] > 0)],
        "reached_funded": [bool(x) for x in o.reached_funded[pick]],
    }

    # --- lifecycle timing / breach stats ------------------------------------ #
    reached = o.reached_funded
    paid = o.payouts_taken > 0
    # eval duration (weeks) for the attempts that CLEARED eval -> mean time to pass.
    eval_weeks = (st.calendar_weeks(o.eval_trading_days, o.trading_days_per_week)
                  if o.eval_trading_days is not None else None)
    mean_pass = (float(np.mean(eval_weeks[reached]))
                 if eval_weeks is not None and reached.any() else None)
    # Time to first payout FROM ZERO, including the accounts that failed first: a
    # failed attempt means buying a fresh eval and starting over, so the realistic
    # wait is a renewal (geometric-retry) expectation, not just the winners' average.
    # With p = P(an attempt reaches a payout), the number of non-paying attempts before
    # the first paying one is geometric with mean (1-p)/p; each such attempt runs to
    # termination (its full duration), then the paying attempt reaches its first payout:
    #   E[T] = mean_time_to_first_payout(paid) + (1-p)/p * mean_full_duration(not paid)
    # (exact in expectation by Wald's identity, attempts being iid).
    p_pay = float(np.mean(paid)) if b else 0.0
    if paid.any():
        t_pay = float(np.mean(st.calendar_weeks(o.first_payout_day[paid],
                                                 o.trading_days_per_week)))
        nonpaid = ~paid
        t_fail = (float(np.mean(weeks[nonpaid])) if nonpaid.any() else 0.0)
        mean_ttfp = t_pay + (1.0 - p_pay) / p_pay * t_fail
    else:
        mean_ttfp = None  # no attempt ever pays -> undefined (shown as "—")
    # lost (breach FAIL_* or wither CAPPED_OUT) with NO payout ever taken.
    lost = ((o.code >= 10) & (o.code < 20)) | (o.code == int(ExitCode.CAPPED_OUT))
    breached_before_payout = float(np.mean((o.payouts_taken == 0) & lost)) if b else None

    numbers = {
        "edge": _f(edge), "breakeven_win_rate": _f(breakeven),
        # "eval pass rate" for a two-phase account = fraction that cleared eval and
        # reached funded (the final code is never PASSED, §H3).
        "reached_funded": _f(float(np.mean(o.reached_funded))),
        "prob_profitable": _f(st.prob_profitable(o)),
        "mean_payout": _f(st.mean_payout(o)),
        "mean_net_payoff": _f(float(np.mean(net_payoff))),
        "payout_velocity_month": _f(st.payout_velocity(o)),
        "return_on_fee_year": _f(st.return_on_fee_per_year(o)),
        "r_renewal_week": _f(rn.r_renewal(o)),
        "fee_bankroll_efficiency": _f(rn.fee_bankroll_efficiency(o)),
        "median_weeks": _f(float(np.median(weeks))) if b else None,
        "mean_weeks_to_pass_eval": _f(mean_pass) if mean_pass is not None else None,
        "mean_weeks_to_first_payout": _f(mean_ttfp) if mean_ttfp is not None else None,
        "breached_before_payout": _f(breached_before_payout)
        if breached_before_payout is not None else None,
        "n_paths": b,
    }
    if feas is not None:
        # whole-attempt withering rate (eval OR funded), matching the outcome bar.
        numbers["nontradable_rate"] = _f(
            float(np.mean(o.code == int(ExitCode.CAPPED_OUT))))

    payoff_qs = st.payoff_quantiles(o).tolist()

    result = {
        "numbers": numbers,
        "outcomes": outcomes,
        "payout_count_dist": [round(float(x), 5) for x in st.payout_count_dist(o)],
        "hist_net_payoff": _hist(net_payoff),
        "hist_return_on_fee": _hist(ret_on_fee[np.isfinite(ret_on_fee)]),
        "hist_weeks": _hist(weeks),
        "hist_time_to_first_payout": _hist(ttfp) if ttfp.size else {"edges": [], "counts": []},
        "payoff_quantiles": {"q": [5, 25, 50, 75, 95],
                             "v": [round(float(x), 2) for x in payoff_qs]},
        "scatter": scatter,
        "equity_paths": _sample_equity_paths(acct, dataset, cfg, feas, policy),
    }
    return result


# --------------------------------------------------------------------------- #
# Sample equity paths (reference trace on a few eval paths)                     #
# --------------------------------------------------------------------------- #


def _sample_equity_paths(acct, dataset, cfg, feas, policy=None, n=6):
    eval_phase = next(p for p in acct.phases if p.role == "eval")
    cp = compile_phase(eval_phase)
    paths = cfg.resampler.generate(dataset.n_days, cfg.L_eval, n, int(cfg.seed) + 7)
    start = float(acct.size)
    pol = np.array([1.0]) if policy is None else np.asarray(policy, dtype=np.float64)
    out = []
    for i in range(n):
        ret, day, low = gather_days(dataset, paths[i])
        r = simulate_reference(cp, ret, day, low, cfg.size_base,
                               pol, start, trace=True, feasibility=feas)
        eq = [start] + [round(float(t["equity"]), 2) for t in r.trace]
        out.append({"equity": eq, "code": _CODE_NAMES.get(int(r.code), str(r.code))})
    return out


# --------------------------------------------------------------------------- #
# win_rate × RR surface (a coarse grid sweep of one metric)                     #
# --------------------------------------------------------------------------- #

_SURFACE_METRICS = {
    "reached_funded": lambda o: float(np.mean(o.reached_funded)),
    "prob_profitable": st.prob_profitable,
    "r_renewal_week": rn.r_renewal,
    "return_on_fee_year": st.return_on_fee_per_year,
}


def _surface(params, acct, feas):
    metric = params.get("surface_metric", "reached_funded")
    fn = _SURFACE_METRICS.get(metric, _SURFACE_METRICS["reached_funded"])
    wr_axis = [round(x, 3) for x in np.linspace(0.30, 0.65, 8)]
    rr_axis = [round(x, 2) for x in np.linspace(0.5, 3.0, 8)]
    n_days = int(params.get("n_days", 160))
    seed = int(params.get("seed", 1))
    n_paths = min(1000, int(params.get("n_paths", 4000)))
    cfg = _run_config(params, n_paths, seed)
    tpd = int(params.get("trades_per_day", 4))
    z = []
    for wr in wr_axis:
        row = []
        for rr in rr_axis:
            g = IIDGenerator(win_rate=wr, rr=rr, trades_per_day=tpd)
            ds = preprocess(g.generate(n_days=n_days, seed=seed).rows)
            o = _ENGINE.run(acct, ds, cfg, feasibility=feas)
            row.append(_f(fn(o)))
        z.append(row)
    return {"metric": metric, "win_rate": wr_axis, "rr": rr_axis, "z": z}
