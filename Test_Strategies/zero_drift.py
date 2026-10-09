"""Matched conditional sign-randomization control for the fixed long-only policy.

Run: python -m Test_Strategies.zero_drift --paths 500
Uses the canonical price replay, not an alternative account simulator.
"""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import timedelta
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from propfirm_engine.execution import BacktestConfig
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.market_replay import Instrument, PriceSession, replay_prices
from propfirm_engine.risk import distribution
from Test_Strategies.open_long import ASSETS, POLICY, TARGETS, load_sessions, summarize, write_records


SCALE = 1_000_000


def innovations(session):
    """Integer price units prevent cumulative floating error altering tick fills."""
    scaled = session.ohlc * SCALE
    rounded = np.rint(scaled)
    if np.any(np.abs(scaled - rounded) > 1e-4) or np.max(np.abs(rounded)) > 1e12:
        raise ValueError("control requires prices representable to six decimal places within safe bounds")
    prices = rounded.astype(np.int64)
    previous = np.r_[prices[0, 0], prices[:-1, 3]]
    return int(prices[0, 0]), prices - previous[:, None]


def reflect(anchor, changes, signs):
    """Mirror whole OHLC innovations, then reconstruct a continuous session.

    Fair independent signs make each close-to-close innovation mean zero.
    This is NOT a continuous-time martingale or an intrabar-order model.
    Sessions re-anchor at their observed opening: there are no overnight holds.
    """
    signs = np.asarray(signs)
    if signs.shape != (len(changes),) or not np.isin(signs, (-1, 1)).all():
        raise ValueError("one -1/+1 sign required per minute")
    signed = changes * signs[:, None]
    previous = anchor + np.r_[0, np.cumsum(signed[:-1, 3], dtype=np.int64)]
    result = signed + previous[:, None]
    negative = signs < 0
    result[negative, 1] = previous[negative] - changes[negative, 2]
    result[negative, 2] = previous[negative] - changes[negative, 1]
    return result.astype(float) / SCALE


def prepare(sessions):
    result = []
    for evening, morning in zip(sessions["18:00"], sessions["09:30"]):
        if evening.session != morning.session:
            raise ValueError("entry-time histories must have identical sessions")
        offset = int(np.searchsorted(evening.timestamps, morning.timestamps[0]))
        if not np.array_equal(evening.timestamps[offset:], morning.timestamps):
            raise ValueError("morning history must be a suffix of the evening history")
        anchor, changes = innovations(evening)
        result.append((evening, offset, anchor, changes))
    if len(result) != len(sessions["18:00"]) or len(result) != len(sessions["09:30"]):
        raise ValueError("entry-time histories must have equal lengths")
    return result


def random_sessions(prepared, seed, product, path):
    """Stable per-session streams, independent of actual trades and skipped days."""
    histories = {"09:30": [], "18:00": []}
    asset_key = int.from_bytes(sha256(product.encode()).digest()[:4], "little")
    for session, offset, anchor, changes in prepared:
        rng = np.random.default_rng(np.random.SeedSequence([seed, asset_key, path, session.session.toordinal()]))
        signs = rng.integers(0, 2, len(changes), dtype=np.int8) * 2 - 1
        prices = reflect(anchor, changes, signs)
        histories["18:00"].append(PriceSession(session.session, session.close_at, session.timestamps, prices))
        histories["09:30"].append(PriceSession(session.session, session.close_at, session.timestamps[offset:], prices[offset:]))
    return histories


def execute(product, variant, history, approval_hours):
    full, pv, tick, full_cost, micro_cost, micro_pv, micro_tick, _ = ASSETS[product]
    micro = variant == "commission_equivalent_micro"
    quantity = int(Fraction(str(full_cost)) // Fraction(str(micro_cost))) if micro else 1
    instrument = Instrument(product if micro else full, micro_pv if micro else pv, micro_tick if micro else tick)
    config = BacktestConfig(micro_cost if micro else full_cost, timedelta(hours=approval_hours), timedelta(0), timedelta(0))
    return replay_prices(replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro" if micro else "mini"),
                         history, POLICY, TARGETS, instrument, config, quantity=quantity, collision_policy="stop_first")


def compare(observed, samples):
    """Descriptive null distribution, plus a finite-simulation upper-tail rank."""
    values = np.asarray(samples, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("finite nonempty control sample required")
    tail = int(np.count_nonzero(values >= observed))
    return {"observed": observed, "control": distribution(values.tolist()),
            "observed_minus_control_mean": float(observed - values.mean()),
            "control_mean_mc_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None,
            "upper_tail_exceedances": tail, "upper_tail_rank": (tail + 1) / (len(values) + 1)}


def holm(pvalues):
    """Holm multiplicity correction, valid without independence across tests."""
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    result, previous = [0.] * len(pvalues), 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(pvalues) - rank) * pvalues[index]))
        result[index] = previous
    return result


METRICS = ("net_cash", "net_cash_per_day", "receipts", "fees", "attempts", "failed_attempts",
           "payout_count", "max_cash_drawdown", "required_bankroll", "trades", "evaluation_passes",
           "collision_rate", "live_handoffs")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("Data"))
    parser.add_argument("--output", type=Path, default=Path("results/zero_drift"))
    parser.add_argument("--assets", nargs="+", choices=sorted(ASSETS), default=list(ASSETS))
    parser.add_argument("--paths", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--approval-hours", type=float, default=0)
    args = parser.parse_args()
    if args.paths < 2 or args.seed < 0 or args.approval_hours < 0 or not np.isfinite(args.approval_hours):
        parser.error("paths >= 2, nonnegative seed and finite nonnegative approval hours required")
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons, quality = [], {}
    for product in dict.fromkeys(args.assets):
        started = perf_counter()
        sessions, quality[product] = load_sessions(args.data, product)
        split = int(len(sessions["18:00"]) * .7)
        oos = {entry: history[split:] for entry, history in sessions.items()}
        prepared = prepare(oos)
        # These variants were already inspected historically; not selected by null results.
        variants = ["full_size"] + (["commission_equivalent_micro"] if product in ("MGC", "SIL") else [])
        historical = {(variant, entry): summarize(execute(product, variant, history, args.approval_hours))
                      for variant in variants for entry, history in oos.items()}
        samples = {key: [] for key in historical}
        print(f"{product}: {len(prepared)} OOS sessions, {args.paths} paths, {len(samples)} variants/times", flush=True)
        for index in range(args.paths):
            synthetic = random_sessions(prepared, args.seed, product, index)
            for variant, entry in samples:
                stats = summarize(execute(product, variant, synthetic[entry], args.approval_hours))
                samples[variant, entry].append({"path": index, **{k: stats[k] for k in METRICS}})
            if (index + 1) % 25 == 0 or index == 0:
                elapsed = perf_counter() - started
                print(f"{product}: {index + 1}/{args.paths}, {elapsed:.1f}s elapsed", flush=True)
        for (variant, entry), rows in samples.items():
            observed = historical[variant, entry]
            row = {"asset": product, "execution_symbol": product if variant != "full_size" else ASSETS[product][0],
                   "variant": variant, "entry_time": entry, "exploratory": variant != "full_size",
                   "sessions": len(prepared), "first_session": str(oos[entry][0].session),
                   "last_session": str(oos[entry][-1].session), "historical": observed,
                   "metrics": {k: compare(observed[k], [r[k] for r in rows]) for k in METRICS}}
            comparisons.append(row)
            write_records(args.output / f"{product}_{variant}_{entry.replace(':', '')}_paths.csv", rows)
        # Per-asset checkpoint: completed assets remain inspectable during longer runs.
        (args.output / f"{product}.json").write_text(json.dumps(comparisons[-len(samples):], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    primary = [r for r in comparisons if not r["exploratory"]]
    adjusted = holm([r["metrics"]["net_cash_per_day"]["upper_tail_rank"] for r in primary])
    for row, value in zip(primary, adjusted):
        row["cash_per_day_upper_tail_holm"] = value
    report = {"model": "conditional symmetric minute-innovation control", "paths": args.paths, "seed": args.seed,
              "approval_hours": args.approval_hours, "policy": {"risk": [asdict(r) for r in POLICY.regimes], "targets": TARGETS},
              "data_quality": quality, "primary_comparisons": len(primary), "results": comparisons,
              "implementation_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in (
                  Path(__file__), Path(__file__).with_name("open_long.py"), Path("src/propfirm_engine/market_replay.py"),
                  Path("src/propfirm_engine/backtest.py"), Path("src/propfirm_engine/engine.py"))},
              "assumptions": [
                  "Independent fair reflections of each observed minute's entire OHLC innovation relative to its preceding close; first previous close is session opening.",
                  "Preserves timestamps, each bar range, absolute body/gap/close innovation, missing-minute pattern and volatility clustering; destroys directional dependence as well as drift.",
                  "Zero expected close-to-close changes, not a continuous-time martingale. Retained intrabar shape and stop-first approximation can affect fills.",
                  "Each session re-anchors at its observed open; all positions close within session. Integer micro-price units avoid accumulated floating tick-rounding error.",
                  "Both entry times and execution sizes share the same randomized price tape per asset/path; account states are independent fresh OOS replays.",
                  "No optimization. Fixed previously specified policy, same final 30% eligible sessions, canonical account engine and execution rules as historical runs.",
                  "Current frozen Lucid commissions, evaluation 105.20/reset 105, all payouts approved, receipt and activation immediate; approval delay is the configured scenario.",
                  "Conditional on observed OOS volatility and data eligibility; this is a retrospective benchmark, NOT a forecast or proof of exploitable drift.",
                  "Upper-tail ranks include ties with (exceedances+1)/(paths+1). Holm correction covers full-size cash/day comparisons only; other metrics are descriptive.",
                  "Micro variants are exploratory because late-period collisions motivated inspecting them. No significance or selection claims for these alternatives.",
                  "Required bankroll is each path's maximum funding deficit over this horizon, NOT capital for a specified lifetime gambler's ruin probability.",
              ]}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Matched zero-drift control", "", f"{args.paths} conditional simulations per asset; seed {args.seed}. Fixed policy; fresh OOS accounts.", "",
             "All money/day values use calendar days. P5–P95 describes simulated outcomes, not a confidence interval for the mean.", "",
             "| Asset | Variant | Entry NY | Historical/day | Control mean/day | Control SD/day | Control P5–P95/day | Upper-tail rank | Holm |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    flat = []
    for row in comparisons:
        metric = row["metrics"]["net_cash_per_day"]
        dist = metric["control"]
        flat.append({"asset": row["asset"], "variant": row["variant"], "entry_time": row["entry_time"],
                     "historical_per_day": metric["observed"], **{f"control_{k}": v for k, v in dist.items() if not isinstance(v, dict)},
                     "difference_per_day": metric["observed_minus_control_mean"], "mean_mc_se": metric["control_mean_mc_se"],
                     "upper_tail_rank": metric["upper_tail_rank"], "holm": row.get("cash_per_day_upper_tail_holm")})
        # Distribution field names are shared with the engine risk report.
        lines.append(f"| {row['execution_symbol']} | {'exploratory micro' if row['exploratory'] else 'full-size'} | {row['entry_time']} | "
                     f"{metric['observed']:.2f} | {dist['mean']:.2f} | {dist['standard_deviation']:.2f} | "
                     f"{dist['percentiles']['0.05']:.2f}–{dist['percentiles']['0.95']:.2f} | {metric['upper_tail_rank']:.4f} | "
                     f"{row.get('cash_per_day_upper_tail_holm', '—')} |")
    lines += ["", "## Interpretation limits", ""] + ["- " + a for a in report["assumptions"]]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_records(args.output / "summary.csv", flat)
    print(f"Results: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
