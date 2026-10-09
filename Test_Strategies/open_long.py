"""Reproducible, fixed-policy 09:30 vs 18:00 long-only historical research.

Run from the repository root: python -m Test_Strategies.open_long
DuckDB queries local Parquet; no data downloads, strategy search or parallelism.
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, time, timedelta
from fractions import Fraction
from hashlib import sha256
import csv
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from propfirm_engine.execution import BacktestConfig, DollarPolicy, RiskRegime
from propfirm_engine.firms.lucidflex import replay_50k
from propfirm_engine.market_replay import Instrument, PriceSession, replay_prices
from propfirm_engine.risk import cash_risk_path, distribution


NY = ZoneInfo("America/New_York")
# symbol, full-size point value/tick, full/micro round-trip commission,
# micro point value/tick, exchange-calendar identifier.
# Commissions: official Lucid schedule, checked 2026-10-03.
# Instrument economics are explicit research inputs, not inferred from prices.
ASSETS = {
    "MES": ("ES", 50, .25, 3.50, 1, 5, .25, "CME Globex Equity"),
    "MNQ": ("NQ", 20, .25, 3.50, 1, 2, .25, "CME Globex Equity"),
    "M2K": ("RTY", 50, .10, 3.50, 1, 5, .10, "CME Globex Equity"),
    "MCL": ("CL", 1000, .01, 4, 1, 100, .01, "CMEGlobex_CL"),
    "MGC": ("GC", 100, .10, 4.60, 1.60, 10, .10, "CMEGlobex_GC"),
    "SIL": ("SI", 5000, .005, 4.60, 3.20, 1000, .005, "CMEGlobex_SI"),
    "6E": ("6E", 125000, .00005, 4.80, None, None, None, "CMEGlobex_FX"),
}
POLICY = DollarPolicy((
    RiskRegime("evaluation", "eval", 2000),
    RiskRegime("funded_build", "funded", 1000, days_to_payout=5, after_payout=False),
    RiskRegime("funded_fallback", "funded", 1500),
))
TARGETS = {"evaluation": 1500, "funded_build": 3000, "funded_fallback": 500}
SOURCES = [
    "https://support.lucidtrading.com/en/articles/11508978-approved-products-and-commissions",
    "https://support.lucidtrading.com/en/articles/11404729-allowed-trading-times",
    "https://support.lucidtrading.com/en/articles/12945796-lucidflex-payouts",
]


def load_sessions(root, product, *, execution_context=False):
    """Paired complete-case sample, common to both entry-time experiments.

    Calendar-date windows deliberately do not trust vendor trade_date around
    holidays. Only the supplied volume-roll/raw-price layer is used. Known bad
    days and absent entry/final bars exclude BOTH timings before any policy run.
    """
    path = root / "continuous" / "roll_rule=volume" / f"product={product}" / "data.parquet"
    with duckdb.connect() as con:
        con.execute("SET threads=1")
        con.execute("SET TimeZone='UTC'")
        bounds = con.execute("SELECT min(ts_event), max(ts_event), count(*) FROM read_parquet(?)", [str(path)]).fetchone()
        schedule = mcal.get_calendar(ASSETS[product][-1]).schedule(bounds[0].date(), bounds[1].date())
        rows = []
        for day, row in schedule.iterrows():
            d = day.date()
            pm = datetime.combine(d - timedelta(days=1), time(18), NY)
            am = datetime.combine(d, time(9, 30), NY)
            cutoff = datetime.combine(d, time(16, 45), NY)
            exchange_close = row.market_close.to_pydatetime()
            if exchange_close <= cutoff:
                cutoff = exchange_close - timedelta(minutes=1)
            if row.market_open > pm or cutoff <= am:
                continue
            rows.append((d, pm, am, cutoff))
        calendar = pd.DataFrame(rows, columns=["day", "entry_pm", "entry_am", "cutoff"])
        for name in ("entry_pm", "entry_am", "cutoff"):
            calendar[name] = pd.to_datetime(calendar[name], utc=True)
        con.register("calendar", calendar)
        con.execute("""CREATE TEMP TABLE bars AS
            SELECT c.day, epoch_ns(p.ts_event) AS ts, p.instrument_id,
                   p.open, p.high, p.low, p.close, p.degraded_day
            FROM read_parquet(?) p JOIN calendar c
              ON p.ts_event >= c.entry_pm AND p.ts_event < c.cutoff
            ORDER BY p.ts_event""", [str(path)])
        quality = con.execute("""SELECT c.day, c.entry_pm, c.entry_am, c.cutoff,
            count(b.ts) AS bars, count(DISTINCT b.ts) AS unique_bars,
            count(DISTINCT instrument_id) AS contracts,
            coalesce(bool_or(degraded_day), false) AS degraded,
            count(*) FILTER (WHERE ts = epoch_ns(entry_pm)) > 0 AS has_pm,
            count(*) FILTER (WHERE ts = epoch_ns(entry_am)) > 0 AS has_am,
            count(*) FILTER (WHERE ts = epoch_ns(cutoff) - 60000000000) > 0 AS has_close
            FROM calendar c LEFT JOIN bars b USING(day)
            GROUP BY ALL ORDER BY day""").fetchdf()
        reasons = Counter()
        eligible = []
        for row in quality.itertuples(index=False):
            bad = []
            if row.bars != row.unique_bars: bad.append("duplicate_minutes")
            if row.contracts != 1: bad.append("not_one_contract")
            if row.degraded: bad.append("degraded_session")
            if not row.has_pm: bad.append("missing_1800")
            if not row.has_am: bad.append("missing_0930")
            if not row.has_close: bad.append("missing_final_minute")
            reasons.update(bad)
            if not bad: eligible.append(row.day)
        eligible_frame = pd.DataFrame({"day": eligible})
        con.register("eligible", eligible_frame)
        data = con.execute("SELECT b.* EXCLUDE(degraded_day) FROM bars b JOIN eligible e USING(day) ORDER BY ts").fetchnumpy()
    sessions = {"09:30": [], "18:00": []}
    if not eligible:
        raise ValueError(f"{product}: no paired complete sessions")
    days, starts = np.unique(data["day"], return_index=True)
    ends = np.r_[starts[1:], len(data["ts"])]
    closes = {r.day.date(): r.cutoff.to_pydatetime() for r in quality.itertuples()}
    previous_t, previous_p = None, None
    for day, start, end in zip(days, starts, ends):
        d = pd.Timestamp(day).date()
        t = data["ts"][start:end]
        p = np.column_stack([data[k][start:end] for k in ("open", "high", "low", "close")])
        morning = round(datetime.combine(d, time(9, 30), NY).timestamp() * 1e9)
        offset = int(np.searchsorted(t, morning))
        evening_context = {"warmup_timestamps": previous_t, "warmup_ohlc": previous_p} if execution_context else {}
        morning_context = {"warmup_timestamps": t[max(0, offset-60):offset],
                           "warmup_ohlc": p[max(0, offset-60):offset]} if execution_context and offset else {}
        sessions["18:00"].append(PriceSession(d, closes[d], t, p, **evening_context))
        sessions["09:30"].append(PriceSession(d, closes[d], t[offset:], p[offset:], **morning_context))
        previous_t, previous_p = t[-60:], p[-60:]
    return sessions, {
        "path": str(path), "source_rows": bounds[2], "first_bar": str(bounds[0]), "last_bar": str(bounds[1]),
        "calendar_candidates": len(quality), "paired_eligible_sessions": len(eligible),
        "excluded_sessions": len(quality) - len(eligible), "exclusion_reason_counts_overlap": dict(reasons),
        "first_session": str(sessions["18:00"][0].session), "last_session": str(sessions["18:00"][-1].session),
    }


def summarize(path):
    result, decisions = path.replay, path.decisions
    summary = asdict(cash_risk_path(result))
    counts = Counter(e.kind for e in result.events)
    evaluation_failures = sum(e.kind == "failure" and e.phase == "eval" for e in result.events)
    resolved_evals = counts["evaluation_pass"] + evaluation_failures
    pnl = [d.net_pnl for d in decisions]
    yearly = defaultdict(float)
    for e in result.events:
        yearly[str(e.at.astimezone(NY).year)] += e.cash
    summary.update({
        "start": str(result.start), "end": str(result.end), "status": result.status,
        "trades": len(decisions), "wins": sum(x > 0 for x in pnl),
        "win_rate": sum(x > 0 for x in pnl) / len(pnl) if pnl else None,
        "evaluation_passes": counts["evaluation_pass"], "resolved_evaluations": resolved_evals,
        "resolved_eval_pass_rate": counts["evaluation_pass"] / resolved_evals if resolved_evals else None,
        "live_handoffs": counts["live_handoff"], "exit_counts": dict(Counter(d.reason for d in decisions)),
        "collisions": sum(d.collision for d in decisions),
        "collision_rate": sum(d.collision for d in decisions) / len(decisions) if decisions else 0,
        "trade_net_pnl_distribution": distribution(pnl),
        "holding_hours_distribution": distribution((d.exit_at - d.entry_at).total_seconds() / 3600 for d in decisions),
        "yearly_external_net_cash": dict(yearly),
        "history_fingerprint": result.history_fingerprint,
    })
    return summary


def write_records(path, records):
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_variant(product, variant, sessions, out, approval_hours=0, *, posthoc=False):
    full, pv, tick, full_cost, micro_cost, micro_pv, micro_tick, _ = ASSETS[product]
    micro = variant == "commission_equivalent_micro"
    quantity = int(Fraction(str(full_cost)) // Fraction(str(micro_cost))) if micro else 1
    instrument = Instrument(product if micro else full, micro_pv if micro else pv, micro_tick if micro else tick)
    config = BacktestConfig(micro_cost if micro else full_cost, timedelta(hours=approval_hours), timedelta(0), timedelta(0))
    spec = replay_50k(eval_fee=105.20, reset_fee=105, contract_type="micro" if micro else "mini")
    records = []
    for entry_time, history in sessions.items():
        split = int(len(history) * .7)
        for partition, subset in (("full", history), ("IS", history[:split]), ("OOS", history[split:])):
            path = replay_prices(spec, subset, POLICY, TARGETS, instrument, config,
                                 quantity=quantity, collision_policy="stop_first")
            row = {"asset": product, "execution_symbol": instrument.symbol, "variant": variant,
                   "evaluation_label": "posthoc_sensitivity_not_untouched_OOS" if posthoc else "fixed_policy",
                   "quantity": quantity, "position_round_trip_commission": quantity * config.cost_per_contract,
                   "entry_time": entry_time, "partition": partition, "sessions": len(subset),
                   "first_session": str(subset[0].session), "last_session": str(subset[-1].session), **summarize(path)}
            records.append(row)
            prefix = f"{product}_{variant}_{entry_time.replace(':', '')}_{partition}"
            write_records(out / f"{prefix}_trades.csv", [asdict(d) for d in path.decisions])
            write_records(out / f"{prefix}_events.csv", [asdict(e) for e in path.replay.events])
            print(f"{product:3} {variant:27} {entry_time} {partition:4} cash={row['net_cash']:10.2f} "
                  f"cash/day={row['net_cash_per_day']:7.2f} trades={row['trades']:5} collisions={row['collision_rate']:.2%}", flush=True)
    return records


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("Data"))
    ap.add_argument("--output", type=Path, default=Path("results/open_long"))
    ap.add_argument("--assets", nargs="+", choices=sorted(ASSETS), default=list(ASSETS))
    ap.add_argument("--approval-hours", type=float, default=0, help="Scenario elapsed hours; not an official processing guarantee")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows, quality, selections = [], {}, {}
    for product in args.assets:
        sessions, quality[product] = load_sessions(args.data, product)
        print(json.dumps({"asset": product, "data_quality": quality[product]}), flush=True)
        baseline = run_variant(product, "full_size", sessions, args.output, args.approval_hours)
        all_rows.extend(baseline)
        triggered = [r["entry_time"] for r in baseline if r["partition"] == "IS"
                     and r["collisions"] >= 5 and r["collision_rate"] >= .01]
        late_triggered = [r["entry_time"] for r in baseline if r["partition"] == "OOS"
                          and r["collisions"] >= 5 and r["collision_rate"] >= .01]
        posthoc = not triggered and bool(late_triggered)
        selections[product] = {"IS_triggered_entry_times": triggered,
                               "late_collision_warning_entry_times": late_triggered,
                               "micro_is_posthoc_sensitivity": posthoc,
                               "micro_available": ASSETS[product][4] is not None}
        # If either entry-time experiment triggers, run BOTH micro timings on
        # identical dates. Never select the variant using OOS profit or collisions.
        if (triggered or late_triggered) and ASSETS[product][4] is not None:
            all_rows.extend(run_variant(product, "commission_equivalent_micro", sessions, args.output,
                                        args.approval_hours, posthoc=posthoc))
    report = {
        "policy": {"risk": [asdict(r) for r in POLICY.regimes], "targets": TARGETS},
        "sources": SOURCES, "data_quality": quality, "micro_selection": selections,
        "versions": {"duckdb": duckdb.__version__, "pandas_market_calendars": mcal.__version__},
        "implementation_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).parents[1] / "src/propfirm_engine/market_replay.py",
            Path(__file__).parents[1] / "src/propfirm_engine/backtest.py")},
        "assumptions": [
            "Fixed policy, no historical optimization. IS is the first 70% of eligible sessions; OOS is the final 30% with fresh account/wallet.",
            "Full history is a separate uninterrupted replay; its cash is not the sum of fresh IS/OOS runs.",
            "Matched complete-case dates for both entry times; filtering on session availability can introduce selection bias, especially in thin micro histories.",
            "Raw prior-session-volume continuous prices; micro data is only a full-size execution proxy.",
            "Long at exact 09:30 or previous-calendar-day 18:00 New York; one fixed-size position per session.",
            "User-selected convention: every first-touch bar spanning both levels is a stop, even if its opening suggests target-first; stop gaps still receive the worse opening fill.",
            "Market sells rounded down to ticks; buy entry rounded up; no spread or additional slippage beyond observed gaps/tick rounding.",
            "Force flat at 16:45 New York, or one minute before the exchange calendar's earlier close.",
            "Current Lucid rules and commissions applied retrospectively, not historical firm offerings or guaranteed future fills.",
            f"Approval delay={args.approval_hours} elapsed hours; receipt and activation immediate. All payouts assumed approved; pending accounts do not trade.",
            "Unlimited external funding for performance. Required bankroll is the observed path deficit, NOT capital for 5% lifetime ruin.",
            "Micro trigger: IS collisions >=5 and >=1% of executed trades in either timing; both alternatives shown without OOS selection.",
            "When only late-period collisions trigger a micro run, that entire variant is labelled posthoc sensitivity, not untouched OOS. Original full-size OOS stays unchanged.",
            "Within each asset dates match; different assets have different coverage and cannot be ranked by total cash alone.",
        ],
        "results": all_rows,
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    simple = [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in all_rows]
    write_records(args.output / "summary.csv", simple)
    lines = ["# Fixed-policy long-only price replay", "", "## Fresh-account OOS results", "",
             "Micro rows marked exploratory were added after inspecting late-period collisions; they are not untouched OOS.", "",
             "| Asset | Variant | Entry NY | Sessions | Net external cash | Cash/calendar day | Payouts | Collisions |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in all_rows:
        if r["partition"] == "OOS":
            label = " (exploratory)" if r["evaluation_label"].startswith("posthoc") else ""
            lines.append(f"| {r['asset']} | {r['quantity']} {r['execution_symbol']}{label} | {r['entry_time']} | {r['sessions']} | "
                         f"${r['net_cash']:,.2f} | ${r['net_cash_per_day']:,.2f} | {r['payout_count']} | {r['collision_rate']:.2%} |")
    lines += ["", "## Coverage", "", "All last sessions are 2026-08-03 for the supplied data; inspect the recorded dates for other inputs.", "",
              "| Asset | First eligible session | OOS first session | Eligible sessions | Excluded / candidate sessions |",
              "|---|---|---|---:|---:|"]
    for product, q in quality.items():
        oos = next(r for r in all_rows if r["asset"] == product and r["partition"] == "OOS")
        lines.append(f"| {product} | {q['first_session']} | {oos['first_session']} | {q['paired_eligible_sessions']} | "
                     f"{q['excluded_sessions']} / {q['calendar_candidates']} |")
    lines += ["", "## Uninterrupted full-history results", "",
              "| Asset | Variant | Entry NY | Net external cash | Cash/calendar day |",
              "|---|---|---|---:|---:|"]
    for r in all_rows:
        if r["partition"] == "full":
            label = " (exploratory)" if r["evaluation_label"].startswith("posthoc") else ""
            lines.append(f"| {r['asset']} | {r['quantity']} {r['execution_symbol']}{label} | {r['entry_time']} | "
                         f"${r['net_cash']:,.2f} | ${r['net_cash_per_day']:,.2f} |")
    lines += ["", "## Interpretation limits", ""] + ["- " + a for a in report["assumptions"]]
    lines += ["", "## Sources", ""] + ["- " + s for s in SOURCES]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Results: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
