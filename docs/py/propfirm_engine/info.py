"""Readable configuration reports, without running or changing an account."""
from dataclasses import fields
from enum import Enum

from .execution import BacktestConfig, LifecycleSpec
from .model import Account
from .rules import RuleKind


def _value(value):
    if isinstance(value, Enum):
        return value.name
    if value is None:
        return "not set"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        text = format(value, ",")
        return text[:-2] if text.endswith(".0") else text
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_value(item) for item in value) + "]"
    return str(value)


def _fields(obj, *, exclude=()):
    return [f"  {f.name.replace('_', ' ')}: {_value(getattr(obj, f.name))}"
            for f in fields(obj) if f.name not in exclude]


def _rule(rule, dated):
    compiled = rule.compile()
    names = {
        RuleKind.PROFIT_TARGET: "Profit target",
        RuleKind.TRAILING_DD: "Trailing drawdown / MLL",
        RuleKind.STATIC_DD: "Static drawdown",
        RuleKind.DAILY_LOSS: "Daily loss limit",
        RuleKind.MIN_DAYS: "Minimum trading days",
        RuleKind.MIN_WINNING_DAYS: "Winning days",
        RuleKind.CONSISTENCY_GATE: "Consistency gate",
        RuleKind.CONSISTENCY_ADJUST: "Consistency target adjustment",
    }
    meanings = {
        RuleKind.PROFIT_TARGET: "Profit >= target; dated replay requires closed balance and a flat book." if dated
            else "Profit >= target; observation fidelity depends on the executor.",
        RuleKind.TRAILING_DD: "Breach at equity <= floor. Update timing ratchets the floor; check timing detects breach.",
        RuleKind.STATIC_DD: "Breach at equity <= opening balance minus amount; a later request lock may raise the floor.",
        RuleKind.DAILY_LOSS: "Breach at day P&L <= -amount; HARD ends the account, SOFT suspends the session.",
        RuleKind.MIN_DAYS: "All pass gates must hold; too few days withholds passing, not a breach.",
        RuleKind.MIN_WINNING_DAYS: "Count sessions with closed net daily profit >= threshold; days need not be consecutive.",
        RuleKind.CONSISTENCY_GATE: "Largest net day <= threshold * cycle profit + cushion; gates eligibility, not a breach.",
        RuleKind.CONSISTENCY_ADJUST: "If largest day > threshold * total profit, raise target to at least raise_to.",
    }
    label = names.get(compiled.kind, type(rule).__name__)
    params = "; ".join(f"{f.name}={_value(getattr(rule, f.name))}" for f in fields(rule))
    lines = [f"  {label}: {params}", f"    Action: {_value(compiled.action)}. " +
             meanings.get(compiled.kind, "Custom rule; inspect its implementation and executor support.")]
    if compiled.kind not in names:
        lines.extend("  " + line for line in _fields(compiled))
    return lines


def _report(spec, config=None, *, title=None, sources=(), unknown_fees=(), notes=()):
    if not isinstance(spec, (Account, LifecycleSpec)):
        raise TypeError("info requires an Account or LifecycleSpec")
    if config is not None and not isinstance(config, BacktestConfig):
        raise TypeError("config must be BacktestConfig or None")
    dated = isinstance(spec, LifecycleSpec)
    account = spec.account if dated else spec
    lines = [title or f"Account: {account.name}",
             "Configured model, not a live rule lookup or a guarantee of firm compliance.",
             f"Amounts in {account.currency}; account size {_value(account.size)}.", "", "Account fees"]
    for name in ("eval_fee", "activation_fee"):
        value = "UNSPECIFIED (not free)" if name in unknown_fees else _value(getattr(account, name))
        lines.append(f"  {name.replace('_', ' ')}: {value}")
    for index, phase in enumerate(account.phases, 1):
        opening = account.size if phase.start_equity is None else phase.start_equity
        lines.extend(["", f"Phase {index}: {phase.name} ({phase.role})",
                      f"  Opening balance: {_value(opening)}" +
                      (" (nominal fallback; a summary run can override it)" if not dated and phase.start_equity is None else "")])
        kinds = {r.compile().kind for r in phase.rules}
        if RuleKind.DAILY_LOSS not in kinds:
            lines.append("  Daily loss limit: not configured")
        if RuleKind.MIN_DAYS not in kinds:
            lines.append("  Separate minimum trading days: not configured")
        if not kinds.intersection((RuleKind.CONSISTENCY_GATE, RuleKind.CONSISTENCY_ADJUST)):
            lines.append("  Consistency: not configured")
        for rule in phase.rules:
            lines.extend(_rule(rule, dated))
            compiled = rule.compile()
            if compiled.kind in (RuleKind.TRAILING_DD, RuleKind.STATIC_DD):
                lines.append(f"    Initial floor: {_value(opening - compiled.p0)}")
            if compiled.kind == RuleKind.PROFIT_TARGET:
                lines.append(f"    Initial target balance: {_value(opening + compiled.p0)}")
        if phase.payout_schema is None:
            lines.append("  Payouts: not configured for this phase")
        else:
            lines.append("  Payout schema (gross request limits, before trader split)")
            lines.extend("  " + line for line in _fields(phase.payout_schema))
            lines.append(f"    Trader share: {phase.payout_schema.split * 100:g}% of gross (after any first-tier split).")
            lines.extend([
                "    Maximum gross request = min(dollar cap, fraction * basis profit, balance - buffer floor).",
                "    Gross must be positive and >= min request; cycle profit must separately meet min cycle profit.",
                "    Retained-profit basis uses balance - profit reference balance (phase opening if not set).",
                "    Dollar caps follow payout order; the last cap repeats. Split is the trader's share.",
                "    Buffer floor is an absolute non-withdrawable balance, not a required profit amount.",
                "    Cycle-profit eligibility is distinct from retained profit; withdrawals reset the cycle basis.",
            ])
    lines.extend(["", "Lifecycle and limits"])
    if dated:
        for line in _fields(spec, exclude=("account", "assumptions")):
            if line.startswith("  reset fee:") and "reset_fee" in unknown_fees:
                line = "  reset fee: UNSPECIFIED (not free)"
            lines.append(line)
        lines.extend([
            "  Tier pairs are (closed profit above phase opening, lower bound inclusive; contract-unit limit).",
            "  Scaling uses closed profit; payout reductions can lower the limit at approval.",
            "  Weekdays label session close dates (0=Monday); opening is the previous local date.",
            "  When enabled, inactivity renews on a trade with absolute net P&L >= activity threshold.",
            "  An inactivity close uses local date + calendar days; not set means exact elapsed days.",
            "  Qualifying closes win exact inactivity ties. Holidays/early closes need supplied session overrides.",
            "  Default payout policy requests the maximum at an eligible flat session close and waits for approval.",
            "  Request lock floor, when set, locks drawdown at request time; it is not a payout buffer.",
            "  Default decisions approve requests; net cash arrives on receipt, not approval.",
            "  With withdraw reduces equity=yes, gross is deducted from account balance at approval.",
            "  Maximum approved payouts end simulated funding; a fresh evaluation can start next session.",
            "  Failed evals use eligible resets; expired resets and funded failures use a new evaluation.",
            "  Fees require available wallet cash. Retries honor retry delay and the next-session restriction.",
        ])
    else:
        lines.append("  Not supplied. An Account alone does not define dated payouts, scaling, hours or inactivity.")
    lines.extend(["", "Runtime scenario (not firm rules)"])
    if config is None:
        lines.append("  Not supplied: trading costs, payment fees, delays and wallet are unspecified.")
    else:
        lines.extend(_fields(config))
        lines.append("  Initial wallet not set means unrestricted funding, not zero capital.")
    lines.extend([
        "  Recorded fill prices/fees, valuation marks, calendars and payout overrides are separate inputs.",
        "  Recorded replay uses supplied fill fees; bracket replay uses configured trading costs.",
        "  BacktestConfig trading costs apply to bracket logs, not recorded fills.",
        "  CONTINUOUS checks only supplied observations; sparse data cannot certify an unseen price path.",
        "  This report does not validate executor support. Use Engine.check_replay for dated preflight.",
        "", "Assumptions and limitations",
    ])
    assumptions = spec.assumptions if dated else ()
    lines.extend(f"  - {note}" for note in (*assumptions, *notes))
    if not assumptions and not notes:
        lines.append("  No profile-specific assumptions supplied; that is not evidence of complete fidelity.")
    if sources:
        lines.extend(["", "Official source references (not fetched by info)"])
        lines.extend(f"  {name}: {url}" for name, url in sources)
    return "\n".join(lines)


def _emit(text, print_output):
    if type(print_output) is not bool:
        raise ValueError("print_output must be bool")
    if print_output:
        print(text)
    return text


def account_info(spec, config=None, *, print_output=True):
    """Print and return all configured phases, rules, payouts and lifecycle fields."""
    return _emit(_report(spec, config), print_output)
