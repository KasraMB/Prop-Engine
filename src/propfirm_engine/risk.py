"""Cash-based risk reporting for complete lifecycle paths, not trading balances."""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from math import ceil, floor, fsum, isfinite, log, log1p
from numbers import Real
from statistics import NormalDist

import numpy as np

from .statistics import wilson_ci
from .uncertainty import _from_distributions


def _finite(value, name, low=0, high=float("inf")):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")


@dataclass(frozen=True)
class RiskConfig:
    bankroll: float | None = None
    target_ruin_probability: float = 0.01
    confidence: float = 0.95
    tail_probability: float = 0.05
    percentiles: tuple[float, ...] = (.01, .05, .10, .25, .50, .75, .90, .95, .99)

    def __post_init__(self):
        if self.bankroll is not None:
            _finite(self.bankroll, "bankroll")
            if Fraction(str(self.bankroll)) * 100 % 1:
                raise ValueError("bankroll must be expressed in whole cents")
        _finite(self.target_ruin_probability, "target_ruin_probability", 0, 1)
        _finite(self.confidence, "confidence", 0, 1)
        _finite(self.tail_probability, "tail_probability", 0, 1)
        if not 0 < self.confidence < 1 or not 0 < self.tail_probability <= 1:
            raise ValueError("confidence must be in (0,1) and tail_probability in (0,1]")
        object.__setattr__(self, "percentiles", tuple(self.percentiles))
        if not self.percentiles or len(set(self.percentiles)) != len(self.percentiles):
            raise ValueError("percentiles must be nonempty and unique")
        for q in self.percentiles:
            _finite(q, "percentile", 0, 1)


def distribution(values, percentiles=RiskConfig().percentiles):
    """Descriptive population variance (ddof=0); sample variance is separate."""
    a = np.asarray(tuple(values), dtype=float)
    if a.ndim != 1 or not np.all(np.isfinite(a)):
        raise ValueError("distribution requires finite scalar values")
    if not len(a):
        return None
    try:
        with np.errstate(over="raise", invalid="raise"):
            percentiles = tuple(percentiles)
            quantiles = np.quantile(a, percentiles)
            return {"count": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
                    "minimum": float(a.min()), "maximum": float(a.max()),
                    "variance": float(a.var()), "standard_deviation": float(a.std()),
                    "sample_variance": float(a.var(ddof=1)) if len(a) > 1 else None,
                    "percentiles": {str(float(q)): float(v) for q, v in zip(percentiles, quantiles)},
                    "percentile_method": "linear interpolation; not confidence limits"}
    except FloatingPointError as exc:
        raise ValueError("distribution exceeds the finite numeric reporting range") from exc


def _cash_extremes(result):
    cash = peak = drawdown = deficit = Fraction(0)
    underwater = None
    longest = 0.0
    for event in result.events:
        cash += Fraction(str(event.cash))
        deficit = max(deficit, -cash)
        peak = max(peak, cash)
        drawdown = max(drawdown, peak - cash)
        if cash < peak and underwater is None:
            underwater = event.at
        if cash >= peak and underwater is not None:
            longest = max(longest, (event.at - underwater).total_seconds() / 86400)
            underwater = None
    if underwater is not None:
        longest = max(longest, (result.end - underwater).total_seconds() / 86400)
    return float(drawdown), longest, ceil(deficit * 100) / 100


@dataclass(frozen=True)
class CashRiskPath:
    net_cash: float
    net_cash_per_day: float
    calendar_days: float
    receipts: float
    fees: float
    outstanding_payouts: float
    attempts: int
    failed_attempts: int
    payout_count: int
    max_cash_drawdown: float
    longest_underwater_days: float
    days_to_first_receipt: float | None
    required_bankroll: float | None
    observed_funding_shortfall: bool
    performance_initial_wallet: float | None
    return_on_initial_wallet: float | None


def cash_risk_path(result, *, unrestricted=None, capital_identified=True):
    """Summarize a run; a wallet-truncated run needs an unrestricted counterpart.

    The counterpart must replay identical inputs/policy/random tape with only
    initial_wallet changed to None. Never estimate capital from a stopped path.
    A run with no wallet_wait already follows the unrestricted trajectory.
    """
    shortfall = any(e.kind == "wallet_wait" for e in result.events)
    capital = result if unrestricted is None else unrestricted
    if capital_identified and any(e.kind == "wallet_wait" for e in capital.events):
        raise ValueError("bankroll analysis needs an unrestricted-wallet replay of the same path")
    if unrestricted is not None and (
        capital.start != result.start or capital.end != result.end or capital.spec != result.spec
        or getattr(capital, "history_fingerprint", None) != getattr(result, "history_fingerprint", None)
        or capital.policy != result.policy or capital.config.initial_wallet is not None
        or replace(capital.config, initial_wallet=result.config.initial_wallet) != result.config
    ):
        raise ValueError("unrestricted replay must match horizon, account and policy")
    drawdown, underwater, _ = _cash_extremes(result)
    required = _cash_extremes(capital)[2] if capital_identified else None
    receipts = [e for e in result.events if e.kind == "receipt"]
    wallet = result.config.initial_wallet
    return CashRiskPath(result.net_cash, result.net_cash_per_day, result.calendar_days,
        result.receipts, result.fees, result.outstanding_payouts, result.attempts,
        result.failed_attempts, len(receipts), drawdown, underwater,
        (receipts[0].at - result.start).total_seconds()/86400 if receipts else None,
        required, shortfall, wallet, result.net_cash/wallet if wallet else None)


def _required_capital(requirements, alpha):
    """Smallest cent-valued capital with empirical P(required > capital) <= alpha."""
    n = len(requirements)
    allowed = (Fraction(str(alpha)) * n).__floor__()
    return 0.0 if allowed >= n else float(requirements[n - allowed - 1])


def _confidence_capital(requirements, alpha, confidence):
    """One-sided distribution-free order-statistic tolerance bound for IID paths.

    Choose the largest fixed k with BinomialCDF(k; n, alpha) <= 1-confidence;
    the (n-k)th ordered capital requirement then bounds the population ruin tail.
    This is not a plug-in confidence interval at a data-selected threshold.
    """
    if alpha == 1:
        return 0.0
    if alpha == 0:
        return None
    n, allowed = len(requirements), -1
    log_pmf, log_cdf = n * log1p(-alpha), -float("inf")
    for k in range(min(n, floor(n * alpha) + 1)):
        if k:
            log_pmf += log(n-k+1) - log(k) + log(alpha) - log1p(-alpha)
        log_cdf = float(np.logaddexp(log_cdf, log_pmf))
        if log_cdf > log1p(-confidence):
            break
        allowed = k
    return None if allowed < 0 else float(requirements[n-allowed-1])


def _tail_mean(values, fraction):
    ordered = sorted(values)
    mass = len(ordered) * fraction
    whole = floor(mass)
    return (fsum(ordered[:whole]) + (mass-whole)*ordered[min(whole, len(ordered)-1)]) / mass


def risk_report(paths, *, options=RiskConfig(), sample_kind="historical_windows"):
    """Fixed-horizon cash report. Only independent_model enables inference.

    Historical windows (including overlapping starts) describe this history;
    their counts are not an independent future sample. Do not label training
    results from a selected policy independent_model.
    """
    paths = tuple(paths)
    if not paths or any(not isinstance(p, CashRiskPath) for p in paths):
        raise ValueError("risk_report requires CashRiskPath observations")
    if not isinstance(options, RiskConfig):
        raise TypeError("options must be RiskConfig")
    if sample_kind not in ("historical_windows", "single_history", "independent_model", "training_model"):
        raise ValueError("unknown sample_kind")
    n = len(paths)
    independent = sample_kind == "independent_model" and n > 1
    capital_identified = all(p.required_bankroll is not None for p in paths)
    requirements = sorted(p.required_bankroll for p in paths) if capital_identified else []
    fields = ("net_cash", "net_cash_per_day", "receipts", "fees", "outstanding_payouts",
              "calendar_days", "attempts", "failed_attempts", "payout_count",
              "max_cash_drawdown", "longest_underwater_days", "required_bankroll",
              "days_to_first_receipt", "return_on_initial_wallet")
    distributions = {field: distribution((getattr(p, field) for p in paths if getattr(p, field) is not None),
                                         options.percentiles) for field in fields}
    curve = []
    from bisect import bisect_right
    capitals = sorted(set([0.0] + requirements + ([] if options.bankroll is None else [options.bankroll])))
    for capital in capitals if capital_identified else ():
        count = n - bisect_right(requirements, capital)
        curve.append({"bankroll": capital, "ruined_paths": count, "ruin_probability": count/n})
    alpha = options.target_ruin_probability
    required = _required_capital(requirements, alpha) if capital_identified else None
    supported = _confidence_capital(requirements, alpha, options.confidence) if independent and capital_identified else None
    if options.bankroll is None or not capital_identified:
        ruined = probability = None
    else:
        ruined = n - bisect_right(requirements, options.bankroll)
        probability = ruined/n
    losses = [max(0.0, -p.net_cash) for p in paths]
    tail = options.tail_probability
    intervals = None
    if independent:
        z = -NormalDist().inv_cdf((1-options.confidence)/2)
        counts = {"profitable": sum(p.net_cash > 0 for p in paths),
                  "loss": sum(p.net_cash < 0 for p in paths),
                  "payout": sum(p.receipts > 0 for p in paths)}
        if ruined is not None:
            counts["ruin"] = ruined
        intervals = {"confidence": options.confidence, "method": "two-sided marginal Wilson intervals",
                     "bounds": {name: wilson_ci(count, n, z=z) for name, count in counts.items()}}
    return {"sample_kind": sample_kind, "paths": n, "options": asdict(options),
        "uncertainty": _from_distributions(distributions, sample_kind=sample_kind,
                                           count=n, confidence=options.confidence),
        "performance_wallets": sorted(set(p.performance_initial_wallet for p in paths), key=lambda x: -1 if x is None else x),
        "distributions": distributions,
        "probability_profitable": sum(p.net_cash > 0 for p in paths)/n,
        "probability_loss": sum(p.net_cash < 0 for p in paths)/n,
        "probability_break_even": sum(p.net_cash == 0 for p in paths)/n,
        "payout_probability": sum(p.receipts > 0 for p in paths)/n,
        "observed_funding_shortfall_frequency": sum(p.observed_funding_shortfall for p in paths)/n,
        "loss_var": float(np.quantile(losses, 1-tail, method="inverted_cdf")),
        "loss_expected_shortfall": max(0.0, -_tail_mean([-loss for loss in losses], tail)),
        "worst_tail_mean_net_cash": _tail_mean([p.net_cash for p in paths], tail),
        "ruined_paths": ruined, "ruin_probability": probability,
        "required_bankroll": required,
        "capital_status": "conditional_wallet_invariance" if capital_identified else "not_identified",
        "achieved_empirical_ruin_probability": (n-bisect_right(requirements, required))/n if capital_identified else None,
        "confidence_supported_bankroll": supported,
        "confidence_status": ("capital_not_identified" if not capital_identified else "supported" if supported is not None else
                              "insufficient_independent_paths" if independent else "no_independent_sample"),
        "probability_intervals": intervals,
        "best_case_zero_failure_upper_bound": float(-np.expm1(log1p(-options.confidence)/n)) if independent else None,
        "bankroll_curve": curve, "path_records": [asdict(p) for p in paths],
        "definitions": {
            "ruin": "First inability to pay a required evaluation/reset/activation within the observed horizon; equality with the fee is sufficient. A prop-account breach is not investor ruin.",
            "bankroll": "Maximum external cash deficit on the same policy's unrestricted-wallet path, rounded up to cents. Requires decisions invariant to available wallet until funding stops. Otherwise capital and counterfactual ruin are not identified. Receipts finance later fees; approvals and trading balances are not spendable cash.",
            "horizon": "Finite observed horizon only; no perpetual ruin probability or extrapolated annual return.",
            "variance": "Descriptive variance uses ddof=0; sample variance uses ddof=1. Dollar variance has units USD squared.",
            "tail": "Loss=max(0,-net cash). VaR is the empirical inverse-CDF loss threshold; expected shortfall averages the worst tail with fractional observation weights.",
            "timing": "Days to first receipt is conditional on receiving one; underwater duration includes unrecovered drawdowns through the horizon.",
            "confidence": "Independent-model bankroll uses a one-sided order-statistic tolerance bound. Probability intervals are two-sided marginal Wilson intervals at a predeclared bankroll. The zero-failure bound describes best-case sample resolution, not current-wallet risk when failures occur. Dependent historical windows receive no inferential confidence bound.",
            "limits": "Empirical zero failures do not establish zero future risk. No confidence statement covers model error, future rule changes or repeated tuning on this report.",
        }}
