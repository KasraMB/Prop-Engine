"""Conditional uncertainty of means, distinct from outcome dispersion."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from numbers import Real

import numpy as np


_KINDS = ("independent_model", "execution_model", "historical_series",
          "historical_windows", "single_history", "training_model")


@dataclass(frozen=True)
class UncertaintyConfig:
    confidence: float = .95
    resamples: int = 0
    seed: int = 0
    block_length: int | None = None

    def __post_init__(self):
        if (isinstance(self.confidence, bool) or not isinstance(self.confidence, Real)
                or not isfinite(self.confidence) or not 0 < self.confidence < 1):
            raise ValueError("confidence must be finite in (0, 1)")
        if type(self.resamples) is not int or self.resamples < 0 or self.resamples == 1:
            raise ValueError("resamples must be zero or an integer >= 2")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.block_length is not None and (type(self.block_length) is not int or self.block_length < 1):
            raise ValueError("block_length must be a positive integer")


def _check(sample_kind, selected_on_sample):
    if sample_kind not in _KINDS:
        raise ValueError(f"sample_kind must be one of {_KINDS}")
    if type(selected_on_sample) is not bool:
        raise TypeError("selected_on_sample must be bool")


def _report(sample_kind, selected_on_sample, options):
    _check(sample_kind, selected_on_sample)
    selected = selected_on_sample or sample_kind == "training_model"
    return {
        "sample_kind": sample_kind, "selected_on_sample": selected,
        "options": asdict(options), "metrics": {},
        "distribution_family": "not identified; no Gaussian outcome law is assumed",
        "scope": ("execution draws conditional on the same historical market tape"
                  if sample_kind == "execution_model" else
                  "model draws conditional on fixed parameters and a frozen policy"
                  if sample_kind == "independent_model" else
                  "the supplied ordered scalar series under stationarity and weak dependence"
                  if sample_kind == "historical_series" else "descriptive historical or training outcomes"),
        "unmeasured": {
            "parameters": "not quantified; requires calibration/refit experiments",
            "model": "not quantified; scenario sensitivity is not a probability law",
            "selection": "not corrected; training selection and repeated holdout reuse invalidate naive inference",
            "execution": "only included if execution uncertainty was varied in the supplied samples",
            "numerical": "not quantified; testing and numerical validation are separate from sampling error",
        },
        "warnings": (["Few bootstrap draws in each tail; increase resamples before interpreting interval endpoints."]
                     if options.resamples and options.resamples*(1-options.confidence)/2 < 10 else []),
        "limits": [
            "Independence and stationarity are caller declarations, not detected properties.",
            "Intervals cover a mean, not individual outcomes, ruin tails or required bankroll.",
            "Intervals are approximate and marginal, not simultaneous across metrics or experiments.",
            "No observed variation or failures does not prove zero population variance or risk.",
            "Finite-variance assumptions and representative samples are required; unseen tails remain unknown.",
        ],
    }


def _metric(count, missing, mean, variance, sample_kind, selected):
    sd = None if variance is None else sqrt(variance)
    reason = ("selected_on_sample" if selected else "missing_observations" if missing else
              "insufficient_observations" if count < 2 else
              "no_observed_variation" if not variance else
              "dependent_windows" if sample_kind == "historical_windows" else
              "single_history" if sample_kind == "single_history" else
              "block_bootstrap_required" if sample_kind == "historical_series" else None)
    independent = sample_kind in ("independent_model", "execution_model")
    return {"count": count, "missing": missing, "mean": mean,
            "sample_standard_deviation": sd,
            "mean_standard_error": sd / sqrt(count) if independent and reason is None else None,
            "mean_interval": None, "method": "iid_mean_standard_error" if independent and reason is None else None,
            "status": reason or "bootstrap_not_requested"}


def _from_distributions(distributions, *, sample_kind, count, selected_on_sample=False, confidence=.95):
    """Reuse existing sufficient statistics without allocating path arrays."""
    report = _report(sample_kind, selected_on_sample, UncertaintyConfig(confidence=confidence))
    for name, stats in distributions.items():
        n = stats["count"] if stats else 0
        report["metrics"][name] = _metric(n, count-n, stats["mean"] if stats else None,
            stats["sample_variance"] if stats else None, sample_kind, report["selected_on_sample"])
    return report


def _array(values):
    try:
        a = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("samples must contain scalar numbers or missing values") from exc
    if a.ndim != 1 or np.isinf(a).any():
        raise ValueError("samples must be one-dimensional, with no infinite values")
    return a


def _bootstrap_means(a, block_length, resamples, rng):
    n, mean = len(a), float(a.mean())
    centered = a-mean
    blocks, remainder = divmod(n, block_length)
    if block_length == 1:
        full, partial = centered, None
    else:
        prefix = np.concatenate(([0.0], np.cumsum(np.concatenate((centered, centered[:block_length-1])))))
        full = prefix[block_length:block_length+n]-prefix[:n]
        partial = prefix[remainder:remainder+n]-prefix[:n] if remainder else None
    width = blocks+bool(remainder)
    batch = max(1, 1_000_000 // width)
    means = np.empty(resamples)
    for begin in range(0, resamples, batch):
        end = min(resamples, begin+batch)
        starts = rng.integers(0, n, size=(end-begin, width))
        sums = full[starts[:, :blocks]].sum(axis=1)
        if remainder:
            sums += partial[starts[:, -1]]
        means[begin:end] = mean+sums/n
    return means


def uncertainty_report(samples, *, sample_kind, options=UncertaintyConfig(), selected_on_sample=False):
    """Report mean uncertainty, optionally resampling supplied scalar outcomes.

    historical_series requires ordered, equally spaced observations, an explicit
    block length and stationarity/weak dependence. It does not rerun the engine
    or refit a strategy. Missing values suppress inference rather than compress
    time. IID kinds assert independent draws under a fixed model, not truth.
    """
    if not isinstance(options, UncertaintyConfig):
        raise TypeError("options must be UncertaintyConfig")
    if not isinstance(samples, Mapping) or not samples:
        raise ValueError("samples must be a nonempty mapping of metric names to observations")
    if any(not isinstance(name, str) or not name for name in samples):
        raise ValueError("metric names must be nonempty strings")
    report = _report(sample_kind, selected_on_sample, options)
    if options.block_length is not None and sample_kind != "historical_series":
        raise ValueError("block_length applies only to historical_series")
    lengths = set()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for name, values in samples.items():
                a = _array(values)
                lengths.add(len(a))
                valid = a[~np.isnan(a)]
                n, missing = len(valid), len(a)-len(valid)
                stats = _metric(n, missing, float(valid.mean()) if n else None,
                    float(valid.var(ddof=1)) if n > 1 else None,
                    sample_kind, report["selected_on_sample"])
                report["metrics"][name] = stats
                block = options.block_length if sample_kind == "historical_series" else 1
                if options.block_length is not None and block > len(a):
                    raise ValueError("block_length exceeds sample length")
                if stats["status"] not in ("bootstrap_not_requested", "block_bootstrap_required"):
                    continue
                if not options.resamples or block is None:
                    continue
                if sample_kind == "historical_series" and n // block < 2:
                    stats["status"] = "insufficient_blocks"
                    continue
                # Each metric uses the same indices; adding metrics cannot change a result.
                means = _bootstrap_means(a, block, options.resamples, np.random.default_rng(options.seed))
                if np.all(means == means[0]):
                    stats["status"] = "degenerate_bootstrap"
                    continue
                if sample_kind == "historical_series":
                    stats["mean_standard_error"] = float(means.std(ddof=1))
                tail = (1-options.confidence)/2
                stats.update(mean_interval=list(map(float, np.quantile(means, [tail, 1-tail]))),
                             method="circular_block_percentile" if sample_kind == "historical_series" else "iid_percentile",
                             status="estimated")
    except FloatingPointError as exc:
        raise ValueError("uncertainty exceeds the finite numeric reporting range") from exc
    if len(lengths) != 1:
        raise ValueError("metrics must share the same observation count")
    return report


def paired_uncertainty(selected, baseline, *, sample_kind, options=UncertaintyConfig(), selected_on_sample=False):
    """Selected minus baseline on the same aligned scenarios, not independent groups."""
    if not isinstance(selected, Mapping) or not isinstance(baseline, Mapping) or selected.keys() != baseline.keys():
        raise ValueError("paired metrics must have matching names")
    differences = {}
    for name in selected:
        left, right = _array(selected[name]), _array(baseline[name])
        if left.shape != right.shape:
            raise ValueError("paired observations must have matching lengths and order")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = left-right
        if np.isinf(delta).any():
            raise ValueError("paired differences exceed the finite numeric range")
        differences[name] = delta
    report = uncertainty_report(differences, sample_kind=sample_kind, options=options,
                                selected_on_sample=selected_on_sample)
    report["comparison"] = "selected minus baseline; caller must align identical scenarios and horizons"
    return report


def scenario_sensitivity(scenarios, *, baseline):
    """Compare named assumptions without assigning probabilities to scenarios."""
    if not isinstance(scenarios, Mapping) or baseline not in scenarios:
        raise ValueError("scenarios must contain the named baseline")
    reference = scenarios[baseline]
    if not isinstance(reference, Mapping) or not reference:
        raise ValueError("each scenario must provide the same nonempty metric mapping")
    if any(not isinstance(key, str) or not key for key in reference):
        raise ValueError("metric names must be nonempty strings")
    rows = {}
    for name, values in scenarios.items():
        if not isinstance(name, str) or not name or not isinstance(values, Mapping) or values.keys() != reference.keys():
            raise ValueError("named scenarios must have matching metric names")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in values.values()):
            raise ValueError("scenario metrics must be finite numbers")
        rows[name] = {key: float(value) for key, value in values.items()}
    deltas = {name: {key: value-rows[baseline][key] for key, value in row.items()} for name, row in rows.items()}
    if any(not isfinite(v) for row in deltas.values() for v in row.values()):
        raise ValueError("scenario differences exceed the finite numeric range")
    return {"baseline": baseline, "values": rows, "deltas": deltas,
            "envelope": {key: [min(row[key] for row in rows.values()), max(row[key] for row in rows.values())]
                         for key in reference},
            "scope": "supplied scenarios only; not a confidence interval, probability law or worst-case guarantee"}
