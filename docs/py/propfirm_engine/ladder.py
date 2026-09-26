"""Generator ladder — model-sensitivity bands (ARCHITECTURE §11.7/§G1;
BUILD_SPEC Step 13; MODEL_RISKS §G1, §I1).

Every headline number this project reports is **model-conditional**: it depends on
the path-generating model assumed for the world, and the most optimistic model
(i.i.d.) systematically flatters the account (no clustered losing streaks, thin
tails). Reporting a single-generator point is therefore dishonest — it hides the
model risk. This layer evaluates **one frozen strategy, one contract, one
objective** across a *ladder* of world-models and returns each headline number as
a :class:`Band` across the ladder, never a single point (§G1). That is the whole
contract: *a result is only reported with its band.*

**A rung is a model of the world**, holding the strategy fixed:

* ``iid`` — independent trades, days shuffled independently. The optimistic floor
  (no serial dependence at all), like the i.i.d. day bootstrap.
* ``block-{m}`` — a serially-dependent (regime) stream resampled with the
  **stationary block bootstrap** at mean-block ``m``. Sweeping ``m`` is the
  **block-length range** (§C7): longer blocks preserve more of the regime's serial
  dependence — persistent runs of *both* wins and losses. Block length is a
  first-class modeling parameter whose sensitivity *must* be examined, which is
  what this sweep does, and it is meaningful only on serially dependent data, so
  these rungs resample the regime stream (blocking an i.i.d. stream would recover
  i.i.d. by construction). **The direction is contract-dependent and not
  assumed:** a drawdown rule is hurt by clustered losses, but a profit-target race
  can be *helped* by clustered wins, so the sweep reports the *range* it produces,
  never a monotone worsening.
* ``regime`` — the regime stream resampled i.i.d. across days: regime persistence
  survives *within* a day but day order is shuffled, isolating the day-serial
  effect from the within-day one when compared to ``block-*``.
* ``stochvol`` — a stochastic-volatility stream (clustered magnitude, fatter
  tails) resampled i.i.d.

The strategy ``(win_rate, rr)`` is **frozen across every rung** (a single strategy
seen under many world-models); only the world-model changes. The objective is a
set of metric functions over :class:`~propfirm_engine.engine.Outcomes`.

**Interpretation guard (§I1).** The honest summary is the *band* and, if one must
be named, the *pessimistic* rung — never the i.i.d. point. The band width is the
model risk. This module deliberately exposes no "the answer" accessor: you get the
per-rung values and the spread, and you decide.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .data import preprocess
from .engine import Engine, RunConfig
from .renewal import fee_bankroll_efficiency, r_renewal
from .resampling import DayResampler, IIDDayBootstrap, StationaryDayBootstrap
from .statistics import (
    pass_rate,
    payout_velocity,
    prob_profitable,
    return_on_fee_per_year,
)
from .synthetic import (
    IIDGenerator,
    RegimeSwitchingGenerator,
    StochasticVolGenerator,
    TradeStreamGenerator,
)


# --------------------------------------------------------------------------- #
# Headline metrics — each maps Outcomes -> float                               #
# --------------------------------------------------------------------------- #

#: The default headline set banded across the ladder. Each is a decision-relevant
#: scalar from Step 10 (distribution/time axes) or Step 11 (renewal). All are
#: pure functions of the raw :class:`Outcomes`, so a rung needs only one engine run.
DEFAULT_METRICS: dict = {
    "pass_rate": lambda o: pass_rate(o.code),
    "prob_profitable": prob_profitable,
    "payout_velocity": payout_velocity,
    "return_on_fee_per_year": return_on_fee_per_year,
    "r_renewal": r_renewal,
    "fee_bankroll_efficiency": fee_bankroll_efficiency,
}


# --------------------------------------------------------------------------- #
# Rungs and bands                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rung:
    """One world-model: a trade-stream generator + a day resampler.

    The generator manufactures the dataset (its dependence model); the resampler
    strings source days into attempts (its serial-dependence model). Together they
    define how Monte Carlo paths are produced for the *fixed* strategy the
    generator carries.

    ``crn_group`` ties rungs into a **Common Random Numbers** family (§G1/§C7): all
    rungs sharing a non-empty group label draw the *same* generated dataset and the
    *same* resample seed, so a rung-to-rung difference reflects only the parameter
    that actually varies between them (e.g. ``mean_block`` across a block-length
    sweep), not dataset-realization or resampling noise. Rungs with ``crn_group=None``
    are independent (each gets its own seeds) — correct for genuinely different
    world-models (iid vs regime vs stochvol) where independent draws are the point."""

    name: str
    generator: TradeStreamGenerator
    resampler: DayResampler
    crn_group: str | None = None


@dataclass(frozen=True)
class Band:
    """A single headline number *across the ladder* — the reportable unit (§G1).

    Holds every rung's value (``per_rung``); ``lo``/``hi``/``spread`` summarize the
    model-sensitivity range over the finite values. There is deliberately no
    "point estimate": the band *is* the estimate."""

    metric: str
    per_rung: dict  # rung name -> float (may be nan where a metric is undefined)

    @property
    def values(self) -> np.ndarray:
        return np.array(list(self.per_rung.values()), dtype=np.float64)

    @property
    def _finite(self) -> np.ndarray:
        v = self.values
        return v[np.isfinite(v)]

    @property
    def lo(self) -> float:
        f = self._finite
        return float(np.min(f)) if f.size else float("nan")

    @property
    def hi(self) -> float:
        f = self._finite
        return float(np.max(f)) if f.size else float("nan")

    @property
    def spread(self) -> float:
        """``hi − lo`` over finite rungs — the width of the model-risk band."""
        f = self._finite
        return float(np.max(f) - np.min(f)) if f.size else float("nan")

    @property
    def iid(self) -> float:
        """The optimistic-floor value, surfaced *only* so a caller can see how far
        the band falls below it — never as the headline (§I1)."""
        return self.per_rung.get("iid", float("nan"))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        pts = ", ".join(f"{k}={v:.4g}" for k, v in self.per_rung.items())
        return f"Band({self.metric}: [{self.lo:.4g}, {self.hi:.4g}] | {pts})"


@dataclass(frozen=True)
class LadderResult:
    """The whole ladder's output: one :class:`Band` per headline metric.

    Exposes the bands and the rung order — and, by construction, **no** accessor
    that collapses a metric to a single generator's value. To read a number you
    read its band."""

    bands: dict  # metric name -> Band
    rungs: tuple  # rung names in ladder order
    strategy: dict  # caller-supplied frozen-strategy metadata (win_rate, rr, ...),
    #                 {} if not passed. The freeze itself is guaranteed structurally
    #                 by default_ladder feeding ONE (win_rate, rr) to every generator;
    #                 this field only records it for reporting.
    n_paths: int

    def band(self, metric: str) -> Band:
        return self.bands[metric]

    def as_table(self) -> dict:
        """``metric -> {rung: value, ..., 'lo':, 'hi':, 'spread':}`` for reporting."""
        out = {}
        for name, band in self.bands.items():
            row = dict(band.per_rung)
            row.update(lo=band.lo, hi=band.hi, spread=band.spread)
            out[name] = row
        return out


# --------------------------------------------------------------------------- #
# The default ladder                                                           #
# --------------------------------------------------------------------------- #


def default_ladder(
    win_rate: float,
    rr: float,
    *,
    trades_per_day: int = 4,
    block_lengths=(2, 5, 10),
    regime_kwargs: dict | None = None,
    stochvol_kwargs: dict | None = None,
) -> list[Rung]:
    """Build the standard i.i.d. → block(range) → regime → stochvol ladder for one
    frozen strategy ``(win_rate, rr)`` (§G1).

    The strategy is identical on every rung; only the world-model differs. The
    ``block-{m}`` rungs resample the **regime** stream (so block length is
    meaningful — see the module docstring) at each ``m`` in ``block_lengths``,
    giving the block-length range the spec requires. ``regime_kwargs`` /
    ``stochvol_kwargs`` tune the dependence models (persistence, spread, vol_phi…)
    without touching the frozen strategy."""
    rk = dict(persistence=0.9, spread=0.2)
    rk.update(regime_kwargs or {})
    sk = dict(vol_phi=0.9, vol_sigma=0.6)
    sk.update(stochvol_kwargs or {})

    def iid_gen():
        return IIDGenerator(win_rate=win_rate, rr=rr, trades_per_day=trades_per_day)

    def regime_gen():
        return RegimeSwitchingGenerator(
            win_rate=win_rate, rr=rr, trades_per_day=trades_per_day, **rk
        )

    def stochvol_gen():
        return StochasticVolGenerator(
            win_rate=win_rate, rr=rr, trades_per_day=trades_per_day, **sk
        )

    # The block sweep + the regime rung form ONE Common-Random-Numbers family: they
    # all share the same regime dataset and resample seed, so block-2 vs block-10 (and
    # vs the iid-resampled regime rung) differ ONLY in their resampler — the isolated
    # block-length signal §C7 asks for, not dataset/resample noise. The iid and
    # stochvol rungs are independent world-models and keep their own draws.
    rungs = [Rung("iid", iid_gen(), IIDDayBootstrap())]
    for m in block_lengths:
        rungs.append(Rung(f"block-{m}", regime_gen(), StationaryDayBootstrap(float(m)),
                          crn_group="regime-family"))
    rungs.append(Rung("regime", regime_gen(), IIDDayBootstrap(), crn_group="regime-family"))
    rungs.append(Rung("stochvol", stochvol_gen(), IIDDayBootstrap()))
    return rungs


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def run_ladder(
    account,
    rungs: list[Rung],
    *,
    n_days: int,
    config: RunConfig,
    metrics: dict | None = None,
    strategy: dict | None = None,
    seed: int | None = None,
    engine: Engine | None = None,
) -> LadderResult:
    """Evaluate ``account`` (the fixed contract) under every rung and band each metric.

    For each rung: generate ``n_days`` sessions from its generator, ``preprocess``
    them into a :class:`TradeDataset`, run :class:`Engine` with the rung's resampler
    (overriding ``config.resampler``), and evaluate every metric on the resulting
    :class:`Outcomes`. Each rung draws an independent, well-separated seed from
    ``seed`` (so rungs are not accidentally correlated) — determinism holds under a
    fixed ``seed``. Returns a :class:`LadderResult` of one :class:`Band` per metric.

    ``config``'s own ``resampler`` and ``seed`` are ignored per rung (each rung
    supplies its resampler; the root seed is ``seed`` if given, else ``config.seed``,
    and each rung's seed is derived from it); every other field
    (``n_paths``, ``L_eval``/``L_funded``, ``size_base``, fees via the account…) is
    held fixed across the ladder so only the world-model varies."""
    metrics = metrics if metrics is not None else DEFAULT_METRICS
    engine = engine if engine is not None else Engine()
    # One seed knob: `seed` if given, else `config.seed` (so the ladder responds to
    # the run config's seed the way the rest of the pipeline does). Seeds are keyed
    # per CRN family: rungs sharing a `crn_group` draw the SAME (gen_seed, run_seed)
    # pair (Common Random Numbers — a controlled single-parameter comparison), while
    # ungrouped rungs each get their own independent pair. Deterministic under seed.
    root = np.random.SeedSequence(config.seed if seed is None else seed)
    keys = []
    for k, rung in enumerate(rungs):
        key = ("group", rung.crn_group) if rung.crn_group else ("rung", k)
        if key not in keys:
            keys.append(key)
    child_states = root.generate_state(2 * len(keys))
    seed_of = {key: (int(child_states[2 * j]), int(child_states[2 * j + 1]))
               for j, key in enumerate(keys)}

    # Within a CRN family the dataset is generated once and shared, so every grouped
    # rung sees the identical source days (not just identical seeds).
    dataset_cache: dict = {}

    per_metric: dict[str, dict] = {name: {} for name in metrics}
    for k, rung in enumerate(rungs):
        key = ("group", rung.crn_group) if rung.crn_group else ("rung", k)
        gen_seed, run_seed = seed_of[key]
        if key in dataset_cache:
            dataset = dataset_cache[key]
        else:
            dataset = preprocess(rung.generator.generate(n_days=n_days, seed=gen_seed).rows)
            dataset_cache[key] = dataset
        cfg = replace(config, resampler=rung.resampler, seed=run_seed)
        outcomes = engine.run(account, dataset, cfg)
        for name, fn in metrics.items():
            try:
                val = float(fn(outcomes))
            except (ZeroDivisionError, ValueError):
                val = float("nan")
            per_metric[name][rung.name] = val

    bands = {name: Band(name, per_rung) for name, per_rung in per_metric.items()}
    return LadderResult(
        bands=bands,
        rungs=tuple(r.name for r in rungs),
        strategy=dict(strategy or {}),
        n_paths=config.n_paths,
    )


__all__ = [
    "Rung",
    "Band",
    "LadderResult",
    "DEFAULT_METRICS",
    "default_ladder",
    "run_ladder",
]
