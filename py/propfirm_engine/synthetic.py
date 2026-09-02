"""Synthetic trade-stream generators (ARCHITECTURE §11.7; BUILD_SPEC Step 3b).

A generator *manufactures* raw trade rows from statistical parameters, as an
alternative source to a real backtest, for two purposes: deterministic,
known-property streams to drive the Step 6–11 tests without real data; and
future breakeven-mapping research (§11.7.4, not built here).

**The one hard contract (§11.7 / §11.1):** a generator emits the *exact raw-row
schema* — ``timestamp``, per-unit ``return``, ``mae``, ``symbol`` — that
:func:`propfirm_engine.data.preprocess` accepts unchanged. The engine cannot tell
synthetic input from real input; that is what makes synthetic data valid for
testing the whole stack, Step 3 included.

**Strategy parameterization (§11.7.1): ``win_rate`` and ``RR`` only.** Risk is the
unit — every trade risks exactly ``−1R`` and returns ``+RR`` on a win. Edge is
**derived, never an input** (accepting it would over-determine the stream and
admit inconsistent triples, MODEL_RISKS §I2): ``edge = win_rate*(RR+1) − 1`` and
``breakeven_win_rate = 1/(RR+1)`` are reported as provenance, not accepted.

**The generator ladder (§11.7.2)** — three dependence models behind one interface:

* :class:`IIDGenerator` — independent ``+RR``/``−1`` draws. The optimistic
  baseline (no clustered losing streaks), like the i.i.d. day-bootstrap.
* :class:`RegimeSwitchingGenerator` — a Markov chain over regimes with different
  win-rates: persistent winning/losing periods (positive autocorrelation) that
  i.i.d. cannot make. The chain's stationary mix reproduces the target
  ``win_rate``.
* :class:`StochasticVolGenerator` — trade magnitude scales with a slow-moving
  volatility process: volatility clustering (autocorrelated |return|) and fatter
  tails than the fixed-size case, while the sign process preserves ``win_rate``.

Every stream carries **provenance** (type, all parameters, seed, derived
edge/breakeven) so a synthetic number is never mistaken for a real-data result
(the §I1 discipline). Generation is deterministic under a fixed seed.

**mae synthesis.** ``mae`` is the per-unit worst adverse excursion (a loss
magnitude, same units as ``return``). A *winning* trade draws down
``intraday_excursion × risk`` before closing positive — deep enough to exercise
``check_timing=CONTINUOUS`` rules (a continuous check sees the floating low an
EOD check misses), but never exceeding its risk. A *losing* trade hit its stop,
so its worst excursion equals its risk (``1R``). Both scale with the trade's
volatility so ``mae ≤ risk`` holds per trade regardless of the vol process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class Provenance:
    """What produced a synthetic stream — rides through to ``Results`` (§11.7.3)."""

    generator: str
    params: dict
    seed: int
    edge: float
    breakeven_win_rate: float


@dataclass(frozen=True)
class SyntheticStream:
    """A raw-row table (``rows``, the §11.1 schema as a column dict) plus provenance.

    ``rows`` is exactly what :func:`propfirm_engine.data.preprocess` consumes.
    """

    rows: dict
    provenance: Provenance

    @property
    def n_trades(self) -> int:
        return len(self.rows["timestamp"])


# Jan 1 2024 is a Monday — a clean default first session for weekday cadence.
_DEFAULT_START = datetime(2024, 1, 1, 9, 30)


@dataclass(frozen=True)
class TradeStreamGenerator(ABC):
    """Common base: parameters + a per-type dependence model → a raw-row table.

    Free strategy parameters are ``win_rate`` and ``rr`` only (§11.7.1). The rest
    shape the *session structure* (``trades_per_day``, ``start``, ``symbol``) and
    the *mae depth* (``intraday_excursion``); win/loss dispersion is out of the
    base case (fixed-size outcomes). Edge is derived, not a field — constructing
    a generator with an ``edge=`` keyword raises ``TypeError`` (§I2).
    """

    win_rate: float
    rr: float
    trades_per_day: int = 4
    intraday_excursion: float = 0.5  # fraction of risk a WIN draws down (mae depth)
    symbol: str = "SYN"
    start: datetime = _DEFAULT_START

    def __post_init__(self) -> None:
        if not 0.0 <= self.win_rate <= 1.0:
            raise ValueError(f"win_rate must be in [0, 1], got {self.win_rate}")
        if self.rr <= 0.0:
            raise ValueError(f"rr (risk-reward) must be > 0, got {self.rr}")
        if self.trades_per_day < 1:
            raise ValueError(f"trades_per_day must be >= 1, got {self.trades_per_day}")
        if not 0.0 <= self.intraday_excursion < 1.0:
            # < 1 so a winning trade's excursion never reaches (let alone exceeds)
            # its risk — "adverse excursion does not exceed risk" (BUILD_SPEC 3b).
            raise ValueError(
                f"intraday_excursion must be in [0, 1), got {self.intraday_excursion}"
            )

    # --- derived, reported, never input (§11.7.1) --------------------------- #

    @property
    def edge(self) -> float:
        """Per-trade expectancy in R: ``win_rate*(RR+1) − 1``."""
        return self.win_rate * (self.rr + 1.0) - 1.0

    @property
    def breakeven_win_rate(self) -> float:
        """The win rate at which ``edge == 0``: ``1/(RR+1)``."""
        return 1.0 / (self.rr + 1.0)

    # --- per-type dependence model ------------------------------------------ #

    @abstractmethod
    def _simulate(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(win, vol)``: a boolean win/loss array and a positive
        volatility-scale array (mean ≈ 1), both length ``n``. The base class turns
        these into returns and mae; each generator differs only here."""
        raise NotImplementedError

    # --- assembly (shared) -------------------------------------------------- #

    def generate(self, n_days: int, seed: int) -> SyntheticStream:
        """Emit ``n_days`` weekday sessions of ``trades_per_day`` trades each."""
        if n_days < 1:
            raise ValueError(f"n_days must be >= 1, got {n_days}")
        rng = np.random.default_rng(seed)
        n = n_days * self.trades_per_day
        win, vol = self._simulate(n, rng)
        base = np.where(win, self.rr, -1.0)  # +RR on a win, −1 on a loss (R units)
        ret = vol * base
        # mae: win draws down intraday_excursion × risk; loss hits its stop (1×risk).
        # Both scale with vol, so mae ≤ risk (= vol×1) holds per trade.
        mae = vol * np.where(win, self.intraday_excursion, 1.0)
        timestamps = self._timestamps(n_days)
        rows = {
            "timestamp": timestamps,
            "return": ret.astype(float).tolist(),
            "mae": mae.astype(float).tolist(),
            "symbol": [self.symbol] * n,
        }
        return SyntheticStream(rows=rows, provenance=self._provenance(seed))

    def _provenance(self, seed: int) -> Provenance:
        params = asdict(self)
        params.pop("start", None)  # a datetime; kept out of the compact record
        return Provenance(
            generator=type(self).__name__,
            params=params,
            seed=seed,
            edge=self.edge,
            breakeven_win_rate=self.breakeven_win_rate,
        )

    def _session_days(self, n_days: int) -> list[datetime]:
        """The first ``n_days`` weekday sessions from ``start`` (skips Sat/Sun), so
        preprocessing derives a ~5 trading-days-per-week cadence (§11.5)."""
        out: list[datetime] = []
        d = self.start
        while len(out) < n_days:
            if d.weekday() < 5:  # Mon–Fri
                out.append(d)
            d += timedelta(days=1)
        return out

    def _timestamps(self, n_days: int) -> list[datetime]:
        """``trades_per_day`` timestamps per session, spread 09:30–16:00 so every
        trade falls before the default 17:00 session reset (one session/weekday)."""
        out: list[datetime] = []
        window_minutes = int(6.5 * 60)  # 09:30 → 16:00
        for day in self._session_days(n_days):
            base = datetime(day.year, day.month, day.day, 9, 30)
            for k in range(self.trades_per_day):
                frac = (k + 1) / (self.trades_per_day + 1)
                out.append(base + timedelta(minutes=int(frac * window_minutes)))
        return out


# --------------------------------------------------------------------------- #
# 1. IID — independent draws (the optimistic baseline)                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IIDGenerator(TradeStreamGenerator):
    """Each trade is an independent ``win_rate`` coin; unit volatility (§11.7.2)."""

    def _simulate(self, n, rng):
        win = rng.random(n) < self.win_rate
        vol = np.ones(n, dtype=np.float64)
        return win, vol


# --------------------------------------------------------------------------- #
# 2. Regime-switching — persistent good/bad periods                             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RegimeSwitchingGenerator(TradeStreamGenerator):
    """A symmetric two-regime Markov chain: a *good* regime (win-rate
    ``win_rate + s``) and a *bad* one (``win_rate − s``), each entered half the
    time in the stationary distribution, so the long-run win-rate is *exactly* the
    target ``win_rate`` (§11.7.2). ``persistence`` is the probability of staying in
    the current regime each trade; > 0.5 produces the positive autocorrelation and
    long runs i.i.d. cannot make.

    ``spread`` is a *requested maximum*: the effective half-spread ``s`` is capped
    at ``min(spread, win_rate, 1 − win_rate)`` so **both** regime win-rates stay in
    ``[0, 1]** while the 50/50 mix still averages to ``win_rate``. A naive
    ``min``/``max`` clip of the two rates would instead skew the mean above/below
    target near the boundaries (and make the realized edge disagree with the
    reported one), which the Must-pass forbids — so near win-rate 0 or 1 there is
    simply less regime variation, never a broken mean."""

    spread: float = 0.25  # requested ± around win_rate; capped to keep rates in [0,1]
    persistence: float = 0.95  # P(stay in current regime) each trade

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.persistence <= 1.0:
            raise ValueError(f"persistence must be in [0, 1], got {self.persistence}")
        if self.spread < 0.0:
            raise ValueError(f"spread must be >= 0, got {self.spread}")

    def _simulate(self, n, rng):
        # Cap the effective spread symmetrically so BOTH regime rates are in [0, 1]
        # and the 50/50 stationary mix averages EXACTLY to win_rate (see docstring).
        s = min(self.spread, self.win_rate, 1.0 - self.win_rate)
        bad = self.win_rate - s
        good = self.win_rate + s
        regime_wr = np.array([bad, good])
        # Both states share one `persistence`, so switch probabilities are equal →
        # the stationary distribution is uniform (0.5/0.5) and the time-averaged
        # win-rate is (good + bad)/2 == win_rate exactly.
        state = 0 if rng.random() < 0.5 else 1
        win = np.empty(n, dtype=bool)
        u_stay = rng.random(n)
        u_win = rng.random(n)
        for i in range(n):
            if u_stay[i] >= self.persistence:  # switch regimes
                state = 1 - state
            win[i] = u_win[i] < regime_wr[state]
        vol = np.ones(n, dtype=np.float64)
        return win, vol


# --------------------------------------------------------------------------- #
# 3. Stochastic volatility — clustered magnitude, fatter tails                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StochasticVolGenerator(TradeStreamGenerator):
    """Independent win/loss *signs* (so ``win_rate`` is preserved exactly), but
    trade *magnitude* scales with a slow-moving log-AR(1) volatility process
    (§11.7.2). ``vol_phi`` is the persistence of log-volatility and ``vol_sigma``
    its shock scale; the process is mean-normalized so ``E[vol] ≈ 1`` and the edge
    is preserved. Autocorrelated |return| gives volatility clustering; the
    lognormal scaling gives fatter tails than the fixed-size case."""

    vol_phi: float = 0.9  # AR(1) persistence of log-volatility
    vol_sigma: float = 0.5  # shock scale of log-volatility

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.vol_phi < 1.0:
            raise ValueError(f"vol_phi must be in [0, 1), got {self.vol_phi}")
        if self.vol_sigma < 0.0:
            raise ValueError(f"vol_sigma must be >= 0, got {self.vol_sigma}")

    def _simulate(self, n, rng):
        win = rng.random(n) < self.win_rate  # IID signs preserve win_rate
        # log-AR(1): x_t = phi x_{t-1} + sigma eps_t, started from its stationary
        # distribution. Stationary variance is sigma^2 / (1 - phi^2).
        eps = rng.standard_normal(n)  # eps[0] intentionally unused: x[0] is seeded
        x = np.empty(n, dtype=np.float64)  # separately from the stationary distribution
        if self.vol_phi < 1.0:
            stat_std = self.vol_sigma / np.sqrt(1.0 - self.vol_phi**2)
        else:  # pragma: no cover - guarded out in __post_init__
            stat_std = self.vol_sigma
        x[0] = stat_std * rng.standard_normal()  # start ON the stationary marginal
        for t in range(1, n):
            x[t] = self.vol_phi * x[t - 1] + self.vol_sigma * eps[t]
        # vol = exp(x - var/2) so E[vol] = 1 (lognormal mean correction).
        stat_var = stat_std**2
        vol = np.exp(x - 0.5 * stat_var)
        return win, vol


__all__ = [
    "Provenance",
    "SyntheticStream",
    "TradeStreamGenerator",
    "IIDGenerator",
    "RegimeSwitchingGenerator",
    "StochasticVolGenerator",
]
