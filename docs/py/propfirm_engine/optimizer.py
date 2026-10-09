"""The bet-sizing optimizer (ARCHITECTURE §16; BUILD_SPEC Step 14).

Searches over the **sizing policy** — never the strategy (§16.0) — to maximize the
**renewal reward rate** ``E[reward]/E[cycle time]`` under survival constraints
(§16.2), using the built sizing hook (§16.5): a policy is a ``policy_params`` array
the kernel already consumes, so a new candidate re-runs the existing Monte Carlo
with no recompilation. This is Level 5 of the trust hierarchy — trustworthy only
once everything below it holds, and only on **held-out** data (§16.7/§G7).

What is built here (the staged plan, §16.4 / Step 14):

* **Tier 1 only** — a risk-multiplier schedule over the *stage* state axis (the two
  reachable bits, ``IN_PROFIT`` × ``PRE_FIRST_PAYOUT`` → four regimes). Continuous,
  low-dimensional, CMA-ES-friendly. **Start here** (§16.4). Tiers 2–3 (a
  cease-trading gate, a mode machine) are a real kernel change and are *not* built
  until Tier 1 demonstrably plateaus out-of-sample (§16.4/§16.7).
* **Feasibility projection** (§16.4b) is passed through to every evaluation, so the
  ``r → 0`` exploit is closed: a microscopic-risk policy withers (``CAPPED_OUT``),
  earns no payout, and is *punished* by the renewal objective rather than rewarded
  for surviving.
* **CMA-ES** (§16.6) — the primary black-box search for this noisy,
  non-differentiable objective — with **Common Random Numbers** within a generator
  (every candidate on the *same* resampled paths, so the objective *difference*
  reflects the policy, not RNG noise), **multi-fidelity screening** (cheap paths to
  cull, the full count to select), and **nested out-of-sample** selection: the
  optimizer never sees the data the reported number is computed on (§16.7).

The governing law (§16.7): *every increase in policy capacity must be matched by an
increase in validation rigor.* Here that is enforced structurally — the only public
entry point that reports a performance number, :func:`walk_forward`, computes it on
a held-out partition the search never touched, across a whole generator ladder if
one is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from numbers import Integral, Real
import warnings

import numpy as np

from .data import slice_days
from .engine import Engine, Outcomes, RunConfig
from .enums import ExitCode
from .resampling import IIDDayBootstrap, StationaryDayBootstrap
from .statistics import attributable_fee

_FAIL_LO = 10  # FAILURE_THRESHOLD
_TIMED_OUT = int(ExitCode.TIMED_OUT)
_PASSED = int(ExitCode.PASSED)

# The kernel's phase-aware sizing regime index (see kernels/reference): 0 = eval
# (a single regime), and funded splits into 1..4 over in-profit × pre/post-first-
# payout. A Tier-1 policy carries one multiplier per regime, indexed directly.
_N_REGIMES = 5
_POLICY_LEN = _N_REGIMES  # length the kernel indexes by the regime index
#: human-readable regime labels, in policy/theta order (index == regime index).
REGIME_LABELS = (
    "eval",
    "funded · flat · pre-payout",
    "funded · in-profit · pre-payout",
    "funded · flat · post-payout",
    "funded · in-profit · post-payout",
)


# --------------------------------------------------------------------------- #
# Tier-1 policy space                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicySpace:
    """Tier-1 risk-multiplier schedule over the phase/stage regimes (§16.4).

    A parameter vector ``theta`` carries one multiplier per regime (:data:`REGIME_LABELS`
    order): the eval regime, then the four funded regimes. It maps directly to the
    ``policy_params`` array the kernel indexes by its regime index. Multipliers are
    bounded to ``[lo, hi]`` — a *mechanical* bound (an executable sizing multiplier is
    non-negative and finite), not a learned one, so it never consumes search capacity
    (§16.3). ``lo = 0`` is deliberately allowed so the search *can* propose vanishing
    risk; the feasibility projection + renewal objective are what make that
    unattractive, not a hidden floor."""

    lo: float = 0.0
    hi: float = 5.0

    def __post_init__(self):
        for name in ("lo", "hi"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.hi <= self.lo:
            raise ValueError("hi must be greater than lo")

    @property
    def n_params(self) -> int:
        return _N_REGIMES

    def clip(self, theta) -> np.ndarray:
        try:
            values = np.asarray(theta)
        except (TypeError, ValueError) as exc:
            raise ValueError("theta must be a finite one-dimensional numeric vector") from exc
        if (values.ndim != 1 or not 1 <= values.size <= self.n_params
                or values.dtype.kind not in "fiu" or not np.all(np.isfinite(values))):
            raise ValueError(f"theta must contain 1 to {self.n_params} finite numeric values")
        return np.clip(values.astype(np.float64), self.lo, self.hi)

    def to_policy(self, theta) -> np.ndarray:
        """``theta`` (one multiplier per regime) -> the ``policy_params`` array the
        kernel indexes by regime index. Short vectors are padded with the unit
        multiplier clipped into [lo, hi]; excess components are rejected."""
        t = self.clip(theta)
        arr = np.full(_POLICY_LEN, np.clip(1.0, self.lo, self.hi), dtype=np.float64)
        arr[: min(t.shape[0], _POLICY_LEN)] = t[:_POLICY_LEN]
        return arr

    def x0(self) -> np.ndarray:
        """Unit-multiplier baseline clipped to the declared bounds."""
        return np.full(self.n_params, np.clip(1.0, self.lo, self.hi), dtype=np.float64)


# --------------------------------------------------------------------------- #
# The renewal objective under survival constraints (§16.2)                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenewalObjective:
    """Scalar objective to **maximize**: renewal reward rate under survival
    constraints (§16.2). ``E[R]/E[T]`` (reward per cadence-estimated week) with soft-penalty
    constraints — ``P(profitable attempt) ≥ p_min`` and ``P(rule breach) ≤
    max_breach_rate`` — folded in as a large linear penalty on violation, so an
    violation affects ranking. A finite linear penalty does not guarantee every
    feasible policy outranks every infeasible one. This is one optional objective,
    not a universal replacement for finite-horizon cash or outcome-tail risk.

    **The defaults leave the constraints inert** (``p_min=0``, ``max_breach_rate=1``),
    so the out-of-the-box objective is the bare renewal rate ``E[R]/E[T]``. That is a
    deliberate opt-in: the ``r → 0`` exploit is closed by the *feasibility projection*
    (a withered attempt earns nothing), not by these penalties, so they are only
    needed when a caller wants to additionally forbid, say, high-breach policies.
    Set ``p_min`` / ``max_breach_rate`` to activate them.

    ``include_fees=False`` drops BOTH fee terms (eval + activation) from the reward, so
    the value is the pure conditional funded rate ``E[net_payout]/E[T]`` — used for the
    fee-free comparison when explicitly requested. It does not justify factoring
    a whole-account objective into separate phase objectives.

    **``cvar_q`` — robust (tail-scored) renewal (§16.11).** With ``cvar_q >= 1`` (default)
    the rate is the nominal ``E[R]/E[T]``. With ``cvar_q < 1`` it is instead the **CVaR of
    the left tail** of the renewal rate's *bootstrap* distribution: resample the per-attempt
    ``(reward, time)`` pairs ``n_boot`` times, recompute the RATIO ``ΣR*/ΣT*`` on each
    resample (ratio-correct — never a quantile of per-path ``R_i/T_i``, whose ``T_i≈0``
    blows up), and average the worst ``cvar_q`` fraction. This does not exist to force
    ceiling-sizing down: it *separates* two policies that look identical under the mean —
    a genuine convexity optimum (ceiling-sizing that wins across resamplings, keeps its
    value under tail-scoring) from a mean-blind-to-tail artifact (looks optimal only
    because a few lucky realizations hide a fat left tail, which the CVaR reveals). The
    bootstrap is **CRN-seeded** (``boot_seed`` fixed) so every candidate in a generation
    is scored on the same resample indices — the objective *difference* reflects the
    policy, not the bootstrap RNG."""

    p_min: float = 0.0  # minimum acceptable P(profitable attempt); 0 = inert
    max_breach_rate: float = 1.0  # maximum acceptable P(terminal rule breach); 1 = inert
    penalty: float = 1e6  # weight on constraint violation (dwarfs any real rate)
    include_fees: bool = True  # False -> pure funded rate (no eval/activation fee), §16.10
    cvar_q: float = 1.0  # <1 -> CVaR of the left q-tail of the bootstrap rate; >=1 -> mean
    n_boot: int = 200  # bootstrap resamples of the (reward, time) pairs (cvar_q<1 only)
    boot_seed: int = 12345  # CRN seed: identical resample indices for every candidate

    def __post_init__(self):
        for name in ("p_min", "max_breach_rate"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name, positive in (("penalty", False), ("cvar_q", True)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                    or value < 0 or positive and value == 0):
                raise ValueError(f"invalid {name}")
        if type(self.include_fees) is not bool:
            raise ValueError("include_fees must be bool")
        for name, minimum in (("n_boot", 1), ("boot_seed", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"invalid {name}")

    def breach_rate(self, o) -> float:
        code = o.code
        n = code.shape[0]
        if n == 0:
            return float("nan")
        return float(np.mean((code >= _FAIL_LO) & (code < _TIMED_OUT)))

    def _rate(self, reward, time):
        """The renewal rate: nominal ``E[R]/E[T]`` (``cvar_q>=1``), else the CVaR of the
        left ``cvar_q`` tail of the ratio's bootstrap distribution (§16.11). Returns
        ``None`` when undefined (non-positive total time / empty tail)."""
        reward, time = np.asarray(reward, dtype=float), np.asarray(time, dtype=float)
        if (reward.ndim != 1 or reward.shape != time.shape
                or not np.all(np.isfinite(reward)) or not np.all(np.isfinite(time)) or np.any(time < 0)):
            raise ValueError("reward/time must be aligned finite vectors with nonnegative time")
        if not len(time):
            return None
        t_mean = float(np.mean(time))
        if t_mean <= 0.0:
            return None
        if self.cvar_q >= 1.0:
            return float(np.mean(reward)) / t_mean  # E[R]/E[T] == renewal.r_renewal
        n = reward.shape[0]
        if n == 0:
            return None
        rng = np.random.default_rng(self.boot_seed)  # CRN: same indices for every candidate
        r = np.empty(int(self.n_boot))
        batch = max(1, 1_000_000 // n)
        for first in range(0, len(r), batch):
            stop = min(first+batch, len(r))
            idx = rng.integers(0, n, size=(stop-first, n))
            den = time[idx].sum(axis=1)
            if np.any(den <= 0):
                return None
            r[first:stop] = reward[idx].sum(axis=1) / den
        mass = self.cvar_q * len(r)
        whole = int(np.floor(mass))
        if not whole:
            return float(r.min())
        ordered = np.partition(r, min(whole, len(r)-1))
        return float((ordered[:whole].sum() + (mass-whole)*ordered[min(whole, len(r)-1)]) / mass)

    def value(self, o) -> float:
        """The scalar to maximize (rate minus constraint penalties).

        Computes the shared derived arrays (attributable fee, per-attempt reward and
        time) ONCE — ``r_renewal``/``prob_profitable``/``breach_rate`` would each
        re-derive the fee otherwise — matching their definitions exactly."""
        if o.net_payout.size == 0:
            raise ValueError("renewal rate is undefined for empty outcomes")
        # eval_fee + activation_fee*reached_funded (§H1); dropped for the funded leg.
        fee = attributable_fee(o) if self.include_fees else 0.0
        reward = o.net_payout - fee
        time = o.total_trading_days.astype(np.float64) / o.trading_days_per_week
        rate = self._rate(reward, time)  # nominal E[R]/E[T] or CVaR of the left tail
        if rate is None or not np.isfinite(rate):
            raise ValueError("renewal rate is undefined; use a cash objective or a supported positive-time model")
        pen = 0.0
        pp = float(np.mean(o.net_payout > fee))  # == statistics.prob_profitable
        if np.isfinite(pp) and pp < self.p_min:
            pen += self.penalty * (self.p_min - pp)
        br = self.breach_rate(o)
        if np.isfinite(br) and br > self.max_breach_rate:
            pen += self.penalty * (br - self.max_breach_rate)
        return rate - pen


# --------------------------------------------------------------------------- #
# Evaluation (Common Random Numbers via a fixed seed)                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PolicyEvaluation:
    """Retained evaluation evidence; direct arrays are copied and read-only.

    Metadata is descriptive, not a signed data provenance/certification record.
    Caller-owned provenance objects and nested feasibility diagnostics are not
    deeply frozen. Custom resampler state is not serialized.
    """
    score: float
    outcomes: Outcomes
    theta: np.ndarray
    policy: np.ndarray
    seed: int
    n_paths: int
    run_parameters: tuple
    data_summary: tuple


def _readonly_copy(values):
    result = np.array(values, copy=True)
    result.flags.writeable = False
    return result


def _evaluation_record(score, outcomes, theta, policy, config, dataset):
    arrays = {f.name: _readonly_copy(getattr(outcomes, f.name))
              for f in fields(outcomes) if isinstance(getattr(outcomes, f.name), np.ndarray)}
    snapshot = replace(outcomes, **arrays)
    parameters = tuple((name, getattr(config, name)) for name in (
        "n_paths", "L_eval", "L_funded", "seed", "size_base", "trade_cost",
        "start_equity", "version", "session_reset", "session_timezone", "batch_size",
        "intraday_mode"))
    parameters += (
        ("resampler", type(config.resampler).__module__ + "." + type(config.resampler).__qualname__),
        ("mean_block", config.resampler.mean_block if type(config.resampler) is StationaryDayBootstrap else None),
        ("resampler_state_complete", type(config.resampler) in (IIDDayBootstrap, StationaryDayBootstrap)),
    )
    clocks = dataset.exit_timestamps
    summary = (
        ("n_days", dataset.n_days), ("n_trades", dataset.n_trades),
        ("first_exit", str(clocks[0]) if clocks is not None else None),
        ("last_exit", str(clocks[-1]) if clocks is not None else None),
        ("clock_basis", "UTC" if dataset.session_timezone is not None else "caller-local"),
        ("session_timezone", dataset.session_timezone), ("session_reset", dataset.session_reset),
        ("trading_days_per_week", dataset.trading_days_per_week),
        ("cadence_source", dataset.cadence_source),
        ("execution_model", outcomes.execution_model),
        ("intraday_mode", outcomes.intraday_mode),
        ("approximation_reasons", outcomes.approximation_reasons),
    )
    return PolicyEvaluation(score, snapshot, _readonly_copy(theta), _readonly_copy(policy),
                            int(config.seed), int(config.n_paths), parameters, summary)


def evaluate_policy(engine, account, dataset, config, theta, space, objective,
                    feasibility=None, *, prepared=None, path_cache=None,
                    return_details=False) -> float | PolicyEvaluation:
    """Objective value of one policy ``theta`` on ``dataset`` under ``config``.

    Determinism is the Common-Random-Numbers device (§16.6): because the resampled
    paths are generated from ``config.seed``, calling this with the *same* config on
    two candidates couples their random draws within one resampling model.
    This can reduce comparison variance; it does not eliminate simulation noise.

    ``prepared`` (an :class:`~propfirm_engine.engine.PreparedRun`) and ``path_cache``
    (a dict) are the sweep fast-path: they skip the per-candidate validate/compile
    and reuse random day indices across candidates. Omitting
    them falls back to a full :meth:`Engine.run` — identical result either way."""
    policy = space.to_policy(theta)
    if prepared is not None:
        o = engine.run_prepared(prepared, dataset, config, policy_params=policy,
                                feasibility=feasibility, path_cache=path_cache)
    else:
        o = engine.run(account, dataset, config, policy_params=policy,
                       feasibility=feasibility)
    score = float(objective.value(o) if hasattr(objective, "value") else objective(o))
    if not np.isfinite(score):
        raise ValueError("objective must return a finite scalar")
    if return_details:
        return _evaluation_record(score, o, theta, policy, config, dataset)
    return score


# --------------------------------------------------------------------------- #
# CMA-ES — compact (μ/μ_w, λ) with step-size control (§16.6)                    #
# --------------------------------------------------------------------------- #


@dataclass
class CMAResult:
    x: np.ndarray  # best mean found (compact theta)
    score: float  # objective (maximization) at x, on the search data
    history: list = field(default_factory=list)  # best score per generation


class CMAES:
    """A compact (μ/μ_w, λ)-CMA-ES for a *maximization* objective (Hansen's standard
    update; we internally minimize ``-f``). Chosen over gradient/grid/RL because the
    objective is a noisy, non-differentiable, moderate-dim black box (§16.6). Small
    and dependency-free — adequate for the Tier-1 parameter count (~4)."""

    def __init__(self, x0, sigma0, *, popsize=None, bounds=None, seed=0,
                 max_gen=30):
        self.x = np.asarray(x0, dtype=np.float64)
        self.N = self.x.shape[0]
        self.sigma = float(sigma0)
        self.bounds = bounds
        self.max_gen = max_gen
        self.rng = np.random.default_rng(seed)

        self.lam = popsize or (4 + int(3 * np.log(self.N)))
        self.mu = self.lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w / w.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)

        N = self.N
        self.cc = (4 + self.mueff / N) / (N + 4 + 2 * self.mueff / N)
        self.cs = (self.mueff + 2) / (N + self.mueff + 5)
        self.c1 = 2 / ((N + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((N + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (N + 1)) - 1) + self.cs
        self.chiN = np.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

        self.pc = np.zeros(N)
        self.ps = np.zeros(N)
        self.C = np.eye(N)

    def _clip(self, x):
        if self.bounds is None:
            return x
        lo, hi = self.bounds
        return np.clip(x, lo, hi)

    def optimize(self, f) -> CMAResult:
        """Maximize ``f`` (a callable ``theta -> float``)."""
        best_x, best_score = self.x.copy(), -np.inf
        history = []
        for _gen in range(self.max_gen):
            # sample lambda candidates ~ N(x, sigma^2 C)
            try:
                A = np.linalg.cholesky(self.C)
            except np.linalg.LinAlgError:
                self.C = np.eye(self.N)
                A = np.eye(self.N)
            zs = self.rng.standard_normal((self.lam, self.N))
            ys = zs @ A.T
            xs = self._clip(self.x + self.sigma * ys)
            scores = np.array([f(x) for x in xs])  # maximization scores

            order = np.argsort(-scores)  # best (highest) first
            xs, ys, scores = xs[order], ys[order], scores[order]
            if scores[0] > best_score:
                best_score, best_x = scores[0], xs[0].copy()
            history.append(float(scores[0]))

            # recombination
            x_old = self.x.copy()
            self.x = np.sum(self.weights[:, None] * xs[:self.mu], axis=0)
            y_w = np.sum(self.weights[:, None] * ys[:self.mu], axis=0)

            # step-size path
            try:
                C_invsqrt = self._inv_sqrt(self.C)
            except np.linalg.LinAlgError:
                C_invsqrt = np.eye(self.N)
            self.ps = ((1 - self.cs) * self.ps
                       + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (C_invsqrt @ y_w))
            ps_norm = np.linalg.norm(self.ps)
            hsig = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (_gen + 1))) / self.chiN < 1.4 + 2 / (self.N + 1)

            # covariance path + rank-1/rank-mu update
            self.pc = ((1 - self.cc) * self.pc
                       + (1.0 if hsig else 0.0)
                       * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w)
            rank_mu = np.zeros((self.N, self.N))
            for i in range(self.mu):
                rank_mu += self.weights[i] * np.outer(ys[i], ys[i])
            ch = (1 - hsig) * self.cc * (2 - self.cc)
            self.C = ((1 - self.c1 - self.cmu) * self.C
                      + self.c1 * (np.outer(self.pc, self.pc) + ch * self.C)
                      + self.cmu * rank_mu)
            self.C = np.triu(self.C) + np.triu(self.C, 1).T  # keep symmetric

            # step-size update
            self.sigma *= np.exp((self.cs / self.damps) * (ps_norm / self.chiN - 1))
            if not np.isfinite(self.sigma) or self.sigma <= 0:
                self.sigma = 1e-8

        # ensure the returned mean is evaluated too (recombination may beat samples)
        mean_score = f(self._clip(self.x))
        if mean_score > best_score:
            best_score, best_x = mean_score, self._clip(self.x).copy()
        return CMAResult(x=best_x, score=float(best_score), history=history)

    @staticmethod
    def _inv_sqrt(C):
        vals, vecs = np.linalg.eigh(C)
        vals = np.maximum(vals, 1e-14)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


# --------------------------------------------------------------------------- #
# Optimize (search on ONE partition) — no reported number here (§16.7)         #
# --------------------------------------------------------------------------- #


@dataclass
class OptConfig:
    """Search knobs, distinct from the model :class:`RunConfig`."""

    sigma0: float = 0.25  # initial search spread as a FRACTION of policy-space width
    max_gen: int = 20
    popsize: int | None = None
    seed: int = 0
    screen_paths: int = 400  # cheap fidelity for the CMA-ES inner loop
    select_paths: int = 4000  # full fidelity for final selection (§16.6 guardrail)
    finalists: int = 8  # top distinct screening candidates, plus incumbent and baseline

    def __post_init__(self):
        self.validate()

    def validate(self):
        for name in ("max_gen", "seed", "screen_paths", "select_paths", "finalists", "popsize"):
            value = getattr(self, name)
            if name == "popsize" and value is None:
                continue
            minimum = 0 if name in ("max_gen", "seed") else (2 if name == "popsize" else 1)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if (isinstance(self.sigma0, bool) or not isinstance(self.sigma0, Real)
                or not np.isfinite(self.sigma0) or self.sigma0 <= 0):
            raise ValueError("sigma0 must be a finite positive fraction of policy-space width")


def _mll_amount(account):
    """The account's tightest trailing-drawdown amount ($), or None if it has none."""
    from .rules import TrailingDrawdownRule
    amts = [r.amount for ph in account.phases for r in ph.rules
            if isinstance(r, TrailingDrawdownRule)]
    return min(amts) if amts else None


def policy_space_for(account, size_base, feasibility=None):
    """A :class:`PolicySpace` whose maximum multiplier is the one that risks the
    **entire MLL** in a single worst-case trade — an account property, so the
    reachable *effective* size (``size_base × multiplier``) is
    ``[0, MLL / stop-per-unit]`` **independent of ``size_base``** (§16.3, the
    R-normalized-policy intent). Falls back to a plain ``[0, 5]`` space when the
    account has no trailing floor to normalize against."""
    mll = _mll_amount(account)
    if mll is None or size_base <= 0:
        return PolicySpace()
    unit = feasibility.unit_loss if feasibility is not None else 1.0  # worst loss / unit
    # Multiplier at which one stop = the MLL. NO floor: clamping hi up to 1.0 would let
    # size_base > MLL bet MORE than the MLL (unrealistic) and break size_base-invariance
    # — the reachable *effective* size must stay exactly [0, MLL/unit] for every size_base.
    hi = (mll / unit) / size_base
    return PolicySpace(lo=0.0, hi=max(hi, 1e-9))


def optimize(account, train_dataset, run_config, *, space=None, objective=None,
             opt_config=None, feasibility=None, engine=None, prepared=None):
    """Run CMA-ES on the **training** partition only and return the best policy.

    This function reports *no* performance number to trust — it only searches. The
    inner loop uses the cheap ``screen_paths`` fidelity (multi-fidelity screening,
    §16.6) with Common Random Numbers (a fixed seed → every candidate on the same
    paths). Final selection among the incumbent(s) is re-scored at the full
    ``select_paths`` fidelity, never the optimistic screening one (the §G1/§I1 trap).
    The returned score is a *training* score; a trustworthy number comes only from
    :func:`walk_forward` on held-out data."""
    # Default to an ACCOUNT-AWARE multiplier bound: the ceiling is the size that
    # risks the whole MLL in one trade, so the reachable effective size does not
    # depend on the size_base knob (§16.3). A caller-supplied space is respected.
    space = space if space is not None else policy_space_for(
        account, run_config.size_base, feasibility)
    objective = objective if objective is not None else RenewalObjective()
    oc = opt_config or OptConfig()
    oc.validate()
    engine = engine or Engine()
    # Prepare the account ONCE (validate/fingerprint/compile is policy-independent,
    # §18) and share one path cache so every candidate reuses the resampled-path
    # indices instead of regenerating them; trade gathering stays compiled.
    prep = prepared if prepared is not None else engine.prepare(account, run_config)
    path_cache: dict = {}

    # CRN: fix the seed so every candidate sees the same screening paths.
    screen_cfg = _with(run_config, n_paths=oc.screen_paths, seed=oc.seed)

    screened = {}

    def f(theta):
        candidate = space.clip(theta).copy()
        score = evaluate_policy(engine, account, train_dataset, screen_cfg, candidate,
                                space, objective, feasibility, prepared=prep,
                                path_cache=path_cache)
        screened[tuple(candidate)] = (score, candidate)
        return score

    # Start the search from the MIDDLE of the range with a step scaled to the range,
    # so the whole (now account-sized) [lo, hi] window is explored in a fixed
    # generation budget regardless of how wide it is — otherwise a search started at
    # the neutral multiplier=1 barely probes a wide bound, and the result would still
    # track size_base. The neutral baseline still anchors final selection below.
    mid = space.lo + (space.hi - space.lo) / 2.0
    x_start = np.full(space.n_params, mid, dtype=np.float64)
    # Step scaled PURELY to the range (no floor): a floor would make the multiplier-
    # space search stop scaling proportionally with size_base and the result would
    # once again track size_base. In effective ($/R) terms this is a fixed step.
    sigma0 = (space.hi - space.lo) * oc.sigma0
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 times policy-space width must be finite and positive")
    es = CMAES(x_start, sigma0, popsize=oc.popsize,
               bounds=(space.lo, space.hi), seed=oc.seed, max_gen=oc.max_gen)
    res = es.optimize(f)

    # Select at full fidelity on TRAINING data, not the final holdout. Keep a
    # bounded shortlist so a screening runner-up can win on the fresh CRN paths.
    # Baseline participates; equal final scores prefer it deterministically.
    if isinstance(oc.finalists, bool) or not isinstance(oc.finalists, (int, np.integer)) or oc.finalists < 1:
        raise ValueError("finalists must be a positive integer")
    select_cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 1)
    base_theta = space.clip(space.x0())
    candidates = [base_theta, space.clip(res.x)]
    candidates.extend(item[1] for item in sorted(
        screened.values(), key=lambda item: item[0], reverse=True)[:oc.finalists])
    unique = {tuple(candidate): candidate for candidate in candidates}
    scores = []
    best_theta, best_score = base_theta, -np.inf
    for candidate in unique.values():
        score = evaluate_policy(engine, account, train_dataset, select_cfg,
                                candidate, space, objective, feasibility,
                                prepared=prep, path_cache=path_cache)
        scores.append((candidate.copy(), float(score)))
        if score > best_score:
            best_theta, best_score = candidate.copy(), float(score)
    return OptOutcome(theta=best_theta, policy=space.to_policy(best_theta),
                      train_score=best_score, baseline_score=scores[0][1],
                      history=res.history, space=space, selection_scores=scores)


@dataclass
class OptOutcome:
    theta: np.ndarray
    policy: np.ndarray  # the length-_POLICY_LEN array for Engine.run(policy_params=)
    train_score: float
    baseline_score: float
    history: list
    space: PolicySpace
    selection_scores: list = field(default_factory=list)  # training-only finalists


# --------------------------------------------------------------------------- #
# Walk-forward / nested OOS — the ONLY reported number (§16.7)                  #
# --------------------------------------------------------------------------- #


def walk_forward(account, train_dataset, test_dataset, run_config, *, space=None,
                 objective=None, opt_config=None, feasibility=None, engine=None,
                 ladder_datasets=None, fitter=None):
    """Optimize on ``train_dataset``, report **only** the held-out score(s).

    Nested / walk-forward out-of-sample is non-negotiable (§16.7): the optimizer is
    fitted on ``train_dataset`` and never sees ``test_dataset``; the reported number
    is computed on the test partition alone, at full ``select_paths`` fidelity — never
    the cheap screening fidelity the search used (the §16.6/§I1 trap). If
    ``ladder_datasets`` (a dict ``name -> dataset`` of the generator ladder, §G1) is
    given, the held-out score is additionally reported as a **band across the whole
    ladder** on data the search never saw.

    Scope note (Tier-1): the policy is *selected* on the training partition under a
    single generator (``config.resampler``) at full fidelity — not chosen by its
    held-out ladder performance. Selecting across the ladder (BUILD_SPEC Step 14 item
    5 in its strongest form) is a Tier-2+ refinement; here the ladder is a held-out
    *reporting* band, and the reported numbers are never taken from the screening
    fidelity."""
    space = space if space is not None else policy_space_for(
        account, run_config.size_base, feasibility)
    objective = objective if objective is not None else RenewalObjective()
    oc = opt_config or OptConfig()
    oc.validate()
    engine = engine or Engine()
    # One prepared account for the whole walk-forward (search + OOS), §18.
    prep = engine.prepare(account, run_config)

    # ``fitter`` lets a caller swap in a different search (e.g. factorized_optimize)
    # while keeping the OOS contract identical — the held-out score below is always the
    # whole-cycle objective on data the search never saw.
    fit = (fitter or optimize)(account, train_dataset, run_config, space=space,
                               objective=objective, opt_config=oc, feasibility=feasibility,
                               engine=engine, prepared=prep)

    # OOS evaluation at full fidelity on data the search never touched. A distinct
    # seed from any used in the search, so the held-out paths are genuinely fresh.
    # The fitted policy and the baseline share one cache (same dataset+cfg → the
    # second eval reuses the first's paths).
    oos_cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 991)
    oos_cache: dict = {}
    oos_evaluation = evaluate_policy(engine, account, test_dataset, oos_cfg, fit.theta,
                                space, objective, feasibility, prepared=prep,
                                path_cache=oos_cache, return_details=True)
    baseline_evaluation = evaluate_policy(engine, account, test_dataset, oos_cfg,
                                   space.x0(), space, objective, feasibility,
                                   prepared=prep, path_cache=oos_cache, return_details=True)

    ladder_band = None
    ladder_evaluations = {}
    if ladder_datasets:
        ladder_band = {}
        for k, (name, ds) in enumerate(ladder_datasets.items()):
            cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 991 + k + 1)
            evaluation = evaluate_policy(engine, account, ds, cfg, fit.theta,
                                                space, objective, feasibility,
                                                prepared=prep, return_details=True)
            ladder_band[name] = evaluation.score
            ladder_evaluations[name] = evaluation

    return WalkForwardResult(
        theta=_readonly_copy(fit.theta), policy=_readonly_copy(fit.policy),
        train_score=fit.train_score, baseline_train_score=fit.baseline_score,
        oos_score=oos_evaluation.score, baseline_oos_score=baseline_evaluation.score,
        oos_ladder=ladder_band, history=list(fit.history),
        oos_evaluation=oos_evaluation, baseline_evaluation=baseline_evaluation,
        ladder_evaluations=ladder_evaluations,
    )


@dataclass
class WalkForwardResult:
    theta: np.ndarray
    policy: np.ndarray
    train_score: float
    baseline_train_score: float
    oos_score: float  # the honest headline: fitted on train, scored on held-out
    baseline_oos_score: float  # the neutral policy on the same held-out data
    oos_ladder: dict | None  # OOS score per generator rung, when a ladder is given
    history: list
    oos_evaluation: PolicyEvaluation | None = None
    baseline_evaluation: PolicyEvaluation | None = None
    ladder_evaluations: dict = field(default_factory=dict)

    @property
    def oos_improvement(self) -> float:
        """How much the fitted policy beats the neutral baseline out-of-sample — the
        only improvement number that is not an overfitting artifact (§16.7)."""
        return self.oos_score - self.baseline_oos_score


def _with(cfg: RunConfig, **changes) -> RunConfig:
    return replace(cfg, **changes)


# --------------------------------------------------------------------------- #
# Rolling time-separated out-of-sample (§16.7 hardened)                        #
# --------------------------------------------------------------------------- #


@dataclass
class RollingFold:
    """One walk-forward fold: fit on an earlier window, score on a strictly later one."""
    idx: int
    train_days: tuple  # (0, t) — expanding train, chronological
    test_days: tuple   # (t, t') — the held-out, strictly-later block
    oos: list          # OOS renewal score per seed
    baseline_oos: list  # neutral policy on the same held-out block, per seed
    theta: np.ndarray  # compatibility field: fitted policy for last seed only
    evaluations: tuple = ()  # complete WalkForwardResult for every seed, in input order


@dataclass
class RollingResult:
    """The OOS **distribution** across folds (and seeds), because a single held-out
    number on this data is dominated by which period it landed in. Read the median and
    the spread, never one point."""
    folds: list
    n_folds: int
    warmup_frac: float
    mean_block: float
    seeds: tuple

    @property
    def oos_all(self) -> list:
        return [v for f in self.folds for v in f.oos]

    @property
    def oos_median(self) -> float:
        return float(np.median(self.oos_all)) if self.oos_all else float("nan")

    @property
    def oos_iqr(self) -> tuple:
        a = self.oos_all
        return (float(np.percentile(a, 25)), float(np.percentile(a, 75))) if a else (float("nan"),) * 2

    @property
    def oos_spread(self) -> float:
        lo, hi = self.oos_iqr
        return hi - lo

    @property
    def fold_medians(self) -> list:
        return [float(np.median(f.oos)) for f in self.folds]

    @property
    def improvement_median(self) -> float:
        d = [v - b for f in self.folds for v, b in zip(f.oos, f.baseline_oos)]
        return float(np.median(d)) if d else float("nan")


def factorized_optimize(account, train_dataset, run_config, *, space=None, objective=None,
                        opt_config=None, feasibility=None, engine=None, prepared=None,
                        eval_grid=25, funded_objective=None):
    """Deprecated compatibility entry: jointly optimize the whole account.

    A fresh funded state does NOT make an arbitrary whole-cycle objective
    separable. The previous implementation ignored `objective` and substituted
    a conditional funded rate. This entry now delegates to `optimize`; it does
    not perform factorization. `eval_grid` is retained only for call compatibility
    and has no effect. Separate funded-only objectives are rejected explicitly.
    """
    if funded_objective is not None:
        raise ValueError("funded_objective is unsupported; supply the whole-account objective instead")
    warnings.warn(
        "factorized_optimize no longer factorizes: using joint whole-account optimize; "
        "eval_grid has no effect. Use optimize directly.",
        FutureWarning, stacklevel=2)
    return optimize(account, train_dataset, run_config, space=space, objective=objective,
                    opt_config=opt_config, feasibility=feasibility, engine=engine,
                    prepared=prepared)


def rolling_walk_forward(account, dataset, run_config, *, n_folds=4, warmup_frac=0.38,
                         mean_block=5.0, seeds=(0,), min_test_days=90, space=None,
                         objective=None, opt_config=None, feasibility=None, engine=None,
                         fitter=None):
    """Rolling **time-separated** OOS with a **block bootstrap** — the honest harness.

    ``dataset`` days are chronological, so the folds are genuine time splits: an
    *expanding* train window ``[0, t)`` and the strictly-later, non-overlapping test
    block ``[t, t')``. Reason (§16.7): on real data a single held-out score is dominated
    by which period it fell in — the same fitted policy can look great on one year and
    poor on the next — so we report the **distribution** of OOS across folds, not a point.

    Within each period the paths are resampled with a **stationary block bootstrap**
    (``mean_block`` > 1) instead of IID single days, so vol-clustering / drawdown-run
    persistence survives the resampling rather than being averaged away.

    ``n_folds`` and ``mean_block`` are explicit so their sensitivity is visible; the
    warm-up (first ``warmup_frac`` of days, train-only) keeps every test block strictly
    out-of-sample. Raises if a test block would be shorter than ``min_test_days`` (too
    short to hold a representative set of drawdown episodes)."""
    days = int(dataset.n_days)
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    warm = int(round(warmup_frac * days))
    test_len = (days - warm) // n_folds
    if test_len < min_test_days:
        raise ValueError(
            f"each test fold would be {test_len} days (< min_test_days={min_test_days}); "
            f"reduce n_folds or warmup_frac (n_days={days})")
    resampler = (StationaryDayBootstrap(float(mean_block))
                 if mean_block and mean_block > 1.0 else IIDDayBootstrap())
    cfg = _with(run_config, resampler=resampler)
    space = space if space is not None else policy_space_for(
        account, run_config.size_base, feasibility)
    objective = objective if objective is not None else RenewalObjective()
    engine = engine or Engine()

    folds = []
    for k in range(n_folds):
        t0 = warm + k * test_len
        t1 = days if k == n_folds - 1 else warm + (k + 1) * test_len
        train, test = slice_days(dataset, 0, t0), slice_days(dataset, t0, t1)
        oos, base, theta, evaluations = [], [], None, []
        for sd in seeds:
            oc = replace(opt_config, seed=sd) if opt_config is not None else OptConfig(seed=sd)
            wf = walk_forward(account, train, test, cfg, space=space, objective=objective,
                              opt_config=oc, feasibility=feasibility, engine=engine,
                              fitter=fitter)
            oos.append(float(wf.oos_score))
            base.append(float(wf.baseline_oos_score))
            theta = wf.theta
            evaluations.append(wf)
        folds.append(RollingFold(k, (0, t0), (t0, t1), oos, base, theta, tuple(evaluations)))
    return RollingResult(folds, n_folds, float(warmup_frac), float(mean_block), tuple(seeds))


__all__ = [
    "PolicySpace",
    "REGIME_LABELS",
    "policy_space_for",
    "RenewalObjective",
    "CMAES",
    "CMAResult",
    "OptConfig",
    "OptOutcome",
    "optimize",
    "walk_forward",
    "WalkForwardResult",
    "rolling_walk_forward",
    "RollingResult",
    "RollingFold",
    "factorized_optimize",
    "evaluate_policy",
    "PolicyEvaluation",
]
