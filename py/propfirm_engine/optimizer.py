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

from dataclasses import dataclass, field

import numpy as np

from .engine import Engine, RunConfig
from .enums import ExitCode
from .statistics import attributable_fee

_FAIL_LO = 10  # FAILURE_THRESHOLD
_TIMED_OUT = int(ExitCode.TIMED_OUT)

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

    @property
    def n_params(self) -> int:
        return _N_REGIMES

    def clip(self, theta) -> np.ndarray:
        return np.clip(np.asarray(theta, dtype=np.float64), self.lo, self.hi)

    def to_policy(self, theta) -> np.ndarray:
        """``theta`` (one multiplier per regime) -> the ``policy_params`` array the
        kernel indexes by regime index. A short ``theta`` is padded with 1.0 (neutral)
        so a length-1 baseline still reproduces constant size."""
        t = self.clip(theta)
        arr = np.ones(_POLICY_LEN, dtype=np.float64)
        arr[: min(t.shape[0], _POLICY_LEN)] = t[:_POLICY_LEN]
        return arr

    def x0(self) -> np.ndarray:
        """A neutral start: unit multiplier everywhere (the constant-size baseline)."""
        return np.ones(self.n_params, dtype=np.float64)


# --------------------------------------------------------------------------- #
# The renewal objective under survival constraints (§16.2)                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RenewalObjective:
    """Scalar objective to **maximize**: renewal reward rate under survival
    constraints (§16.2). ``E[R]/E[T]`` (reward per calendar week) with soft-penalty
    constraints — ``P(profitable attempt) ≥ p_min`` and ``P(rule breach) ≤
    max_breach_rate`` — folded in as a large linear penalty on violation, so an
    infeasible policy is ranked strictly below every feasible one without making the
    landscape discontinuous enough to break CMA-ES.

    Maximizing bare ``E[payout]`` is the wrong target (§16.2): it is blind to
    convexity, velocity and renewal structure. The ratio form fixes all three, and
    the constraints keep the search from buying rate with ruinous breach risk.

    **The defaults leave the constraints inert** (``p_min=0``, ``max_breach_rate=1``),
    so the out-of-the-box objective is the bare renewal rate ``E[R]/E[T]``. That is a
    deliberate opt-in: the ``r → 0`` exploit is closed by the *feasibility projection*
    (a withered attempt earns nothing), not by these penalties, so they are only
    needed when a caller wants to additionally forbid, say, high-breach policies.
    Set ``p_min`` / ``max_breach_rate`` to activate them."""

    p_min: float = 0.0  # minimum acceptable P(profitable attempt); 0 = inert
    max_breach_rate: float = 1.0  # maximum acceptable P(terminal rule breach); 1 = inert
    penalty: float = 1e6  # weight on constraint violation (dwarfs any real rate)

    def breach_rate(self, o) -> float:
        code = o.code
        n = code.shape[0]
        if n == 0:
            return float("nan")
        return float(np.mean((code >= _FAIL_LO) & (code < _TIMED_OUT)))

    def value(self, o) -> float:
        """The scalar to maximize (rate minus constraint penalties).

        Computes the shared derived arrays (attributable fee, per-attempt reward and
        time) ONCE — ``r_renewal``/``prob_profitable``/``breach_rate`` would each
        re-derive the fee otherwise — matching their definitions exactly."""
        if o.net_payout.size == 0:
            return -self.penalty
        fee = attributable_fee(o)  # eval_fee + activation_fee*reached_funded (§H1)
        reward = o.net_payout - fee
        time = o.total_trading_days.astype(np.float64) / o.trading_days_per_week
        t_mean = float(np.mean(time))
        if t_mean <= 0.0:
            return -self.penalty
        rate = float(np.mean(reward)) / t_mean  # E[R]/E[T] == renewal.r_renewal
        if not np.isfinite(rate):
            return -self.penalty
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


def evaluate_policy(engine, account, dataset, config, theta, space, objective,
                    feasibility=None, *, prepared=None, path_cache=None) -> float:
    """Objective value of one policy ``theta`` on ``dataset`` under ``config``.

    Determinism is the Common-Random-Numbers device (§16.6): because the resampled
    paths are generated from ``config.seed``, calling this with the *same* config on
    two candidates evaluates both on the *same* paths, so the objective *difference*
    is pure policy signal — never forced across generators, only within one.

    ``prepared`` (an :class:`~propfirm_engine.engine.PreparedRun`) and ``path_cache``
    (a dict) are the sweep fast-path: they skip the per-candidate validate/compile
    and reuse the resampled-path materialization across candidates (§18). Omitting
    them falls back to a full :meth:`Engine.run` — identical result either way."""
    policy = space.to_policy(theta)
    if prepared is not None:
        o = engine.run_prepared(prepared, dataset, config, policy_params=policy,
                                feasibility=feasibility, path_cache=path_cache)
    else:
        o = engine.run(account, dataset, config, policy_params=policy,
                       feasibility=feasibility)
    return objective.value(o)


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

    sigma0: float = 0.6
    max_gen: int = 20
    popsize: int | None = None
    seed: int = 0
    screen_paths: int = 400  # cheap fidelity for the CMA-ES inner loop
    select_paths: int = 4000  # full fidelity for final selection (§16.6 guardrail)


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
    objective = objective or RenewalObjective()
    oc = opt_config or OptConfig()
    engine = engine or Engine()
    # Prepare the account ONCE (validate/fingerprint/compile is policy-independent,
    # §18) and share one path cache so every candidate reuses the resampled-path
    # materialization instead of regenerating it.
    prep = prepared if prepared is not None else engine.prepare(account, run_config)
    path_cache: dict = {}

    # CRN: fix the seed so every candidate sees the same screening paths.
    screen_cfg = _with(run_config, n_paths=oc.screen_paths, seed=oc.seed)

    def f(theta):
        return evaluate_policy(engine, account, train_dataset, screen_cfg, theta,
                               space, objective, feasibility, prepared=prep,
                               path_cache=path_cache)

    # Start the search from the MIDDLE of the range with a step scaled to the range,
    # so the whole (now account-sized) [lo, hi] window is explored in a fixed
    # generation budget regardless of how wide it is — otherwise a search started at
    # the neutral multiplier=1 barely probes a wide bound, and the result would still
    # track size_base. The neutral baseline still anchors final selection below.
    mid = 0.5 * (space.lo + space.hi)
    x_start = np.full(space.n_params, mid, dtype=np.float64)
    # Step scaled PURELY to the range (no floor): a floor would make the multiplier-
    # space search stop scaling proportionally with size_base and the result would
    # once again track size_base. In effective ($/R) terms this is a fixed step.
    sigma0 = (space.hi - space.lo) / 4.0
    es = CMAES(x_start, sigma0, popsize=oc.popsize,
               bounds=(space.lo, space.hi), seed=oc.seed, max_gen=oc.max_gen)
    res = es.optimize(f)

    # Re-score at FULL fidelity (still training data), never the cheap screening
    # fidelity the incumbent was chosen on (§16.6/§I1 trap). We ALWAYS return the
    # incumbent — the best policy the search found — never falling back to the neutral
    # baseline. The baseline (multiplier 1 = a *constant size_base* bet) is
    # size_base-relative, so swapping to it would make the fitted policy track the
    # size_base knob; the incumbent is size_base-invariant (the search runs in a
    # multiplier range that scales as MLL/size_base, i.e. a fixed effective-$/R range).
    # The baseline is still scored, but only reported as the improvement reference.
    select_cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 1)
    incumbent = space.clip(res.x)
    base_theta = space.x0()
    incb_score = evaluate_policy(engine, account, train_dataset, select_cfg,
                                 incumbent, space, objective, feasibility,
                                 prepared=prep, path_cache=path_cache)
    base_full = evaluate_policy(engine, account, train_dataset, select_cfg,
                                base_theta, space, objective, feasibility,
                                prepared=prep, path_cache=path_cache)
    return OptOutcome(theta=incumbent, policy=space.to_policy(incumbent),
                      train_score=float(incb_score), baseline_score=float(base_full),
                      history=res.history, space=space)


@dataclass
class OptOutcome:
    theta: np.ndarray
    policy: np.ndarray  # the length-_POLICY_LEN array for Engine.run(policy_params=)
    train_score: float
    baseline_score: float
    history: list
    space: PolicySpace


# --------------------------------------------------------------------------- #
# Walk-forward / nested OOS — the ONLY reported number (§16.7)                  #
# --------------------------------------------------------------------------- #


def walk_forward(account, train_dataset, test_dataset, run_config, *, space=None,
                 objective=None, opt_config=None, feasibility=None, engine=None,
                 ladder_datasets=None):
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
    objective = objective or RenewalObjective()
    oc = opt_config or OptConfig()
    engine = engine or Engine()
    # One prepared account for the whole walk-forward (search + OOS), §18.
    prep = engine.prepare(account, run_config)

    fit = optimize(account, train_dataset, run_config, space=space,
                   objective=objective, opt_config=oc, feasibility=feasibility,
                   engine=engine, prepared=prep)

    # OOS evaluation at full fidelity on data the search never touched. A distinct
    # seed from any used in the search, so the held-out paths are genuinely fresh.
    # The fitted policy and the baseline share one cache (same dataset+cfg → the
    # second eval reuses the first's paths).
    oos_cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 991)
    oos_cache: dict = {}
    oos_score = evaluate_policy(engine, account, test_dataset, oos_cfg, fit.theta,
                                space, objective, feasibility, prepared=prep,
                                path_cache=oos_cache)
    baseline_oos = evaluate_policy(engine, account, test_dataset, oos_cfg,
                                   space.x0(), space, objective, feasibility,
                                   prepared=prep, path_cache=oos_cache)

    ladder_band = None
    if ladder_datasets:
        ladder_band = {}
        for k, (name, ds) in enumerate(ladder_datasets.items()):
            cfg = _with(run_config, n_paths=oc.select_paths, seed=oc.seed + 991 + k + 1)
            ladder_band[name] = evaluate_policy(engine, account, ds, cfg, fit.theta,
                                                space, objective, feasibility,
                                                prepared=prep)

    return WalkForwardResult(
        theta=fit.theta, policy=fit.policy,
        train_score=fit.train_score, baseline_train_score=fit.baseline_score,
        oos_score=float(oos_score), baseline_oos_score=float(baseline_oos),
        oos_ladder=ladder_band, history=fit.history,
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

    @property
    def oos_improvement(self) -> float:
        """How much the fitted policy beats the neutral baseline out-of-sample — the
        only improvement number that is not an overfitting artifact (§16.7)."""
        return self.oos_score - self.baseline_oos_score


def _with(cfg: RunConfig, **changes) -> RunConfig:
    from dataclasses import replace
    return replace(cfg, **changes)


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
    "evaluate_policy",
]
