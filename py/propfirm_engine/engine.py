"""The engine — orchestration tying the pipeline together (ARCHITECTURE §17;
BUILD_SPEC Step 9).

``Engine.run`` is pure orchestration: validate → preprocess/cache → fingerprint →
compile/cache → per-phase **survivors-only** simulation → aggregate rich raw
per-attempt outcomes → return an :class:`Outcomes` object (which Step 10's
``Results`` wraps with lazy statistics).

Survivors-only (§17): the eval phase runs for all attempts; the funded phase runs
only for the attempts that **passed** it, each drawing its own independent funded
path (eval and funded may use different ``L``, §C7). A failed-eval attempt never
produces funded outcomes. A direct-funded account (``phases=(funded,)``) runs the
funded phase for every attempt (``reached_funded`` trivially true, §14.1).

Time spans the **whole attempt** (§14.4/§H4): ``total_trading_days`` sums the eval
and funded phases' days, because the fee is tied up from the eval's first day
until the attempt terminates. ``max_payouts`` comes from the funded phase's
schema, not the dataset (§H2). The default ``policy_params`` (length-1) reproduces
the constant-size case and threads unchanged to the kernel (§16.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cache import Caches
from .data import TradeDataset
from .enums import ExitCode
from .fingerprint import fingerprint
from .resampling import DayResampler, IIDDayBootstrap, materialize
from .simulate import simulate_phase_batch

_PASSED = int(ExitCode.PASSED)


@dataclass
class RunConfig:
    """Monte Carlo run parameters. ``L_eval``/``L_funded`` are the per-phase path
    lengths (§C7); ``batch_size`` is a memory knob only — because the resampled
    paths are generated once under ``seed``, it never changes results."""

    n_paths: int = 10_000
    L_eval: int = 60
    L_funded: int = 120
    seed: int = 0
    resampler: DayResampler = field(default_factory=IIDDayBootstrap)
    size_base: float = 1.0
    trade_cost: float = 0.0  # per-executed-trade cost ($), subtracted from each trade's P&L
    start_equity: float | None = None  # defaults to the account size
    version: str = "v1"
    session_reset: str = "17:00"
    batch_size: int = 10_000
    provenance: object = None  # caller-supplied input origin (e.g. "OOS 2023"), §G7


@dataclass
class Outcomes:
    """Rich per-attempt raw outcomes for a whole batch (§13, §14.4).

    Length-``B`` arrays plus the scalars the statistics layer needs. Nothing is
    pre-aggregated away: the payout-count support (``max_payouts``), the
    path-dependent fees, and the time axis all ride through.
    """

    code: np.ndarray  # int32[B] — final terminal ExitCode of the attempt
    reached_funded: np.ndarray  # bool[B] — did it clear the eval phase (§H1)
    net_payout: np.ndarray  # float64[B] — sum of net payouts (funded)
    payouts_taken: np.ndarray  # int32[B]
    first_payout_day: np.ndarray  # int32[B] — funded-phase day-index, -1 if none
    total_trading_days: np.ndarray  # int32[B] — eval + funded days (§H4)
    size: int  # account nominal balance
    size_base: float  # the $/unit-return scalar that actually drives P&L (§16.1)
    max_payouts: int  # from the funded PayoutSchema (§H2), 0 if no funded schema
    eval_fee: float
    activation_fee: float
    trading_days_per_week: float  # from the dataset, for the time axis (§11.5)
    fingerprint: str
    provenance: object = None  # the input's origin, carried through for reporting (§G7)
    feas_agg: object = None  # funded-phase FeasibilityAgg (§16.9), when active
    eval_feas_agg: object = None  # eval-phase FeasibilityAgg (§16.9) — eval usually
    #                              carries the tightest buffer, so its wither stats matter
    eval_trading_days: np.ndarray | None = None  # int32[B] — per-attempt eval-phase
    #    days (days-to-pass for survivors, days-to-fail otherwise); 0 for direct-funded

    @property
    def n_attempts(self) -> int:
        return int(self.code.shape[0])


@dataclass
class PreparedRun:
    """The per-account work of a run hoisted out of the Monte Carlo loop (§18).

    ``validate`` + ``fingerprint`` + compile/cache-lookup depend only on the account
    and ``config.version`` — not on the dataset, seed, path count, or sizing policy —
    so an optimizer sweeping hundreds of policies over one fixed account computes
    this ONCE via :meth:`Engine.prepare` and reuses it for every candidate."""

    fingerprint: str
    eval_ph: object
    funded_ph: object
    size: int
    eval_fee: float
    activation_fee: float
    max_payouts: int


class Engine:
    """Runs a compiled account over resampled Monte Carlo paths (§17)."""

    def __init__(self, caches: Caches | None = None):
        self.caches = caches if caches is not None else Caches()

    def prepare(self, account, config: RunConfig) -> PreparedRun:
        """Validate, fingerprint and compile the account once (§9/§10), returning a
        reusable :class:`PreparedRun` — the account-only prologue of :meth:`run`."""
        from .validate import validate

        validate(account)
        fp = fingerprint(account, config.version)
        compiled = self.caches.accounts.get(account, config.version, key=fp)
        eval_ph = next((p for p in compiled.phases if p.role == "eval"), None)
        funded_ph = next((p for p in compiled.phases if p.role == "funded"), None)
        max_payouts = (
            funded_ph.payout.max_payouts
            if funded_ph is not None and funded_ph.payout is not None
            else 0
        )
        return PreparedRun(fp, eval_ph, funded_ph, account.size, account.eval_fee,
                           account.activation_fee, max_payouts)

    def run(self, account, trades, config: RunConfig, policy_params=None,
            feasibility=None) -> Outcomes:
        prepared = self.prepare(account, config)
        # dataset: accept a ready TradeDataset, or preprocess (cached) from raw rows
        if isinstance(trades, TradeDataset):
            dataset = trades
        else:
            dataset = self.caches.trades.get(trades, session_reset=config.session_reset)
        return self.run_prepared(prepared, dataset, config, policy_params, feasibility)

    def _resample(self, path_cache, dataset, resampler, n_days, L, n_paths, seed):
        """Resampled day-index paths and (optionally) their materialization.

        With ``path_cache=None`` (a plain :meth:`run`) it returns the index paths and
        ``None`` materials — the batch then gathers per attempt exactly as before.
        With a ``path_cache`` dict (an optimizer sweep) it memoizes both the paths and
        their policy-independent materialization keyed on the resampler/dataset/shape/
        seed, so every candidate reuses them (§18)."""
        if path_cache is None:
            return resampler.generate(n_days, L, n_paths, seed), None
        key = (id(resampler), id(dataset), n_days, L, n_paths, seed)
        cached = path_cache.get(key)
        if cached is None:
            paths = resampler.generate(n_days, L, n_paths, seed)
            cached = (paths, materialize(dataset, paths))
            path_cache[key] = cached
        return cached

    def run_prepared(self, prepared: PreparedRun, dataset: TradeDataset,
                     config: RunConfig, policy_params=None,
                     feasibility=None, path_cache=None) -> Outcomes:
        """Run the Monte Carlo batch for an already-:meth:`prepare`d account. This is
        the hot entry the optimizer calls per candidate — no re-validate/fingerprint/
        compile, only the sizing-policy-dependent simulation. An optional
        ``path_cache`` dict reuses resampled paths + their materialization across
        candidates that share ``(resampler, dataset, shape, seed)`` (§18)."""
        policy = np.asarray(
            policy_params if policy_params is not None else [1.0], dtype=np.float64
        )
        fp = prepared.fingerprint

        start_equity = (
            config.start_equity if config.start_equity is not None else float(prepared.size)
        )
        b = config.n_paths
        n_days = dataset.n_days
        # Independent, well-separated seeds for the two phases (a plain seed+1 would
        # alias one run's funded stream onto the next run's eval stream in a sweep).
        eval_seed, funded_seed = (int(x) for x in np.random.SeedSequence(config.seed).generate_state(2))

        eval_ph = prepared.eval_ph
        funded_ph = prepared.funded_ph
        funded_feas_agg = None  # filled if the funded phase runs under feasibility
        eval_feas_agg = None

        code = np.zeros(b, dtype=np.int32)
        reached_funded = np.zeros(b, dtype=bool)
        net_payout = np.zeros(b, dtype=np.float64)
        payouts_taken = np.zeros(b, dtype=np.int32)
        first_payout_day = np.full(b, -1, dtype=np.int32)
        total_days = np.zeros(b, dtype=np.int32)
        eval_days = np.zeros(b, dtype=np.int32)  # per-attempt eval duration, for the offset

        # --- eval phase (all attempts) ---
        if eval_ph is not None:
            eval_paths, eval_mats = self._resample(
                path_cache, dataset, config.resampler, n_days, config.L_eval, b, eval_seed
            )
            er = simulate_phase_batch(
                eval_ph, dataset, eval_paths, config.size_base, policy, start_equity,
                config.batch_size, feasibility=feasibility, materials=eval_mats,
                trade_cost=config.trade_cost,
            )
            code[:] = er.code
            total_days[:] = er.total_trading_days
            eval_days[:] = er.total_trading_days
            eval_feas_agg = er.feas_agg  # eval-phase §16.9 aggregate (may be None)
            survivors = np.where(er.code == _PASSED)[0]
        else:
            survivors = np.arange(b)  # direct-funded: every attempt runs funded

        # --- funded phase (survivors only) ---
        if funded_ph is not None and survivors.shape[0] > 0:
            # Common Random Numbers across policies (§16.6): draw a funded path for
            # EVERY attempt keyed by its global index, then pick the survivors' rows —
            # so a given attempt always sees the SAME funded path no matter which
            # policy let it survive. Survivor-position indexing (generate n_surv rows)
            # would hand attempt #7 a different path when the survivor set changes,
            # breaking CRN exactly on the reward-producing leg. Only survivors are
            # simulated; generating b index rows is negligible beside the sims.
            funded_paths_all, funded_mats_all = self._resample(
                path_cache, dataset, config.resampler, n_days, config.L_funded, b, funded_seed
            )
            funded_paths = funded_paths_all[survivors]
            funded_mats = (None if funded_mats_all is None
                           else [funded_mats_all[i] for i in survivors])
            fr = simulate_phase_batch(
                funded_ph, dataset, funded_paths, config.size_base, policy, start_equity,
                config.batch_size, feasibility=feasibility, materials=funded_mats,
                trade_cost=config.trade_cost,
            )
            funded_feas_agg = fr.feas_agg
            reached_funded[survivors] = True
            code[survivors] = fr.code
            net_payout[survivors] = fr.net_payout
            payouts_taken[survivors] = fr.payouts_taken
            # first_payout_day is funded-phase-relative; offset by the attempt's eval
            # days so it is measured from ATTEMPT start, consistent with the whole-
            # attempt time axis (§H4). -1 (no payout) stays -1.
            fpd = fr.first_payout_day
            offset = np.where(fpd >= 0, eval_days[survivors] + fpd, -1)
            first_payout_day[survivors] = offset
            total_days[survivors] += fr.total_trading_days  # eval + funded (§H4)

        return Outcomes(
            code=code,
            reached_funded=reached_funded,
            net_payout=net_payout,
            payouts_taken=payouts_taken,
            first_payout_day=first_payout_day,
            total_trading_days=total_days,
            size=prepared.size,
            size_base=config.size_base,
            max_payouts=prepared.max_payouts,
            eval_fee=prepared.eval_fee,
            activation_fee=prepared.activation_fee,
            trading_days_per_week=dataset.trading_days_per_week,
            fingerprint=fp,
            provenance=config.provenance,
            feas_agg=funded_feas_agg,
            eval_feas_agg=eval_feas_agg,
            eval_trading_days=eval_days,
        )


__all__ = ["Engine", "RunConfig", "Outcomes", "PreparedRun"]
