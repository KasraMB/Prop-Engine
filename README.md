# propfirm_engine

**A high-performance Monte Carlo engine for valuing prop-firm trading accounts as convex structured products.**

`propfirm_engine` answers a deceptively hard question: *given a trading strategy's historical performance, what is a prop-firm evaluation account actually worth?* Not the marketing number, not the mean payout — the full distribution of outcomes, normalized by time, across a range of plausible futures, with the assumptions it rests on made explicit.

---

## Why this exists

A prop-firm account is not a trading account. It is a **structured financial product** with a floored downside and a capped, path-dependent upside:

- **Downside is bounded at the fee.** A −$1m day and a −$50k day are the same outcome — the account is lost and the evaluation fee is forfeit. Nothing worse can happen.
- **Upside is a capped, path-dependent sequence of payouts.** Profit is extracted through withdrawals, each subject to splits, per-request caps, buffers, and a finite payout count — until a trailing-drawdown knockout ends the account.

That framing changes everything. The account is a **path-dependent knockout**: the *order* of returns matters as much as their distribution. The same set of trading days in a different sequence can knock the account out on day 3 (worthless) or survive to five payouts (full value). A single expected-value number cannot describe such an instrument — its worth lives in the *shape* of the payoff distribution and the *velocity* at which capital is returned.

`propfirm_engine` is built from the ground up around this reality.

---

## What it does

- **Models real firm rules exactly.** Profit targets, trailing and static drawdowns, daily-loss limits, consistency rules, minimum trading days, winning-day payout gates, target adjustments, soft/hard breaches, and dual-axis timing (a floor that trails at end-of-day but breaches intraday) are all expressible as **configuration**, not new code.
- **Simulates at scale.** A compiled, JIT-accelerated kernel runs hundreds of thousands of resampled account attempts in parallel, retaining rich per-attempt outcomes — every payout, its timing, and the terminal state.
- **Reports the product, not a point estimate.** Probability of profit, the full payout-count distribution, payoff quantiles, return-on-fee, payout velocity, and time-to-first-payout — each as a distribution, each time-normalized to real calendar units.
- **Composes attempts into renewal economics.** Because the real activity is *repeated* — fail, pay another fee, retry — the engine analyzes the sequence as a renewal-reward process, producing the long-run reward rate and finite-horizon cashflow distributions that make otherwise-incomparable accounts comparable.
- **Quantifies its own uncertainty.** Every headline number is reported as a **model-sensitivity band** across a ladder of return-generating models, never as a single false-precision figure.

---

## Design philosophy

### Two systems, cleanly separated

```
   PROP-FIRM DSL              COMPILE            NUMERICAL ENGINE           STATISTICS
   (expressive,      ───►   (once, cached)  ───►  (NumPy + Numba,    ───►  (separate,
    object-oriented)                               zero Python objs)        swappable)
```

Firms are described in an expressive, Pythonic DSL. That description is **compiled once** into primitive integer-coded arrays. The hot loop never touches a Python object — it is a single predicate/action evaluator in which every rule (pass, fail, payout, target-adjust, soft breach, stage) is one instance of the same mechanism. Statistics are pure post-processing over raw outcomes and never require re-running the simulation.

The result: **adding a firm is writing configuration, not code.** A genuinely new mechanic is a bounded, registry-enforced addition to the kernel — and the registry refuses to simulate any rule it cannot yet execute, so nothing fails silently.

### Correctness before speed, trust before compute

The engine is governed by an explicit **trust hierarchy**. No layer is believed until the layers beneath it hold:

| Level | Question | Guarantee |
|------:|----------|-----------|
| **0** | Is the strategy frozen and not overfit to the data it's tested on? | Held-out validation; frozen spec; nested out-of-sample for the optimizer |
| **1** | Does the kernel reproduce the firm's rules *exactly*? | A slow, obviously-correct reference oracle that the fast kernel must match **bit-for-bit** |
| **2** | Does the trade data faithfully represent what could have happened? | Holding-interval-clipped adverse excursion; session-calendar day assignment |
| **3** | Does the conclusion survive plausible alternative futures? | A generator ladder (i.i.d. → block → stationary → regime → stochastic-vol), reported as a band |
| **4** | Does sizing remain valid away from historical position size? | A market-impact model before any scaled-up policy is trusted |
| **5** | Is the resulting return-on-fee actually attractive? | Renewal-and-survival objectives, evaluated only on unseen data |

The operating rule is simple: **do not spend compute before you have earned trust.** A million simulations of an unproven kernel, or an optimizer exploiting an overfit strategy, only produce a more precise wrong answer.

### Honest about what it cannot know

The engine ships with a companion catalogue of its own assumptions and limitations — where it is deliberately simplified, where a number could be confidently wrong, and where a result is conditional on a model choice. A bootstrap confidence interval is stated for exactly what it is: sampling uncertainty *conditional on one historical realization and one resampling model*, not a claim about the future. This candor is a feature, not a disclaimer.

---

## Architecture at a glance

```
propfirm_engine/
├── model.py         # Firm → Program → Variant → Account → Phase (frozen, hashable DSL)
├── rules.py         # Rule types + the registry that enforces kernel support
├── enums.py         # The integer vocabulary shared by the DSL and the kernel
├── config.py        # Firm configuration: expressive tables + scaling sugar
├── validate.py      # Role-aware sanity checks (permissive, not a rigid schema)
├── compiler.py      # DSL objects → compiled arrays; "compute only what is required"
├── fingerprint.py   # Structural hash = cache key = reproducibility record
├── data.py          # Raw trades → simulation-ready dataset (session days, excursions)
├── synthetic.py     # Parametric trade-stream generators for testing & research
├── kernels.py       # The JIT hot path: single-attempt + batched Monte Carlo
├── reference.py     # The pure-Python oracle the kernel is proven against
├── resampling.py    # Whole-day, joint-asset bootstraps (i.i.d. + stationary block)
├── simulate.py      # Batch orchestration → rich raw outcomes
├── statistics.py    # Decision statistics over outcomes (distribution + time axes)
├── renewal.py       # Repeated-attempt renewal-reward economics
├── engine.py        # Engine.run(): the orchestrator that ties it together
└── firms/           # One file per firm — pure configuration, no logic
```

---

## Roadmap

The build is sequenced so that each stage is provably correct before the next depends on it. The two hard gates — **oracle parity** (the kernel must match the reference) and **frozen-strategy discipline** (the input must not be overfit) — precede everything above them.

- [ ] **Foundations** — the integer vocabulary and the immutable DSL tree
- [ ] **Rules & registry** — the predicate/action rule set and the safety net
- [ ] **Data pipeline** — trade preprocessing, session calendar, adverse-excursion clipping
- [ ] **Synthetic generators** — known-property data streams that exercise the whole stack
- [ ] **Configuration & validation** — firms as pure config; role-aware validation
- [ ] **Compiler** — struct-of-arrays emission and the requirements resolver
- [ ] **Kernel + oracle** — *(Level-1 gate)* the fast kernel proven bit-for-bit against the reference
- [ ] **Fingerprint & caching** — reproducibility and no-recompute locked in early
- [ ] **Resampling** — whole-day, joint-asset i.i.d. and stationary-block bootstraps
- [ ] **Batch Monte Carlo & the engine** — the full orchestrated run
- [ ] **Decision statistics** — the distribution and time axes of a single attempt
- [ ] **Renewal economics** — the long-run rate and finite-horizon cashflow distribution
- [ ] **Generator ladder** — every headline number as a model-sensitivity band
- [ ] **Sizing optimizer** — a renewal-and-survival objective under strict out-of-sample discipline

---

## Documentation

| Document | Purpose |
|----------|---------|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | The complete design — the DSL, the predicate/action kernel, the payout schema, statistics, and renewal economics. **How it is built.** |
| [`BUILD_SPEC.md`](./BUILD_SPEC.md) | Each build step as a contract: the component, its observable behavior, and the test cases that prove it. **What must pass.** |
| [`MODEL_RISKS.md`](./MODEL_RISKS.md) | Every known assumption, simplification, and open decision, ranked by how much it distorts the numbers. **Where it is knowingly wrong.** |

---

## Status

**In active development.** The design is complete and the implementation is proceeding step by step against the contracts in `BUILD_SPEC.md`. Correctness gates are enforced before scale: no simulated number is trusted until the kernel has passed oracle parity, and no economic conclusion is drawn from a strategy that has not been frozen on held-out data.
