# propfirm_engine — Model Risks, Assumptions & Open Design Holes

Companion to `ARCHITECTURE.md`. This catalogues where the engine is knowingly wrong, where it is silently assuming something, and where the spec is still incomplete. Context: the trading is **algorithmic/systematic**, so trader-psychology risks (tilt, freezing, revenge trading, discretion near the breach line) are out of scope by construction and are not listed here.

Each item is tagged:
- **[HOLE]** — genuinely undefined in the architecture; must be specified before the relevant part is built.
- **[WRONG-ON-PURPOSE]** — a deliberate simplification; acceptable if stated, dangerous if forgotten.
- **[CORRECTNESS]** — a latent bug risk in the design as written.
- **[FRAMING]** — the design is fine but the doc oversells it; correct the claim, not the code.

Severity is the practical one: how much it distorts the numbers you'll actually act on.

---

## 0. The governing frame: this is a convex structured product

The engine values a **structured product with a floored downside and a capped, path-dependent upside**. Downside is bounded at the fees paid (evaluation + any activation fee): a −$1m day and a −$50k day are the *same outcome* — account lost, fee forfeited. Upside is the sum of released payouts, subject to split, per-request cap, cap-fraction, buffer, and a finite payout count. This framing is not decoration; it re-weights every risk below.

Two consequences dominate everything else in this document:

- **Loss magnitude beyond the knockout boundary has no additional payoff consequence — but the frequency, timing, and cross-asset clustering of adverse observations remain economically material.** Once a path breaches, a −$10k and a −$1m loss are identical attempt economics, so under-sampling loss *depth past the barrier* is fine. What is *not* irrelevant is how often and when a path *reaches* the barrier: the frequency, timing, and same-day cross-asset clustering of adverse observations determine knockout probability and time-to-resolution, which are the whole game for a path-dependent barrier. So the structured-product insight removes the left-*depth* concern, not the extreme-event *arrival* concern — the latter is exactly what G1's dependence modeling must get right.
- **The account is a path-dependent knockout, so the upside leg and loss *ordering* are the whole game.** Payoff accrues as payouts until a trailing-drawdown knockout ends it. The value of a knockout depends on the *sequence* of returns, not just their distribution — the same days in a different order can knock out early (worthless) or survive to five payouts (full value). Therefore the two things that matter most are (1) the payout schema (§A, now specified) and (2) inter-day path dependence in resampling (§B1). Everything else is secondary to these.

Scope: **one account at a time.** Sequential *renewal* economics — fail → pay another fee → retry, analysed as a renewal-reward process above the engine (ARCHITECTURE §15) — is in scope, because it composes one-at-a-time completed attempt outcomes. **Simultaneous** correlated accounts (same strategy, same market, failing together) are a distinct portfolio layer and remain explicitly deferred and not modeled here. The per-account attempt is the simulation unit; the renewal sequence is the analysis unit; the portfolio is out of scope.

**Two reporting consequences of the frame (now built into ARCHITECTURE §14):**
- **The mean is an insufficient summary.** For a convex payoff, value lives in the distribution's shape, so `E[payout]` alone hides the product. The engine reports the payoff distribution and decision statistics — `P(profitable)` (not derivable from the mean), the payout-count histogram, payoff quantiles, and return-on-fee — with the mean as one number among them.
- **Payouts must be time-normalized, and *only* by calendar time.** The product is a rate: the fee is tied up until the account resolves, and stacking/rotating depends on velocity. Calendar time is reconstructed from trade cadence (trading-days-per-week, §11.5). **The tempting "fraction of the source dataset" proxy is explicitly rejected** — a bootstrap can produce a simulated path longer or shorter than its source history, so dataset-fraction measures simulation run-length, not product duration, and would mislead. Time-to-first-payout and payout-per-month/-year are the honest velocity statistics.

These re-point the deferred optimizer's objective (§15): maximizing bare `E[payout]` is wrong for a convex fee-per-attempt instrument because it is blind to both convexity and velocity; the objective should be distribution- and time-aware (e.g. return-on-fee per year, or expected payout subject to a `P(profitable)` floor).

---

## A. Payout leg — the product's upside (now specified)

The payout schema (ARCHITECTURE §6b) closes the two critical holes that previously made funded-phase output meaningless. They are retained here as resolved entries so the history and remaining care are visible.

### A1. Payout amount / profit-split schema  **[RESOLVED — see ARCHITECTURE §6b]**
Previously undefined. Now specified as `PayoutSchema` on the funded phase, grounded in the actual Lucid rules: per-payout-index dollar cap tuple (flat for Flex, stepping for Pro/Direct), a `cap_fraction` for Flex's "50% of profit up to" ceiling, `min_request` threshold, `buffer_floor` for the Pro/Daily non-withdrawable balance, and a possibly-tiered split for the grandfathered "100% first $10k" case. Payout amount is `net(split, min(dollar_cap[i], cap_fraction × cycle_profit))` gated by min-request and buffer. This is the product's upside leg and is now explicit rather than elided.

Remaining care: the schema is validated against Lucid; confirm each *new* firm's payout arithmetic maps onto these fields before assuming it fits (some firms may need a field not yet present — that is a bounded schema addition, caught the same way the rule registry catches new rule kinds).

### A2. Post-payout equity and drawdown-floor reset  **[RESOLVED — see ARCHITECTURE §6b.1]**
Previously undefined. Now an explicit config-controlled transition on the payout fire-action: `reset_fields` (counters zeroed), `withdraw_reduces_equity` (whether the paid amount leaves the balance, with retained above-cap profit staying as buffer per the Lucid "excess rolls forward" behavior), and `recompute_floor_on_payout` (whether the trailing floor re-derives off the post-withdrawal balance, with locked floors persisting). The design also captures the structurally important consequence that retained excess profit makes the account *safer* on the next cycle, which directly affects survival to later payouts.

Remaining care: whether the floor recomputes on payout is the single most consequential field and is firm-specific — verify it per account type, since it decides whether the second payout is easier or harder than the first.

### A3. Multiple distinct payout mechanisms  **[NON-ISSUE] — not applicable to any known firm**
No known prop firm runs two separate payout mechanisms on one account; every account type has a single payout path gated by one conjunction of conditions. The single-mechanism model is therefore treated as complete. Recorded only so that if a genuine multi-mechanism firm ever appears, the assumption is visible rather than silent — it is not an open design question.

---

## B. Statistical model risks — the numbers can be confidently wrong

### B1. Inter-day dependence is destroyed by day-block resampling  **[WRONG-ON-PURPOSE] — severity: critical**
Day-block resampling preserves structure *within* a day but shuffles days independently. It therefore destroys **inter-day dependence**: multi-day losing streaks, volatility clustering, autocorrelation, and mean-reversion-after-drawdown. Prop accounts die primarily from *streaks* against a trailing drawdown, and a hard drawdown barrier is acutely sensitive to the ordering and clustering of losses. If real drawdowns are streakier than independent day-shuffling reproduces — which is the norm for most strategies — the bootstrap **understates failure rates and overstates pass rates and payouts**. This is the single most impactful statistical assumption in the system.

**Mitigation path:** the stationary bootstrap's geometric block length should operate at the *day* level with a mean block length > 1 day, so multi-day blocks preserve some cross-day dependence, rather than shuffling single days. The mean block length becomes a modeling parameter that should be tuned to the strategy's actual autocorrelation, not left at 1. Regime-aware resampling (already deferred in the architecture) is the fuller fix. Until then, treat pass-rate and drawdown-survival numbers as optimistic bounds.

### B2. Bootstrap resamples the empirical distribution — matters on the RIGHT tail now  **[WRONG-ON-PURPOSE] — severity: moderate (re-pointed by §0)**
Originally flagged as a left-tail (catastrophic-loss) problem. Under the structured-product frame (§0) the left-tail *depth* is economically irrelevant — an unsampled −$1m day is no worse than a −$10k breach at the attempt level. (Note the §0 qualification: the *arrival* of adverse days — their frequency, timing, and cross-asset clustering — still matters for knockout probability; only loss depth past the barrier is void.) What *additionally* matters is the **right tail**: the bootstrap can only reproduce favorable streaks that occurred in-sample, and expected value is dominated by paths that survive to multiple payouts. If the data lacks long enough winning runs, the upside leg is under-valued; if a lucky in-sample run is over-represented in a short history, it is over-valued. So the tail concern flips toward the right and softens in severity, but does not vanish — it bears on payout frequency, while adverse *arrival* (not depth) still bears on knockout via G1.

**Mitigation path:** ensure sufficient history to represent realistic winning-streak lengths; report sensitivity of expected payout to block length (§B1). Parametric right-tail modeling is optional and lower priority than getting inter-day dependence right.

### B3. Finite trade history → block resampling reuses the same days  **[WRONG-ON-PURPOSE] — severity: moderate**
If the input history is short (a few hundred days), day-block resampling draws heavily from a small pool, so simulated paths are highly correlated across the Monte Carlo set. Confidence intervals will look tighter than they should — the effective sample size is the number of *distinct days*, not the number of simulations. Running 100k sims over 200 days of data does not give 100k independent pieces of evidence.

**Mitigation path:** report effective sample size / distinct-day count alongside CIs; widen CIs to reflect data-limited resampling, or use a subsampling correction.

### B4. Independence of the strategy's returns from account size  **[WRONG-ON-PURPOSE] — severity: high (once the optimizer runs)**
`trade_low = size × mae` and `pnl = size × ret` assume returns scale linearly with position size. This is fine at historical size but breaks under the optimizer, which will push size away from historical. Slippage, market impact, and fill quality are nonlinear in size; a doubled position does not experience a doubled MAE in practice. The further the sizing policy departs from historical size, the more the intraday low and realized P&L are **fiction that flatters larger sizes** (impact is underestimated, so big bets look safer and more profitable than they'd be).

**Mitigation path:** a size-impact model (even a simple concave slippage term) before trusting any optimized policy that scales size materially; bound the optimizer's size range to something near historical until such a model exists.

---

## C. Correctness risks in the design as written

### C1. `fastmath=True` on equality-boundary comparisons  **[CORRECTNESS] — severity: moderate**
Breach and target checks are equality-boundary comparisons (`equity <= floor`, `equity - start >= target`). `fastmath=True` permits floating-point reassociation and relaxed IEEE semantics, which can flip a result that sits exactly on the boundary. A breach that should/shouldn't fire by a fraction of a cent could invert. Presented in the architecture as a pure speed win, it is actually a correctness-vs-speed trade on exactly the comparisons that decide outcomes.

**Mitigation path:** either drop `fastmath` on the kernel that does breach comparisons, or introduce a small explicit epsilon and verify the epsilon (not the compiler flag) governs boundary behavior. Test both ways and confirm outcome parity.

### C2. Capped-out detection peeks at trade outcome — **[RESOLVED BY DEFERRAL]**
`_no_allowable_trade(equity, dd_floor, size)` as sketched decided disallowal using the trade's own `trade_low` — hindsight, not a pre-trade constraint. A real capped-out condition is about **minimum expressible position size vs remaining buffer**, and the model has no minimum-size concept while sizing is a constant. **Decision:** capped-out is **deferred with the sizing policy**. `CAPPED_OUT` remains a reserved code but is never emitted under a constant policy, the hindsight check is removed from the kernel, and it is **off the Step 6 boundary-test list** (which is why Step 6 is now writable — `BUILD_SPEC` A2). When the sizing policy lands, define capped-out against a configured minimum position size and worst-case per-unit loss evaluated *before* the trade, and add its boundary test then.

### C3. Drawdown-floor initialization and first-trade behavior  **[RESOLVED — decided]**
Previously the kernel sketch and this item disagreed (`- INF` sentinel vs `start − amount`). **Decision:** `peak = start_equity`, `dd_floor = start_equity - amount`, **live from trade 1** — a trailing-DD account can breach on its first trade, and there is no `-INF` sentinel or "seeded on first update" delay. For an EOD-*update* rule the floor still *advances* only at `_close_day`, but it *exists and can be breached* intraday on day 1 at its initial level. The `-INF` line is removed from the §12 sketch. Step 6 tests the first-trade and first-day boundaries against the oracle.

### C4. Within-trade event ordering is a silent precedence decision  **[CORRECTNESS/FRAMING] — severity: low-moderate**
On a single trade, equity can breach a drawdown *and* satisfy a payout *and* hit the target simultaneously. The kernel resolves this by check order (fail → adjust → pass → payout). That is a real, results-affecting precedence rule (a trade that both breaches and hits target is counted as a *failure*), but it's implicit. Correct for most firms — a breach on the same trade should kill regardless of target — but it should be stated and justified, not left to loop order.

**Mitigation path:** state the precedence explicitly in the architecture as a deliberate rule; confirm it matches each firm's actual adjudication (some firms honor a target hit that coincides with a soft breach differently).

### C5. Soft-breach day-truncation interaction with EOD rules  **[RESOLVED — decided]**
Two independent quantities must not be conflated: **breach detection reads the day's intraday low-water mark** (the floating P&L low, derived from `mae`), while **EOD-timed rules and the EOD floor update read the day's true closing equity**. A soft breach is *detected intraday* (the floating low crossed the line) and truncates the remaining same-day trades — but it does **not** redefine closing equity. **Decision:** closing equity is always the day's actual end-of-day equity (the equity after its last *executed* trade — which on a truncated day is the trade at which the soft breach fired, since subsequent trades are skipped), never a synthetic "breach-point" value. EOD rules on a truncated day evaluate against that real closing equity, with the same fold-then-evaluate order and fail→adjust→pass→payout precedence as any close (§C9). A soft-breached day: counts as a trading day, does *not* count as a winning day, has its partial loss stand, resumes next day. This is an exact Step 6 oracle case ("a day that soft-breaches *and* carries an EOD rule").

**Intraday detection is approximate and meant to be built out.** Because the intraday low comes from `mae` (or synthetic excursion), not a tick path, `check_timing=CONTINUOUS` detection is *non-exact* by construction — it catches a breach if the trade's floating low crosses the line, without knowing the exact moment. This is a deliberate, buildable-out surface: the two-timing-field model (§6a) already carries the `CONTINUOUS` path, so an intraday detector can be refined later without changing the interface; the baseline is that a breach is counted when the floating low crosses, and its fidelity is bounded by the `mae` input (`MODEL_RISKS.md` §D1).

### C6. The final day needs an explicit end-of-path close  **[CORRECTNESS] — severity: high, now specified**
The per-attempt loop closes a day only on rollover or soft breach; the last day has no rollover, so without an explicit close after the final trade its EOD checks, EOD floor update, and winning-day finalization never run. Concrete failures: an EOD-timed breach on the last day is missed; the last day never counts toward `N_QUALIFYING_DAYS`, so a payout whose final qualifying day *is* the last day is silently lost. **Fixed** in the kernel sketch (ARCHITECTURE §12) with an end-of-path `_close_day` whose terminal return is honoured, and pinned by the Step 6 "breach/qualify on the final trade of a day" tests. Listed because it is exactly the kind of boundary bug the oracle exists to catch.

### C7. Per-attempt path length `L` is an undefined modeling parameter  **[HOLE] — severity: high**
An attempt runs until pass/fail/terminal-payout or it *runs out of resampled days* (`TIMED_OUT`). The resampler emits `[B, L]` day-gathers, but **`L` is specified nowhere** — not in the resampling contract, not in `BUILD_SPEC` Step 8. `L` is a first-class modeling parameter masquerading as an implementation detail: too short and payout counts are biased down by *path length* rather than by the rules (accounts time out before they could have reached later payouts); too long and accounts effectively "live forever," inflating payout counts and distorting the renewal cycle-time `T_i` that §15 consumes. `L` therefore directly shapes both the payout-count distribution (§14) and every renewal velocity number (§15).

**Mitigation path:** make `L` (or the calendar horizon it represents) an explicit config parameter with a stated economic meaning — e.g. "the horizon over which one attempt is evaluated," chosen to match the real account's practical lifetime, not left implicit. Report payout-count and renewal metrics' sensitivity to `L` the same way block length is reported (§G1). Never let a headline number depend on an unstated `L`. **Note the truncation direction (`§C3`-review):** truncating the *longest-surviving* funded accounts at `L` shortens exactly the best cycles' `T_i`, which *inflates* `R_renewal = E[R]/E[T]` (the good attempts look faster than they are). This is the opposite direction from "too long → inflates counts"; both must be reported.

**When to decide:** not before the kernel (Step 6 doesn't depend on `L` — it simulates one fixed path). `L` becomes load-bearing at Step 8 (the resampler takes it as an explicit input) and must be *decided with a stated economic rationale* before any payout-count (Step 10) or renewal (Step 11) number is trusted. `BUILD_SPEC` Step 8 already requires `L` as an explicit per-phase input; this item is the reminder to give it a *justified value*, not just a parameter slot.

### C8. Continuous consistency check is degenerate — must be EOD-timed and activation-gated  **[CORRECTNESS] — severity: high, now specified**
The consistency predicate is `max_day_pnl > threshold × total_pnl`. Evaluated *continuously* (per-trade) it breaks in two ways: (a) when `total_pnl ≤ 0` early in a phase the ratio is undefined/negative and the test is meaningless; (b) once a single good day has closed (`max_day_pnl ≈ 100`) while `total_pnl ≈ 100`, the ratio is ≈ 1.0 > threshold, so the account **fails on the first trade of the next day** — a spurious kill. Real firms evaluate consistency only at day close, and only once profit is meaningful (a minimum-profit floor, or at payout-eligibility). **Fixed** by making `ConsistencyRule` default `check_timing = EOD` (matching the ADJUST variant) and adding an `activate_above` profit floor below which the rule does not evaluate (ARCHITECTURE §5). Both the timing and the gate are required; either alone leaves a live spurious-failure path. Step 6 tests: consistency does not fire below `activate_above`, and fires only at day close above it.

### C9. `_close_day` internal fold-then-evaluate order  **[CORRECTNESS] — severity: moderate, now specified**
At a day close, `_close_day` both (i) folds the just-closed day's `day_pnl` into `max_day_pnl` and the winning-day counter, and (ii) evaluates EOD breach/adjust/pass/payout predicates that *read those very fields*. Whether the closing day's own pnl counts toward its own EOD test depends entirely on the order of (i) vs (ii), and two reasonable implementations diverge here. **Decided:** (i) folds *first*, then (ii) evaluates against the updated state — the closing day's pnl counts toward its own EOD consistency test and can complete its own winning-day payout (ARCHITECTURE §12). Within (ii) the precedence is fail→adjust→pass→payout (§C4). This is an explicit Step 6 oracle case ("closing day's own pnl included in its own EOD predicate").

---

## D. Data-fidelity risks

### D1. Intraday low from 1m bars must be restricted to the holding interval  **[WRONG-ON-PURPOSE / CORRECTNESS] — severity: high**
`trade_low` is faithful to 1-minute data **only after restricting bar excursions to the position's actual holding interval**. A bar's low does not tell you the position was exposed at that low: the position may open partway through a bar (the bar's low occurred *before* entry), or exit before the bar's low is reached. If `trade_low` is derived from every spanned bar's low without respecting the open/close timestamps, it becomes *too conservative* — it charges the account for excursions it was never exposed to — which is a correctness error, not merely a resolution limit. After the holding-interval restriction, the residual uncertainty is only intrabar path ordering and genuine sub-minute excursion (small for these liquid contracts, non-zero for SIL/MCL on news).

Note this is separate from the `mae`-absent case: if the backtest supplies no per-trade adverse-excursion figure at all, the intraday low degrades to realized between-trade lows and true within-trade gap-through is invisible, so `check_timing=CONTINUOUS` rules understate failure.

**Mitigation path:** compute `trade_low` from bar excursions clipped to `[entry_time, exit_time]`, not from whole spanned bars; make "bar low outside the holding interval must not count" an explicit oracle test (G6). Treat presence of a holding-interval-restricted adverse excursion as the prerequisite for trusting any intraday-timing firm.

### D2. Concurrent positions collapse the equity model  **[WRONG-ON-PURPOSE] — severity: strategy-dependent, up to critical**
The single sequential-equity series assumes no overlapping positions. For a strategy that holds multiple positions at once, the true intraday low is the joint mark of the open book, not the sum of independent per-trade lows, and `size × ret` per closed trade is not the equity path. For such strategies the drawdown numbers aren't lower-fidelity — they can be **structurally wrong**. The architecture states this as one out-of-scope line; its severity depends entirely on whether the target strategies are single-position.

**Mitigation path:** confirm the strategies being simulated are effectively single-position (or serialize-able without material overlap). If not, this is a redesign of the data model, not a footnote.

### D3. Session-boundary / calendar messiness  **[WRONG-ON-PURPOSE] — severity: moderate**
"Assign each trade a day by one session-boundary time" is clean in the spec and messy in reality: DST shifts, holidays, half-days, weekend gaps, and near-24h instruments make day assignment ambiguous. A misassigned trade silently corrupts every day-scoped rule (daily loss, winning-day count, EOD drawdown). Treated as a one-parameter setting, it's actually a small calendar problem.

**Mitigation path:** use an explicit trading calendar for the instrument, not a fixed clock offset; test days around DST and holidays.

### D4. The trade-stream producer is out of scope but owns Level 0 and Level 2  **[HOLE / boundary statement] — severity: high**
The engine's input (§11) is a per-*closed-trade* table with entry/exit timestamps and a holding-interval-clipped `trade_low` (D1). Producing that table from the raw ~1-minute OHLCV bars — running the strategy backtest, pairing entries to exits, and computing clipped adverse excursion from the bars over each holding interval — is a **separate layer that none of the three documents specifies**. Yet it is where two of the most important things live: the **frozen strategy itself (Level 0, G7)** and the **single most fidelity-critical field (D1's clipped `trade_low`, Level 2)**. Consequences: the engine cannot be exercised on real data until this producer exists; and its correctness is **not testable by the engine's own suite** — a bug in entry/exit pairing or excursion clipping produces a perfectly-simulated wrong answer. This is an explicit boundary the engine docs stop at, not a component the engine provides. **What it does *not* block:** the synthetic generators (Step 3b, §11.7) emit the same raw-row contract, so the *entire engine and its full test suite* (Steps 1–11) can be built and validated without the producer. Only *real-strategy conclusions* wait on D4 — everything mechanical is unblocked by synthetic data.

**Mitigation path:** treat the trade-stream producer as its own project with its own tests (entry/exit pairing correctness; `trade_low` clipped to `[entry, exit]` verified against hand-checked bars — the same case as the Step 3 / Step 6 clipping tests, but at the producer level). Freeze its strategy output before the engine consumes it (G7). Record, with every engine result, which producer version and which data period generated the trades, so a producer bug is traceable rather than silent.

---

## E. Framing corrections — claims the doc oversells

### E1. "New firms are config, not code" is conditional  **[FRAMING] — severity: low**
True only for firms that reuse already-implemented mechanics. Every genuinely new mechanic so far (soft breach, dual-axis timing, floor locking, target adjust, payout, capped-out) required a new kernel branch plus registry entry. The honest claim is: *new firms are config if they reuse existing mechanics; a new mechanic is a bounded four-step kernel addition the registry forces you to make.* The registry hard-fail keeps this safe, so it's not a flaw — but the stronger phrasing should be softened wherever it appears.

### E2. "Near-linear core scaling" is an upper bound  **[FRAMING] — severity: low**
`prange` over sims parallelizes cleanly in principle, but the per-trade gathers over the resampled trade arrays are memory-bandwidth bound, which commonly caps scaling below linear well before core count is exhausted. Fine to expect good scaling; don't promise near-linear without measuring.

### E3. JIT/compilation realities unstated  **[FRAMING] — severity: low**
First-call JIT latency, `cache=True` invalidation quirks, and the requirement that the whole hot loop stay nopython (no accidental object-mode fallback) are real operational costs not mentioned. They don't change the design but should be expected during build.

---

## F. Priority summary

**Resolved (were blocking; now specified in ARCHITECTURE §6b):**
- A1 payout/split schema — done, grounded in Lucid's actual terms
- A2 post-payout equity & floor reset — done, config-controlled transition
- A3 multiple-payout mechanisms — non-issue; no known firm has them

**Know you are wrong on purpose (state loudly, revisit when the optimizer runs):**
- B1 inter-day dependence (tune day-block length > 1)
- B4 size-linear scaling of returns/MAE (needs an impact model before trusting optimized sizes)
- B2 empirical-tail limit on ruin probability
- D2 single-position assumption (verify it holds for your strategies)

**Correctness items — all gated behind oracle parity (G6), which is Level 1 of the trust hierarchy in §G:**
- C1 fastmath on boundary comparisons
- C3 floor initialization / first-day
- C5 soft-breach × EOD-rule interaction
- C2 capped-out definition (or drop until sizing exists)

These are validated *by* the G6 reference-oracle equivalence test (its boundary-case list targets exactly C1–C5), which must pass before any Monte Carlo at scale.

**Framing to soften in the architecture:**
- E1 config-not-code is conditional
- E2 scaling is an upper bound

The through-line: the engine's *mechanics* are in good shape, but its *economic output* (payouts) and its *tail realism* (inter-day dependence, empirical tails, size-scaling) are where the real risk lives. The mechanics will give you precise answers; these items determine whether the precise answers are also correct.

---

## G. Methodological priorities — the science, ranked (validate before believing any output)

The engineering is ahead of the science. The machinery is precise and general, but it rests on statistical assumptions that currently bear more weight than they've been examined for. This section ranks what to validate, with the minimum check that would make the headline numbers trustworthy. **Data context:** trades are generated from a backtest over **1-minute OHLCV futures data**, worst-fill assumed, across **6E, M2K, MCL, MES, MGC, MNQ, SIL**. That context is baked into the assessment below.

### G1. Inter-day AND cross-asset dependence in the bootstrap — **the top priority**
The account is a path-dependent knockout; what kills it is autocorrelated, clustered drawdown. Day-block resampling only preserves dependence up to the block length, and account-killing days are disproportionately *correlated-asset* days: MES/MNQ/M2K (US equity index) move together, MGC/SIL (metals) move together. Two concrete failure modes:
- **Block length too short** under-produces multi-day losing streaks → overstates survival and payouts.
- **Resampling assets independently** destroys the cross-asset correlation that creates the worst aggregate days → badly understates drawdown risk.

**Minimum checks:** (1) do **not** assume the block length can be read off a single autocorrelation statistic — for drawdown survival the relevant dependence is not necessarily captured by linear ACF (it can live in the volatility process or in nonlinear/tail dependence). Instead: estimate several dependence diagnostics, test a range of mean block lengths, compare survival/payout outputs across that range, and **select a defensible block-length range, reporting sensitivity across it** rather than pretending one length is objectively correct. (2) Resample **whole calendar/session days across all 7 assets jointly**, with the atom defined precisely:

> **The resampling atom is one canonical session day, containing all seven assets' trades assigned to that day — not requiring every asset to have traded.** Missingness (an asset with no trades that day) is legitimate and must be preserved, not filled. **Day identity must be fixed *before* resampling using the canonical session calendar**, because if "day" is defined inconsistently across assets the bootstrap can *manufacture* correlation by mis-aligning independently-defined days.

(3) Report survival/payout as a band across the block-length range, not a point. This is the single most output-determining item in either document.

**Block-length sensitivity is a first-class output, not a mitigation.** For every headline result, report the metric *across* the generator ladder and block-length range, so the reader sees how much the answer moves when dependence assumptions move:

| Generator  | Block mean | P(profitable) | P(5 payouts) | E[payout] | Payout/yr |
|------------|-----------:|--------------:|-------------:|----------:|----------:|
| IID        | 1          | …             | …            | …         | …         |
| Stationary | 2          | …             | …            | …         | …         |
| Stationary | 5          | …             | …            | …         | …         |
| Stationary | 10         | …             | …            | …         | …         |
| Regime     | —          | …             | …            | …         | …         |
| Stoch-vol  | —          | …             | …            | …         | …         |

The point is not to pick the block length that gives the "right" answer — it is to show how much the answer moves. The resulting interval is a **model-sensitivity band, not a confidence interval**: it measures dependence-assumption uncertainty, which is a different (and usually larger) thing than the sampling uncertainty a CI captures.

**The band is itself data-selected — a second overfitting channel, one level up from G7.** The block length (and the endpoints of the range the band spans, and the choice of generators in the ladder) are all chosen by looking at the *same single historical realization* that is then resampled. So the dependence model is fit to the data it models, and the band's width is data-selected — it brackets dependence-assumption uncertainty *only within a model family chosen on that data*, not the true uncertainty. This is the direct analogue of G7 (strategy selection) applied to the *resampling* model: just as a strategy overfits when selected on the data, the dependence model overfits when tuned on it. Honest consequence: the model-sensitivity band is a floor on model uncertainty, not a full accounting of it; a dependence structure the data never exhibited (but the future might) is outside every rung of the ladder. State the band as "sensitivity across a data-chosen model family," never as "the" uncertainty.

**The generator ladder (methodological requirement, not a feature).** A single resampler — however well tuned — is one hypothesis about the return-generating process, and the prop-firm barriers make results acutely sensitive to that choice. **An apparent edge is not considered robust unless it survives materially different plausible return-generating mechanisms.** Evaluate the *same* strategy, contract, and objective across a ladder of generators:

```
IID  →  block bootstrap  →  stationary bootstrap  →  regime-conditioned bootstrap  →  stochastic-volatility / parametric
```

Each rung relaxes an assumption the previous one imposed (IID ignores dependence; block/stationary add short-range dependence; regime-conditioned adds regime switching; stochastic-vol adds volatility clustering and fatter tails than the empirical sample contains). An edge that is real under IID but evaporates under regime-conditioned or stochastic-vol resampling was an artifact of the resampler, not a property of the strategy. Report the headline renewal and survival numbers *across the whole ladder* and keep only what survives.

**CRN scope (interacts with the optimizer).** Common Random Numbers is a paired-variance-reduction tool for comparing policies *within a single generator* — same resample seeds across candidate policies so the objective difference reflects the policy, not RNG noise. It must **not** be forced *across* generators: different rungs represent fundamentally different stochastic processes, so identical randomness is neither meaningful nor desirable there. Within a generator, CRN for policy comparison; across generators, the goal is robustness, not paired comparison.

### G2. Single-path sufficiency — the bootstrap measures conditional, not total, uncertainty
You have one historical realization per strategy. The bootstrap generates *variation* but cannot add *information* the single path lacks; with a few years of data the effective number of independent drawdown episodes may be a dozen. The precise statement is a *scope* claim, not a direction: **a bootstrap CI measures sampling uncertainty conditional on the empirical history and the chosen resampling model; it does not measure uncertainty about whether the historical sample itself is representative of the future.** That second component — sample-representativeness / model uncertainty — is simply not in the CI at all, which is why a tight bootstrap interval must not be read as "we are confident about the future."

**Minimum checks:** report the effective sample size (distinct days, and rough count of independent drawdown episodes) alongside every CI; state explicitly that intervals are conditional on the history and resampling model; use the generator ladder (§G1) and the block-length range as the honest breadth of model uncertainty, since spread *across* generators/lengths is a better proxy for representativeness uncertainty than any single-model CI.

### G3. Size-invariance breaks under the optimizer — asset-specific severity
`pnl = size × ret`, `trade_low = size × mae` assume linear scaling. Worst-fill-within-a-1m-bar handles slippage *at backtested size*, but **not market impact from larger positions** — a bigger order can fill worse than any price in the bar. Severity is asset-dependent: MES and MNQ are deep enough that impact stays small even at size; **6E, MCL, M2K, MGC, and especially SIL** are thinner, and an optimizer scaling contracts up will manufacture fills that don't exist. The optimizer is the stated purpose of the project and it breaks precisely this assumption.

**Minimum checks:** before trusting any optimized policy that materially scales size, add at least a simple concave market-impact term calibrated per contract (liquid vs thin); until then, bound the optimizer's size range near backtested size. Flag SIL and 6E results as impact-sensitive.

### G4. Backtest fidelity is the ceiling — the engine cannot exceed its input
The "historical data" is itself a conservative model, not ground truth. From 1m OHLCV you cannot know intrabar path order — when a stop and target sit in the same bar, which filled first is unknown. Assuming the adverse fill (consistent with worst-fill) is the right conservative choice, but it means the trade sequence fed to the engine is already conservative, and that conservatism propagates untouched through every downstream number.

**Minimum checks:** state the intrabar assumption explicitly with each result set; where it matters, sanity-check against a finer data source on a sample; treat outputs as conservative estimates, and know that the sub-minute spike a 1m bar-low understates is the residual (small for these liquid contracts, non-zero for SIL/MCL on news).

### G5. Stationarity — is the edge still live
The engine treats a 5-year-old edge and a current edge identically. Strategies decay and regimes shift; a payout estimate off stale data is confidently wrong.

**Minimum checks:** add a stationarity / regime flag on the input (e.g. compare recent-window return distribution to the full history; flag drift); optionally weight or restrict to recent data for forward-looking estimates. At minimum, timestamp the edge and don't trust forward payout numbers from data whose regime no longer holds.

### G7. Strategy-selection / backtest overfitting — the input the engine cannot repair  **[HOLE] — critical, sits BELOW everything (Level 0)**
Every other item assumes the strategy arriving at the engine is an *exogenous, unbiased* input. That assumption is false the moment a strategy is selected or tuned by looking at the same historical data that is then resampled. The chain **strategy selection → historical overfit → bootstrap → precise-looking payout distribution** produces confident output from a selection artifact, and the bootstrap *cannot* repair it — the bias is upstream of everything the engine models. Worse, bootstrapping makes the artifact look *more* trustworthy (tight bands around an overfit edge), so this is the most dangerous risk in the document precisely because it is invisible in the outputs.

The optimizer makes this acute. **CRN reduces comparison noise but does not reduce overfitting** — an efficient optimizer with common random numbers becomes *better at exploiting the specific peculiarities of the one historical sample*. So the layer you most want to build (the sizing optimizer) is exactly where selection bias does the most damage.

**Mitigations (required, not optional):**
- **Freeze the strategy before engine evaluation.** The specification the engine sees must be fixed independently of the data being resampled.
- **Keep a genuinely untouched validation / forward period** the strategy was never selected or tuned on; report engine results on that period separately.
- **Account for the search.** When multiple strategies or parameter sets were tried, the winning specification is not an exogenous input — correct for the multiplicity of the search rather than treating the survivor as if it were pre-registered.
- **For the optimizer: nested / walk-forward out-of-sample is mandatory.** The optimizer must never see the same historical sample used to establish the strategy's viability; otherwise it (helped by CRN) simply learns the sample's noise. This is the concrete rule that stops Level 5 optimizing through the Level 0 selection problem.

This is Level 0 of the trust hierarchy below: a frozen, non-overfit strategy is the input on which mechanical correctness, data fidelity, and everything above them condition.

### G6. Reference-oracle equivalence — the precondition for trusting the kernel  **[CORRECTNESS] — critical (Level 1)**
Every number in this document is meaningless if the kernel does not reproduce the firm's contract exactly. The slow, obviously-correct pure-Python reference implementation (`reference.py`) is therefore not just a debugging aid — it is a **methodological precondition**: the optimized Numba kernel must match the oracle before any large Monte Carlo, any generator-ladder work, or any optimization is trusted. This catches the C-section boundary bugs (C1–C5) far more reliably than reasoning about the kernel in isolation.

The equivalence test must deliberately include the boundary cases where the bugs live:
- equity exactly equal to the drawdown floor; one tick above; one tick below
- profit target exactly reached
- breach and target satisfied on the same trade (precedence)
- breach on the first trade; breach on the final trade of a day
- soft breach followed by end-of-day processing (the truncated-day close)
- payout exactly at `min_request`; exactly at a dollar cap; exactly at the `cap_fraction` bound
- withdrawal followed by drawdown-floor recomputation
- payout immediately followed by a breach
- trailing-floor lock transition
- minimum-size / capped-out boundary
- `trade_low` derived only from bar excursions *within the position's holding interval* (see D1)

**The requirement is two-tier, but the default is bitwise, because this kernel has no within-sim reduction:**
- **Per-sim path — bitwise equality (the default here).** Each simulation is an *independent sequential accumulation* (`equity += size*ret[t]`, day by day); `prange` parallelizes *across* simulations, never *within* one, so there is no cross-lane floating-point reduction on the per-sim path. With `fastmath` off (which it is, §C1), pure-Python float64 and Numba float64 performing the same operations in the same order are bitwise identical. Therefore the per-sim comparison defaults to **bitwise equality**, and any observed non-bitwise difference is evidence of a *real* divergence (a genuine logic or ordering bug), not benign reassociation — a blanket tolerance here would risk masking exactly the C1-class boundary flip the gate guards against.
- **Genuine reductions only — tolerance.** Numerical tolerance is reserved for a *true* cross-lane reduction if one is ever introduced (e.g. a vectorized sum over sims in the statistics layer). It is not granted on the per-sim simulation path.
- **Boundary bar — exact.** At the enumerated contract boundaries, behavior must match exactly; with bitwise per-sim equality this is automatic, and it is the reason `fastmath` is off (a reassociation could flip a breach at the boundary).
- **Boundary bar — exact.** At the enumerated contract boundaries (equity exactly on the floor, payout exactly at a cap, etc.) the behavior must match *exactly*, because those are the comparisons where a last-ULP difference *is* a semantic difference. This is also the direct argument for **dropping `fastmath` on the branch-deciding comparisons (C1)**: with `fastmath` you cannot separate a benign last-ULP difference from a reassociation that flipped a breach at the boundary, which destroys the ability to test the boundary exactly. No `fastmath` on comparisons; then boundary-exact equality is well-defined.

### The trust hierarchy — why the priority order is what it is
The items above are not a flat list; they are levels, each meaningful only if the ones below it hold. The governing principle: **do not let a higher level optimize or conclude through an unresolved lower level** — do not let a Level-5 optimizer search through Level-3 stochastic-model uncertainty, never do any of it on a Level-1-unproven kernel, and never treat a Level-0-overfit strategy as an unbiased input.

0. **Level 0 — strategy validity.** Is the strategy frozen and not overfit to the data being resampled? → freeze the spec, hold out a genuine forward period, account for the search, nested OOS for the optimizer (G7). *Everything above conditions on this; the engine cannot repair a selection-biased input.*
1. **Level 1 — mechanical correctness.** Does the kernel reproduce the firm's rules exactly? → oracle parity: **bitwise on the per-sim path** (no within-sim reduction exists), boundary-exact at contract edges (G6).
2. **Level 2 — data fidelity.** Does the input trade stream reasonably represent what could have happened? → 1m OHLCV, worst-fill, holding-interval-restricted `trade_low`, session calendar (G4/D1, and the data contract in ARCHITECTURE §11).
3. **Level 3 — stochastic-model robustness.** Does the conclusion survive plausible alternative futures? → the generator ladder IID → block → stationary → regime → stochastic-vol (G1), reported as a model-sensitivity band, with conditional-vs-total uncertainty stated honestly (G2).
4. **Level 4 — parameter extrapolation.** Does the optimizer remain valid away from historical position size? → market-impact / slippage model (G3).
5. **Level 5 — economic decision.** Is the resulting payout velocity / return-on-fee attractive? → renewal economics and the optimizer objective (ARCHITECTURE §15–16), evaluated only on data the strategy and optimizer never saw (Level 0).

Stationarity (G5) sits alongside Level 2–3 as an input-validity gate on the whole stack. The full stack in one line: **frozen non-overfit strategy → correct rules → faithful historical representation → realistic dependence → realistic parameter scaling → economic optimization.**

### Priority order
**G7 (strategy validity, Level 0) and G6 (oracle parity, Level 1) precede everything.** Then **G1 ≫ G3 ≈ G2 > G5 > G4.** G1 (joint day-block resampling, block-length *range* reported as a model-sensitivity band) determines whether any survival or payout number is real, and the cross-asset point makes it urgent for the 7-instrument, correlated-cluster setup. G3 gates the optimizer specifically. G2 states the honest scope of the CIs. G5 and G4 are lower but should be reported on every result rather than assumed away. Recommended build order: (1) **frozen strategy + held-out forward period**, (2) reference oracle + boundary tests, (3) joint seven-asset calendar-day representation, (4) block-length sensitivity as a first-class output, (5) generator ladder, (6) uncertainty / model-sensitivity-band reporting, (7) size/impact model, (8) stationarity/regime diagnostics, (9) optimizer *with nested out-of-sample*.

**The operating rule:** *do not spend compute before you've earned trust.* A million simulations of a kernel that hasn't passed the oracle, or a parallel optimizer exploiting IID bootstrap paths on an overfit strategy, just produces a more precise wrong answer.

### What is NOT a priority (downgraded by the data setup)
- **MAE fidelity** (was a concern): you compute `trade_low` directly from 1m bar lows under worst-fill, so it is as faithful as 1m allows — not a noisy broker column — **provided the excursion is clipped to the position's holding interval (D1)**. With that clip, only intrabar ordering and sub-minute excursion remain, and they're small for these contracts. Without it, `trade_low` is not merely limited but too conservative.
- **Left-tail loss *depth*:** void under the structured-product frame (§0) — floored downside. (Adverse-day *arrival* — frequency/timing/clustering — is not void; it drives knockout and is handled by G1.)

### The one-line version
Prove the kernel against the reference oracle first (G6); then spend effort on the resampling — **joint whole-day, all-asset blocks with a defensible block-length *range* (not a single derived length), evaluated across the generator ladder**, with uncertainty reported as conditional-on-history. The machinery is ready and, once the oracle proves it correct, the dominant question is no longer engineering but whether the stochastic process feeding the engine is realistic enough for a knockout-and-payout product.

---

## H. Reporting & fee semantics — correctness of the statistics layer

These are correctness items in the §14/§15 statistics layer (as opposed to the kernel). They were surfaced while closing the payout-schema gaps and are given their own entries here so `ARCHITECTURE.md` §14 can cite them precisely.

### H1. Fees are path-dependent  **[CORRECTNESS] — severity: moderate**
The fee attributable to an attempt is not a single scalar: an attempt that fails the **eval** phase pays the evaluation fee but **never** pays the activation fee (it never reached the funded account). Folding both into one constant biases every fee-denominated statistic — `prob_profitable`, `payoff_quantiles`, `return_on_fee`, `return_on_fee_per_year`, and the composite objective. The fee must be resolved per attempt as `eval_fee + (activation_fee if reached_funded else 0)`, which is why the raw outcomes expose a `reached_funded` flag (§13/§14.4). All §14 functions take `(eval_fee, activation_fee)` and compute the attributable fee per attempt, never a scalar.

### H2. `max_payouts` is schema config, not a dataset field  **[CORRECTNESS] — severity: low**
The payout-count distribution's support (`P(0)…P(max)`) is bounded by `max_payouts`, which is a property of the funded phase's `PayoutSchema` (§6b), **not** of `TradeDataset`. `Results` must receive `max_payouts` resolved from the account's schema at construction and must not read it off the dataset (the dataset has no such field). Plumbing that reads it from the dataset is a bug.

### H3. `pass_rate` is not a funded-phase success metric  **[CORRECTNESS] — severity: moderate, now structurally addressed**
A funded attempt is an economic *success* whether it ran out of path with payouts banked (`TIMED_OUT`) or hit the payout ceiling (`MAXED_OUT`) — neither is `PASSED`, which means "cleared the eval." `pass_rate` (keyed on `code == PASSED`) is therefore an **eval-phase** metric only. The overloading is now reduced structurally: reaching `max_payouts` returns the distinct `ExitCode.MAXED_OUT` rather than a `PASSED`-equivalent (ARCHITECTURE §6b.2), so filtering `PASSED` on a funded batch cannot silently mean "hit the ceiling." Funded economic performance is still read from `net_payout` / `payouts_taken` / the payout-count distribution, never from `pass_rate`. (Interacts with path length `L`, §C7: a longer `L` converts some `TIMED_OUT` funded successes into additional payouts or into `MAXED_OUT`, so code semantics and `L` must be held together when reading funded results.)

### H4. `total_trading_days` and renewal cycle-time span the whole attempt (eval + funded)  **[CORRECTNESS] — severity: high**
The fee is tied up from the eval phase's first day until the attempt terminates, so the time used in every rate — payout velocity, `R_renewal`, fee-bankroll efficiency (§14.2, §15) — must be the **whole attempt's** calendar duration, eval + funded, not funded-only. Because §17 runs eval and funded as separate survivors-only phases each drawing their own `[B, L]` path, the aggregator must sum day-counts across every phase the attempt ran. Using funded-only understates capital tie-up and overstates the rate — a first-order error for a product §0 defines as a rate. (`ARCHITECTURE.md` §14.4, §15.2.)

### H5. `r_path` (i.i.d. attempt draws) cannot diagnose cross-cycle correlation  **[WRONG-ON-PURPOSE / CLAIM CORRECTION] — severity: moderate**
`R_renewal = E[R]/E[T]` can diverge from the realized long-run rate for two independent reasons: (1) ratio-estimator / finite-horizon (Jensen) bias, and (2) cross-cycle correlation (attempts from the same strategy and data are not independent). The `r_path` estimator in §15.2 resamples completed attempts **i.i.d.**, which removes correlation by construction — so it isolates and measures only (1). It must **not** be read as evidence that cycles are uncorrelated. Diagnosing (2) requires generating attempt *sequences that preserve order* — running the whole renewal chain on one continuous resampled day-path rather than stitching independently-drawn attempts — which is deferred with the generator ladder (§G1). Until then, cross-cycle correlation is a **named, uncaptured gap**, not something the two-rate comparison closes.

**Mitigation path:** when the generator ladder is built, add an order-preserving renewal simulation (resample a long day-path, run attempts back-to-back on it) and compare its rate to the i.i.d. `r_path`; the difference is the correlation effect. Until then, state that renewal rates assume cross-cycle independence.

---

## I. Synthetic trade-stream risks

The synthetic generator (`ARCHITECTURE.md` §11.7) manufactures trade rows from parameters, for testing and for future breakeven-mapping research. Its risks are distinct from the real-data risks above because the data itself is invented.

### I1. Synthetic results are model-conditional — a floor for mapping, never a claim about real strategies  **[WRONG-ON-PURPOSE] — severity: high if misread**
A synthetic stream is only as honest as its statistical model of returns, and the §G1/§B1 warning applies *doubly*: an i.i.d. synthetic generator cannot produce the clustered losing streaks that actually knock accounts out, so every breakeven line it draws is **optimistic**. Two disciplines are therefore mandatory:

- **Never read a single-generator synthetic result as "the" answer.** Run the synthetic ladder (i.i.d. → regime-switching → stochastic-vol) exactly as the resampling ladder is run for real data, and report the breakeven surface *across* generators. A requirement that holds under i.i.d. but fails under regime-switching was an artifact of the benign generator.
- **Keep the two ladders distinct in interpretation.** The resampling ladder varies dependence over a *fixed empirical distribution* (real trades); the synthetic ladder varies *both* distribution and dependence (invented trades). They share a dependence vocabulary so results can be cross-checked, but a synthetic breakeven line is a statement about a *parameter region*, not about any real strategy. Provenance (generator type, parameters, seed, derived edge) rides through to `Results` so the two are never conflated.

The correct framing of any synthetic breakeven result: "under *this* return model, a system with these stats clears firm X" — a map of strategy-space, not a verdict on a real edge. Treated that way it is a powerful research instrument; read as an unconditional minimum-requirement it is misleading in the optimistic direction.

### I2. Derived edge must not be an input  **[CORRECTNESS] — severity: low, now specified]**
The generator takes `win_rate` and `RR` only; per-trade edge is *derived* (`win_rate*(RR+1) − 1`) and reported, never accepted as input. Accepting edge as a third parameter would over-determine the trade distribution and admit inconsistent triples (an edge disagreeing with the win-rate/RR it is built from), silently producing a stream whose realized edge is not the one requested. The breakeven line (`win_rate = 1/(RR+1)`) is only internally consistent because edge is derived from the same two parameters the stream uses. (ARCHITECTURE §11.7.1.)
