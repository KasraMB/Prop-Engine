# Contributing

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

Production code belongs in `src/propfirm_engine`; regression tests belong in
`tests`. Keep strategy producers outside the engine package. Do not commit local
market data, generated trade exports, caches, credentials or scratch scripts.

## Correctness before optimization

Read the [design overview](docs/DESIGN.md) and the
[execution contract](docs/BRACKET_BACKTEST.md) before changing simulation semantics.
Add an independently calculated boundary example before relying on parity with
another implementation. Keep failure, non-tradability, horizon end and live
handoff distinct.

Use official firm sources only. Date research changes and distinguish official
evidence, user decisions and implementation assumptions. Never silently weaken
input capability validation to make a preset run.

Keep named policy regimes causal. Split by complete sessions; never use OOS to
choose policies, regime definitions, hyperparameters or stopping criteria.

## Repository layout

- Canonical Python: `src/propfirm_engine/`.
- Published browser mirror: `docs/py/propfirm_engine/`, with its manifest.
  Keep changed source modules synchronized; source equality is not full browser
  runtime certification.
- Maintained specifications and official-source evidence: `docs/`.
- Reproducible benchmarks: `benchmarks/`. Avoid one-off variants of the same program.
- Standalone strategy examples: `Test_Strategies/`.

Make small, tested commits. Do not rewrite existing history, delete user inputs,
or push remote changes as a side effect of local implementation.
