# Strategy examples

`open_long.py` is the DuckDB-backed fixed-policy historical experiment comparing
09:30 and 18:00 New York long entries, separately per asset. It uses the engine's
price replay API and writes local trade/event ledgers and complete summaries.
See [historical price replay](../docs/PRICE_REPLAY.md) for execution assumptions,
commission-equivalent micro sizing, data exclusions and the reproducible command.

`zero_drift.py` runs a matched conditional control for that experiment:

```powershell
$env:PYTHONPATH = 'src;.'
python -m Test_Strategies.zero_drift --paths 500
```

It independently reflects minute OHLC innovations, preserving observed movement
magnitudes and timestamps. Both entry times share each randomized tape. The fixed
policy runs through the same account engine on fresh OOS accounts. Local outputs
in `results/zero_drift` include per-path metrics, distributions, historical
comparisons and a Markdown report. This retrospective sign-symmetry benchmark
removes directional dependence as well as drift; it is not a future-price model.
Gold/silver micro alternatives remain explicitly exploratory. Required bankroll
is a finite-path funding deficit, not a lifetime ruin estimate.

`execution_study.py` compares commission-only execution, uncompensated slippage,
slippage-aware brackets, and IS-refitted policies. It holds the selected policy
fixed for low/stressed execution scenarios. It reads prices through the same
DuckDB loader and writes per-stage distributions and sample fill ledgers:

```powershell
python -m Test_Strategies.execution_study --assets MES --generations 5 --paths 3 --evaluation-paths 20
```

Use `--assets MES MNQ M2K MCL MGC SIL 6E` for all assets, or `--max-sessions 120`
for a bounded integration run. Profiles are illustrative assumptions, not
measured Lucid fills. The previously inspected OOS period is not newly untouched
evidence. No market data or output files belong in version control.

`orb_reference.py` provides a dependency-free, causal opening-range breakout
input producer. It is separate from the engine API, installed package and browser
bundle.

See the [ORB guide](../docs/ORB_REFERENCE.md) for the input contract and a runnable
example. Tests cover signal timing and validation, not execution realism or
strategy profitability.

Keep market data and generated trade exports local. The engine evaluates supplied
trade histories; strategy selection and signal generation remain separate concerns.
