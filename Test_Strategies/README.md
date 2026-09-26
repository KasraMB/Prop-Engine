# Strategy example

`orb_reference.py` provides a dependency-free, causal opening-range breakout
input producer. It is separate from the engine API, installed package and browser
bundle.

See the [ORB guide](../docs/ORB_REFERENCE.md) for the input contract and a runnable
example. Tests cover signal timing and validation, not execution realism or
strategy profitability.

Keep market data and generated trade exports local. The engine evaluates supplied
trade histories; strategy selection and signal generation remain separate concerns.
