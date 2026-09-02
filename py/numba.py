"""Browser (Pyodide) numba shim: ``@njit`` becomes a no-op so the compiled kernels
run as plain Python. This is bit-identical to the real numba path — the engine's
Level-1 parity gate proves kernel == pure-Python reference for every input, and the
whole-pipeline golden hash reproduces under this shim."""


def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # bare @njit
    def _decorate(fn):
        return fn
    return _decorate                        # @njit(cache=True, ...)


prange = range
