"""Browser (Pyodide) numba shim: ``@njit`` becomes a no-op so the compiled kernels
run as plain Python. Regression and browser tests cover supported workflows;
this shim is not a guarantee of universal runtime or numerical equivalence."""


def njit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]                      # bare @njit
    def _decorate(fn):
        return fn
    return _decorate                        # @njit(cache=True, ...)


prange = range
