from copy import deepcopy

import pytest

from benchmarks.research import check_budget


def test_performance_budget_gates_only_the_selected_hardware_and_scale():
    report = dict(hardware={"cpu": "fixture"}, scale=1,
                  results={"replay": {"warm_median_seconds": 1, "peak_python_bytes": 100}})
    budget = dict(hardware={"cpu": "fixture"}, scale=1,
                  limits={"replay": {"warm_median_seconds": 2, "peak_python_bytes": 200}})
    check_budget(report, budget)
    slow = deepcopy(report)
    slow["results"]["replay"]["warm_median_seconds"] = 3
    with pytest.raises(ValueError, match="exceeded"):
        check_budget(slow, budget)
    with pytest.raises(ValueError, match="does not match"):
        check_budget(report | {"hardware": {"cpu": "other"}}, budget)
