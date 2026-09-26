"""Keep active public examples executable without external data or downloads."""
from pathlib import Path
import csv
from datetime import date, timedelta
import re

import pytest


@pytest.mark.parametrize("name", ["README.md", "docs/CASHFLOW_SCENARIOS.md",
                                 "docs/PAYOUT_LIFECYCLE.md", "docs/ORB_REFERENCE.md",
                                 "docs/ANALYTICAL_MODEL.md"])
def test_current_documented_python_examples(name, tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    examples = re.findall(r"```python\n(.*?)\n```", (root / name).read_text(encoding="utf-8"),
                          flags=re.DOTALL)
    assert examples, f"No runnable examples found in {name}"
    namespace = {}
    if name == "README.md":
        # README snippets form one workflow and read the user's CSV. Supply a
        # small deterministic fixture, rather than skipping the public example.
        monkeypatch.chdir(tmp_path)
        with (tmp_path / "trades.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "entry_at", "exit_at", "session", "stop_loss", "take_profit", "won",
            ])
            writer.writeheader()
            day = date(2026, 9, 1)
            for _ in range(20):
                while day.weekday() >= 5:
                    day += timedelta(days=1)
                writer.writerow(dict(entry_at=f"{day}T10:00:00-04:00",
                                     exit_at=f"{day}T10:05:00-04:00",
                                     session=str(day), stop_loss=100,
                                     take_profit=200, won="true"))
                day += timedelta(days=1)
    for index, example in enumerate(examples):
        exec(compile(example, f"{name}:example-{index}", "exec"),
             namespace if name == "README.md" else {})
    if name == "README.md":
        assert namespace["fit"].score == namespace["fit"].out_of_sample_score
        assert len(namespace["fit"].train_sessions) == 14
        assert len(namespace["fit"].test_sessions) == 6
