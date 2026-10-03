"""Opt-in Chromium tests covering both local HTTP and real browser Python.

Run with RUN_BROWSER_TESTS=1 after installing playwright and its chromium.
The static test downloads the pinned Python/NumPy/timezone runtime from CDNs.
"""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_BROWSER_TESTS") != "1",
                                reason="opt-in browser/CDN integration test")


@pytest.fixture(params=["local", "static"])
def site(request):
    if request.param == "static" and os.environ.get("BROWSER_BASE_URL"):
        yield os.environ["BROWSER_BASE_URL"].rstrip("/") + "/", "static"
        return
    from dashboard.server import Handler
    root = Path(__file__).resolve().parents[1]
    handler = Handler if request.param == "local" else partial(SimpleHTTPRequestHandler, directory=str(root / "docs"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f"http://127.0.0.1:{server.server_port}/", request.param
    server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_upload_fit_export_and_errors_in_real_browser(site, tmp_path):
    from playwright.sync_api import sync_playwright
    from dashboard.replay import generate, run
    url, runtime = site
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("!document.getElementById('run').disabled || !document.getElementById('error').hidden", timeout=180_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        assert page.locator('input[type="checkbox"]').count() == 0
        params = page.evaluate("syntheticParameters()")
        page.locator("#generate").click()
        page.wait_for_function("!busy && csvText.length > 0", timeout=30_000)
        csv = page.evaluate("csvText")
        assert csv == generate(params)["csv"]
        page.locator("#sourceType").select_option("upload")
        page.locator("#csvFile").set_input_files({"name": "history.csv", "mimeType": "text/csv", "buffer": csv.encode()})
        page.wait_for_function("document.getElementById('fileStatus').textContent.includes('history.csv')")
        page.locator("#generations").fill("2")
        page.locator("#population").fill("4")
        assert page.locator('[data-field="maximum"]').count() > 0
        for field in page.locator('[data-field="maximum"]').all():
            field.fill("2000")
        request = page.evaluate("collect()")
        expected = run(request)
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('results').hidden || !document.getElementById('error').hidden", timeout=90_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        assert "Out of sample" in page.locator("#scope").inner_text()
        assert "56 sessions" in page.locator("#split").inner_text()
        assert "24 sessions" in page.locator("#split").inner_text()
        with page.expect_download() as download:
            page.locator("#download").click()
        result = json.loads(Path(download.value.path()).read_text())
        assert result["score"] == pytest.approx(expected["score"], abs=1e-8)
        assert result["headline"]["events"] == expected["headline"]["events"]
        for actual_regime, expected_regime in zip(result["policy"]["regimes"], expected["policy"]["regimes"], strict=True):
            # Browser and native linear algebra may differ in the last bits.
            # Executed event ledgers above must still agree exactly.
            assert actual_regime["risk_dollars"] == pytest.approx(expected_regime["risk_dollars"], abs=1e-10, rel=0)
            assert {k:v for k,v in actual_regime.items() if k != "risk_dollars"} == {k:v for k,v in expected_regime.items() if k != "risk_dollars"}
        assert len(page.locator("#cashChart polyline").all()) == 2
        chart = page.evaluate("cashSeries(latest)")
        assert chart["start"] == result["training"]["start"]
        assert chart["end"] == result["headline"]["end"]
        assert chart["boundary"] == result["headline"]["start"]
        assert chart["series"][0][0][1] == 0
        is_cash = result["training"]["net_cash"]
        assert chart["series"][0][-1][1] == pytest.approx(is_cash + result["headline"]["net_cash"])
        assert chart["series"][1][0][1] == pytest.approx(is_cash)
        assert chart["series"][1][-1][1] == pytest.approx(is_cash + result["baseline"]["net_cash"])
        for points in chart["series"]:
            assert [p[0] for p in points] == sorted(p[0] for p in points)
        marker = page.locator("#oosBoundary")
        assert marker.get_attribute("data-at") == result["headline"]["start"]
        assert marker.get_attribute("stroke-dasharray") == "5 4"
        assert marker.get_attribute("x1") == marker.get_attribute("x2")
        from datetime import datetime
        start, end, boundary = (datetime.fromisoformat(chart[k]) for k in ("start", "end", "boundary"))
        assert float(marker.get_attribute("x1")) == pytest.approx(70 + (boundary - start) / (end - start) * 610)
        assert "Headline metrics remain OOS-only" in page.locator("#cashCaption").inner_text()
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            target = Path(os.environ["BROWSER_SCREENSHOT_DIR"]); target.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(target / f"replay-{runtime}.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        # Input changes must clear stale results, and malformed data must surface errors.
        page.locator("#csvFile").set_input_files({"name": "invalid.csv", "mimeType": "text/csv", "buffer": b"timestamp,pnl\nx,100"})
        page.wait_for_function("document.getElementById('fileStatus').textContent.includes('invalid.csv')")
        assert page.locator("#results").is_hidden()
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('error').hidden", timeout=30_000)
        assert "CSV requires exactly" in page.locator("#error").inner_text()
        assert page.locator("#run").is_enabled()
        # Fixed-history reporting must not be advertised as OOS.
        page.locator("#sourceType").select_option("synthetic")
        page.locator("#mode").select_option("backtest")
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('results').hidden", timeout=30_000)
        assert "not OOS" in page.locator("#scope").inner_text()
        assert page.locator("#oosBoundary").count() == 0
        assert page.locator("#cashChart polyline").count() == 1
        # Statistics controls must affect the actual generated history, not just labels.
        page.locator("#gen_win_rate").fill("100")
        page.locator("#gen_rr").fill("3")
        page.locator("#gen_sessions").fill("10")
        page.locator("#gen_trades_per_day").fill("2")
        for model in ("iid", "regime", "stochvol"):
            page.locator("#gen_generator").select_option(model)
            params = page.evaluate("syntheticParameters()")
            page.locator("#generate").click()
            page.wait_for_function("!busy && csvText.length > 0", timeout=30_000)
            actual = page.evaluate("sourceMetadata")
            assert actual["parameters"] == params
            assert actual["statistics"]["trades"] == 20
            assert actual["statistics"]["win_rate"] == 1
            assert actual["statistics"]["mean_rr"] == pytest.approx(3)
            assert "20 trades" in page.locator("#fileStatus").inner_text()
            assert "100.0%" in page.locator("#fileStatus").inner_text()
            assert "3.00" in page.locator("#fileStatus").inner_text()
        page.locator("#gen_generator").select_option("iid")
        page.locator("#gen_sessions").fill("80")
        page.locator("#gen_trades_per_day").fill("4")
        if runtime == "static":
            page.locator("#mode").select_option("fit")
            page.locator("#rollingMode").select_option("rolling")
            page.locator("#generations").fill("100")
            page.locator("#population").fill("32")
            page.locator("#run").click()
            page.wait_for_function("document.getElementById('status').textContent.includes('candidate evaluations') || !document.getElementById('error').hidden", timeout=60_000)
            assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
            page.locator("#cancel").click()
            assert page.locator("#results").is_hidden()
            page.wait_for_function("!document.getElementById('run').disabled", timeout=180_000)
            # A stale/mixed deployment must fail closed, never execute partial code.
            page.route("**/py/replay.py", lambda route: route.fulfill(body="tampered", content_type="text/plain"))
            page.evaluate("startEngine()")
            page.wait_for_function("!document.getElementById('error').hidden", timeout=180_000)
            assert "bundle changed" in page.locator("#error").inner_text()
        assert not errors
        browser.close()


def test_rolling_results_export_chart_and_scope_in_real_browser(site):
    from playwright.sync_api import sync_playwright
    from dashboard.replay import run
    url, runtime = site
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("ready || !document.getElementById('error').hidden", timeout=180_000)
        assert page.locator("#error").is_hidden()
        page.locator("#generate").click()
        page.wait_for_function("!busy && csvText.length > 0", timeout=30_000)
        page.locator("#rollingMode").select_option("rolling")
        page.locator("#window_sessions").fill("10")
        page.locator("#stride_sessions").fill("3")
        page.locator("#generations").fill("1")
        page.locator("#population").fill("4")
        page.locator("#target_ruin").fill("5")
        page.locator("#risk_tail").fill("10")
        page.locator("#ruinMode").evaluate("e => e.closest('details').open = true")
        page.locator("#ruinMode").select_option("bootstrap")
        page.locator("#ruin_paths").fill("4")
        page.locator("#ruin_sessions").fill("20")
        expected = run(page.evaluate("collect()"))
        page.locator("#run").click()
        page.wait_for_function("!busy && latest !== null", timeout=120_000)
        actual = page.evaluate("latest")
        assert actual["rolling"] == expected["rolling"]
        assert actual["score"] == pytest.approx(expected["score"])
        assert page.locator("#rollingPanel").is_visible()
        assert actual["risk"] == expected["risk"]
        assert actual["ruin"] == expected["ruin"]
        assert page.locator("#ruinPanel").is_visible()
        assert "OOS sessions only" in page.locator("#ruinScope").inner_text()
        assert page.locator("#ruinBankrollChart polyline").count() == 1
        assert page.locator("#ruinHorizons tr").count() >= 2
        assert "independent-cycle approximation" in page.locator("#ruinPanel").inner_text()
        assert actual["risk"]["options"]["target_ruin_probability"] == .05
        assert page.locator("#riskPanel").is_visible()
        assert "OOS only" in page.locator("#riskScope").inner_text()
        assert "Synthetic trade history" in page.locator("#riskScope").inner_text()
        assert page.locator("#bankrollChart polyline").count() == 1
        assert page.locator("#bankrollChart circle").count() > 0
        assert "P99" in page.locator("#riskDistributionHead").inner_text()
        assert "Variance" in page.locator("#riskDistributionHead").inner_text()
        assert page.locator("#riskDistributions tr").count() == 12
        assert "Worst 10.00%" in page.locator("#riskTailNote").inner_text()
        assert actual["risk"]["confidence_supported_bankroll"] is None
        for dot in page.locator("#bankrollChart circle").all():
            assert 70 <= float(dot.get_attribute("cx")) <= 680
            assert 25 <= float(dot.get_attribute("cy")) <= 205
            assert dot.locator("title").text_content()
        assert "OOS" in page.locator("#rollingTitle").inner_text()
        assert page.locator("#rollingChart circle").count() == 5
        assert page.locator("#rollingWindows tr").count() == 5
        assert page.locator("#cashChart polyline").count() == 2
        assert page.locator("#oosBoundary").count() == 1
        assert "mean window" in page.locator("#metrics").inner_text()
        assert "16 IS windows" in page.locator("#rollingScope").inner_text()
        for dot in page.locator("#rollingChart circle").all():
            assert 70 <= float(dot.get_attribute("cx")) <= 680
            assert 25 <= float(dot.get_attribute("cy")) <= 190
            assert dot.locator("title").text_content()
        with page.expect_download() as download:
            page.locator("#download").click()
        exported = json.loads(Path(download.value.path()).read_text())
        assert exported["rolling"] == actual["rolling"]
        assert exported["risk"] == actual["risk"]
        assert exported["ruin"] == actual["ruin"]
        page.locator("#rollingPanel details").evaluate("e => e.open = true")
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            target = Path(os.environ["BROWSER_SCREENSHOT_DIR"]); target.mkdir(parents=True, exist_ok=True)
            page.locator("#rollingPanel").screenshot(path=str(target / f"rolling-{runtime}.png"))
            page.locator("#riskPanel").screenshot(path=str(target / f"risk-{runtime}.png"))
            page.locator("#ruinPanel").screenshot(path=str(target / f"ruin-{runtime}.png"))
        page.emulate_media(media="print")
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        assert page.locator("#riskPanel").is_visible()
        assert page.locator("#ledgerPanel").is_hidden()
        assert page.locator("#provenancePanel").is_hidden()
        assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(255, 255, 255)"
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            page.screenshot(path=str(target / f"report-print-{runtime}.png"), full_page=True)
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        page.emulate_media(media="screen")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.locator("#window_sessions").fill("999")
        assert page.locator("#results").is_hidden()
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('error').hidden", timeout=30_000)
        assert "no complete" in page.locator("#error").inner_text()
        page.locator("#window_sessions").fill("10")
        page.locator("#mode").select_option("backtest")
        page.locator("#run").click()
        page.wait_for_function("!busy && latest !== null", timeout=90_000)
        assert "not OOS" in page.locator("#rollingTitle").inner_text()
        assert page.locator("#rollingChart circle").count() == 24
        page.locator("#rollingMode").select_option("single")
        page.locator("#run").click()
        page.wait_for_function("!busy && latest !== null", timeout=30_000)
        assert page.locator("#rollingPanel").is_hidden()
        assert page.locator("#chronologicalNote").is_hidden()
        page.locator("#gen_win_rate").fill("0")
        for field in page.locator('[data-field="risk_dollars"]').all():
            field.fill("2000")
        page.locator("#generate").click()
        page.wait_for_function("!busy && csvText.length > 0", timeout=30_000)
        page.locator("#run").click()
        page.wait_for_function("!busy && latest !== null", timeout=90_000)
        cycle = page.evaluate("latest.ruin.cycle_approximation")
        assert cycle["status"] == "certain_ruin"
        assert cycle["confidence_bounds"] == [1, 1]
        assert "100.00%" in page.locator("#ultimateMetrics").inner_text()
        assert "No finite bound" in page.locator("#ultimateMetrics").inner_text()
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            page.set_viewport_size({"width": 1440, "height": 1000})
            page.locator("#ruinPanel details").evaluate("e => e.open = true")
            page.locator("#ruinPanel").screenshot(path=str(target / f"ruin-cycles-{runtime}.png"))
        assert not errors
        browser.close()


def test_target_discovery_without_manual_history_in_real_browser(site):
    from playwright.sync_api import sync_playwright
    from dashboard.replay import run
    url, runtime = site
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url + "research.html")
        page.wait_for_function("ready || !document.getElementById('error').hidden", timeout=180_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        assert page.locator('input[type="file"]').count() == 0
        for key, value in {"paths": "10", "sessions": "7", "generations": "2", "population": "4"}.items():
            page.locator("#" + key).fill(value)
        assert page.locator("#risk_max").input_value() == "2000"
        expected = run(page.evaluate("collectResearch()"))
        page.locator("#run").click()
        page.wait_for_function("!busy && (latest !== null || !document.getElementById('error').hidden)", timeout=120_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        actual = page.evaluate("latest")
        assert actual["fit"]["policy"] == expected["fit"]["policy"]
        assert actual["fit"]["holdout"]["score"] == pytest.approx(expected["fit"]["holdout"]["score"])
        assert actual["fit"]["candidate_seeds"] == 0
        assert actual["reference_used_for_selection"] is False
        assert actual["risk"] == expected["risk"]
        assert "7 training / 3 holdout" in page.locator("#researchScope").inner_text()
        assert page.locator("#researchPolicy tr").count() == 3
        assert page.locator("#researchComparison tr").count() == 3
        assert page.locator("#researchDecisions tr").count() > 0
        assert "model" in page.locator("#riskScope").inner_text()
        assert page.locator("#bankrollChart polyline").count() == 1
        page.locator("#riskDistributionHead").evaluate("e => e.closest('details').open = true")
        assert "Variance" in page.locator("#riskDistributionHead").inner_text()
        with page.expect_download() as download:
            page.locator("#download").click()
        exported = json.loads(Path(download.value.path()).read_text())
        assert exported == actual
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            target = Path(os.environ["BROWSER_SCREENSHOT_DIR"])
            target.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(target / f"research-{runtime}.png"), full_page=True)
        page.emulate_media(media="print")
        page.evaluate("window.dispatchEvent(new Event('beforeprint'))")
        assert page.locator("#riskPanel").is_visible()
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        page.emulate_media(media="screen")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            page.screenshot(path=str(target / f"research-mobile-{runtime}.png"), full_page=True)
        page.locator("#risk_min").fill("2000")
        assert page.locator("#results").is_hidden()
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('error').hidden", timeout=30_000)
        assert "bounds" in page.locator("#error").inner_text()
        assert page.locator("#run").is_enabled()
        assert not errors
        browser.close()


def test_manual_trace_uses_current_engine_and_steps_payout_events(site):
    from playwright.sync_api import sync_playwright
    from dashboard.replay import run
    url, runtime = site
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(url + "trace.html")
        page.wait_for_function("ready || !document.getElementById('error').hidden", timeout=180_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        assert page.locator("#pageTitle").inner_text() == "Account trace"
        assert page.locator("#sourceType").input_value() == "manual"
        assert page.locator("#modeField").is_hidden()
        for key in ("cost_per_contract", "approval_delay_hours", "receipt_delay_hours"):
            page.locator("#" + key).fill("0")
        for field in page.locator('[data-field="risk_dollars"]').all():
            field.fill("100")
        for i in range(7):
            if i:
                page.locator("#nextSession").click()
            page.locator("#manual_take_profit").fill("1500" if i < 2 else "200")
            page.locator("#addWin").click()
            page.wait_for_function("!busy && (!document.getElementById('results').hidden || !document.getElementById('error').hidden)", timeout=30_000)
            assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        request = page.evaluate("collect()")
        expected = run(request)
        actual = page.evaluate("latest")
        assert actual["headline"]["events"] == expected["headline"]["events"]
        assert actual["headline"]["receipts"] == 450
        assert actual["headline"]["net_cash"] == pytest.approx(344.8)
        events = actual["headline"]["events"]
        for kind in ("trade", "evaluation_pass", "session_close", "request", "approval", "receipt"):
            index = next(i for i, e in enumerate(events) if e["kind"] == kind)
            page.locator("#traceIndex").fill(str(index + 1))
            page.locator("#traceIndex").press("Tab")
            assert json.loads(page.locator("#traceDetails").inner_text()) == events[index]
        page.locator("#traceFirst").click()
        page.locator("#traceNextTrade").click()
        assert json.loads(page.locator("#traceDetails").inner_text())["kind"] == "trade"
        page.locator("#traceNextSession").click()
        assert json.loads(page.locator("#traceDetails").inner_text())["kind"] == "session_close"
        if os.environ.get("BROWSER_SCREENSHOT_DIR"):
            target = Path(os.environ["BROWSER_SCREENSHOT_DIR"]); target.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(target / f"trace-{runtime}.png"), full_page=True)
        page.locator("#undoTrade").click()
        page.wait_for_function("!busy && latest !== null", timeout=30_000)
        assert page.evaluate("manualTrades.length") == 6
        assert page.evaluate("latest.headline.receipts") == 0
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.locator("#clearTrades").click()
        assert page.locator("#results").is_hidden()
        assert page.evaluate("manualTrades.length") == 0
        # Two complete handoff cycles must renew in both runtimes, not plateau forever.
        from datetime import date, timedelta
        from dashboard.replay import manual
        day, rows = date(2026, 9, 1), []
        for i in range(54):
            while day.weekday() >= 5:
                day += timedelta(days=1)
            rows.append(dict(session=day.isoformat(), entry_time="10:00", duration_minutes=5,
                             stop_loss=100, take_profit=1500 if i % 27 < 2 else 200, won=True))
            day += timedelta(days=1)
        payload = manual({"trades": rows})
        page.locator("#sourceType").select_option("upload")
        page.locator("#csvFile").set_input_files({"name": "renewal.csv", "mimeType": "text/csv",
                                                 "buffer": payload["csv"].encode()})
        page.wait_for_function("document.getElementById('fileStatus').textContent.includes('renewal.csv')")
        expected = run(page.evaluate("collect()"))
        page.locator("#run").click()
        page.wait_for_function("!busy && latest !== null", timeout=30_000)
        actual = page.evaluate("latest.headline")
        assert actual["events"] == expected["headline"]["events"]
        assert actual["attempts"] == 2 and actual["failed_attempts"] == 0
        assert actual["status"] == "RESTART_PENDING"
        assert actual["net_cash"] == pytest.approx(7045.85)
        index = next(i for i, e in enumerate(actual["events"])
                     if e["kind"] == "phase_start" and e["attempt"] == 2)
        page.locator("#traceIndex").fill(str(index + 1))
        page.locator("#traceIndex").press("Tab")
        assert json.loads(page.locator("#traceDetails").inner_text())["balance"] == 50_000
        assert page.request.get(url + "montecarlo.html").status == 404
        browser.close()
