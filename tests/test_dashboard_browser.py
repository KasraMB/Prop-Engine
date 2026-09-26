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
    from dashboard.replay import run
    url, runtime = site
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []; page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function("!document.getElementById('run').disabled || !document.getElementById('error').hidden", timeout=180_000)
        assert page.locator("#error").is_hidden(), page.locator("#error").inner_text()
        assert page.locator("#accept").is_checked() is False
        csv = page.evaluate("demoCsv()")
        page.locator("#csvFile").set_input_files({"name": "history.csv", "mimeType": "text/csv", "buffer": csv.encode()})
        page.wait_for_function("document.getElementById('fileStatus').textContent.includes('history.csv')")
        page.locator("#accept").check()
        page.locator("#generations").fill("2")
        page.locator("#population").fill("4")
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
        assert result["policy"] == expected["policy"]
        assert len(page.locator("#cashChart polyline").all()) == 2
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
        page.locator("#demo").click()
        page.locator("#mode").select_option("backtest")
        page.locator("#run").click()
        page.wait_for_function("!document.getElementById('results').hidden", timeout=30_000)
        assert "not OOS" in page.locator("#scope").inner_text()
        if runtime == "static":
            page.locator("#mode").select_option("fit")
            page.locator("#generations").fill("100")
            page.locator("#population").fill("32")
            page.locator("#run").click()
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
