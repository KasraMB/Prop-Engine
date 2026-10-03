"use strict";
// Display canonical per-window summaries; never simulate or select a policy here.
$("rollingControls").hidden = traceView;
$("rollingMode").disabled = traceView;
$("rollingMode").onchange = () => {
  const enabled = !traceView && $("rollingMode").value === "rolling";
  $("rollingParameters").hidden = $("rollingParameters").disabled = !enabled;
  invalidate();
};

function renderRolling(result) {
  const report = result.rolling?.headline;
  $("rollingPanel").hidden = $("chronologicalNote").hidden = !report;
  if (!report) return;
  const s = report.summary,
    d = s.distributions,
    c = s.first_evaluation_counts;
  const rate = (v) =>
    v == null ? "Not applicable" : `${(100 * v).toFixed(1)}%`;
  $("rollingTitle").textContent =
    result.mode === "fit"
      ? "OOS rolling historical starts"
      : "Full-history rolling starts (not OOS)";
  $("rollingScope").textContent =
    `${s.windows} complete windows · ${report.rolling.window_sessions} sessions each · start every ${report.rolling.stride_sessions} sessions · ${report.excluded_incomplete_starts} incomplete tail starts excluded.${result.rolling.training ? ` Optimizer used ${result.rolling.training.windows.length} IS windows; no window crosses the split.` : ""}`;
  $("rollingMetrics").replaceChildren();
  metric(
    "Mean window cash / day",
    money(d.net_cash_per_day.mean),
    result.rolling.baseline
      ? `Initial policy: ${money(result.rolling.baseline.summary.distributions.net_cash_per_day.mean)}`
      : "Equal-weight mean of per-window rates",
    d.net_cash_per_day.mean,
    "rollingMetrics",
  );
  metric(
    "Mean window net cash",
    money(d.net_cash.mean),
    `Median ${money(d.net_cash.median)}`,
    d.net_cash.mean,
    "rollingMetrics",
  );
  metric(
    "First-evaluation pass rate",
    rate(s.first_evaluation_pass_rate),
    `${c.passed} passed / ${s.first_evaluation_started} started`,
    undefined,
    "rollingMetrics",
  );
  metric(
    "Received-payout frequency",
    rate(s.payout_probability),
    "Across complete windows, including retries",
    undefined,
    "rollingMetrics",
  );
  const passTime = s.days_to_first_pass_among_passes;
  $("rollingOutcomes").textContent =
    `First evaluations: ${c.passed} passed, ${c.failed} failed, ${c.unresolved} unresolved, ${c.not_started} not started, ${c.not_applicable} not applicable. Any MLL breach: ${rate(s.any_mll_breach_probability)}. Mean elapsed days to first pass among passes: ${passTime ? passTime.mean.toFixed(2) : "not available"}.`;
  $("rollingDistributions").replaceChildren();
  for (const [field, label] of [
    ["net_cash_per_day", "Net cash / calendar day"],
    ["net_cash", "Net cash"],
    ["receipts", "Received payouts"],
    ["fees", "Fees"],
    ["max_external_cash_drawdown", "External cash drawdown"],
    ["executed_trades", "Executed trades"],
    ["calendar_days", "Elapsed calendar days"],
  ]) {
    const row = document.createElement("tr");
    cell(row, label);
    for (const stat of ["mean", "median", "p05", "p95"])
      cell(
        row,
        ["executed_trades", "calendar_days"].includes(field)
          ? d[field][stat].toFixed(2)
          : money(d[field][stat]),
      );
    $("rollingDistributions").append(row);
  }
  $("rollingWindows").replaceChildren();
  for (const w of report.windows.slice(0, 500)) {
    const row = document.createElement("tr");
    [
      w.first_session,
      w.last_session,
      w.first_evaluation,
      money(w.net_cash),
      money(w.net_cash_per_day),
      money(w.receipts),
      w.attempts,
    ].forEach((v) => cell(row, v));
    $("rollingWindows").append(row);
  }
  drawRolling(report.windows);
}

function drawRolling(windows) {
  const svg = $("rollingChart"),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const first = Date.parse(windows[0].first_session),
    last = Date.parse(windows.at(-1).first_session);
  let low = Math.min(0, ...windows.map((w) => w.net_cash_per_day)),
    high = Math.max(0, ...windows.map((w) => w.net_cash_per_day));
  if (low === high) {
    low--;
    high++;
  }
  const x = (v) =>
    first === last
      ? 375
      : 70 + ((Date.parse(v) - first) / (last - first)) * 610;
  const y = (v) => 190 - ((v - low) / (high - low)) * 165;
  function add(tag, attrs, text) {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    svg.append(e);
    return e;
  }
  for (const value of [low, (low + high) / 2, high]) {
    add("line", {
      x1: 70,
      x2: 680,
      y1: y(value),
      y2: y(value),
      stroke: "#293340",
    });
    add(
      "text",
      {
        x: 64,
        y: y(value) + 4,
        fill: "#9aa9b9",
        "font-size": 10,
        "text-anchor": "end",
      },
      money(value),
    );
  }
  for (const w of windows) {
    const description = `${w.first_session} to ${w.last_session}: ${money(w.net_cash_per_day)} per calendar day; net cash ${money(w.net_cash)}; first evaluation ${w.first_evaluation}`;
    const dot = add("circle", {
      cx: x(w.first_session),
      cy: y(w.net_cash_per_day),
      r: 3,
      fill: "#63ddc0",
      tabindex: 0,
      "aria-label": description,
    });
    const title = document.createElementNS(ns, "title");
    title.textContent = description;
    dot.append(title);
  }
  add(
    "text",
    { x: 70, y: 218, fill: "#9aa9b9", "font-size": 10 },
    windows[0].first_session,
  );
  add(
    "text",
    { x: 680, y: 218, fill: "#9aa9b9", "font-size": 10, "text-anchor": "end" },
    windows.at(-1).first_session,
  );
}
