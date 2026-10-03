"use strict";
// Display the canonical cash report; no simulation or confidence calculations here.
const percentage = (v) =>
  v == null ? "Not available" : `${(100 * v).toFixed(2)}%`;
function renderRisk(result) {
  const r = result.risk;
  $("riskPanel").hidden = !r;
  if (!r) return;
  const single = r.paths < 2,
    d = r.distributions,
    o = r.options;
  const model = result.mode === "model_search";
  const scope = result.mode === "fit" ? "OOS only" : "Full history; not OOS";
  const source =
    result.request.history_source?.kind === "synthetic"
      ? "Synthetic trade history. "
      : "Provided trade history. ";
  $("riskScope").textContent = single
    ? `${source}${scope}. One observed path cannot estimate a result distribution or future ruin probability. Select rolling historical starts to compare multiple windows.`
    : `${source}${scope}. ${r.paths} historical windows; ${d.calendar_days.minimum.toFixed(2)}–${d.calendar_days.maximum.toFixed(2)} elapsed calendar days per window. Overlapping windows are dependent. Performance uses the configured wallet; bankroll requirements use the same policy with funding constraints removed.`;
  if (model)
    $("riskScope").textContent =
      `${r.paths} independent model holdout paths; fixed selected policy, configured wallet. Confidence describes Monte Carlo sampling conditional on the model, not uncertainty about real-market performance.`;
  $("riskDistributionViews").hidden = single;
  $("riskMetrics").replaceChildren();
  if (single) {
    metric(
      "Cash needed for this path",
      money(r.required_bankroll),
      "Observed fee/receipt sequence; not a future safety guarantee",
      undefined,
      "riskMetrics",
    );
    metric(
      "Maximum cash drawdown",
      money(d.max_cash_drawdown.mean),
      "External cash, not trading balance",
      undefined,
      "riskMetrics",
    );
    metric(
      "Longest underwater period",
      `${d.longest_underwater_days.mean.toFixed(2)} days`,
      "Includes unrecovered drawdown at horizon",
      undefined,
      "riskMetrics",
    );
    metric(
      "Funding shortfall on this path",
      r.observed_funding_shortfall_frequency ? "Yes" : "No",
      "Configured-wallet replay",
      undefined,
      "riskMetrics",
    );
  } else {
    metric(
      "Net-loss frequency",
      percentage(r.probability_loss),
      `Profitable: ${percentage(r.probability_profitable)} · break-even: ${percentage(r.probability_break_even)}`,
      undefined,
      "riskMetrics",
    );
    metric(
      "Finite-horizon funding failure",
      percentage(r.ruin_probability),
      o.bankroll == null
        ? "No finite analysis bankroll specified"
        : `${r.ruined_paths} / ${r.paths} ${model ? "paths" : "windows"} at ${money(o.bankroll)}`,
      undefined,
      "riskMetrics",
    );
    metric(
      "Required starting bankroll",
      money(r.required_bankroll),
      `Empirical ruin ≤ ${percentage(o.target_ruin_probability)} over this horizon`,
      undefined,
      "riskMetrics",
    );
    metric(
      "Net cash standard deviation",
      money(d.net_cash.standard_deviation),
      `Across ${r.paths} ${model ? "model-path" : "window"} outcomes`,
      undefined,
      "riskMetrics",
    );
    drawBankroll(r);
    $("riskCapitalNote").textContent =
      `Bankroll covers the largest cash deficit before receipts arrive, not just the final loss. At ${money(r.required_bankroll)}, observed shortfall frequency is ${percentage(r.achieved_empirical_ruin_probability)}. No independent-sample confidence guarantee is available from these historical windows; zero observed failures does not mean zero future risk.`;
    if (model)
      $("riskCapitalNote").textContent =
        `Capital covers interim cash deficits, not only terminal losses. ${r.confidence_supported_bankroll == null ? "Too few independent paths to support the requested bankroll risk at this confidence level." : `${percentage(o.confidence)} confidence-supported bankroll: ${money(r.confidence_supported_bankroll)}.`} This is finite-horizon model risk; zero observed failures does not establish zero future risk.`;
    const head = document.createElement("tr");
    for (const label of [
      "Measure",
      "N",
      "Mean",
      "Std dev",
      "Variance",
      ...o.percentiles.map((q) => `P${(q * 100).toLocaleString()}`),
    ]) {
      const th = document.createElement("th");
      th.textContent = label;
      head.append(th);
    }
    $("riskDistributionHead").replaceChildren(head);
    $("riskDistributions").replaceChildren();
    for (const [key, label, dollars] of [
      ["net_cash", "Net cash ($)", true],
      ["net_cash_per_day", "Cash / calendar day ($/day)", true],
      ["max_cash_drawdown", "Maximum cash drawdown ($)", true],
      ["required_bankroll", "Uninterrupted-policy bankroll ($)", true],
      ["receipts", "Received payouts ($)", true],
      ["fees", "Account fees ($)", true],
      ["outstanding_payouts", "Outstanding payouts ($)", true],
      ["payout_count", "Received payout count", false],
      ["failed_attempts", "Failed attempts", false],
      ["longest_underwater_days", "Longest underwater period (days)", false],
      [
        "days_to_first_receipt",
        "Days to first receipt (recipients only)",
        false,
      ],
      ["return_on_initial_wallet", "Net cash / initial wallet (ratio)", false],
    ]) {
      const row = document.createElement("tr"),
        stats = d[key];
      cell(row, label);
      cell(row, stats?.count ?? 0);
      const format = (v) =>
        v == null ? "—" : dollars ? money(v) : v.toFixed(3);
      cell(row, format(stats?.mean));
      cell(row, format(stats?.standard_deviation));
      cell(
        row,
        stats
          ? stats.variance.toLocaleString(undefined, {
              maximumFractionDigits: 2,
            }) + " (units²)"
          : "—",
      );
      const quantiles = new Map(
        Object.entries(stats?.percentiles || {}).map(([key, value]) => [
          Number(key),
          value,
        ]),
      );
      for (const q of o.percentiles) cell(row, format(quantiles.get(q)));
      $("riskDistributions").append(row);
    }
    $("riskTailNote").textContent =
      `Worst ${percentage(o.tail_probability)}: loss VaR ${money(r.loss_var)}; expected shortfall ${money(r.loss_expected_shortfall)}; mean net cash ${money(r.worst_tail_mean_net_cash)}. Loss is max(0, −net cash); expected shortfall averages the worst tail with fractional observation weights.`;
  }
  $("riskDefinitions").replaceChildren();
  for (const [name, definition] of Object.entries(r.definitions)) {
    const term = document.createElement("dt"),
      text = document.createElement("dd");
    term.textContent = name[0].toUpperCase() + name.slice(1);
    text.textContent = definition;
    $("riskDefinitions").append(term, text);
  }
}

function drawBankroll(report, targetId = "bankrollChart") {
  const svg = $(targetId),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const rows = report.bankroll_curve,
    maximum = Math.max(1, rows.at(-1).bankroll);
  const x = (value) => 70 + (610 * value) / maximum,
    y = (value) => 205 - 180 * value;
  function add(tag, attrs, text) {
    const e = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs))
      e.setAttribute(key, value);
    if (text !== undefined) e.textContent = text;
    svg.append(e);
    return e;
  }
  for (const value of [0, 0.5, 1]) {
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
        "font-size": 11,
        "text-anchor": "end",
      },
      percentage(value),
    );
  }
  let points = [],
    previous = rows[0].ruin_probability;
  for (const row of rows) {
    points.push(
      `${x(row.bankroll)},${y(previous)}`,
      `${x(row.bankroll)},${y(row.ruin_probability)}`,
    );
    previous = row.ruin_probability;
  }
  points.push(`${x(maximum)},${y(previous)}`);
  add("polyline", {
    points: points.join(" "),
    fill: "none",
    stroke: "#63ddc0",
    "stroke-width": 2,
  });
  const target = report.options.target_ruin_probability;
  add("line", {
    x1: 70,
    x2: 680,
    y1: y(target),
    y2: y(target),
    stroke: "#e7b868",
    "stroke-dasharray": "5 4",
  });
  // All curve records stay in the export; bound interactive marks for large histories.
  const stride = Math.max(1, Math.ceil(rows.length / 250));
  rows
    .filter((_, i) => i % stride === 0 || i === rows.length - 1)
    .forEach((row) => {
      const description = `${money(row.bankroll)} bankroll: ${row.ruined_paths}/${report.paths} shortfalls (${percentage(row.ruin_probability)})`;
      const mark = add("circle", {
        cx: x(row.bankroll),
        cy: y(row.ruin_probability),
        r: 3,
        fill: "#63ddc0",
        tabindex: 0,
        "aria-label": description,
      });
      const title = document.createElementNS(ns, "title");
      title.textContent = description;
      mark.append(title);
    });
  for (const value of [0, maximum / 2, maximum])
    add(
      "text",
      {
        x: x(value),
        y: 233,
        fill: "#9aa9b9",
        "font-size": 11,
        "text-anchor":
          value === 0 ? "start" : value === maximum ? "end" : "middle",
      },
      money(value),
    );
  add(
    "text",
    { x: 680, y: 16, fill: "#e7b868", "font-size": 11, "text-anchor": "end" },
    `Dashed: target ${percentage(target)}`,
  );
}

function renderRuin(result) {
  const r = result.ruin;
  $("ruinPanel").hidden = !r;
  if (!r) return;
  const risk = r.risk,
    cycle = r.cycle_approximation;
  const scope =
    result.mode === "fit"
      ? "OOS sessions only; frozen IS-selected policy"
      : "Full history; fixed policy, not OOS";
  $("ruinScope").textContent =
    `${scope}. ${r.settings.paths} independent bootstrap paths, ${r.settings.sessions} sessions each, mean block ${r.settings.mean_block}. All payouts retained. Future outcomes are conditional on this resampled history; path count does not create more historical evidence.`;
  $("ruinMetrics").replaceChildren();
  metric(
    "Finite-horizon funding failure",
    percentage(risk.ruin_probability),
    `At ${money(risk.options.bankroll)} starting cash`,
    undefined,
    "ruinMetrics",
  );
  metric(
    "Finite-horizon bankroll",
    money(risk.required_bankroll),
    `Empirical target ${percentage(risk.options.target_ruin_probability)}`,
    undefined,
    "ruinMetrics",
  );
  metric(
    "Confidence-supported bankroll",
    risk.confidence_supported_bankroll == null
      ? "Insufficient paths"
      : money(risk.confidence_supported_bankroll),
    `${percentage(risk.options.confidence)} confidence; bootstrap model only`,
    undefined,
    "ruinMetrics",
  );
  $("ruinHorizons").replaceChildren();
  $("ruinDistributions").replaceChildren();
  for (const [key, label] of [
    ["net_cash", "Net cash"],
    ["max_cash_drawdown", "Maximum cash drawdown"],
    ["required_bankroll", "Cash needed"],
  ]) {
    const d = risk.distributions[key],
      row = document.createElement("tr");
    const q = d.percentiles;
    [
      label,
      money(d.mean),
      money(d.standard_deviation),
      `${d.variance.toLocaleString(undefined, { maximumFractionDigits: 2 })} USD²`,
      q["0.05"] == null ? "Not requested" : money(q["0.05"]),
      money(d.median),
      q["0.95"] == null ? "Not requested" : money(q["0.95"]),
    ].forEach((value) => cell(row, value));
    $("ruinDistributions").append(row);
  }
  for (const item of r.horizon_curve) {
    const row = document.createElement("tr");
    [
      item.sessions,
      `${item.ruined_paths} / ${r.settings.paths}`,
      percentage(item.probability),
    ].forEach((value) => cell(row, value));
    $("ruinHorizons").append(row);
  }
  drawBankroll(risk, "ruinBankrollChart");
  $("ultimateMetrics").replaceChildren();
  $("ultimateScope").textContent =
    `${r.cycle_records.length} complete, settled account cycles; ${r.excluded_unsettled_or_open_accounts} open or unsettled accounts excluded. This exclusion can bias the model. Cycles are treated as independent, with receipts settled before the next sampled cycle; cross-account receipt overlap and fee-state dependence are not preserved.`;
  if (!cycle) {
    $("ultimateLimits").textContent =
      "Ultimate cycle estimate unavailable: complete settled cycles and a finite analysis bankroll are required. Full-engine survivors are unresolved, not permanently safe.";
    return;
  }
  const range = (bounds) =>
    bounds[0] === bounds[1]
      ? percentage(bounds[0])
      : `${percentage(bounds[0])} – ${percentage(bounds[1])}`;
  metric(
    "Ultimate ruin: model range",
    range(cycle.probability_bounds),
    "Includes unresolved continuation risk",
    undefined,
    "ultimateMetrics",
  );
  metric(
    "Model confidence range",
    range(cycle.confidence_bounds),
    `${percentage(cycle.confidence)}; excludes cycle-law estimation error`,
    undefined,
    "ultimateMetrics",
  );
  metric(
    "Sufficient ultimate bankroll",
    cycle.sufficient_bankroll == null
      ? "No finite bound"
      : money(cycle.sufficient_bankroll),
    `Model upper bound for target ${percentage(cycle.target)}; not the minimum`,
    undefined,
    "ultimateMetrics",
  );
  metric(
    "Mean cash / sampled cycle",
    money(cycle.mean_cycle_cash),
    "All receipts minus account fees",
    undefined,
    "ultimateMetrics",
  );
  $("ultimateLimits").textContent =
    `${cycle.unresolved_paths.toLocaleString()} unresolved paths at the computational limit. Nonpositive mean cash with possible negative cycles implies eventual ruin in this IID model, not proof of real-world ruin. Full-engine ultimate ruin remains unidentified: finite simulation alone cannot establish permanent survival. Full cash distributions and cycle records are included in the JSON export.`;
}

$("printReport").onclick = () => window.print();
let printDetails = [];
window.addEventListener("beforeprint", () => {
  printDetails = [...$("results").querySelectorAll("details")].map((e) => [
    e,
    e.open,
  ]);
  printDetails.forEach(([e]) => (e.open = true));
});
window.addEventListener("afterprint", () =>
  printDetails.forEach(([e, open]) => (e.open = open)),
);
