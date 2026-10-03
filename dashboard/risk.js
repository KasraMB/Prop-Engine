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
  const scope = result.mode === "fit" ? "OOS only" : "Full history; not OOS";
  const source =
    result.request.history_source?.kind === "synthetic"
      ? "Synthetic trade history. "
      : "Provided trade history. ";
  $("riskScope").textContent = single
    ? `${source}${scope}. One observed path cannot estimate a result distribution or future ruin probability. Select rolling historical starts to compare multiple windows.`
    : `${source}${scope}. ${r.paths} historical windows; ${d.calendar_days.minimum.toFixed(2)}–${d.calendar_days.maximum.toFixed(2)} elapsed calendar days per window. Overlapping windows are dependent. Performance uses the configured wallet; bankroll requirements use the same policy with funding constraints removed.`;
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
      "Ruin frequency",
      percentage(r.ruin_probability),
      o.bankroll == null
        ? "No finite analysis bankroll specified"
        : `${r.ruined_paths} / ${r.paths} windows at ${money(o.bankroll)}`,
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
      `Across ${r.paths} window outcomes`,
      undefined,
      "riskMetrics",
    );
    drawBankroll(r);
    $("riskCapitalNote").textContent =
      `Bankroll covers the largest cash deficit before receipts arrive, not just the final loss. At ${money(r.required_bankroll)}, observed shortfall frequency is ${percentage(r.achieved_empirical_ruin_probability)}. No independent-sample confidence guarantee is available from these historical windows; zero observed failures does not mean zero future risk.`;
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

function drawBankroll(report) {
  const svg = $("bankrollChart"),
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
