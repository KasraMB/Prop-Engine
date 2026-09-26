"use strict";
const $ = (id) => document.getElementById(id);
const local = document.body.dataset.runtime === "server";
let worker,
  ready = false,
  busy = false,
  csvText = "",
  sourceName = "",
  latest = null;
let rowNumber = 0;
const money = (v) =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(v);
const configFields = [
  "cost_per_contract",
  "cost_per_trade",
  "payment_fee",
  "initial_wallet",
  "approval_delay_hours",
  "receipt_delay_hours",
  "activation_delay_hours",
  "retry_delay_hours",
];

function cell(row, text) {
  const td = document.createElement("td");
  td.textContent = text;
  row.append(td);
  return td;
}
function addRegime(values) {
  const row = document.createElement("tr");
  const defaults = {
    name: "regime_" + ++rowNumber,
    phase: "funded",
    in_profit: "",
    days_to_payout: "",
    after_payout: "",
    risk_dollars: 150,
    minimum: 50,
    maximum: 500,
  };
  for (const [key, value] of Object.entries({ ...defaults, ...values })) {
    const td = cell(row, "");
    let field;
    if (["phase", "in_profit", "after_payout"].includes(key)) {
      field = document.createElement("select");
      const options =
        key === "phase"
          ? [
              ["eval", "Evaluation"],
              ["funded", "Funded"],
            ]
          : [
              ["", "Any"],
              ["true", "Yes"],
              ["false", "No"],
            ];
      options.forEach(([v, label]) => field.add(new Option(label, v)));
    } else {
      field = document.createElement("input");
      field.type = key === "name" ? "text" : "number";
      if (key !== "name") {
        field.min = "0";
        field.step = key === "days_to_payout" ? "1" : "any";
      }
      field.required = key !== "days_to_payout";
    }
    field.dataset.field = key;
    field.setAttribute("aria-label", key.replaceAll("_", " "));
    field.value = value == null ? "" : value;
    td.append(field);
  }
  const actions = cell(row, "");
  actions.className = "row-actions";
  for (const [label, title, action] of [
    [
      "↑",
      "Move regime up",
      () => {
        if (row.previousElementSibling) row.before(row.previousElementSibling);
      },
    ],
    [
      "↓",
      "Move regime down",
      () => {
        if (row.nextElementSibling) row.after(row.nextElementSibling);
      },
    ],
    ["×", "Remove regime", () => row.remove()],
  ]) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.setAttribute("aria-label", title);
    button.onclick = () => {
      action();
      invalidate();
    };
    actions.append(button);
  }
  $("regimes").append(row);
}
[
  { name: "evaluation", phase: "eval", risk_dollars: 200 },
  { name: "payout_days_complete", days_to_payout: 0 },
  { name: "one_day_to_payout", days_to_payout: 1 },
  { name: "two_days_to_payout", days_to_payout: 2 },
  { name: "after_first_payout", after_payout: true },
  { name: "funded_in_profit", in_profit: true, risk_dollars: 200 },
  { name: "funded_fallback", risk_dollars: 150 },
].forEach(addRegime);

function invalidate() {
  latest = null;
  $("results").hidden = true;
  $("error").hidden = true;
}
function setBusy(value) {
  busy = value;
  $("controls").disabled = value;
  $("run").disabled = !ready || value;
  $("cancel").hidden = !value || local;
}
function fail(message) {
  setBusy(false);
  $("error").textContent = message;
  $("error").hidden = false;
  $("status").textContent =
    "No result produced. Review the inputs or engine error below.";
}
function startEngine() {
  ready = false;
  $("retryBoot").hidden = true;
  $("run").disabled = true;
  if (local) {
    ready = true;
    $("runtime").textContent = "Local Python engine";
    $("privacy").textContent =
      "Trade contents are sent only to your local dashboard server, not to GitHub.";
    $("status").textContent =
      "Ready. Load a history and confirm its input contract.";
    setBusy(false);
    return;
  }
  if (worker) worker.terminate();
  worker = new Worker("replay-worker.js");
  worker.onmessage = ({ data }) => {
    if (data.type === "status") $("runtime").textContent = data.message;
    if (data.type === "ready") {
      ready = true;
      $("runtime").textContent = "Browser engine ready";
      $("build").textContent =
        "Verified bundle " + data.version + " · Pyodide 0.26.4";
      $("status").textContent =
        "Ready. Load a history and confirm its input contract.";
      setBusy(false);
    }
    if (data.type === "result") complete(data.result);
    if (data.type === "error") fail(data.message);
    if (data.type === "boot-error") {
      ready = false;
      fail(data.message);
      $("runtime").textContent = "Engine failed to load";
      $("retryBoot").hidden = false;
    }
  };
  worker.onerror = (event) => {
    ready = false;
    fail(event.message || "Browser worker failed");
    $("retryBoot").hidden = false;
  };
}
function collect() {
  if (!csvText) throw new Error("Load a bracket-history CSV first.");
  const regimes = [],
    risk_bounds = {};
  for (const row of $("regimes").rows) {
    const fields = Object.fromEntries(
      [...row.querySelectorAll("[data-field]")].map((e) => [
        e.dataset.field,
        e.value,
      ]),
    );
    regimes.push({
      name: fields.name,
      phase: fields.phase,
      risk_dollars: Number(fields.risk_dollars),
      in_profit: fields.in_profit === "" ? null : fields.in_profit === "true",
      after_payout:
        fields.after_payout === "" ? null : fields.after_payout === "true",
      days_to_payout:
        fields.days_to_payout === "" ? null : Number(fields.days_to_payout),
    });
    risk_bounds[fields.name] = [Number(fields.minimum), Number(fields.maximum)];
  }
  return {
    profile: "lucidflex_50k_dll_off",
    mode: $("mode").value,
    csv: csvText,
    accept_bracket_contract: $("accept").checked,
    objective: $("objective").value,
    account: {
      eval_fee: Number($("eval_fee").value),
      reset_fee: Number($("reset_fee").value),
      contract_type: $("contract_type").value,
    },
    config: Object.fromEntries(
      configFields.map((key) => [key, Number($(key).value)]),
    ),
    regimes,
    risk_bounds,
    search: Object.fromEntries(
      ["generations", "population", "seed"].map((key) => [
        key,
        Number($(key).value),
      ]),
    ),
  };
}
$("replayForm").onsubmit = async (event) => {
  event.preventDefault();
  if (!ready || busy) return;
  invalidate();
  try {
    const request = collect();
    setBusy(true);
    $("status").textContent =
      request.mode === "fit"
        ? "Fitting on IS only, then evaluating the frozen policy on OOS…"
        : "Replaying the full history with the fixed policy…";
    if (local) {
      const response = await fetch("api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Replay failed");
      complete(result);
    } else worker.postMessage({ type: "run", request });
  } catch (error) {
    fail(String(error));
  }
};
function complete(result) {
  latest = {
    ...result,
    source_name: sourceName,
    runtime: $("runtime").textContent,
    bundle: $("build").textContent,
  };
  setBusy(false);
  $("status").textContent =
    "Completed. Changing inputs clears these results; preserve the JSON before editing.";
  render(latest);
}
function metric(label, value, detail, sign) {
  const box = document.createElement("div");
  box.className = "metric";
  const name = document.createElement("span");
  name.textContent = label;
  const number = document.createElement("strong");
  number.textContent = value;
  if (sign !== undefined)
    number.className = sign >= 0 ? "positive" : "negative";
  const note = document.createElement("small");
  note.textContent = detail;
  box.append(name, number, note);
  $("metrics").append(box);
}
function render(result) {
  const r = result.headline,
    baseline = result.baseline;
  $("scope").textContent = result.headline_scope;
  $("split").textContent = result.split
    ? `IS: ${result.split.train.sessions} sessions (${result.split.train.first_session} – ${result.split.train.last_session}) · OOS: ${result.split.test.sessions} sessions (${result.split.test.first_session} – ${result.split.test.last_session}). ${result.split.boundary_policy}. ${result.evaluations} IS candidates evaluated.`
    : `${result.input.sessions} sessions · ${result.input.trades} input trades. This is a fixed-policy replay, not a held-out estimate.`;
  $("metrics").replaceChildren();
  metric(
    result.objective === "net_cash"
      ? "Objective · net external cash"
      : "Objective · net cash / calendar day",
    money(result.score),
    "Observed result, not population EV",
    result.score,
  );
  metric(
    "Net external cash",
    money(r.net_cash),
    baseline
      ? "Initial policy OOS: " + money(baseline.net_cash)
      : "Receipts minus account fees",
    r.net_cash,
  );
  metric(
    "Cash received / fees paid",
    money(r.receipts),
    money(r.fees) + " in account fees",
  );
  metric(
    "Terminal status",
    r.status,
    `${r.attempts} attempts · ${r.failed_attempts} failed`,
  );
  metric(
    "Executed trades",
    String(r.executed_trades),
    `${r.calendar_days.toFixed(2)} elapsed calendar days`,
  );
  metric(
    "Final trading balance",
    r.final_balance == null ? "—" : money(r.final_balance),
    "Not added to external cash",
  );
  metric(
    "Outstanding payouts",
    money(r.outstanding_payouts),
    "Not counted as cash received",
  );
  metric(
    "OOS baseline cash / day",
    baseline ? money(baseline.net_cash_per_day) : "Not applicable",
    "Initial policy, same scenario and holdout",
  );
  const observed = new Set(
    (result.training?.events || []).map((e) => e.regime).filter(Boolean),
  );
  $("selectedPolicy").replaceChildren();
  result.policy.regimes.forEach((regime, i) => {
    const row = document.createElement("tr");
    [
      regime.name,
      money(result.request.regimes[i].risk_dollars),
      money(regime.risk_dollars),
      result.training
        ? observed.has(regime.name)
          ? "Yes"
          : "No · baseline retained"
        : "Not fitted",
    ].forEach((v) => cell(row, v));
    $("selectedPolicy").append(row);
  });
  $("events").replaceChildren();
  r.events.slice(0, 500).forEach((e) => {
    const row = document.createElement("tr");
    [
      e.at,
      e.kind,
      `${e.attempt} / ${e.phase}`,
      e.quantity,
      money(e.balance),
      money(e.floor),
      money(e.cash),
      [e.regime, e.code].filter(Boolean).join(" / "),
    ].forEach((v) => cell(row, v));
    $("events").append(row);
  });
  $("ledgerCaption").textContent =
    `${r.events.length} events; showing the first ${Math.min(500, r.events.length)}. The JSON download contains every event, including gross payouts and qualifying-day state.`;
  $("provenance").textContent = JSON.stringify(
    {
      source: sourceName,
      csv_sha256: result.csv_sha256,
      history_fingerprint: r.history_fingerprint,
      objective: result.objective,
      direction: result.direction,
      split: result.split,
      training_score: result.training_score,
      seed: result.seed,
      assumptions: r.assumptions,
      spec: r.spec,
      config: r.config,
      policy: result.policy,
    },
    null,
    2,
  );
  drawCash(r, baseline);
  $("results").hidden = false;
}
function drawCash(result, baseline) {
  const svg = $("cashChart"),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const series = [result, ...(baseline ? [baseline] : [])].map((r) => {
    let cash = 0;
    const points = [[Date.parse(r.start), 0]];
    for (const e of r.events)
      if (e.cash) {
        points.push([Date.parse(e.at), cash]);
        cash += e.cash;
        points.push([Date.parse(e.at), cash]);
      }
    points.push([Date.parse(r.end), cash]);
    return points;
  });
  let low = 0,
    high = 0;
  series.forEach((points) =>
    points.forEach(([, y]) => {
      low = Math.min(low, y);
      high = Math.max(high, y);
    }),
  );
  if (low === high) {
    low -= 1;
    high += 1;
  }
  const start = Date.parse(result.start),
    end = Date.parse(result.end);
  const x = (t) => 70 + ((t - start) / Math.max(1, end - start)) * 610;
  const y = (v) => 190 - ((v - low) / (high - low)) * 165;
  const element = (name, attrs, text) => {
    const e = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if (text !== undefined) e.textContent = text;
    svg.append(e);
  };
  [low, (low + high) / 2, high].forEach((v) => {
    element("line", { x1: 70, x2: 680, y1: y(v), y2: y(v), stroke: "#293340" });
    element(
      "text",
      {
        x: 64,
        y: y(v) + 4,
        fill: "#9aa9b9",
        "font-size": 10,
        "text-anchor": "end",
      },
      money(v),
    );
  });
  series.forEach((points, i) =>
    element("polyline", {
      points: points.map(([t, v]) => `${x(t)},${y(v)}`).join(" "),
      fill: "none",
      stroke: i ? "#91a0b2" : "#63ddc0",
      "stroke-width": 2,
    }),
  );
  element(
    "text",
    { x: 70, y: 218, fill: "#9aa9b9", "font-size": 10 },
    result.start.slice(0, 10),
  );
  element(
    "text",
    { x: 680, y: 218, fill: "#9aa9b9", "font-size": 10, "text-anchor": "end" },
    result.end.slice(0, 10),
  );
}
function demoCsv() {
  const lines = ["entry_at,exit_at,session,stop_loss,take_profit,won"];
  const day = new Date("2026-01-05T00:00:00Z");
  let count = 0,
    state = 73;
  while (count < 80) {
    if (day.getUTCDay() !== 0 && day.getUTCDay() !== 6) {
      const date = day.toISOString().slice(0, 10);
      for (let trade = 0; trade < 4; trade++) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        const minute = String(trade * 10).padStart(2, "0"),
          exit = String(trade * 10 + 5).padStart(2, "0");
        lines.push(
          `${date}T15:${minute}:00+00:00,${date}T15:${exit}:00+00:00,${date},100,${trade % 2 ? 150 : 200},${state / 2 ** 32 < 0.55}`,
        );
      }
      count++;
    }
    day.setUTCDate(day.getUTCDate() + 1);
  }
  return lines.join("\n") + "\n";
}
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("csvFile").onchange = async () => {
  invalidate();
  csvText = "";
  const file = $("csvFile").files[0];
  if (!file) return;
  if (file.size > 5_000_000) {
    fail("CSV exceeds the 5 MB dashboard limit.");
    return;
  }
  csvText = await file.text();
  sourceName = file.name;
  $("fileStatus").textContent =
    `${file.name} · ${file.size.toLocaleString()} bytes. Validation runs before replay.`;
};
$("demo").onclick = () => {
  invalidate();
  csvText = demoCsv();
  sourceName = "synthetic-example.csv";
  $("csvFile").value = "";
  $("fileStatus").textContent =
    "Synthetic example · 80 weekday sessions · 320 trades · variable reward/risk. Not market data; no holiday filtering or profitability claim.";
};
$("template").onclick = () =>
  download("synthetic-example.csv", demoCsv(), "text/csv");
$("download").onclick = () => {
  if (latest)
    download(
      "propfirm-replay-result.json",
      JSON.stringify(latest, null, 2),
      "application/json",
    );
};
$("addRegime").onclick = () => {
  addRegime({});
  invalidate();
};
$("replayForm").addEventListener("input", invalidate);
$("mode").onchange = () => {
  const fit = $("mode").value === "fit";
  $("run").textContent = fit ? "Optimize & evaluate" : "Replay fixed policy";
  ["generations", "population", "seed"].forEach(
    (id) => ($(id).disabled = !fit),
  );
};
$("cancel").onclick = () => {
  worker.terminate();
  invalidate();
  setBusy(false);
  startEngine();
  $("status").textContent = "Run canceled. Reloading the isolated engine…";
};
$("retryBoot").onclick = () => {
  invalidate();
  startEngine();
};
startEngine();
