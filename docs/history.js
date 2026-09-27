"use strict";
// Input construction only. All generation and account execution use Python.
let sourceMetadata = null,
  manualTrades = [],
  fileReadSerial = 0;
const generatorIds = [
  "generator",
  "win_rate",
  "rr",
  "stop_loss",
  "trades_per_day",
  "sessions",
  "seed",
  "start_date",
];

function syntheticParameters() {
  const params = Object.fromEntries(
    generatorIds.map((key) => [
      key,
      ["generator", "start_date"].includes(key)
        ? $("gen_" + key).value
        : Number($("gen_" + key).value),
    ]),
  );
  params.win_rate /= 100;
  if (params.generator === "regime") {
    params.persistence = Number($("gen_persistence").value);
    params.spread = Number($("gen_spread").value) / 100;
  }
  if (params.generator === "stochvol") {
    params.vol_phi = Number($("gen_vol_phi").value);
    params.vol_sigma = Number($("gen_vol_sigma").value);
  }
  return params;
}
function generatorHint() {
  const p = syntheticParameters();
  $("regimeParams").hidden = $("regimeParams").disabled =
    p.generator !== "regime";
  $("volParams").hidden = $("volParams").disabled = p.generator !== "stochvol";
  const edge = p.win_rate * (p.rr + 1) - 1;
  $("generatorHint").textContent = Number.isFinite(edge)
    ? `Target gross expectancy: ${edge.toFixed(3)} R/trade · breakeven win rate: ${(100 / (p.rr + 1)).toFixed(1)}%. ${p.generator === "stochvol" ? "Volatility scales both stop and target; their ratio stays fixed." : "Fixed stop and target per contract."} One-minute trades, weekday sessions; no holiday filter.`
    : "Set a win rate and reward/risk ratio.";
}
function clearHistory() {
  csvText = "";
  sourceName = "";
  sourceMetadata = null;
  $("downloadHistory").disabled = true;
}
function applyHistory(payload, name) {
  csvText = payload.csv;
  sourceName = name;
  sourceMetadata = { ...payload.provenance, statistics: payload.stats };
  const s = payload.stats;
  $("fileStatus").textContent =
    `${s.trades} trades · ${s.sessions} sessions · realized win rate ${(s.win_rate * 100).toFixed(1)}% · mean reward/risk ${s.mean_rr.toFixed(2)} · mean stop ${money(s.mean_stop)} per contract`;
  $("downloadHistory").disabled = false;
}
async function ensureHistory() {
  const source = $("sourceType").value;
  if (source === "synthetic" && !csvText) {
    $("status").textContent = "Generating bracket history…";
    applyHistory(
      await rpc("generate", syntheticParameters()),
      "synthetic-history.csv",
    );
  } else if (source === "manual") {
    $("status").textContent = "Preparing manual bracket history…";
    applyHistory(
      await rpc("manual", { trades: manualTrades }),
      "manual-history.csv",
    );
  }
  if (!csvText) throw new Error("Load or enter a trade history first.");
}
function changeSource() {
  ++fileReadSerial;
  clearHistory();
  invalidate();
  const source = $("sourceType").value;
  for (const panel of document.querySelectorAll("[data-source]")) {
    panel.hidden = panel.disabled = panel.dataset.source !== source;
  }
  $("csvFile").value = "";
  $("fileStatus").textContent =
    source === "synthetic"
      ? "Generate a preview or run directly with these parameters."
      : source === "manual"
        ? `${manualTrades.length} hand-entered trades. Add a win or loss to run the trace.`
        : "Choose a bracket-history CSV.";
  generatorHint();
}
$("sourceType").onchange = changeSource;
$("syntheticPanel").addEventListener("input", () => {
  clearHistory();
  generatorHint();
  $("fileStatus").textContent =
    "Parameters changed. Generate or run to create the new history.";
});
$("generate").onclick = async () => {
  if (!ready || busy) return;
  if (
    ![...$("syntheticPanel").querySelectorAll("input,select")].every((e) =>
      e.reportValidity(),
    )
  )
    return;
  const serial = ++operationSerial;
  invalidate();
  clearHistory();
  setBusy(true);
  try {
    const payload = await rpc("generate", syntheticParameters());
    if (serial !== operationSerial) return;
    applyHistory(payload, "synthetic-history.csv");
    setBusy(false);
    $("status").textContent =
      "History generated. Replay it, fit a policy, or download the CSV.";
  } catch (error) {
    if (serial === operationSerial) fail(String(error));
  }
};
$("csvFile").onchange = async () => {
  const serial = ++fileReadSerial;
  invalidate();
  clearHistory();
  const file = $("csvFile").files[0];
  if (!file) return;
  if (file.size > 5_000_000) {
    fail("CSV exceeds the 5 MB dashboard limit.");
    return;
  }
  const text = await file.text();
  if (serial !== fileReadSerial) return;
  csvText = text;
  sourceName = file.name;
  sourceMetadata = { kind: "uploaded", filename: file.name };
  $("fileStatus").textContent =
    `${file.name} · ${file.size.toLocaleString()} bytes`;
  $("downloadHistory").disabled = false;
};
$("downloadHistory").onclick = () => {
  if (csvText) download(sourceName, csvText, "text/csv");
};

function nextSession() {
  const value = $("manual_session").value;
  const day = new Date(value + "T12:00:00Z");
  if (!Number.isFinite(day.getTime())) return;
  do {
    day.setUTCDate(day.getUTCDate() + 1);
  } while ([0, 6].includes(day.getUTCDay()));
  $("manual_session").value = day.toISOString().slice(0, 10);
  $("manual_entry_time").value = "10:00";
}
function renderManualRows() {
  $("manualRows").replaceChildren();
  for (const t of manualTrades.slice(-100)) {
    const row = document.createElement("tr");
    [
      t.session,
      t.entry_time,
      `${money(t.stop_loss)} / ${money(t.take_profit)}`,
      t.won ? "Win" : "Loss",
    ].forEach((value) => cell(row, value));
    $("manualRows").append(row);
  }
}
async function addManual(won) {
  if (!ready || busy) return;
  if (
    ![...$("manualPanel").querySelectorAll("input")].every((e) =>
      e.reportValidity(),
    )
  )
    return;
  const row = Object.fromEntries(
    [
      "session",
      "entry_time",
      "duration_minutes",
      "stop_loss",
      "take_profit",
    ].map((key) => [
      key,
      ["session", "entry_time"].includes(key)
        ? $("manual_" + key).value
        : Number($("manual_" + key).value),
    ]),
  );
  row.won = won;
  manualTrades.push(row);
  renderManualRows();
  clearHistory();
  $("mode").value = "backtest";
  $("mode").onchange();
  await runCurrent("backtest");
  if (!csvText) {
    manualTrades.pop();
    renderManualRows();
    return;
  }
  const [hours, minutes] = row.entry_time.split(":").map(Number);
  const next = hours * 60 + minutes + row.duration_minutes + 1;
  if (next + row.duration_minutes > 16 * 60 + 45) nextSession();
  else
    $("manual_entry_time").value =
      `${String(Math.floor(next / 60)).padStart(2, "0")}:${String(next % 60).padStart(2, "0")}`;
}
$("addWin").onclick = () => addManual(true);
$("addLoss").onclick = () => addManual(false);
$("nextSession").onclick = nextSession;
$("undoTrade").onclick = () => {
  manualTrades.pop();
  renderManualRows();
  clearHistory();
  invalidate();
  if (manualTrades.length) runCurrent("backtest");
  else $("fileStatus").textContent = "No hand-entered trades.";
};
$("clearTrades").onclick = () => {
  manualTrades = [];
  renderManualRows();
  clearHistory();
  invalidate();
  $("fileStatus").textContent = "Manual history cleared.";
};
if (traceView) $("sourceType").value = "manual";
changeSource();
