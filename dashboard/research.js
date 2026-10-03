"use strict";
const $ = (id) => document.getElementById(id);
const local = document.body.dataset.runtime === "server";
const money = (value) =>
  value == null
    ? "Not available"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
      }).format(value);
let ready = false,
  busy = false,
  latest = null,
  worker,
  serial = 0;
function cell(row, value) {
  const td = document.createElement("td");
  td.textContent = value;
  row.append(td);
}
function metric(label, value, detail, sign, target = "metrics") {
  const box = document.createElement("div");
  box.className = "metric";
  const title = document.createElement("span"),
    number = document.createElement("strong"),
    note = document.createElement("small");
  title.textContent = label;
  number.textContent = value;
  note.textContent = detail;
  if (sign !== undefined)
    number.className = sign >= 0 ? "positive" : "negative";
  box.append(title, number, note);
  $(target).append(box);
}
function invalidate() {
  latest = null;
  $("results").hidden = true;
  $("error").hidden = true;
}
function setBusy(value) {
  busy = value;
  $("controls").disabled = value;
  $("run").disabled = value || !ready;
  $("cancel").hidden = !value || local;
}
function fail(error) {
  setBusy(false);
  $("error").textContent = String(error);
  $("error").hidden = false;
  $("status").textContent =
    "Search did not complete. Review the input or engine error.";
}
function collectResearch() {
  const number = (id) => Number($(id).value);
  const levels = (id) =>
    $(id)
      .value.split(",")
      .map((v) => {
        if (!v.trim())
          throw new Error("Dollar levels cannot contain empty entries");
        return Number(v.trim());
      });
  return {
    mode: "model_search",
    profile: "lucidflex_50k_dll_off",
    objective: $("objective").value,
    model: {
      mu: number("mu"),
      sigma: number("sigma"),
      sessions: number("sessions"),
      start_date: $("start_date").value,
    },
    account: {
      eval_fee: number("eval_fee"),
      reset_fee: number("reset_fee"),
      contract_type: "micro",
    },
    config: Object.fromEntries(
      [
        "initial_wallet",
        "cost_per_contract",
        "cost_per_trade",
        "payment_fee",
        "approval_delay_hours",
        "receipt_delay_hours",
        "activation_delay_hours",
        "retry_delay_hours",
      ].map((k) => [k, number(k)]),
    ),
    search: Object.fromEntries(
      ["paths", "generations", "population", "seed", "holdout_seed"].map(
        (k) => [k, number(k)],
      ),
    ),
    initial: { risk: number("initial_risk"), target: number("initial_target") },
    bounds: {
      risk: [number("risk_min"), number("risk_max")],
      target: [number("target_min"), number("target_max")],
    },
    choices:
      $("search_actions").value === "grid"
        ? { risk: levels("risk_levels"), target: levels("target_levels") }
        : null,
    regime_set: $("regime_set").value,
    risk: {
      target_ruin_probability: number("target_ruin") / 100,
      tail_probability: number("risk_tail") / 100,
    },
  };
}
function boot() {
  ready = false;
  setBusy(false);
  $("retryBoot").hidden = true;
  if (worker) worker.terminate();
  if (local) {
    ready = true;
    $("runtime").textContent = "Local Python engine";
    $("status").textContent =
      "Ready. Discover a policy without supplying trades.";
    setBusy(false);
    return;
  }
  worker = new Worker("replay-worker.js");
  worker.onmessage = ({ data }) => {
    if (data.type === "status") $("runtime").textContent = data.message;
    if (data.type === "ready") {
      ready = true;
      $("runtime").textContent = "Browser engine ready";
      $("build").textContent = `Verified bundle ${data.version}`;
      $("status").textContent =
        "Ready. Discover a policy without supplying trades.";
      setBusy(false);
    }
    if (data.type === "progress")
      $("status").textContent =
        data.stage === "search"
          ? `Searching training paths: ${data.evaluations} / up to ${data.maximum_evaluations} policies. Larger searches can take several minutes.`
          : data.stage === "ruin"
            ? `Final holdout diagnostics: ${data.completed_paths} / ${data.paths} paths.`
            : "Search complete. Evaluating untouched model holdout…";
    if (data.type === "result" && data.id === serial) complete(data.result);
    if (data.type === "error" && data.id === serial) fail(data.message);
    if (data.type === "boot-error") {
      ready = false;
      fail(data.message);
      $("retryBoot").hidden = false;
    }
  };
  worker.onerror = (e) => {
    ready = false;
    worker.terminate();
    fail(e.message);
    $("retryBoot").hidden = false;
  };
}
$("researchForm").onsubmit = async (event) => {
  event.preventDefault();
  if (!ready || busy) return;
  invalidate();
  const id = ++serial;
  try {
    const request = collectResearch();
    setBusy(true);
    $("status").textContent =
      "Searching risk and profit targets on training paths…";
    if (!local) {
      worker.postMessage({ action: "run", id, request });
      return;
    }
    const response = await fetch("api/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    if (id === serial) complete(result);
  } catch (error) {
    if (id === serial) fail(error);
  }
};
function complete(result) {
  latest = {
    ...result,
    runtime: $("runtime").textContent,
    bundle: $("build").textContent,
  };
  setBusy(false);
  $("status").textContent =
    "Completed. Preserve the JSON before changing inputs; repeated tuning on holdout is not fresh validation.";
  const f = result.fit,
    held = f.holdout;
  $("researchScope").textContent =
    `${f.training.paths} training / ${held.paths} holdout model paths; ${result.model.sessions} sessions per path. ${f.evaluations} distinct policies evaluated. Reference example was not seeded. All headline statistics describe the selected policy on model holdout—not historical strategy performance.`;
  $("metrics").replaceChildren();
  metric(
    "Holdout mean objective",
    money(held.score),
    `Standard error ${money(held.score_standard_error)}; ${result.objective}`,
    held.score,
  );
  metric(
    "Training mean objective",
    money(f.training.score),
    "Selection only; not the headline estimate",
  );
  metric(
    "Paired gain over flat baseline",
    money(f.paired_holdout_gain),
    `Standard error ${money(f.paired_gain_standard_error)}`,
    f.paired_holdout_gain,
  );
  metric(
    "Holdout payout probability",
    percentage(held.payout_probability),
    "At least one received payout within the horizon",
  );
  $("researchPolicy").replaceChildren();
  f.policy.sizing.regimes.forEach((r, i) => {
    const row = document.createElement("tr");
    const conditions = [
      r.phase,
      r.days_to_payout == null ? "" : `${r.days_to_payout} days left`,
      r.after_payout == null
        ? ""
        : r.after_payout
          ? "after payout"
          : "before first payout",
      r.in_profit == null ? "" : r.in_profit ? "in profit" : "not in profit",
    ]
      .filter(Boolean)
      .join(" · ");
    [
      r.name,
      conditions,
      `${money(result.initial_policy.sizing.regimes[i].risk_dollars)} / ${money(result.initial_policy.targets[i])}`,
      money(r.risk_dollars),
      money(f.policy.targets[i]),
      f.training.visited_regimes.includes(r.name)
        ? "Yes"
        : "No; baseline retained",
    ].forEach((v) => cell(row, v));
    $("researchPolicy").append(row);
  });
  $("researchComparison").replaceChildren();
  for (const [label, s] of [
    ["Discovered policy", held],
    ["Initial flat policy", f.baseline_holdout],
    ["Named example (reference only)", result.reference_holdout],
  ]) {
    const row = document.createElement("tr");
    [
      label,
      money(s.score),
      money(s.score_standard_error),
      money(s.mean_net_cash),
      `${money(s.net_cash_p05)} / ${money(s.net_cash_median)} / ${money(s.net_cash_p95)}`,
      percentage(s.payout_probability),
    ].forEach((v) => cell(row, v));
    $("researchComparison").append(row);
  }
  renderRisk(result);
  const u = result.cycle_approximation;
  $("ultimateScope").textContent =
    `${result.cycle_records.length} settled cycles from unrestricted-wallet holdout counterparts; ${result.excluded_unsettled_or_open_accounts} unfinished/unsettled accounts excluded. ${u ? "Conditional on this empirical cycle law." : "No ultimate estimate: settled cycles and a finite bankroll are required."}`;
  $("ultimateMetrics").replaceChildren();
  if (u) {
    const range = (b) => b.map(percentage).join(" – ");
    metric(
      "Ultimate model range",
      range(u.probability_bounds),
      `${u.unresolved_paths} unresolved simulation paths`,
      undefined,
      "ultimateMetrics",
    );
    metric(
      "Model confidence range",
      range(u.confidence_bounds),
      `${percentage(u.confidence)}; excludes cycle-law estimation error`,
      undefined,
      "ultimateMetrics",
    );
    metric(
      "Sufficient ultimate bankroll",
      u.sufficient_bankroll == null
        ? "No finite bound"
        : money(u.sufficient_bankroll),
      `Target ${percentage(u.target)}; sufficient bound, not the minimum`,
      undefined,
      "ultimateMetrics",
    );
  }
  $("researchDecisions").replaceChildren();
  for (const d of result.representative_path.decisions.slice(0, 500)) {
    const row = document.createElement("tr");
    [
      d.session_index + 1,
      d.regime,
      money(d.buffer),
      money(d.net_risk),
      money(d.net_target),
      percentage(d.probability),
      d.won ? "Win" : "Loss",
    ].forEach((v) => cell(row, v));
    $("researchDecisions").append(row);
  }
  $("provenance").textContent = JSON.stringify(
    {
      request: result.request,
      scope: result.scope,
      assumptions: result.assumptions,
      spec: result.spec,
      policy: f.policy,
      reference_used_for_selection: result.reference_used_for_selection,
    },
    null,
    2,
  );
  $("results").hidden = false;
}
$("researchForm").addEventListener("input", invalidate);
$("search_actions").onchange = () => {
  document.querySelectorAll(".grid-field").forEach((e) => {
    e.hidden = $("search_actions").value !== "grid";
    e.querySelector("input").disabled = e.hidden;
  });
  invalidate();
};
$("download").onclick = () => {
  if (!latest) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(latest, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = "propfirm-target-search.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
$("cancel").onclick = () => {
  ++serial;
  invalidate();
  boot();
  $("status").textContent = "Search canceled. Reloading the isolated engine…";
};
$("retryBoot").onclick = boot;
boot();
