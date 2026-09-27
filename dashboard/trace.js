"use strict";
// A viewer over engine-emitted events. Never calculate payout or rule decisions here.
let traceEvents = [],
  traceCash = [],
  traceCursor = 0;
$(traceView ? "traceLink" : "replayLink").classList.add("active");
if (traceView) {
  $("pageTitle").textContent = "Account trace";
  $("pageDescription").textContent =
    "Enter trades or load a history, then step through the current engine's account events. Same rules, sizing and cashflows as strategy replay.";
  $("resultTitle").textContent = "Account lifecycle";
  $("runHeading").textContent = "Trace the fixed policy";
  $("mode").value = "backtest";
  $("mode").disabled = true;
  $("modeField").hidden = true;
  document.querySelectorAll(".fit-only").forEach((e) => {
    e.hidden = true;
    e.querySelectorAll("input").forEach((input) => (input.disabled = true));
  });
  $("mode").onchange();
}
function renderTrace(result) {
  traceEvents = result.events;
  let cash = 0;
  traceCash = traceEvents.map((e) => (cash += e.cash));
  $("tracePanel").hidden = traceEvents.length === 0;
  $("traceSlider").max = traceEvents.length - 1;
  $("traceIndex").max = traceEvents.length;
  selectTrace(traceEvents.length - 1);
}
function selectTrace(index) {
  if (!traceEvents.length) return;
  traceCursor = Math.max(
    0,
    Math.min(traceEvents.length - 1, Math.trunc(Number(index) || 0)),
  );
  const e = traceEvents[traceCursor];
  $("traceSlider").value = traceCursor;
  $("traceIndex").value = traceCursor + 1;
  $("tracePosition").textContent =
    `${traceCursor + 1} / ${traceEvents.length} EVENTS`;
  $("traceLabel").textContent =
    `${e.at} · ${e.kind} · attempt ${e.attempt} / ${e.phase}`;
  $("traceFirst").disabled = $("tracePrevious").disabled = traceCursor === 0;
  $("traceLast").disabled = $("traceNext").disabled =
    traceCursor === traceEvents.length - 1;
  for (const [id, kind] of [
    ["traceNextTrade", "trade"],
    ["traceNextSession", "session_close"],
  ]) {
    $(id).disabled = !traceEvents
      .slice(traceCursor + 1)
      .some((event) => event.kind === kind);
  }
  $("traceState").replaceChildren();
  metric(
    "Closed balance",
    money(e.balance),
    `Attempt ${e.attempt} · ${e.phase}`,
    undefined,
    "traceState",
  );
  metric(
    "Loss floor",
    money(e.floor),
    "At this engine event",
    undefined,
    "traceState",
  );
  metric(
    "Qualifying days",
    String(e.qualifying_days),
    `Cycle profit ${money(e.cycle_profit)}`,
    undefined,
    "traceState",
  );
  metric(
    "External net cash",
    money(traceCash[traceCursor]),
    `This event ${money(e.cash)}`,
    traceCash[traceCursor],
    "traceState",
  );
  $("traceDetails").textContent = JSON.stringify(e, null, 2);
  drawAccount(e);
}
function jumpTrace(kind) {
  const next = traceEvents.findIndex(
    (event, index) => index > traceCursor && event.kind === kind,
  );
  if (next >= 0) selectTrace(next);
}
function drawAccount(event) {
  const svg = $("accountChart"),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  let first = traceCursor;
  while (
    first > 0 &&
    traceEvents[first].kind !== "phase_start" &&
    traceEvents[first - 1].attempt === event.attempt &&
    traceEvents[first - 1].phase === event.phase
  )
    first--;
  // An upfront fee may precede phase creation (balance/floor zero); omit it from
  // the account graph rather than drawing a fictitious $50K trading gain.
  while (first <= traceCursor && traceEvents[first].kind === "fee") first++;
  const events = traceEvents.slice(first, traceCursor + 1);
  if (!events.length) return;
  let low = Infinity,
    high = -Infinity;
  for (const e of events) {
    low = Math.min(low, e.floor, e.balance);
    high = Math.max(high, e.floor, e.balance);
  }
  if (low === high) {
    low -= 1;
    high += 1;
  }
  const x = (i) => 80 + (i / Math.max(1, events.length - 1)) * 600;
  const y = (value) => 190 - ((value - low) / (high - low)) * 165;
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    if (text) node.textContent = text;
    svg.append(node);
  };
  for (const value of [low, (high + low) / 2, high]) {
    add("line", {
      x1: 80,
      x2: 680,
      y1: y(value),
      y2: y(value),
      stroke: "#293340",
    });
    add(
      "text",
      {
        x: 74,
        y: y(value) + 4,
        fill: "#9aa9b9",
        "font-size": 10,
        "text-anchor": "end",
      },
      money(value),
    );
  }
  for (const [field, color] of [
    ["balance", "#63ddc0"],
    ["floor", "#ff9999"],
  ]) {
    add("polyline", {
      points: events.map((e, i) => `${x(i)},${y(e[field])}`).join(" "),
      fill: "none",
      stroke: color,
      "stroke-width": 2,
    });
    add("circle", {
      cx: x(events.length - 1),
      cy: y(event[field]),
      r: 3,
      fill: color,
    });
  }
  add(
    "text",
    { x: 80, y: 218, fill: "#9aa9b9", "font-size": 10 },
    events[0].at.slice(0, 16),
  );
  add(
    "text",
    { x: 680, y: 218, fill: "#9aa9b9", "font-size": 10, "text-anchor": "end" },
    event.at.slice(0, 16),
  );
}
$("traceSlider").oninput = () => selectTrace($("traceSlider").value);
$("traceIndex").onchange = () => selectTrace(Number($("traceIndex").value) - 1);
$("traceFirst").onclick = () => selectTrace(0);
$("tracePrevious").onclick = () => selectTrace(traceCursor - 1);
$("traceNext").onclick = () => selectTrace(traceCursor + 1);
$("traceLast").onclick = () => selectTrace(traceEvents.length - 1);
$("traceNextTrade").onclick = () => jumpTrace("trade");
$("traceNextSession").onclick = () => jumpTrace("session_close");
