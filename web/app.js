// Saw Sound Meter: wires the microphone, the feature maths and the detector to the screen.
import { FeatureExtractor } from "./features.js";
import { Detector, OFF, TRANSITION, RUNNING, GOOD, ODD, STOP } from "./detector.js";
import { MicCapture, Windower, decodeFile } from "./audio.js";
import { APP_VERSION } from "./version.js";

const $ = (id) => document.getElementById(id);
const COLOURS = { [OFF]: "#9e9e9e", [TRANSITION]: "#9e9e9e", [GOOD]: "#1e8e3e", [ODD]: "#f1b000", [STOP]: "#ff5252" };
const TEXT = {
  idle: ["READY", "Tap Start, then put the phone on the saw at the tape mark."],
  [OFF]: ["WAITING", "Motor off. Not judging."],
  [TRANSITION]: ["STARTING", "Motor just started. Judging in a few seconds."],
  [GOOD]: ["GOOD", "Cut sounds normal."],
  [ODD]: ["ODD", "Sound is unusual. Watch the cut."],
  [STOP]: ["STOP", "Sound is far from normal. Stop and check."],
};
const SAVED_KEY = "sawmeter.savedCuts", SENS_KEY = "sawmeter.sensitivity", MATERIAL_KEY = "sawmeter.lastMaterial";
const DRAFT_KEY = "sawmeter.draft";        // the cut in progress, written every 5 s so a reload cannot lose it
const UNLABELLED = "unlabelled";

let model, fx, det, mic, windower, wakeLock;
let listening = false, lastVibrate = 0, beepCtx;
let currentCut = null;                     // features of the cut in progress (for "That cut was GOOD/ODD")
let live = [];                             // recent {t, score, verdict} for the live timeline
let t0 = 0, motorOnAt = null;
let throughAt = null;                      // window index (since motor on) where the user tapped "Blade is through"
let throughGuess = null;                   // window index where the page's own indicator fired
let pendingIndex = null;
let bannerTimer = null;                    // hides the "I take that back" banner after a few seconds                   // position (in the saved list) of the cut waiting for its GOOD/ODD answer

// ---------- start-up ----------
/** Fetch model.json; if it is newer than the one in memory, swap it in. Called at start-up and
 *  again on every Start tap, so a page left open in the lab still picks up a retrained model. */
async function refreshModel() {
  let fresh;
  try { fresh = await (await fetch("model.json", { cache: "no-cache" })).json(); }
  catch (e) { if (model) return false; throw e; }          // offline: keep what we have
  if (model && fresh.trained_at === model.trained_at) return false;
  model = fresh;
  fx = new FeatureExtractor(model);
  det = new Detector(model, loadSensitivity());
  $("sensitivity").value = det.sensitivity; $("sens-out").textContent = det.sensitivity.toFixed(2) + "×";
  placeMarks();
  $("model-info").textContent = `Page ${APP_VERSION}. Numbers calculated from ${model.trained_on.length} good cut(s) on ${model.trained_at.slice(0, 10)}. ` +
    `Yellow above ${model.thresholds.yellow.toFixed(2)}, red above ${model.thresholds.red.toFixed(2)}.` +
    (model.breakthrough ? " Blade-through indicator: on." : " Blade-through indicator: not learned yet.");
  return true;
}

async function init() {
  await refreshModel();
  if (!("vibrate" in navigator)) $("vibrate-note").textContent = "(not available on iPhone: the screen flashes instead)";
  updateSavedCount();
  recoverDraft();
  $("version").textContent = APP_VERSION;
  document.title = `Saw Sound Meter ${APP_VERSION}`;
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost")) {
    navigator.serviceWorker.register("sw.js", { updateViaCache: "none" }).catch(() => {});
    // A newer page was installed while this one was open: reload when it is safe (not listening).
    navigator.serviceWorker.addEventListener("controllerchange", () => { if (!listening) location.reload(); });
  }
}

// ---------- live listening ----------
async function start() {
  $("mic-error").hidden = true;
  try {
    if (await refreshModel()) $("phase").textContent = "Picked up a newer model.";
    det.reset(); live = []; currentCut = null; motorOnAt = null; throughAt = null; throughGuess = null; t0 = performance.now();
    windower = new Windower(fx.windowLength, fx.hopLength, onWindow);
    mic = new MicCapture(model.audio.sr, (s) => windower.push(s));
    const info = await mic.start();
    listening = true;
    $("start").hidden = true; $("stop").hidden = false;
    setState(OFF);
    $("phase").textContent = `Listening (${info.contextRate} Hz${info.resampling ? ", resampled" : ""})`;
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
  } catch (err) {
    $("mic-error").hidden = false;
    $("mic-error").textContent = "Microphone problem: " + (err.message || err) +
      (location.protocol !== "https:" && location.hostname !== "localhost" ? " (the page must be opened over https for the mic to work)" : "");
  }
}

async function stop() {
  listening = false;
  if (currentCut && currentCut.length >= det.motor.minCutWindows) finishCut();      // stopped mid-cut: keep what we have
  else { currentCut = null; clearDraft(); }
  motorOnAt = null;
  await mic?.stop(); wakeLock?.release?.(); wakeLock = null;
  $("start").hidden = false; $("stop").hidden = true;
  setState("idle"); $("phase").textContent = "Not listening"; $("clock").textContent = ""; $("load").textContent = "";
  showThrough(false); $("through-banner").hidden = true;
}

function onWindow(samples) { onFeatures(fx.extract(samples)); }
window.sawmeterDebug = { feed: (f) => onFeatures(f) };            // lets the automated page test push measurements
function onFeatures(f) {
  const r = det.push(f);
  const t = (performance.now() - t0) / 1000;
  if (r.phase !== OFF && motorOnAt === null) {                                       // motor just started
    motorOnAt = t; currentCut = []; throughAt = null; throughGuess = null; showThrough(true); $("through-banner").hidden = true;
  }
  if (currentCut) {
    currentCut.push(Array.from(f, (v) => Math.round(v * 1000) / 1000));
    if (currentCut.length % 10 === 0) saveDraft();
  }
  if (r.phase === OFF && motorOnAt !== null) {                                       // motor stopped
    if (r.cutDone) finishCut(); else { currentCut = null; clearDraft(); }            // too short to be a cut
    motorOnAt = null; showThrough(false); $("through-banner").hidden = true;
  }
  if (r.through && currentCut) {                                                     // the page's own guess
    throughGuess = currentCut.length;
    $("through-banner").textContent = `Sounds like the blade is starting to come through (${fmt(throughGuess * model.audio.hop_s)}). Still tap the button when YOU hear it come free.`;
    $("through-banner").hidden = false; clearTimeout(bannerTimer);
  }
  if (r.retracted && currentCut) {                                                   // ... taken back
    throughGuess = null;
    $("through-banner").textContent = "Never mind, it still sounds like cutting. I take that guess back.";
    clearTimeout(bannerTimer); bannerTimer = setTimeout(() => ($("through-banner").hidden = true), 8000);
  }
  $("load").textContent = r.load === null ? "" : `blade load ${r.load.toFixed(1)}`;

  setState(r.phase === RUNNING ? r.verdict : r.phase);
  $("score").textContent = r.score.toFixed(2);
  $("bar-fill").style.width = Math.min(100, (r.score / (det.red * 1.4)) * 100) + "%";
  $("clock").textContent = motorOnAt === null ? "" : "motor on " + fmt(t - motorOnAt);
  live.push({ t, score: r.score, verdict: r.phase === RUNNING ? r.verdict : OFF });
  if (live.length > 180) live.shift();               // last 90 s
  drawTimeline($("timeline"), live.map((p) => p.score), live.map((p) => p.verdict), null, det.yellow, det.red);
  if (r.verdict === STOP) alertStop();
}

/** The "Blade is through" marker: shown while the motor runs; a tap stamps the time into the cut. */
function showThrough(visible) {
  $("through-row").hidden = !visible;
  if (visible) $("through-note").textContent = "Tap the moment you hear the blade come through the last of the steel.";
}
function markThrough() {
  if (!currentCut) return;
  throughAt = currentCut.length;
  $("through-note").textContent = `Marked at ${fmt(throughAt * model.audio.hop_s)} (motor time). Tap again to move it.`;
}
/** The motor stopped (or Stop was tapped mid-cut): store the cut NOW, before any button is
 *  tapped, so a wrong tap, a closed tab or a dead battery cannot lose it. The GOOD/ODD panel
 *  then only edits the label of that stored cut. */
function finishCut(startedAt = null, recovered = false) {
  if (!currentCut || currentCut.length < 20) { currentCut = null; clearDraft(); return; }
  const cut = { label: UNLABELLED, material: localStorage.getItem(MATERIAL_KEY) || "", notes: "",
    at: startedAt || new Date().toISOString(), duration_s: Math.round(currentCut.length * model.audio.hop_s),
    through_at_s: throughAt === null ? null : Math.round(throughAt * model.audio.hop_s * 10) / 10,
    through_guess_s: throughGuess === null ? null : Math.round(throughGuess * model.audio.hop_s * 10) / 10,
    sensitivity: det.sensitivity, model_trained_at: model.trained_at, feature_names: model.feature_names, features: currentCut };
  const cuts = loadSaved(); cuts.push(cut);
  if (persist(cuts)) pendingIndex = cuts.length - 1;
  else { pendingIndex = null; downloadCuts([cut], "cut"); alert("Phone storage is full, so this cut was downloaded as a file instead of saved on the phone. Download the saved cuts, then Clear, before the next cut."); }
  currentCut = null; throughAt = null; throughGuess = null; clearDraft(); updateSavedCount();
  if (!recovered) { const banner = $("cut-banner"); banner.hidden = false; setTimeout(() => (banner.hidden = true), 10000); }
  if (pendingIndex !== null) openLabelPanel(pendingIndex, recovered ? "Recovered a cut that was never labelled. It is saved. How was it?" : "");
}

/** Show the GOOD/ODD panel for saved cut number i. */
function openLabelPanel(i, note = "") {
  pendingIndex = i;
  const c = loadSaved()[i];
  $("cut-material").value = c?.material || localStorage.getItem(MATERIAL_KEY) || "";
  $("cut-notes").value = c?.notes || "";
  $("after-cut-note").textContent = note || "This cut is already saved on the phone. Pick GOOD or ODD to label it.";
  $("after-cut").hidden = false;
  $("after-cut").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function persist(cuts) { try { localStorage.setItem(SAVED_KEY, JSON.stringify(cuts)); return true; } catch { return false; } }
function saveDraft() {
  try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ at: new Date().toISOString(), throughAt, throughGuess, features: currentCut })); } catch {}
}
function clearDraft() { try { localStorage.removeItem(DRAFT_KEY); } catch {} }
/** After a reload or a crash: a draft of at least one minimum cut becomes a saved, unlabelled cut. */
function recoverDraft() {
  let d = null;
  try { d = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null"); } catch {}
  if (!d || !Array.isArray(d.features) || d.features.length < det.motor.minCutWindows) { clearDraft(); return; }
  currentCut = d.features; throughAt = d.throughAt ?? null; throughGuess = d.throughGuess ?? null;
  finishCut(d.at, true);
}

function alertStop() {
  const now = Date.now();
  if (now - lastVibrate < 2000) return;
  lastVibrate = now;
  if ($("vibrate").checked && navigator.vibrate) navigator.vibrate([400, 200, 400]);
  if ($("beep").checked) beep();
}

function beep() {
  try {
    beepCtx = beepCtx || new (window.AudioContext || window.webkitAudioContext)();
    const o = beepCtx.createOscillator(), g = beepCtx.createGain();
    o.frequency.value = 880; g.gain.value = 0.2; o.connect(g); g.connect(beepCtx.destination);
    o.start(); o.stop(beepCtx.currentTime + 0.15);
  } catch {}
}

// ---------- after-cut feedback ----------
function saveCut(label) {
  const cuts = loadSaved();
  if (pendingIndex === null || !cuts[pendingIndex]) { $("after-cut").hidden = true; return; }
  const material = $("cut-material").value.trim(), notes = $("cut-notes").value.trim();
  Object.assign(cuts[pendingIndex], { label, material, notes });
  if (!persist(cuts)) alert("Phone storage is full. Download the saved cuts, then Clear.");
  localStorage.setItem(MATERIAL_KEY, material);              // remembered for the next cut
  pendingIndex = null; $("after-cut").hidden = true; $("cut-notes").value = ""; updateSavedCount();
}
function deletePending() {
  const cuts = loadSaved();
  if (pendingIndex !== null && cuts[pendingIndex] && confirm("Delete this cut for good?")) { cuts.splice(pendingIndex, 1); persist(cuts); }
  pendingIndex = null; $("after-cut").hidden = true; updateSavedCount();
}
const loadSaved = () => { try { return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]"); } catch { return []; } };
function updateSavedCount() {
  const cuts = loadSaved();
  $("saved-count").textContent = cuts.length;
  const list = $("saved-list"); list.innerHTML = "";
  cuts.slice().reverse().forEach((c, i) => {
    const li = document.createElement("li");
    const when = new Date(c.at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    const idx = cuts.length - 1 - i;
    li.innerHTML = `<span class="tag tag-${c.label}">${c.label === "normal" ? "GOOD" : c.label === "abnormal" ? "ODD" : "NOT LABELLED"}</span>` +
      `<span><b>${esc(c.material) || "(no material)"}</b> <span class="meta">${when}, ${fmt(c.duration_s || 0)}` +
      (c.through_at_s != null ? `, blade through at ${fmt(c.through_at_s)}` : "") +
      (c.through_guess_s != null ? ` (page guessed ${fmt(c.through_guess_s)})` : "") + `</span>` +
      (c.notes ? `<br><span class="meta">${esc(c.notes)}</span>` : "") +
      (c.label === UNLABELLED ? ` <button class="link relabel">Label it</button>` : "") + `</span>` +
      `<button class="link del" title="Delete this saved cut">✕</button>`;
    li.querySelector(".del").onclick = () => { if (!confirm("Delete this saved cut for good?")) return; cuts.splice(idx, 1); persist(cuts); if (pendingIndex === idx) { pendingIndex = null; $("after-cut").hidden = true; } updateSavedCount(); };
    li.querySelector(".relabel")?.addEventListener("click", () => openLabelPanel(idx));
    list.appendChild(li);
  });
  // material suggestions = ones used before
  const dl = $("material-list"); dl.innerHTML = "";
  [...new Set(cuts.map((c) => c.material).filter(Boolean))].forEach((m) => { const o = document.createElement("option"); o.value = m; dl.appendChild(o); });
  if (!$("cut-material").value) $("cut-material").value = localStorage.getItem(MATERIAL_KEY) || cuts.at(-1)?.material || "";
}
const esc = (s) => String(s || "").replace(/[&<>"]/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch]));
function downloadSaved() { downloadCuts(loadSaved(), "saved-cuts"); }
function downloadCuts(cuts, prefix) {
  const blob = new Blob([JSON.stringify({ exported: new Date().toISOString(), cuts })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `${prefix}-${new Date().toISOString().slice(0, 10)}.json`; a.click();
}

// ---------- test with a recording ----------
async function testFile(file) {
  $("file-result").textContent = "Decoding…";
  try {
    const samples = await decodeFile(file, model.audio.sr);
    const feats = [];
    for (let i = 0; i + fx.windowLength <= samples.length; i += fx.hopLength) feats.push(fx.extract(samples.subarray(i, i + fx.windowLength)));
    const r = det.scoreAll(feats);
    const hop = model.audio.hop_s;
    const judged = r.phases.map((p, i) => (p === RUNNING ? i : -1)).filter((i) => i >= 0);
    const firstOf = (v) => { const i = r.verdicts.indexOf(v); return i < 0 ? null : i * hop; };
    const nOdd = judged.filter((i) => r.verdicts[i] !== GOOD).length;
    let text = `${file.name}: ${fmt(samples.length / model.audio.sr)} long, motor running for ${fmt(judged.length * hop)}. `;
    const stopAt = firstOf(STOP), oddAt = firstOf(ODD);
    text += stopAt !== null ? `Meter would have shown STOP at ${fmt(stopAt)}.` :
      oddAt !== null ? `ODD at ${fmt(oddAt)} (${((nOdd / Math.max(1, judged.length)) * 100).toFixed(1)}% of the cut), never STOP.` :
      judged.length ? "GOOD throughout." : "Motor never detected as running.";
    $("file-result").textContent = text;
    const c = $("file-timeline"); c.hidden = false;
    drawTimeline(c, r.scores, r.verdicts, hop, det.yellow, det.red);
  } catch (err) {
    $("file-result").textContent = "Could not read that file: " + (err.message || err);
  }
}

// ---------- drawing & helpers ----------
function drawTimeline(canvas, scores, verdicts, hop, yellow, red) {
  const W = canvas.clientWidth || 600, H = canvas.height;
  if (canvas.width !== W) canvas.width = W;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const n = scores.length; if (!n) return;
  const ymax = Math.max(red * 1.5, ...scores.slice(-n)) * 1.05;
  const y = (s) => H - 4 - (Math.min(s, ymax) / ymax) * (H - 8);
  ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
  for (const [v, col] of [[yellow, COLOURS[ODD]], [red, COLOURS[STOP]]]) { ctx.strokeStyle = col; ctx.beginPath(); ctx.moveTo(0, y(v)); ctx.lineTo(W, y(v)); ctx.stroke(); }
  ctx.setLineDash([]);
  const xs = (i) => (i / Math.max(1, n - 1)) * W;
  for (let i = 0; i < n; i++) {
    ctx.fillStyle = COLOURS[verdicts[i]] || COLOURS[OFF];
    ctx.fillRect(xs(i) - 1, y(scores[i]) - 2, Math.max(2, W / n), 4);
  }
  if (hop) {                                             // time labels for file mode
    ctx.fillStyle = "#ddd"; ctx.font = "11px system-ui";
    const step = n * hop > 240 ? 60 : 30;
    for (let t = 0; t <= n * hop; t += step) { const i = t / hop; ctx.fillText(fmt(t), Math.min(xs(i), W - 28), 11); }
  }
}

function setState(state) {
  document.body.dataset.state = state;
  const [big, small] = TEXT[state] || TEXT.idle;
  $("verdict").textContent = big; $("hint").textContent = small;
}
function placeMarks() {
  const top = det.red * 1.4;
  $("mark-yellow").style.left = (det.yellow / top) * 100 + "%";
  $("mark-red").style.left = (det.red / top) * 100 + "%";
}
function loadSensitivity() { const v = parseFloat(localStorage.getItem(SENS_KEY)); return isFinite(v) ? v : 1; }
function fmt(s) { s = Math.max(0, Math.round(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; }

// ---------- wiring ----------
$("start").onclick = start;
$("stop").onclick = stop;
$("sensitivity").oninput = (e) => { const v = parseFloat(e.target.value); det.setSensitivity(v); localStorage.setItem(SENS_KEY, v); $("sens-out").textContent = v.toFixed(2) + "×"; placeMarks(); };
$("sens-reset").onclick = () => { $("sensitivity").value = 1; $("sensitivity").dispatchEvent(new Event("input")); };
$("cut-good").onclick = () => saveCut("normal");
$("cut-odd").onclick = () => saveCut("abnormal");
$("cut-skip").onclick = deletePending;
$("through").onclick = markThrough;
$("download").onclick = downloadSaved;
$("clear-saved").onclick = () => { if (confirm("Delete ALL saved cuts on this phone? Download them first if you have not.")) { localStorage.removeItem(SAVED_KEY); pendingIndex = null; $("after-cut").hidden = true; updateSavedCount(); } };
$("file").onchange = (e) => e.target.files[0] && testFile(e.target.files[0]);
init().catch((e) => ($("model-info").textContent = "Could not load model.json: " + e.message));
