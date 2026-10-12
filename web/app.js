// Saw Sound Meter: wires the microphone, the feature maths and the detector to the screen.
import { FeatureExtractor } from "./features.js";
import { Detector, OFF, TRANSITION, RUNNING, GOOD, ODD, STOP } from "./detector.js";
import { MicCapture, Windower, decodeFile } from "./audio.js";

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
const SAVED_KEY = "sawmeter.savedCuts", SENS_KEY = "sawmeter.sensitivity";

let model, fx, det, mic, windower, wakeLock;
let listening = false, lastVibrate = 0, beepCtx;
let currentCut = null;                     // features of the cut in progress (for "That cut was GOOD/ODD")
let live = [];                             // recent {t, score, verdict} for the live timeline
let t0 = 0, motorOnAt = null;

// ---------- start-up ----------
async function init() {
  const res = await fetch("model.json", { cache: "no-cache" });
  model = await res.json();
  fx = new FeatureExtractor(model);
  det = new Detector(model, loadSensitivity());
  $("sensitivity").value = det.sensitivity; $("sens-out").textContent = det.sensitivity.toFixed(2) + "×";
  placeMarks();
  $("model-info").textContent = `Model learned from ${model.trained_on.length} good cut(s), ${model.trained_at.slice(0, 10)}. ` +
    `Yellow above ${model.thresholds.yellow.toFixed(2)}, red above ${model.thresholds.red.toFixed(2)}.`;
  if (!("vibrate" in navigator)) $("vibrate-note").textContent = "(not available on iPhone: the screen flashes instead)";
  updateSavedCount();
  if ("serviceWorker" in navigator && (location.protocol === "https:" || location.hostname === "localhost"))
    navigator.serviceWorker.register("sw.js").catch(() => {});
}

// ---------- live listening ----------
async function start() {
  $("mic-error").hidden = true;
  try {
    det.reset(); live = []; currentCut = null; motorOnAt = null; t0 = performance.now();
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
  await mic?.stop(); wakeLock?.release?.(); wakeLock = null;
  $("start").hidden = false; $("stop").hidden = true;
  setState("idle"); $("phase").textContent = "Not listening"; $("clock").textContent = "";
}

function onWindow(samples) {
  const f = fx.extract(samples);
  const r = det.push(f);
  const t = (performance.now() - t0) / 1000;
  if (r.phase !== OFF && motorOnAt === null) { motorOnAt = t; currentCut = []; }   // motor just started
  if (currentCut) currentCut.push(Array.from(f, (v) => Math.round(v * 1000) / 1000));
  if (r.phase === OFF && motorOnAt !== null) {                                       // motor stopped
    if (r.cutDone) finishCut(); else currentCut = null;                              // too short to be a cut
    motorOnAt = null;
  }

  setState(r.phase === RUNNING ? r.verdict : r.phase);
  $("score").textContent = r.score.toFixed(2);
  $("bar-fill").style.width = Math.min(100, (r.score / (det.red * 1.4)) * 100) + "%";
  $("clock").textContent = motorOnAt === null ? "" : "motor on " + fmt(t - motorOnAt);
  live.push({ t, score: r.score, verdict: r.phase === RUNNING ? r.verdict : OFF });
  if (live.length > 180) live.shift();               // last 90 s
  drawTimeline($("timeline"), live.map((p) => p.score), live.map((p) => p.verdict), null, det.yellow, det.red);
  if (r.verdict === STOP) alertStop();
}

function finishCut() {
  if (!currentCut || currentCut.length < 20) { currentCut = null; return; }
  const banner = $("cut-banner"); banner.hidden = false; setTimeout(() => (banner.hidden = true), 10000);
  $("after-cut").hidden = false;
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
  if (!currentCut) return;
  const cuts = loadSaved();
  cuts.push({ label, at: new Date().toISOString(), model_trained_at: model.trained_at, feature_names: model.feature_names, features: currentCut });
  try { localStorage.setItem(SAVED_KEY, JSON.stringify(cuts)); } catch { alert("Phone storage is full. Download the saved cuts, then Clear."); }
  currentCut = null; $("after-cut").hidden = true; updateSavedCount();
}
const loadSaved = () => { try { return JSON.parse(localStorage.getItem(SAVED_KEY) || "[]"); } catch { return []; } };
function updateSavedCount() { $("saved-count").textContent = loadSaved().length; }
function downloadSaved() {
  const blob = new Blob([JSON.stringify({ exported: new Date().toISOString(), cuts: loadSaved() })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `saved-cuts-${new Date().toISOString().slice(0, 10)}.json`; a.click();
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
$("download").onclick = downloadSaved;
$("clear-saved").onclick = () => { if (confirm("Delete all saved cuts on this phone?")) { localStorage.removeItem(SAVED_KEY); updateSavedCount(); } };
$("file").onchange = (e) => e.target.files[0] && testFile(e.target.files[0]);
init().catch((e) => ($("model-info").textContent = "Could not load model.json: " + e.message));
