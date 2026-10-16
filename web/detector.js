// The detector on the phone: same maths as src/detector.py and src/phases.py.
//
//   score   = distance from the nearest "kind of normal" (about 1-2 is typical)
//   phase   = off / transition / running  (only "running" is judged)
//   verdict = off / good / odd / stop

export const OFF = "off", TRANSITION = "transition", RUNNING = "running";
export const GOOD = "good", ODD = "odd", STOP = "stop";

export class Detector {
  constructor(model, sensitivity = 1.0) {
    this.model = model;
    this.idx = model.distance_index;
    this.mean = model.scaler.mean; this.std = model.scaler.std;
    this.clusters = model.clusters;
    this.n = this.idx.length;
    this.smoothK = model.smooth_windows;
    this.redAfterYellow = model.thresholds.red_after_yellow_windows;
    this.setSensitivity(sensitivity);
    this.iRms = model.feature_names.indexOf("rms_db");
    this.iPulse = model.feature_names.indexOf("pulse_depth");
    this.iCentroid = model.feature_names.indexOf("centroid_hz");
    this.bt = model.breakthrough ? new Breakthrough(model.breakthrough) : null;
    this.reset();
  }

  /** sensitivity > 1 makes the meter stricter (lines move down), < 1 more relaxed. */
  setSensitivity(s) {
    this.sensitivity = s;
    this.yellow = this.model.thresholds.yellow / s;
    this.red = this.model.thresholds.red / s;
  }

  reset() { this._recent = []; this._oddStreak = 0; this.motor = new MotorTracker(this.model); this.bt?.reset(); }

  /** Standardised feature vector (the space the clusters and the break-through direction live in). */
  standardise(f) {
    const z = new Float64Array(this.n);
    for (let j = 0; j < this.n; j++) z[j] = (f[this.idx[j]] - this.mean[j]) / this.std[j];
    return z;
  }

  /** Raw (unsmoothed) distance of one feature vector from normal. */
  rawScore(f) {
    const z = this.standardise(f);
    let best = Infinity;
    for (const c of this.clusters) {
      const d = new Float64Array(this.n);
      for (let j = 0; j < this.n; j++) d[j] = z[j] - c.mean[j];
      let d2 = 0;
      for (let j = 0; j < this.n; j++) {
        const row = c.precision[j];
        let s = 0;
        for (let k = 0; k < this.n; k++) s += row[k] * d[k];
        d2 += d[j] * s;
      }
      if (d2 < best) best = d2;
    }
    return Math.sqrt(Math.max(best, 0) / this.n);
  }

  /** Feed one window's features (live, in order).
   *  Returns {score, raw, phase, verdict, cutDone, load, through}: load is the blade-load number
   *  (about -1 in the steel, +1 through; null when not judging), through is true on the window
   *  the break-through guess fires, retracted on the window it is taken back. */
  push(f) {
    const raw = this.rawScore(f);
    this._recent.push(raw);
    if (this._recent.length > this.smoothK) this._recent.shift();
    const score = this._recent.reduce((a, b) => a + b, 0) / this._recent.length;
    const { phase, cutDone } = this.motor.push(f[this.iRms], f[this.iPulse]);
    let verdict = OFF;
    if (phase !== RUNNING) {
      this._oddStreak = 0;
    } else {
      this._oddStreak = score > this.yellow ? this._oddStreak + 1 : 0;
      verdict = (score > this.red || this._oddStreak >= this.redAfterYellow) ? STOP : score > this.yellow ? ODD : GOOD;
    }
    const bt = this.bt ? this.bt.push(this.standardise(f), phase, f[this.iRms], f[this.iCentroid])
      : { load: null, progress: 0, stage: 0, headsUp: false, through: false, retracted: false };
    return { score, raw, phase, verdict, cutDone, load: bt.load, progress: bt.progress, stage: bt.stage,
             headsUp: bt.headsUp, through: bt.through, retracted: bt.retracted };
  }

  /** Offline: score a whole recording's features (array of vectors) exactly like Python's evaluate. */
  scoreAll(featureList) {
    const raws = featureList.map(f => this.rawScore(f));
    const scores = raws.map((_, i) => {
      const lo = Math.max(0, i - this.smoothK + 1);
      let s = 0; for (let j = lo; j <= i; j++) s += raws[j];
      return s / (i - lo + 1);
    });
    const phases = labelPhasesOffline(featureList, this.model, this.iRms, this.iPulse);
    const verdicts = [];
    let streak = 0;
    for (let i = 0; i < scores.length; i++) {
      if (phases[i] !== RUNNING) { streak = 0; verdicts.push(OFF); continue; }
      streak = scores[i] > this.yellow ? streak + 1 : 0;
      verdicts.push((scores[i] > this.red || streak >= this.redAfterYellow) ? STOP : scores[i] > this.yellow ? ODD : GOOD);
    }
    let loads = null, through = null, headsUp = null;
    if (this.bt) {
      const b = new Breakthrough(this.model.breakthrough);
      loads = [];
      featureList.forEach((f, i) => {
        const r = b.push(this.standardise(f), phases[i], f[this.iRms], f[this.iCentroid]);
        loads.push(r.load);
        if (r.headsUp) { headsUp = i; through = null; }
        if (r.through) through = i;
        if (r.retracted) { through = null; headsUp = null; }
      });
    }
    return { scores, phases, verdicts, loads, through, headsUp };
  }
}

/** Blade break-through indicator: same maths and state machine as src/breakthrough.py.
 *  load = w . (z - mid): about -1 while the blade is in the steel, about +1 once through.
 *  "through" fires only after the load has read "in the steel" for loaded_windows, then
 *  stays above fire_level for hold_windows. Resets whenever the motor is not "running". */
export class Breakthrough {
  constructor(d) { this.d = d; this.reset(); }
  reset() { this.hist = []; this.recent = []; this.rms = []; this.cen = []; this.run = 0; this.hold = 0; this.hold2 = 0; this.fired = false; this.banner = false; this.lowAtFire = 0; }
  /** The user tapped "Blade touched steel": measure the cut's own low from here on. */
  restart() { this.reset(); }
  /** Returns {load, progress, stage, headsUp, through, retracted}. stage 0 = cutting, 1 = heads-up
   *  (sound is changing), 2 = banner (coming through). headsUp / through / retracted are true only on
   *  the window the state changes. */
  push(z, phase, rmsDb, centroid) {
    const d = this.d;
    if (phase !== RUNNING) { this.reset(); return { load: null, progress: 0, stage: 0, headsUp: false, through: false, retracted: false }; }
    const refN = d.smooth_windows + d.quiet_ref_windows;
    // "got quieter" gate: the last 10 s must be quiet_db below the median of the 60 s before that
    let quiet = true, bright = true;
    if (d.quiet_db > 0 && rmsDb !== undefined) {
      this.rms.push(rmsDb); if (this.rms.length > refN) this.rms.shift();
      const ref = this.rms.slice(0, Math.max(0, this.rms.length - d.smooth_windows));
      quiet = ref.length >= d.smooth_windows && mean(this.rms.slice(-d.smooth_windows)) < median(ref) - d.quiet_db;
    }
    if (d.bright_pct > 0 && centroid !== undefined) {
      this.cen.push(centroid); if (this.cen.length > refN) this.cen.shift();
      const ref = this.cen.slice(0, Math.max(0, this.cen.length - d.smooth_windows));
      bright = ref.length >= d.smooth_windows && mean(this.cen.slice(-d.smooth_windows)) > median(ref) * (1 + d.bright_pct);
    }
    let raw = 0;
    for (let j = 0; j < d.w.length; j++) raw += d.w[j] * (z[j] - d.mid[j]);
    this.hist.push(raw);
    if (this.hist.length > d.smooth_windows) this.hist.shift();
    const load = mean(this.hist);
    this.run++;
    this.recent.push(load);
    if (this.recent.length > d.lookback_windows) this.recent.shift();
    const low = percentile(this.recent, d.low_percentile);
    let headsUp = false, through = false, retracted = false;
    if (!this.fired) {
      this.hold = (this.run >= d.min_run_windows && load > low + d.delta && quiet) ? this.hold + 1 : 0;
      if (this.hold >= d.hold_windows) { this.fired = true; this.lowAtFire = low; this.hold = 0; this.hold2 = 0; this.banner = false; headsUp = true; }
    } else {
      this.hold = load < this.lowAtFire + d.retract_delta ? this.hold + 1 : 0;
      if (this.hold >= d.hold_windows) { this.fired = false; this.hold = 0; this.hold2 = 0; this.banner = false; retracted = true; }
      else if (!this.banner) {
        this.hold2 = (quiet && bright) ? this.hold2 + 1 : 0;
        if (this.hold2 >= d.stage2_hold_windows) { this.banner = true; through = true; }
      }
    }
    const progress = this.run >= d.min_run_windows ? Math.max(0, Math.min(1, (load - low) / d.progress_span)) : 0;
    const stage = this.fired ? (this.banner ? 2 : 1) : 0;
    return { load, progress, stage, headsUp, through, retracted };
  }
}

function mean(values) { return values.reduce((a, b) => a + b, 0) / values.length; }

/** numpy.median */
function median(values) { return percentile(values, 50); }

/** numpy.percentile with its default linear interpolation. */
function percentile(values, q) {
  const s = Array.from(values).sort((a, b) => a - b);
  const pos = (s.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Live motor on/off tracker (same hysteresis rule as src/phases.py, but it can't see the future,
 *  so the last couple of windows before OFF are judged instead of being "transition"). */
export class MotorTracker {
  constructor(model) {
    const m = model.motor;
    this.thr = m.threshold_db; this.onWindows = m.on_windows; this.offWindows = m.off_windows;
    this.maxPulse = m.max_pulse; this.spinup = m.spinup_windows;
    this.minCutWindows = Math.round(m.min_cut_s / model.audio.hop_s);
    this.on = false; this.run = 0; this.sinceOn = 0; this.loudHistory = [];
  }
  push(rmsDb, pulseDepth) {
    const loud = rmsDb > this.thr, steady = pulseDepth < this.maxPulse;
    let cutDone = false;
    if (!this.on) {
      this.run = (loud && steady) ? this.run + 1 : 0;
      if (this.run >= this.onWindows) { this.on = true; this.run = 0; this.sinceOn = this.onWindows; this.loudHistory = []; }
      else return { phase: OFF, cutDone };
    } else {
      this.sinceOn++;
      this.loudHistory.push(loud);
      if (this.loudHistory.length > 2 * this.offWindows) this.loudHistory.shift();
      const quiet = this.loudHistory.filter(l => !l).length;
      if (quiet >= this.offWindows) {
        cutDone = this.sinceOn >= this.minCutWindows;
        this.on = false; this.run = 0; this.loudHistory = [];
        return { phase: OFF, cutDone };
      }
    }
    return { phase: this.sinceOn <= this.spinup ? TRANSITION : RUNNING, cutDone };
  }
}

/** Offline phase labels for a whole recording: identical to src/phases.py label_phases(). */
export function labelPhasesOffline(featureList, model, iRms, iPulse) {
  const m = model.motor, n = featureList.length;
  const loud = featureList.map(f => f[iRms] > m.threshold_db);
  const steady = featureList.map(f => f[iPulse] < m.max_pulse);
  const on = new Array(n).fill(false);
  let state = false, run = 0, onSince = 0;
  for (let i = 0; i < n; i++) {
    if (!state) {
      run = (loud[i] && steady[i]) ? run + 1 : 0;
      if (run >= m.on_windows) { state = true; run = 0; onSince = i - m.on_windows + 1; for (let j = onSince; j <= i; j++) on[j] = true; }
      continue;
    }
    on[i] = true;
    const lo = Math.max(onSince, i - 2 * m.off_windows + 1);
    let quiet = 0, firstQuiet = -1;
    for (let j = lo; j <= i; j++) if (!loud[j]) { quiet++; if (firstQuiet < 0) firstQuiet = j; }
    if (quiet >= m.off_windows) { state = false; for (let j = firstQuiet; j <= i; j++) on[j] = false; }
  }
  const labels = on.map(o => o ? RUNNING : OFF);
  for (let i = 0; i < n; i++) {
    const starts = on[i] && (i === 0 || !on[i - 1]);
    if (starts) for (let j = i; j < Math.min(n, i + m.spinup_windows); j++) if (on[j]) labels[j] = TRANSITION;
    const ends = on[i] && (i === n - 1 || !on[i + 1]);
    if (ends) for (let j = Math.max(0, i - m.spindown_windows + 1); j <= i; j++) if (on[j]) labels[j] = TRANSITION;
  }
  return labels;
}
