// Feature extraction for the phone page. This must give the SAME numbers as
// src/features.py (there is an automatic parity test: tests/test_parity.py).
//
// One ~1 s window of audio (Float32Array at model.audio.sr) ->
//   [rms_db, dominant_hz, centroid_hz, rolloff_hz, pulse_depth, pulse_hz, mel_00 ... mel_63]

const EPS = 1e-10;

/** In-place iterative radix-2 FFT. re/im are Float64Arrays whose length is a power of 2. */
export function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {           // bit-reversal permutation
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

/** |FFT(x zero-padded/truncated to nfft)| for bins 0..nfft/2 (like numpy rfft magnitude). */
export function rfftMagnitude(x, nfft) {
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  re.set(x.subarray(0, Math.min(x.length, nfft)));
  fft(re, im);
  const out = new Float64Array(nfft / 2 + 1);
  for (let k = 0; k < out.length; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

const hannCache = new Map();
function hann(n) {                              // periodic Hann, same formula as Python
  if (!hannCache.has(n)) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n);
    hannCache.set(n, w);
  }
  return hannCache.get(n);
}

export class FeatureExtractor {
  constructor(model) {
    const a = model.audio;
    this.sr = a.sr; this.nfft = a.n_fft; this.hop = a.frame_hop; this.nMels = a.n_mels;
    this.rolloff = a.rolloff_fraction; this.pulse = a.pulse;
    this.mel = model.mel;                         // [{start, weights}] per band (sparse filterbank)
    this.featureNames = model.feature_names;
    this.windowLength = Math.round(a.window_s * a.sr);
    this.hopLength = Math.round(a.hop_s * a.sr);
    this._re = new Float64Array(this.nfft); this._im = new Float64Array(this.nfft);
  }

  /** window: Float32Array/Float64Array of windowLength samples -> Float64Array of features */
  extract(window) {
    const x = window, N = x.length, nfft = this.nfft, nb = nfft / 2 + 1;
    const nFrames = 1 + Math.floor((N - nfft) / this.hop);
    const w = hann(nfft);
    const meanPower = new Float64Array(nb);
    const logMelSum = new Float64Array(this.nMels);
    const power = new Float64Array(nb);
    for (let f = 0; f < nFrames; f++) {
      const off = f * this.hop;
      const re = this._re, im = this._im;
      for (let i = 0; i < nfft; i++) { re[i] = x[off + i] * w[i]; im[i] = 0; }
      fft(re, im);
      for (let k = 0; k < nb; k++) { power[k] = re[k] * re[k] + im[k] * im[k]; meanPower[k] += power[k]; }
      for (let m = 0; m < this.nMels; m++) {       // mel energy of this frame, then log (mean of logs, like Python)
        const { start, weights } = this.mel[m];
        let e = 0;
        for (let j = 0; j < weights.length; j++) e += weights[j] * power[start + j];
        logMelSum[m] += Math.log10(e + EPS);
      }
    }
    const mag = new Float64Array(nb);
    let sumMag = 0, sumFMag = 0, sumSq = 0;
    for (let k = 0; k < nb; k++) { meanPower[k] /= nFrames; mag[k] = Math.sqrt(meanPower[k]); }
    for (let i = 0; i < N; i++) sumSq += x[i] * x[i];
    const binHz = this.sr / nfft;
    for (let k = 0; k < nb; k++) { sumMag += mag[k]; sumFMag += k * binHz * mag[k]; }

    const rmsDb = 10 * Math.log10(sumSq / N + EPS);
    const centroid = sumFMag / (sumMag + EPS);
    // dominant frequency: biggest bin (skipping DC and the last bin), refined with a parabola on log magnitude
    let kmax = 1;
    for (let k = 2; k < nb - 1; k++) if (mag[k] > mag[kmax]) kmax = k;
    const la = Math.log(mag[kmax - 1] + EPS), lb = Math.log(mag[kmax] + EPS), lc = Math.log(mag[kmax + 1] + EPS);
    const denom = la - 2 * lb + lc;
    let offset = denom !== 0 ? 0.5 * (la - lc) / denom : 0;
    offset = Math.min(0.5, Math.max(-0.5, offset));
    const dominant = (kmax + offset) * binHz;
    // roll-off: first bin where the cumulative magnitude reaches 85% of the total (numpy searchsorted, left)
    const target = this.rolloff * sumMag;
    let cum = 0, kroll = nb - 1;
    for (let k = 0; k < nb; k++) { cum += mag[k]; if (cum >= target) { kroll = k; break; } }
    const rolloff = kroll * binHz;

    const [pulseDepth, pulseHz] = this.pulseFeatures(x);

    const out = new Float64Array(6 + this.nMels);
    out[0] = rmsDb; out[1] = dominant; out[2] = centroid; out[3] = rolloff; out[4] = pulseDepth; out[5] = pulseHz;
    for (let m = 0; m < this.nMels; m++) out[6 + m] = 10 * logMelSum[m] / nFrames;
    return out;
  }

  /** How strongly and how fast the loudness throbs inside the window (same as Python pulse_features). */
  pulseFeatures(x) {
    const p = this.pulse, frame = Math.round(p.frame_s * this.sr), n = Math.floor(x.length / frame);
    const env = new Float64Array(n);
    let mean = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = i * frame; j < (i + 1) * frame; j++) s += x[j] * x[j];
      env[i] = Math.sqrt(s / frame); mean += env[i];
    }
    mean /= n;
    let varSum = 0;
    for (let i = 0; i < n; i++) { env[i] -= mean; varSum += env[i] * env[i]; }
    const depth = Math.sqrt(varSum / n) / (mean + EPS);
    const spec = rfftMagnitude(env, p.fft);
    const hzPerBin = 1 / (p.fft * p.frame_s);
    let best = -1, bestK = 0;
    for (let k = 0; k < spec.length; k++) {
      const hz = k * hzPerBin;
      if (hz >= p.min_hz && hz <= p.max_hz && spec[k] > best) { best = spec[k]; bestK = k; }
    }
    return [depth, bestK * hzPerBin];
  }
}
