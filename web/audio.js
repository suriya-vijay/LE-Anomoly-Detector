// Microphone capture + windowing for the phone page.
//
// Produces ~1 s windows (Float32Array) at the model's sample rate, a new one every 0.5 s,
// exactly like src/audio_io.split_windows. Also decodes recording files for the test mode.

/** Low-pass + linear-interpolation resampler for when the browser won't give us the rate we want. */
class Resampler {
  constructor(fromRate, toRate) {
    this.ratio = fromRate / toRate;
    this.pos = 0; this.tail = new Float32Array(0);
    // windowed-sinc low-pass just under the new Nyquist (like scipy's resample_poly), so the
    // live page keeps the same top tone band as the Python side without aliasing
    const taps = 95, fc = 0.49 / this.ratio;
    this.kernel = new Float32Array(taps);
    const mid = (taps - 1) / 2;
    let sum = 0;
    for (let i = 0; i < taps; i++) {
      const x = i - mid;
      const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
      const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1));
      this.kernel[i] = sinc * w; sum += this.kernel[i];
    }
    for (let i = 0; i < taps; i++) this.kernel[i] /= sum;
  }
  process(chunk) {
    const taps = this.kernel.length, half = (taps - 1) / 2;
    const x = new Float32Array(this.tail.length + chunk.length);
    x.set(this.tail); x.set(chunk, this.tail.length);
    const usable = x.length - taps;                 // filtered samples available: indices half .. half+usable
    if (usable <= 0) { this.tail = x; return new Float32Array(0); }
    const filtered = new Float32Array(usable);
    for (let i = 0; i < usable; i++) {
      let s = 0;
      for (let k = 0; k < taps; k++) s += x[i + k] * this.kernel[k];
      filtered[i] = s;
    }
    const out = [];
    while (this.pos < usable - 1) {
      const i = Math.floor(this.pos), f = this.pos - i;
      out.push(filtered[i] * (1 - f) + filtered[i + 1] * f);
      this.pos += this.ratio;
    }
    const consumed = Math.floor(this.pos);
    this.pos -= consumed;
    this.tail = x.subarray(consumed);
    return Float32Array.from(out);
  }
}

/** Collects samples and hands out overlapping windows. */
export class Windower {
  constructor(windowLength, hopLength, onWindow) {
    this.win = windowLength; this.hop = hopLength; this.onWindow = onWindow;
    this.buf = new Float32Array(0); this.emitted = 0;
  }
  push(samples) {
    const b = new Float32Array(this.buf.length + samples.length);
    b.set(this.buf); b.set(samples, this.buf.length); this.buf = b;
    while (this.buf.length >= this.win) {
      this.onWindow(this.buf.slice(0, this.win));
      this.buf = this.buf.subarray(this.hop);
    }
  }
  reset() { this.buf = new Float32Array(0); }
}

const WORKLET_SRC = `
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("tap", Tap);`;

export class MicCapture {
  constructor(targetRate, onSamples) { this.targetRate = targetRate; this.onSamples = onSamples; }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser can't access the microphone. Use Chrome (Android) or Safari (iPhone) over https.");
    // Turn OFF the phone's voice-call processing: it would quietly change the loudness we measure.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 }, video: false,
    });
    const AC = window.AudioContext || window.webkitAudioContext;
    try { this.ctx = new AC({ sampleRate: this.targetRate }); } catch { this.ctx = new AC(); }
    await this.ctx.resume();
    this.resampler = this.ctx.sampleRate === this.targetRate ? null : new Resampler(this.ctx.sampleRate, this.targetRate);
    this.source = this.ctx.createMediaStreamSource(this.stream);
    const deliver = (chunk) => this.onSamples(this.resampler ? this.resampler.process(chunk) : chunk);
    if (this.ctx.audioWorklet) {
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      await this.ctx.audioWorklet.addModule(url);
      this.node = new AudioWorkletNode(this.ctx, "tap", { numberOfOutputs: 0 });
      this.node.port.onmessage = (e) => deliver(e.data);
      this.source.connect(this.node);
    } else {                                                  // older Safari
      this.node = this.ctx.createScriptProcessor(4096, 1, 1);
      this.node.onaudioprocess = (e) => deliver(e.inputBuffer.getChannelData(0).slice(0));
      this.source.connect(this.node); this.node.connect(this.ctx.destination);
    }
    const settings = this.stream.getAudioTracks()[0].getSettings?.() || {};
    return { contextRate: this.ctx.sampleRate, resampling: !!this.resampler, settings };
  }

  async stop() {
    this.stream?.getTracks().forEach(t => t.stop());
    try { this.source?.disconnect(); this.node?.disconnect(); } catch {}
    await this.ctx?.close();
    this.stream = this.ctx = this.node = this.source = null;
  }
}

/** Decode a recording file to mono samples at `rate` (the browser resamples for us). */
export async function decodeFile(file, rate) {
  const bytes = await file.arrayBuffer();
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const probe = new OAC(1, 1, rate);
  const buffer = await new Promise((res, rej) => probe.decodeAudioData(bytes, res, rej));
  if (buffer.numberOfChannels === 1 && buffer.sampleRate === rate) return buffer.getChannelData(0);
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const d = buffer.getChannelData(c);
    for (let i = 0; i < d.length; i++) mono[i] += d[i] / buffer.numberOfChannels;
  }
  if (buffer.sampleRate === rate) return mono;
  const r = new Resampler(buffer.sampleRate, rate);
  const out = r.process(mono);
  return out;
}
