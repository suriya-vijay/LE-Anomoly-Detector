// Run by tests/test_parity.py:  node parity_runner.mjs <model.json> <windows.bin> <n_samples_per_window>
// Reads float32 windows, prints JSON {features: [[...]], scores: [...], phases: [...], verdicts: [...]}.
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const { FeatureExtractor } = await import(path.join(here, "..", "web", "features.js").replace(/\\/g, "/").replace(/^([A-Za-z]):/, "file:///$1:"));
const { Detector } = await import(path.join(here, "..", "web", "detector.js").replace(/\\/g, "/").replace(/^([A-Za-z]):/, "file:///$1:"));

const [modelPath, binPath, nStr] = process.argv.slice(2);
const model = JSON.parse(readFileSync(modelPath, "utf8"));
const buf = readFileSync(binPath);
const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const n = parseInt(nStr, 10);

const fx = new FeatureExtractor(model);
const det = new Detector(model);
const features = [];
for (let i = 0; i + n <= all.length; i += n) features.push(Array.from(fx.extract(all.subarray(i, i + n))));
const { scores, phases, verdicts, loads, through, headsUp } = det.scoreAll(features);
process.stdout.write(JSON.stringify({ features, scores, phases, verdicts, loads, through, headsUp }));
