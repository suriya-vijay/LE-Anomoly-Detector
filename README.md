# Saw Sound Meter

**Live page:** https://suriya-vijay.github.io/LE-Anomoly-Detector/web/

I work in a microscopy lab where we cut steel samples on a **Struers Discotom-6** cut-off saw before
mounting, polishing and etching them. Two things kept bothering me:

1. Sometimes a cut goes wrong (the blade deflects, binds, or starts thumping) and you only notice
   if you happen to be standing there listening.
2. You often can't see when the blade has actually come through the steel.

So I built a free tool that runs on a phone. You lay the phone on the saw, tap Start once, and leave it.
It listens to every cut and shows:

- **GOOD / ODD / STOP**: a big color screen that turns yellow when the sound is unusual and flashes
  red (and vibrates) when it stays far from normal for 6 seconds.
- **Cut progress**: a bar that fills as the sound changes near the end of the cut, then a
  "Finishing: blade nearly through" message.

No audio is recorded or uploaded. The phone only keeps measurements of the sound.

## How it works 

1. Twice a second the phone takes the last second of sound and
measures 70 things about it: how loud it is, the strongest pitch, how "bright" it is (spectral centroid
and rolloff), whether it's pulsing like a "duh duh duh" thump, and a 64-band picture of which
frequencies are present (a mel spectrum). Those 70 numbers are the "fingerprint" of that half-second.

2. I only taught it with good cuts and a variety of bad cuts. The training groups the good fingerprints into a few "kinds of normal"
(k-means clustering). For any new sound it measures how far it is from the nearest kind of normal, using a
distance that knows which measurements usually move together (Mahalanobis distance; the covariance is
"shrunk" with Ledoit-Wolf so it stays stable with not much data). 

3. The yellow and red lines come from testing each good cut with a
model that was trained *without* it (leave-one-out). That way the lines reflect how the model treats a cut
it has never heard, which is the real situation in the lab. I use the median plus a robust spread so one
odd recording can't move the lines much. There's also a plain rule anyone can check: STOP if the cut gets
6 dB louder than its own first minute. A STOP only shows once it has lasted 6 seconds with the motor still
running (yellow until then), so the motor winding down after switch-off doesn't flash STOP.

4. When the blade gets through the steel, the sound gets a little quieter and
brighter. I used my "Blade is through" taps to learn one direction in the 70-number space that separates
"in the steel" from "through" (a linear classifier). Each cut is compared with its *own* earlier sound,
not a fixed level, because where the phone sits changes all the numbers. The change is a slow slide over the
last 20-60 seconds, so the page says "Finishing" when the slide starts instead of naming the exact second.

5. Training is in Python. The phone page runs the same
feature maths and model in JavaScript. A test runs both on the same data and checks that they agree.

## Using it at the saw

- Open the page on the phone, tap **Start listening**, allow the microphone, lay the phone on the saw
  housing (same spot every time). That's it. It handles cut after cut by itself: it knows when the motor
  starts and stops, and saves every cut automatically.
- Use "Add to Home Screen" so it opens like an app. It works offline once loaded.
- **Data collection mode** (Settings) is only for training. It shows "Blade touched steel" / "Blade is through"
  buttons and asks GOOD or ODD after each cut. Operators don't need it.

## Retraining with new cuts

1. On the phone: Saved cuts → **Download as a file**.
2. Put the file in `data/saved_cuts/` (add notes in the file if something unusual happened).
3. Run:

```powershell
.venv\Scripts\python -m src.train       # learn from all good cuts, prints the honest scorecard
.venv\Scripts\python -m src.evaluate    # full report + reports/weirdness.png
.venv\Scripts\python -m pytest -q       # 29 automatic checks, incl. phone vs Python agreement
git add data/saved_cuts models web/model.json; git commit -m "Add cuts"; git push
```

The phone picks up the new model the next time Start is tapped with signal.

A good cut with loud shop noise in it (air blasts, the engraving pen) can be kept out of training by
adding `"train": false` to it, so the alarm lines don't loosen.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
.venv\Scripts\python -m http.server -d web 8000   # try the page on the computer: http://localhost:8000
```


## What I learned

- Signal processing: FFTs, mel spectrograms and which sound features actually change when a saw cuts.
- Anomaly detection when you only have examples of "good", and why that's normal for machines.
- Why you must test on data the model never saw, and how easy it is to fool yourself by tuning on
  the same data you score on.
- Building something people can use without training: offline web app, microphone access, keeping
  the screen awake, and never losing a cut if the phone dies.

## Tools

Python (NumPy, SciPy, scikit-learn, librosa, pandas, Matplotlib, pytest), JavaScript (Web Audio API,
AudioWorklet, service worker / PWA), GitHub Pages. Built with the help of an AI coding assistant.
