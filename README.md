# Saw Sound Anomaly Detector

A free tool that learns what **one cut-off saw** sounds like when it runs normally,
and shows **green / yellow / red** when the sound changes, so operators who can't
rely on hearing still get a clear warning.

## Setup (once)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Everyday use

```powershell
.venv\Scripts\python -m src.train       # learn "normal" from the good cuts in data/normal (updates web/model.json)
.venv\Scripts\python -m src.evaluate    # honest report + reports/weirdness.png
.venv\Scripts\python -m pytest -q       # automatic checks (17 tests)
.venv\Scripts\python -m http.server -d web 8000   # try the phone page on this computer: http://localhost:8000
```
The phone page and how to put it online: see [web/README.md](web/README.md).

## Folder layout

| Folder | What goes there |
|---|---|
| `data/normal/` | WAV recordings of normal saw runs |
| `data/abnormal/` | WAV recordings where something was wrong |
| `data/test/` | New recordings you want to check |
| `data/public/mimii/` | Public machine-sound data for testing the pipeline |
| `data/runs.csv` | One row per recording (see below) |
| `data/saved_cuts/` | Files downloaded from the phone page's "That cut was GOOD/ODD" buttons |
| `models/` | The trained model (`saw.json`); a copy goes to `web/model.json` for the phone |
| `reports/` | Pictures from `src.evaluate` |
| `src/` | Reusable Python code |
| `notebooks/` | Step-by-step notebooks with plots |
| `tests/` | Automatic checks (`pytest`) |
| `web/` | Version 2 phone web page |

## Run log (`data/runs.csv`)

One row per recording. Delete the `EXAMPLE` row once you add real ones.

| Column | Example | Notes |
|---|---|---|
| filename | `2026-10-03_run01.wav` | must match the file name exactly |
| date | `2026-10-03` | |
| label | `normal` / `abnormal` | |
| operator | initials | |
| blade_type | `diamond_wafering` | |
| rpm | `3000` | blade speed |
| feed | `low` / `medium` / `high` or a number | feed rate / force |
| coolant | `on` / `off` | |
| material | `steel_1018` | |
| thickness_mm | `10` | |
| notes | free text | |
| train | `yes` / `no` | leave blank = yes for normal runs; `no` keeps a run out of training |
| use_from_s / use_until_s | `316` | only use this time range, e.g. to skip people talking |

## Recording tips (these matter more than any algorithm)

- Use the **same phone, same recording app, same mic position** every time
  (tape a mark on the bench).
- Turn **off** any "audio enhancement", "noise reduction" or "voice" mode in the app.
  Those change loudness on their own and confuse the detector.
- **M4A straight from your phone is fine** (WAV, MP3 also work). Just copy the file
  into `data/normal/` or `data/abnormal/` with File Explorer. No converting needed.
- **Start recording before the saw starts, and stop after it spins down.**
- Aim for **15–20 normal runs** across the materials you usually cut before
  trusting the alert thresholds.
