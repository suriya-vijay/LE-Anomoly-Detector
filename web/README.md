# Saw Sound Meter (phone page)

A single web page that listens through the phone's microphone and shows
**GOOD / ODD / STOP** while the cut-off saw runs. Everything happens on the
phone; nothing is uploaded anywhere.

## Try it on the computer first

```powershell
cd "C:\Audio Anomoly Detector LE"
.venv\Scripts\python -m http.server -d web 8000
```
Open http://localhost:8000 in Chrome or Edge. Scroll to **Test with a recording**
and pick one of the `.m4a` files in `data/`. You'll see the same timeline the
Python report draws.

## Put it on the phone (free, via GitHub Pages)

1. On github.com, create a new **public** repository (e.g. `saw-sound-meter`).
2. In this project folder (once): `git init`, `git add .`, `git commit -m "Saw sound meter"`,
   then follow GitHub's "push an existing repository" lines.
3. In the repository: **Settings → Pages → Build and deployment → Source: Deploy from a branch →
   Branch: main, folder: /docs** … but our page lives in `web/`, so either rename
   `web` to `docs`, or (simpler) keep `web/` and set the folder to **/ (root)**
   and open `https://<you>.github.io/saw-sound-meter/web/`.
4. Open that address on the phone, tap **Start listening**, allow the microphone,
   then use the browser's "Add to Home Screen" so it opens like an app.

The page must be opened over **https** (GitHub Pages is) or the phone refuses
microphone access.

## Using it at the saw

- Lay the phone on the saw housing at the tape mark, screen up, before starting the motor.
- Grey = motor off or just started (not judging). Green = GOOD. Yellow = ODD, watch it.
  Red (flashing, vibrating on Android) = STOP and check.
- After each cut, tap **That cut was GOOD** or **ODD**. The measurements are kept on the phone.
  Every so often tap **Download as a file** and drop the file into `data/saved_cuts/`
  so the model can be retrained with more examples.
- **Sensitivity**: 1.0× uses the lines learned from your good cuts. Raise it if the meter is
  missing things; lower it if it's nagging during cuts you know are fine.

## Updating the model

After adding recordings to `data/normal/` and rows to `data/runs.csv`:
```powershell
.venv\Scripts\python -m src.train
git add web/model.json; git commit -m "Retrained model"; git push
```
The phone picks up the new `model.json` the next time the page is opened online.
