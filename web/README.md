# Saw Sound Meter: the phone page

One web page that listens through the phone's microphone and shows **GOOD / ODD / STOP** and the
cut progress while the saw runs. Everything happens on the phone; nothing is uploaded.

## Two modes

- **Operator mode** (default): tap **Start listening** once, lay the phone on the saw, leave it.
  It notices when the motor starts and stops, judges every cut, and saves each cut's measurements
  automatically (unlabelled). No other buttons, no questions.
- **Data collection mode** (Settings, or open the page with `?collect`): adds the
  "Blade touched steel" / "Blade is through" buttons and asks GOOD or ODD after each cut.
  This is how I collect training data.

Phones only allow a web page to use the microphone after one tap, so Start is needed once per session.
The page keeps the screen awake while listening.

## Try it on the computer

```powershell
.venv\Scripts\python -m http.server -d web 8000
```
Open http://localhost:8000, open **Test with a recording** and pick an .m4a from `data/`.

## Hosting

The page is plain static files (no server code), so any free static host works. It must be **https**,
or the phone refuses microphone access. It's on GitHub Pages now. To move it to Vercel, import the
repository at vercel.com and set **Root Directory** to `web`.

## After changing anything in `web/`

Bump the version in **both** `version.js` and `sw.js` (the test checks they match). The service worker
caches the page for offline use, and the version badge proves the phone loaded the new code.
