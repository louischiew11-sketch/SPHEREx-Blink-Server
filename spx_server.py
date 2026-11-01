#!/usr/bin/env python3
"""
spx_server.py - gives the SPHEREx Blink page pictures for ANY coordinates.

The page asks:   GET /field?ra=133.795&dec=-7.245        (degrees)
This server then
  1. asks NASA/IPAC IRSA which SPHEREx spectral images cover that spot (TAP query),
  2. downloads a small cutout of each one (in parallel),
  3. re-aligns them with spx_convert.py, and
  4. sends them back in the same format as spherex-data.json.
Results are saved in ./cache, so a repeat visit is instant, and when NASA releases
new data only the NEW images are downloaded and added.

Run it:        pip install numpy        then        python spx_server.py
It listens on  http://localhost:8000   (or the PORT environment variable).
Put the server's public address into SERVER near the top of index.html.

Optional query parameters: n (grid size in pixels, 20-100, default 90),
limit (max frames, default 60, spread evenly over time), det (e.g. 5,6 = only arrays D5, D6).
"""
import csv, io, json, os, sys, threading, time
import urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import spx_convert as spx

TAP_URL = os.environ.get("SPX_TAP", "https://irsa.ipac.caltech.edu/TAP/sync")
IRSA_ROOT = os.environ.get("SPX_ROOT", "https://irsa.ipac.caltech.edu/")
CACHE_DIR = os.environ.get("SPX_CACHE", "cache")
FRESH_SECONDS = 6 * 3600       # re-check the archive for new data after this long
DOWNLOAD_WORKERS = 6           # be gentle with the archive
MAX_JOBS = 2                   # fields being built at the same time
UA = "spherex-blink/1.0 (public sky viewer)"

jobs, lock = {}, threading.Lock()
slots = threading.BoundedSemaphore(MAX_JOBS)


def http_get(url, data=None, timeout=180, tries=3):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # network hiccup: wait and retry
            last = e
            time.sleep(1.5 * (k + 1))
    raise last


def find_frames(ra, dec):
    """Ask IRSA which SPHEREx spectral images cover this position."""
    adql = (
        "SELECT a.uri AS uri, p.time_bounds_lower AS mjd, p.energy_bandpassname AS band "
        "FROM spherex.artifact a JOIN spherex.plane p ON a.planeid = p.planeid "
        "WHERE 1 = CONTAINS(POINT('ICRS', %.6f, %.6f), p.poly) ORDER BY p.time_bounds_lower" % (ra, dec)
    )
    q = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql})
    text = http_get(TAP_URL + "?" + q, timeout=300).decode("utf-8", "replace")
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        uri = (r.get("uri") or "").strip()
        if not uri.lower().endswith(".fits"):
            continue
        band = (r.get("band") or "")
        rows.append({"url": uri if uri.startswith("http") else IRSA_ROOT + uri.lstrip("/"),
                     "mjd": float(r["mjd"]) if r.get("mjd") else 0.0,
                     "det": int(band[-1]) if band[-1:].isdigit() else 0})
    return rows


def spread(rows, limit):
    """Keep at most `limit` rows, spread evenly over time."""
    if len(rows) <= limit:
        return rows
    return [rows[int(i * (len(rows) - 1) / (limit - 1))] for i in range(limit)] if limit > 1 else rows[:1]


def cache_path(key):
    return os.path.join(CACHE_DIR, key + ".json")


def load_cache(key):
    try:
        return json.load(open(cache_path(key)))
    except Exception:
        return None


def payload(c, urls):
    frames = [f for f in c["frames"] if f["url"] in urls]
    return {"state": "done", "data": {"version": 1, "pixscale": spx.PIXSCALE, "fields": [
        {"ra": c["ra"], "dec": c["dec"], "n": c["n"], "frames": frames}]}}


def wanted(rows, dets, limit):
    if dets:
        rows = [r for r in rows if r["det"] in dets]
    return spread(rows, limit)


def build(key, ra, dec, n, dets, limit, job):
    try:
        with slots:
            job.update(msg="Searching the SPHEREx archive…")
            rows = wanted(find_frames(ra, dec), dets, limit)
            if not rows:
                job.update(state="error", msg="SPHEREx has no images covering that position yet.")
                return
            c = load_cache(key) or {"ra": ra, "dec": dec, "n": n, "frames": []}
            have = {f["url"] for f in c["frames"]}
            todo = [r for r in rows if r["url"] not in have]
            job.update(done=0, total=len(todo), msg="Downloading %d new images…" % len(todo))
            size = n * spx.PIXSCALE / 3600 * 1.2  # cutout width in degrees, with some margin

            def one(r):
                url = "%s?center=%.6f,%.6fd&size=%.4f" % (r["url"], ra, dec, size)
                try:
                    fr = spx.convert(http_get(url), ra, dec, n, name=os.path.basename(r["url"]))
                    fr["url"] = r["url"]
                    return fr
                except Exception as e:
                    print("  failed", r["url"], e, file=sys.stderr)
                    return None
                finally:
                    job["done"] += 1
                    job["msg"] = "Downloaded %d of %d images…" % (job["done"], job["total"])

            with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
                new = [f for f in pool.map(one, todo) if f]
            if todo and not new and not c["frames"]:
                job.update(state="error", msg="The archive did not return any image files.")
                return
            c["frames"] = sorted(c["frames"] + new, key=lambda f: f["mjd"])
            c["checked"] = time.time()
            os.makedirs(CACHE_DIR, exist_ok=True)
            json.dump(c, open(cache_path(key), "w"), separators=(",", ":"))
            job.update(state="done")
    except Exception as e:
        print("job failed:", e, file=sys.stderr)
        job.update(state="error", msg="Couldn’t reach the SPHEREx archive (%s)." % type(e).__name__)


def field_request(ra, dec, n, limit, dets):
    ra, dec = round(ra, 3), round(dec, 3)  # nearby requests share one cache entry
    key = "%.3f_%.3f_%d" % (ra, dec, n)
    c = load_cache(key)
    if c and time.time() - c.get("checked", 0) < FRESH_SECONDS:
        urls = {f["url"] for f in c["frames"] if not dets or f["det"] in dets}
        keep = {r["url"] for r in spread([{"url": f["url"], "mjd": f["mjd"]} for f in c["frames"] if f["url"] in urls], limit)}
        return payload(c, keep)
    with lock:
        job = jobs.get(key)
        if job and job["state"] == "done":
            c = load_cache(key)
            if c:
                urls = {f["url"] for f in c["frames"] if not dets or f["det"] in dets}
                keep = {r["url"] for r in spread([{"url": f["url"], "mjd": f["mjd"]} for f in c["frames"] if f["url"] in urls], limit)}
                jobs.pop(key, None)
                return payload(c, keep)
        if job and job["state"] == "error":
            jobs.pop(key, None)
            return {"state": "error", "msg": job["msg"]}
        if not job:
            job = jobs[key] = {"state": "running", "done": 0, "total": 0, "msg": "Starting…"}
            threading.Thread(target=build, args=(key, ra, dec, n, dets, limit, job), daemon=True).start()
        return {"state": "running", "done": job["done"], "total": job["total"], "msg": job["msg"]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if u.path != "/field":
            return self._send(404, {"state": "error", "msg": "Use /field?ra=...&dec=..."})
        try:
            ra, dec = float(q["ra"]) % 360, float(q["dec"])
            n, limit = int(q.get("n", 90)), int(q.get("limit", 60))
            dets = {int(x) for x in q["det"].split(",")} if q.get("det") else set()
            assert -90 <= dec <= 90 and 20 <= n <= 100 and 1 <= limit <= 120
        except Exception:
            return self._send(400, {"state": "error", "msg": "Please give ra and dec in degrees."})
        self._send(200, field_request(ra, dec, n, limit, dets))

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", sys.argv[1] if len(sys.argv) > 1 else 8000))
    print("SPHEREx Blink server on port %d  (cache folder: %s)" % (port, CACHE_DIR))
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
