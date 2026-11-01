#!/usr/bin/env python3
"""
spx_convert.py - turn SPHEREx spectral-image cutouts (.fits) into the single
data file (spherex-data.json) that the SPHEREx Blink page loads.

Needs only numpy:   pip install numpy

Each input picture is re-pointed onto one shared grid (north up, east left,
6.15 arcsec pixels) centred on your target, so the same pixel is the same bit of
sky in every frame. That is what makes blinking between dates meaningful.

Examples
  python spx_convert.py *.fits --name "WISE 0855" --ra 133.79513 --dec -7.24515
  python spx_convert.py new_files/*.fits --name "WISE 0855" --merge    # add new data later

--merge keeps the frames already in spherex-data.json and adds only new ones,
so updating for a new data release is: download new cutouts, run this, upload.
Use a different --name for each target; the page lists every field it finds.
"""
import argparse, base64, glob, json, os, re, sys
import numpy as np

PIXSCALE = 6.15  # arcsec per output pixel (SPHEREx native pixel size)
# Pixel-flag bits that mean "do not trust this pixel" (see the SPHEREx Explanatory Supplement).
BAD_BITS = [0, 1, 2, 6, 7, 9, 10, 11, 14, 15, 17, 19]
SOURCE_BIT = 21  # pixel belongs to a known catalog source (informational)


def read_fits(path):
    """Minimal FITS reader (path or raw bytes). Returns a list of (header dict, data)."""
    buf = bytes(path) if isinstance(path, (bytes, bytearray)) else open(path, "rb").read()
    pos, out = 0, []
    while pos < len(buf):
        hdr, end = {}, False
        while not end:
            blk = buf[pos:pos + 2880]
            pos += 2880
            for i in range(0, 2880, 80):
                c = blk[i:i + 80].decode("ascii", "replace")
                if c.startswith("END"):
                    end = True
                    break
                if c[8:10] == "= ":
                    v = c[10:]
                    if v.lstrip().startswith("'"):
                        v = v.lstrip()[1:].split("'")[0]
                    else:
                        v = v.split("/")[0]
                    hdr[c[:8].strip()] = v.strip()
        bp, nax = int(hdr.get("BITPIX", 8)), int(hdr.get("NAXIS", 0))
        shape = [int(hdr["NAXIS%d" % (i + 1)]) for i in range(nax)][::-1]
        n = int(np.prod(shape)) if shape else 0
        size = abs(bp) // 8 * n
        raw = buf[pos:pos + size]
        data = raw
        if hdr.get("XTENSION", "IMAGE") == "IMAGE" and n:
            dt = {8: ">u1", 16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8"}[bp]
            data = np.frombuffer(raw, dtype=dt).reshape(shape)
        pos += ((size + int(hdr.get("PCOUNT", 0)) + 2879) // 2880) * 2880
        out.append((hdr, data))
    return out


class WCS:
    """Celestial WCS: TAN projection with SIP distortion, as used by SPHEREx."""

    def __init__(s, h):
        g = lambda k: float(h[k])
        s.c1, s.c2 = g("CRPIX1"), g("CRPIX2")
        s.M = np.diag([g("CDELT1"), g("CDELT2")]) @ np.array([[g("PC1_1"), g("PC1_2")], [g("PC2_1"), g("PC2_2")]])
        s.a0, s.d0 = np.radians(g("CRVAL1")), np.radians(g("CRVAL2"))
        coef = lambda L: {(int(m.group(1)), int(m.group(2))): g(k) for k in h
                          for m in [re.fullmatch(L + r"_(\d+)_(\d+)", k)] if m}
        s.A, s.B = coef("A"), coef("B")

    def pix2tan(s, x, y):  # 1-based pixel -> tangent-plane degrees
        u, v = x - s.c1, y - s.c2
        u2 = u + sum(c * u**p * v**q for (p, q), c in s.A.items())
        v2 = v + sum(c * u**p * v**q for (p, q), c in s.B.items())
        return s.M[0, 0] * u2 + s.M[0, 1] * v2, s.M[1, 0] * u2 + s.M[1, 1] * v2

    def sky2tan(s, ra, dec):
        a, d = np.radians(ra), np.radians(dec)
        cc = np.sin(s.d0) * np.sin(d) + np.cos(s.d0) * np.cos(d) * np.cos(a - s.a0)
        xi = np.cos(d) * np.sin(a - s.a0) / cc
        eta = (np.cos(s.d0) * np.sin(d) - np.sin(s.d0) * np.cos(d) * np.cos(a - s.a0)) / cc
        return np.degrees(xi), np.degrees(eta)

    def sky2pix(s, ra, dec):
        xi, eta = s.sky2tan(ra, dec)
        uv = np.linalg.inv(s.M) @ np.vstack([xi.ravel(), eta.ravel()])
        x, y = uv[0] + s.c1, uv[1] + s.c2
        for _ in range(6):  # Newton steps to undo the distortion
            fx, fy = s.pix2tan(x, y)
            ex, ey = xi.ravel() - fx, eta.ravel() - fy
            ax, ay = s.pix2tan(x + .5, y), s.pix2tan(x, y + .5)
            J = np.array([[(ax[0] - fx) / .5, (ay[0] - fx) / .5], [(ax[1] - fy) / .5, (ay[1] - fy) / .5]])
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            x = x + (J[1, 1] * ex - J[0, 1] * ey) / det
            y = y + (-J[1, 0] * ex + J[0, 0] * ey) / det
        return x.reshape(ra.shape), y.reshape(ra.shape)


def tan2sky(ra0, dec0, xi, eta):  # tangent-plane degrees -> sky degrees
    a0, d0 = np.radians(ra0), np.radians(dec0)
    xi, eta = np.radians(xi), np.radians(eta)
    rho, c = np.hypot(xi, eta), np.arctan(np.hypot(xi, eta))
    with np.errstate(all="ignore"):
        s = np.where(rho == 0, 0, eta * np.sin(c) * np.cos(d0) / rho)
        d = np.arcsin(np.cos(c) * np.sin(d0) + s)
        a = a0 + np.arctan2(xi * np.sin(c), rho * np.cos(d0) * np.cos(c) - eta * np.sin(d0) * np.sin(c))
    return np.degrees(a) % 360, np.degrees(d)


def bilinear(img, x, y):  # x, y are 1-based pixel positions; NaN outside the picture
    ny, nx = img.shape
    xx, yy = x - 1, y - 1
    x0, y0 = np.floor(xx).astype(int), np.floor(yy).astype(int)
    fx, fy = xx - x0, yy - y0
    ok = (x0 >= 0) & (y0 >= 0) & (x0 < nx - 1) & (y0 < ny - 1)
    x0, y0 = np.clip(x0, 0, nx - 2), np.clip(y0, 0, ny - 2)
    v = (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
         + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)
    return np.where(ok, v, np.nan)


def wavelength_map(tab_raw, h, x, y):
    """Per-pixel wavelength (micron) from the WCS-WAVE lookup table (9 x 9 grid)."""
    X = np.frombuffer(tab_raw, ">i4", 9, 0).astype(float)
    Y = np.frombuffer(tab_raw, ">i4", 9, 36).astype(float)
    V = np.frombuffer(tab_raw, ">f4", 162, 72).reshape(9, 9, 2)[:, :, 0].astype(float)
    # pixel -> table coordinate: CRVAL + CDELT * (pixel - CRPIX)
    px = float(h["CRVAL1W"]) + float(h["CDELT1W"]) * (x - float(h["CRPIX1W"]))
    py = float(h["CRVAL2W"]) + float(h["CDELT2W"]) * (y - float(h["CRPIX2W"]))
    def idx(P, A):
        i = np.clip(np.searchsorted(A, P) - 1, 0, len(A) - 2)
        return i, np.clip((P - A[i]) / (A[i + 1] - A[i]), 0, 1)
    ix, fx = idx(px, X)
    iy, fy = idx(py, Y)
    return (V[iy, ix] * (1 - fx) * (1 - fy) + V[iy, ix + 1] * fx * (1 - fy)
            + V[iy + 1, ix] * (1 - fx) * fy + V[iy + 1, ix + 1] * fx * fy)


def convert(path, ra0, dec0, N, name=None):
    """path may be a file path or the raw bytes of a .fits file (then pass name)."""
    ext = {h.get("EXTNAME"): (h, d) for h, d in read_fits(path)}
    h, img = ext["IMAGE"]
    flags = ext["FLAGS"][1].astype(np.int64)
    wcs = WCS(h)
    j, i = np.mgrid[0:N, 0:N]  # row j from the top, column i from the left
    east = (N / 2 - (i + .5)) * PIXSCALE / 3600  # degrees; east is to the left
    north = (N / 2 - (j + .5)) * PIXSCALE / 3600
    ra, dec = tan2sky(ra0, dec0, east, north)
    x, y = wcs.sky2pix(ra, dec)
    out = bilinear(img.astype(float), x, y).astype("<f4")
    xi = np.clip(np.rint(x - 1).astype(int), 0, flags.shape[1] - 1)
    yi = np.clip(np.rint(y - 1).astype(int), 0, flags.shape[0] - 1)
    f = flags[yi, xi]
    bad = np.zeros(f.shape, bool)
    for b in BAD_BITS:
        bad |= ((f >> b) & 1).astype(bool)
    code = (bad * 1 + ((f >> SOURCE_BIT) & 1) * 2 + np.isnan(out) * 4).astype("u1")
    wav = wavelength_map(ext["WCS-WAVE"][1], h, x, y)
    wav = np.where(np.isnan(out), 0, np.round(wav * 1e4)).astype("<u2")
    b64 = lambda a: base64.b64encode(a.tobytes()).decode()
    name = name or os.path.basename(path)
    m = re.search(r"level2_(\d{4}W\d+_\w+?)_(\d+)_(\d)D(\d)", name)
    return {
        "id": "%s_D%s" % (h.get("OBSID", name), h["DETECTOR"]),  # one exposure = up to 6 detector frames
        "det": int(h["DETECTOR"]),
        "date": h["DATE-OBS"][:10],
        "mjd": float(h["MJD-OBS"]),
        "survey": m.group(1) if m else "",
        "img": b64(out), "flag": b64(code), "wav": b64(wav),
    }


def main():
    ap = argparse.ArgumentParser(description="Convert SPHEREx cutouts to spherex-data.json")
    ap.add_argument("files", nargs="+", help=".fits files (wildcards ok)")
    ap.add_argument("-o", "--out", default="spherex-data.json")
    ap.add_argument("--name", default="Field", help="name shown on the page")
    ap.add_argument("--ra", type=float, help="target RA in degrees")
    ap.add_argument("--dec", type=float, help="target Dec in degrees")
    ap.add_argument("--size", type=int, default=90, help="output size in pixels")
    ap.add_argument("--merge", action="store_true", help="add to an existing data file")
    a = ap.parse_args()
    files = sorted({f for p in a.files for f in glob.glob(p)})
    if not files:
        sys.exit("No .fits files found.")
    data = {"version": 1, "pixscale": PIXSCALE, "fields": []}
    if a.merge and os.path.exists(a.out):
        data = json.load(open(a.out))
    field = next((f for f in data["fields"] if f["name"] == a.name), None)
    if field is None:
        if a.ra is None or a.dec is None:
            h = read_fits(files[0])[1][0]
            ctr = WCS(h)
            n = int(h["NAXIS1"]) / 2
            t = ctr.pix2tan(n, n)
            a.ra, a.dec = [float(v) for v in tan2sky(np.degrees(ctr.a0), np.degrees(ctr.d0), *t)]
            print("No --ra/--dec given; using the centre of the first file: %.5f %.5f" % (a.ra, a.dec))
        field = {"name": a.name, "ra": a.ra, "dec": a.dec, "n": a.size, "frames": []}
        data["fields"].append(field)
    have = {f["id"] for f in field["frames"]}
    for p in files:
        try:
            fr = convert(p, field["ra"], field["dec"], field["n"])
        except Exception as e:  # keep going if one file is odd
            print("  skipped %s (%s)" % (os.path.basename(p), e))
            continue
        if fr["id"] in have:
            print("  already have", fr["id"])
            continue
        field["frames"].append(fr)
        print("  added %s  D%d  %s" % (fr["id"], fr["det"], fr["date"]))
    field["frames"].sort(key=lambda f: f["mjd"])
    json.dump(data, open(a.out, "w"), separators=(",", ":"))
    print("Wrote %s: %d field(s), %.1f MB" % (a.out, len(data["fields"]), os.path.getsize(a.out) / 1e6))


if __name__ == "__main__":
    main()
