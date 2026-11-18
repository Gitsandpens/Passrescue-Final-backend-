import json, urllib.request
from datetime import datetime, timezone

IDS = [25994, 27424, 33591, 37849, 39084, 49260, 40697, 42063]
HDR = {"User-Agent": "PassRescue/1.0 (academic project)"}


def _get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers=HDR), timeout=20).read().decode()


def download():
    out, errs = {}, []
    for n in IDS:
        for src in ("celestrak", "backup"):
            try:
                if src == "celestrak":
                    l = _get(f"https://celestrak.org/NORAD/elements/gp.php?CATNR={n}&FORMAT=tle").strip().splitlines()[:3]
                    assert len(l) == 3 and l[1].strip().startswith("1 ")
                else:
                    j = json.loads(_get(f"https://tle.ivanstanojevic.me/api/tle/{n}"))
                    l = [j["name"], j["line1"], j["line2"]]
                out[str(n)] = [x.strip() for x in l]
                break
            except Exception as e:
                errs.append(f"{n} {src}: {e}")
    if len(out) < len(IDS):
        raise RuntimeError("Orbital data download failed: " + "; ".join(errs[-4:]))
    out["_fetched"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    return out
