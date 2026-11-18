import json
from tle_source import download

out = download()
json.dump(out, open("tles.json", "w"))
print("Saved", len(out) - 1, "satellites, fetched", out["_fetched"])
