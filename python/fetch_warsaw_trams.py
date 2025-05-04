import requests
import json


OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def fetch_warsaw_trams():
    """
    Query Overpass API for Warsaw tram relations and save them as warsaw_trams.geojson
    (relations without a `ref=*` tag are skipped)
    """
    print("Fetching Warsaw tram data…")

    query = """
    [out:json];
    area["name"="Warszawa"]->.warsaw;
    relation["type"="route"]["route"="tram"](area.warsaw);
    out body;
    >;
    out skel qt;
    """

    rsp = requests.post(OVERPASS_URL, data={"data": query})
    if rsp.status_code != 200:
        raise RuntimeError(f"Overpass error {rsp.status_code}: {rsp.text[:200]}")

    data = rsp.json()

    # Collect nodes and ways
    nodes = {}
    ways = {}
    tram_lines = {}

    for el in data["elements"]:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])

        elif el["type"] == "way" and "nodes" in el:
            ways[el["id"]] = el["nodes"]

        elif el["type"] == "relation" and el["tags"].get("route") == "tram":
            ref = el["tags"].get("ref")          # <-- None if missing
            if not ref:                          # skip unnamed lines
                continue

            tram_lines.setdefault(
                ref,
                {"ref": ref, "name": el["tags"].get("name", f"Tram {ref}"), "ways": []},
            )

            tram_lines[ref]["ways"].extend(
                m["ref"] for m in el["members"] if m["type"] == "way" and m["role"] == ""
            )

    # Build GeoJSON
    features = []
    for ref, line in tram_lines.items():
        for way_id in line["ways"]:
            if way_id not in ways:
                continue
            coords = [
                [nodes[n_id][1], nodes[n_id][0]]  # lon, lat for GeoJSON
                for n_id in ways[way_id]
                if n_id in nodes
            ]
            if coords:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {"ref": ref, "name": line["name"]},
                        "geometry": {"type": "LineString", "coordinates": coords},
                    }
                )

    out = {"type": "FeatureCollection", "features": features}

    with open("warsaw_trams.geojson", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    # keep raw dump for debugging
    with open("warsaw_trams_raw.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"Saved {len(tram_lines)} tram lines "
        f"({len(features)} LineStrings) to warsaw_trams.geojson"
    )
    return len(features)


if __name__ == "__main__":
    fetch_warsaw_trams()
