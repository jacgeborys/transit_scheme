#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict

from shapely.geometry    import shape, mapping, LineString, MultiLineString
from shapely.ops         import linemerge

# Default folder for input/output
DEFAULT_FOLDER = "D:/QGIS/transit_scheme/python"

def load_and_process_centerlines(input_path):
    """
    1) Chop every original LineString into its raw vertex-to-vertex segments.
    2) Tag each raw segment with the tram_line refs that use it.
    3) Group segments by identical tram_line sets.
    4) Merge contiguous runs within each group into longer topo-segments.
    5) Return a GeoJSON FeatureCollection of those merged service-segments.
    """
    print(f"Loading centerlines from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        gj = json.load(f)

    # 1) Build raw vertex→vertex segments
    raw_seg_map = defaultdict(set)  # { (ptA,ptB): {ref,...} }
    for feat in gj["features"]:
        ref = feat["properties"].get("tram_line")
        if not ref:
            continue
        line = shape(feat["geometry"])
        coords = list(line.coords)
        for a, b in zip(coords, coords[1:]):
            seg = (tuple(a), tuple(b))
            # normalize orientation so (A,B)==(B,A)
            norm = seg if seg[0] <= seg[1] else (seg[1], seg[0])
            raw_seg_map[norm].add(str(ref))

    # 2) Group raw segments by the exact set of tram lines
    service_groups = defaultdict(list)  # { frozenset(refs): [ (ptA,ptB), ... ] }
    for seg_coords, refs in raw_seg_map.items():
        key = frozenset(int(r) for r in refs)
        service_groups[key].append(seg_coords)

    # 3) For each service-group, merge contiguous raw segments
    features = []
    for refs_set, seg_list in service_groups.items():
        # sorted list of tram numbers, e.g. [10,11,17]
        lines = sorted(refs_set)
        lines_key = "_".join(str(n) for n in lines)

        # build a MultiLineString of all raw pieces
        mls = MultiLineString([LineString(s) for s in seg_list])
        merged = linemerge(mls)
        if isinstance(merged, LineString):
            parts = [merged]
        else:
            # could be MultiLineString or GeometryCollection
            geoms = getattr(merged, "geoms", [])
            parts = [g for g in geoms if isinstance(g, LineString)]


        # emit one Feature per merged part
        for idx, part in enumerate(parts):
            serv_sec_id = f"sseg_{lines_key}_{idx}"
            features.append({
                "type": "Feature",
                "properties": {
                    "serv_sec_id":  serv_sec_id,
                    "segmentType":  "service",
                    "split":        "vertex-merged",
                    "lines":        lines
                },
                "geometry": mapping(part)
            })

    return {"type": "FeatureCollection", "features": features}


def main(input_file, output_file):
    print(f"Processing {input_file} → {output_file}")
    viz = load_and_process_centerlines(input_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(viz, f, ensure_ascii=False)
    print(f"Done: {len(viz['features'])} service-segments written.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create tram service-segments by merging raw vertex slices")
    p.add_argument(
        "--input",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_centerlines.geojson"),
        help="Input centerlines GeoJSON"
    )
    p.add_argument(
        "--output",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_viz.geojson"),
        help="Output service-segment GeoJSON"
    )
    args = p.parse_args()
    main(args.input, args.output)
