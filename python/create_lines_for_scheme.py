#!/usr/bin/env python3
import argparse, json, os, sys, time, multiprocessing
from functools import partial
from collections import defaultdict

import numpy as np
from shapely.geometry   import (
    shape, mapping, LineString, MultiLineString, Polygon, Point, MultiPoint
)
from shapely.ops        import transform, unary_union, split, linemerge
from pyproj             import Transformer

# ——————————————————————————————————————————————
# CRS definitions
# ——————————————————————————————————————————————
WGS84 = "EPSG:4326"
PLM   = "EPSG:2180"  # Polish metric CRS

FWD   = Transformer.from_crs(WGS84, PLM, always_xy=True).transform
BACK  = Transformer.from_crs(PLM, WGS84, always_xy=True).transform

DEFAULT_FOLDER = "D:/QGIS/transit_scheme/python"

# ——————————————————————————————————————————————
# UTILITY: Simple progress bar
# ——————————————————————————————————————————————
def show_progress(i, n, msg="Working"):
    pct = int(100 * (i+1)/n)
    bar = "#" * int(pct/2) + "-" * (50 - int(pct/2))
    sys.stdout.write(f"\r{msg}: [{bar}] {pct}%")
    sys.stdout.flush()
    if i+1 == n:
        print()

# ——————————————————————————————————————————————
# STAGE 1: Load & Buffer raw tram lines
# ——————————————————————————————————————————————
def load_tram_lines(path):
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    ways = []
    for ft in gj["features"]:
        ref = ft["properties"].get("ref") or ft["properties"].get("tram_line")
        if not ref:
            continue
        geom_ll = shape(ft["geometry"])
        geom_m  = transform(FWD, geom_ll)
        if not geom_m.is_empty and geom_m.length:
            ways.append((str(ref), geom_m))
    return ways

def create_buffers(ways, buffer_dist=30, res=16, cap_style=1, join_style=1):
    """
    buffer_dist: meters
    res: resolution of circle approximation (16 is default round)
    cap_style=1 (round), join_style=1 (round) avoids sharp triangles
    """
    by_ref = defaultdict(list)
    for ref, geom in ways:
        by_ref[ref].append(geom)

    buffers = {}
    for i,(ref,geoms) in enumerate(by_ref.items()):
        show_progress(i, len(by_ref), "Buffering lines")
        merged = unary_union(geoms)
        # simplify *after* or *before* buffer if you need
        buf = merged.buffer(
            buffer_dist,
            resolution=res,
            cap_style=cap_style,
            join_style=join_style
        )
        buffers[ref] = buf
    return buffers

# ——————————————————————————————————————————————
# STAGE 2: Planar partition & segment separation
# ——————————————————————————————————————————————
def safe_op(op, a, b):
    try:
        return op(a, b)
    except:
        # fallbacks
        a0 = a.simplify(0.1, True).buffer(0)
        b0 = b.simplify(0.1, True).buffer(0)
        return op(a0, b0)

def partition_segments(buffers, min_area):
    """
    Build a dict: frozenset(refs) → Polygon union of their overlap region.
    Uses incremental binary‐coded union like your original.
    """
    refs      = list(buffers.keys())
    bit_mask  = {r:1<<i for i,r in enumerate(refs)}
    regions   = {0: unary_union(list(buffers.values()))}

    for i,ref in enumerate(refs):
        show_progress(i, len(refs), "Partitioning")
        newr = {}
        buf = buffers[ref]
        for code,poly in regions.items():
            if poly.is_empty: continue
            inside  = safe_op(lambda x,y: x.intersection(y), poly, buf)
            outside = safe_op(lambda x,y: x.difference(y), poly, buf)

            if inside.area >= min_area:
                c2 = code | bit_mask[ref]
                newr.setdefault(c2, []).append(inside)

            if outside.area >= min_area:
                newr.setdefault(code, []).append(outside)

        # dissolve each code back into a single polygon
        regions = {c: unary_union(ps) for c,ps in newr.items()}
    show_progress(len(refs)-1, len(refs), "Partitioning")
    # drop code 0 (no lines)
    return {
        frozenset(r for r in refs if code & bit_mask[r]): geom
        for code,geom in regions.items() if code
    }

def extract_polygon_parts(geom, min_area):
    parts = []
    if geom.is_empty: return parts
    if isinstance(geom, Polygon):
        if geom.area>=min_area: parts.append(geom)
    else:
        for g in getattr(geom, "geoms", []):
            if isinstance(g, Polygon) and g.area>=min_area:
                parts.append(g)
    return parts

def separate_all(corridors, min_area):
    """
    Flatten each (refs,Polygon) into a list of dicts with discrete parts.
    """
    out = []
    for i,(refs,poly) in enumerate(corridors.items()):
        show_progress(i, len(corridors), "Separating")
        for p in extract_polygon_parts(poly, min_area):
            out.append({
                "refs":      sorted(refs, key=int),
                "area_m2":   p.area,
                "geometry":  p
            })
    return out

# ——————————————————————————————————————————————
# STAGE 3: Create centerlines (unclipped)
# ——————————————————————————————————————————————
def follow_skeleton(polygon, **kwargs):
    """
    Your existing follow_skeleton() implementation goes here unchanged.
    """
    # … copy all the code from your old follow_skeleton() …
    raise NotImplementedError("Paste your skeleton‐follow code here")

def process_segment(seg, params):
    """Wrap follow_skeleton + per‐line expansion"""
    line_objs = []
    try:
        skeleton = follow_skeleton(seg["geometry"], **params)
        for ref in seg["refs"]:
            line_objs.append({
                "segment_id": seg.get("id",""),
                "tram_line":  ref,
                "geometry":   skeleton
            })
    except Exception as e:
        print(f"Skipping {seg.get('id')}—{e}")
    return line_objs

def create_centerlines(segments, params, max_join=300):
    # parallel map
    with multiprocessing.Pool(max(1, multiprocessing.cpu_count()-1)) as pool:
        f = partial(process_segment, params=params)
        all_cl = []
        for i, lst in enumerate(pool.imap_unordered(f, segments)):
            all_cl += lst
            show_progress(i, len(segments), "Centerlining")
    # TODO: paste your join_centerlines_by_ref() here if you still need it
    return all_cl

# ——————————————————————————————————————————————
# STAGE 4: Split & merge service‐segments on centerlines
# ——————————————————————————————————————————————
def split_service_segments(centerline_fc):
    """
    Paste your final “vertex→service” code from 04_split_centerlines.py here.
    Return a GeoJSON FC of LineString features with properties
    { serv_sec_id, segmentType, split, lines: […] }.
    """
    raise NotImplementedError("Paste your 04_split… logic here")

# ——————————————————————————————————————————————
# I/O
# ——————————————————————————————————————————————
def save_geojson(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"Wrote {path}")

# ——————————————————————————————————————————————
# MAIN
# ——————————————————————————————————————————————
def main(args):
    t0 = time.time()

    # 1) load + buffer
    ways    = load_tram_lines(args.input)
    buffers = create_buffers(ways, args.buffer, args.res, cap_style=1, join_style=1)
    save_geojson({
        "type":"FeatureCollection",
        "features":[
          {"type":"Feature","properties":{"ref":r},"geometry":mapping(g)}
          for r,g in buffers.items()
        ]
    }, args.buffers)

    # 2) partition & separate
    corridors = partition_segments(buffers, args.min_area)
    pieces    = separate_all(corridors, args.min_area)
    save_geojson({
        "type":"FeatureCollection",
        "features":[
          {"type":"Feature",
           "properties":{
             "refs":p["refs"],"area_m2":round(p["area_m2"],1)
           },
           "geometry":mapping(transform(BACK,p["geometry"]))
          }
          for p in pieces
        ]
    }, args.segments)

    # 3) centerlines
    centerlines = create_centerlines(pieces, {
        "min_width":    args.min_width,
        "max_width":    args.max_width,
        "sample_dist":  args.sample_dist,
        "simplify":     args.simplify,
        "smooth":       args.smooth
    }, args.max_join)
    save_geojson({
        "type":"FeatureCollection",
        "features":[
          {"type":"Feature",
           "properties":{
             "segment_id":cl["segment_id"],
             "tram_line": cl["tram_line"]
           },
           "geometry": mapping(transform(BACK, cl["geometry"]))
          }
          for cl in centerlines
        ]
    }, args.centerlines)

    # 4) split & merge into service‐segments
    viz = split_service_segments({
        "type":"FeatureCollection",
        "features":[
          {"type":"Feature","properties":{"tram_line":cl["tram_line"]},"geometry":mapping(cl["geometry"])}
          for cl in centerlines
        ]
    })
    save_geojson(viz, args.viz)

    print(f"All done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    DEFAULT_FOLDER = "D:/QGIS/transit_scheme/python"
    p = argparse.ArgumentParser(description="All-in-one tram pipeline")
    p.add_argument(
        "--input",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_trams.geojson"),
        help="Raw tram lines GeoJSON"
    )
    p.add_argument(
        "--buffers",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_buffers.geojson"),
        help="Buffered tram corridors GeoJSON"
    )
    p.add_argument(
        "--segments",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_segments.geojson"),
        help="Separated corridor segments GeoJSON"
    )
    p.add_argument(
        "--centerlines",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_centerlines.geojson"),
        help="Centerlines GeoJSON"
    )
    p.add_argument(
        "--viz",
        default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_viz.geojson"),
        help="Visualization-ready service segments GeoJSON"
    )
    # … other args …
    args = p.parse_args()
    main(args)

