#!/usr/bin/env python3
"""
tram_segment_separation.py — Creates properly separated tram corridor segments

This script:
1. Loads tram lines from GeoJSON and creates a buffer for each line
2. Creates a planar partition where each point belongs to exactly one polygon
3. Properly separates all physically disconnected segments
4. Filters out small areas (< MIN_AREA sq meters)

Parameters:
    --buffer    Buffer width (meters)      default: 30m
    --min_area  Minimum area               default: 5000 sq meters
    --simplify  Simplification tolerance   default: 1m
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer

# Default parameters
BUFFER_M = 30.0        # Buffer width in meters
MIN_AREA_M2 = 5000.0   # Minimum area in square meters
SIMPLIFY_M = 1.0       # Simplification tolerance in meters

# CRS
WGS84 = "EPSG:4326"
PLM = "EPSG:2180"  # Polish projected CRS (metres)

FWD = Transformer.from_crs(WGS84, PLM, always_xy=True).transform
BACK = Transformer.from_crs(PLM, WGS84, always_xy=True).transform


def show_progress(current, total, message="Processing", bar_length=50):
    """Display a progress bar in the terminal"""
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r{message}: [{arrow + spaces}] {int(percent * 100)}%")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write('\n')


def load_ref_ways(src: str):
    """Load tram lines from GeoJSON and convert to metric projection"""
    with open(src, encoding="utf-8") as f:
        gj = json.load(f)

    ways = []
    for ft in gj["features"]:
        ref = ft["properties"].get("ref")
        if not ref:
            continue
        geom_ll = shape(ft["geometry"])
        geom_m = transform(FWD, geom_ll)
        if not geom_m.is_empty and geom_m.length:
            ways.append((ref, geom_m))
    return ways


def create_buffers(ways: List[Tuple[str, LineString]], buffer_size: float, simplify_tolerance: float):
    """Create a buffer for each tram line reference with simplification"""
    ref_buffers = defaultdict(list)
    
    print("Grouping geometries by reference...")
    # Group geometries by ref
    for ref, geom in ways:
        ref_buffers[ref].append(geom)
    
    # Merge geometries for each ref and create buffer
    merged_buffers = {}
    
    total = len(ref_buffers)
    print(f"Creating buffers for {total} tram lines...")
    
    for i, (ref, geoms) in enumerate(ref_buffers.items()):
        show_progress(i, total, "Creating buffers")
        try:
            merged = unary_union(geoms)
            # Simplify before buffering to reduce complexity
            if simplify_tolerance > 0:
                merged = merged.simplify(simplify_tolerance, preserve_topology=True)
            buffer = merged.buffer(buffer_size)
            merged_buffers[ref] = buffer
        except Exception as e:
            print(f"\nError processing line {ref}: {e}")
            continue
    
    show_progress(total, total, "Creating buffers")
    return merged_buffers


def safe_operation(operation, geom1, geom2):
    """Safely perform a geometric operation with error handling"""
    try:
        return operation(geom1, geom2)
    except Exception:
        # Try with simplified geometries
        try:
            simple1 = geom1.simplify(0.1, preserve_topology=True)
            simple2 = geom2.simplify(0.1, preserve_topology=True)
            return operation(simple1, simple2)
        except Exception:
            # If that still fails, try with buffer(0) to clean up geometry
            try:
                clean1 = geom1.buffer(0)
                clean2 = geom2.buffer(0)
                return operation(clean1, clean2)
            except Exception:
                # Last resort: just return an empty geometry
                print(f"\nWarning: Operation failed after multiple attempts")
                if operation.__name__ == 'intersection':
                    return Polygon()  # Empty polygon for intersection
                else:
                    return geom1      # Original geometry for difference


def safe_intersection(geom1, geom2):
    """Safely perform intersection operation"""
    return safe_operation(lambda a, b: a.intersection(b), geom1, geom2)


def safe_difference(geom1, geom2):
    """Safely perform difference operation"""
    return safe_operation(lambda a, b: a.difference(b), geom1, geom2)


def create_planar_partition(buffers, min_area):
    """
    Create a planar partition where each point in space belongs to exactly one polygon,
    tagged with all tram lines that pass through it.
    """
    print("Creating planar partition of space...")
    
    # Get all buffer geometries and create the universe
    buffer_geoms = list(buffers.values())
    universe = unary_union(buffer_geoms)
    
    # Create a dictionary to track which refs are present at each point in space
    # We'll do this by binary encoding: each ref gets a position in the binary number
    refs = list(buffers.keys())
    ref_to_bit = {ref: 1 << i for i, ref in enumerate(refs)}
    
    # Initialize with empty space (no refs)
    regions = {0: universe}
    
    # For each tram line, update the regions
    for i, ref in enumerate(refs):
        show_progress(i, len(refs), "Processing tram lines")
        
        buffer_geom = buffers[ref]
        new_regions = {}
        
        # For each existing region...
        for code, region in regions.items():
            if region.is_empty:
                continue
                
            # Split into part that's inside this buffer and part that's outside
            inside = safe_intersection(region, buffer_geom)
            outside = safe_difference(region, buffer_geom)
            
            # Area inside buffer gets this ref added
            if not inside.is_empty and inside.area >= min_area:
                new_code = code | ref_to_bit[ref]
                if new_code in new_regions:
                    new_regions[new_code] = unary_union([new_regions[new_code], inside])
                else:
                    new_regions[new_code] = inside
            
            # Area outside buffer keeps same refs
            if not outside.is_empty and outside.area >= min_area:
                if code in new_regions:
                    new_regions[code] = unary_union([new_regions[code], outside])
                else:
                    new_regions[code] = outside
        
        regions = new_regions
    
    show_progress(len(refs), len(refs), "Processing tram lines")
    
    # Convert binary codes back to ref sets
    result = {}
    for code, geom in regions.items():
        if code == 0:  # Skip areas with no tram lines
            continue
            
        # Decode which refs are in this region
        region_refs = frozenset([ref for ref in refs if code & ref_to_bit[ref]])
        
        # Add to result
        if region_refs in result:
            result[region_refs] = unary_union([result[region_refs], geom])
        else:
            result[region_refs] = geom
    
    return result


def get_polygon_parts(geometry, min_area):
    """
    Extract individual polygons from a geometry, ensuring proper separation
    of disconnected parts and filtering by minimum area.
    """
    parts = []
    
    # Handle empty geometries
    if geometry is None or geometry.is_empty:
        return parts
    
    # Handle multi-polygons explicitly
    if isinstance(geometry, MultiPolygon):
        for poly in geometry.geoms:
            if poly.area >= min_area:
                parts.append(poly)
    # Handle single polygons
    elif isinstance(geometry, Polygon):
        if geometry.area >= min_area:
            parts.append(geometry)
    # Try to get polygons from other geometry types
    else:
        try:
            # Try to get parts by converting to polygons
            if hasattr(geometry, 'geoms'):
                for geom in geometry.geoms:
                    if isinstance(geom, Polygon) and geom.area >= min_area:
                        parts.append(geom)
            else:
                # Last resort - buffer by 0 to create polygons
                buffered = geometry.buffer(0)
                return get_polygon_parts(buffered, min_area)
        except Exception as e:
            print(f"Error extracting polygons: {e}")
    
    return parts


def separate_all_disconnected_segments(corridors, min_area):
    """
    Thoroughly separate all physically disconnected segments into individual features,
    even if they have the same tram line combination.
    """
    print("Separating all disconnected segments...")
    result = []
    
    # Process each corridor with its reference set
    for i, (refs, geom) in enumerate(corridors.items()):
        show_progress(i, len(corridors), "Separating segments")
        
        tram_lines = sorted(list(refs))
        
        # Get all separate polygon parts
        parts = get_polygon_parts(geom, min_area)
        
        # Add each part as a separate feature
        for j, part in enumerate(parts):
            result.append({
                "refs": tram_lines,
                "line_count": len(tram_lines),
                "area_m2": round(part.area, 1),
                "geometry": part
            })
    
    show_progress(len(corridors), len(corridors), "Separating segments")
    
    print(f"Created {len(result)} separate corridor segments")
    return result


def save_geojson(features, filename):
    """Save features to GeoJSON file"""
    print(f"Saving to {filename}...")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"type": "FeatureCollection", "features": features},
            f,
            ensure_ascii=False,
        )
    print(f"Wrote {filename}")


def main(src, buffers_dst, corridors_dst, buffer_size, min_area, simplify_tolerance):
    start_time = time.time()
    
    # Load tram lines
    print("Loading tram lines...")
    ways = load_ref_ways(src)
    print(f"Loaded {len(ways):,} ways")
    
    # Create a buffer for each tram line reference
    buffers = create_buffers(ways, buffer_size, simplify_tolerance)
    print(f"Created {len(buffers):,} reference buffers")
    
    # Create planar partition
    corridors = create_planar_partition(buffers, min_area)
    print(f"Found {len(corridors):,} unique tram line combinations")
    
    # Separate ALL disconnected segments
    separated_segments = separate_all_disconnected_segments(corridors, min_area)
    print(f"Separated into {len(separated_segments)} distinct corridor segments")
    
    # Create GeoJSON features for individual buffers
    print("Creating GeoJSON features for buffers...")
    buffer_features = []
    for i, (ref, buffer_m) in enumerate(buffers.items()):
        show_progress(i, len(buffers), "Processing buffers")
        buffer_ll = transform(BACK, buffer_m)
        buffer_features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"buffer_{i}",
                    "tram_line": ref,
                    "area_m2": round(buffer_m.area, 1),
                },
                "geometry": mapping(buffer_ll),
            }
        )
    show_progress(len(buffers), len(buffers), "Processing buffers")
    
    # Create GeoJSON features for corridor sections
    print("Creating GeoJSON features for segments...")
    segment_features = []
    for i, segment in enumerate(separated_segments):
        show_progress(i, len(separated_segments), "Processing segments")
        try:
            geom_ll = transform(BACK, segment["geometry"])
            segment_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": f"segment_{i}",
                        "tram_lines": segment["refs"],
                        "line_count": segment["line_count"],
                        "area_m2": segment["area_m2"],
                    },
                    "geometry": mapping(geom_ll),
                }
            )
        except Exception as e:
            print(f"\nError processing segment {i}: {e}")
            continue
    
    show_progress(len(separated_segments), len(separated_segments), "Processing segments")
    
    # Save to GeoJSON files
    save_geojson(buffer_features, buffers_dst)
    save_geojson(segment_features, corridors_dst)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tram segment separation")
    p.add_argument("--src", default="warsaw_trams.geojson")
    p.add_argument("--buffers", default="warsaw_tram_buffers.geojson")
    p.add_argument("--segments", default="warsaw_tram_segments.geojson")
    p.add_argument("--buffer", type=float, default=BUFFER_M, help="buffer size (m)")
    p.add_argument("--min_area", type=float, default=MIN_AREA_M2, help="minimum area (sq m)")
    p.add_argument("--simplify", type=float, default=SIMPLIFY_M, help="simplification tolerance (m)")
    a = p.parse_args()
    main(a.src, a.buffers, a.segments, a.buffer, a.min_area, a.simplify)