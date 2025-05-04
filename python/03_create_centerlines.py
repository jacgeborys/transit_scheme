#!/usr/bin/env python3
"""
unclipped_tram_centerlines.py — Creates centerlines without clipping to boundaries

This script:
1. Loads corridor segments from a GeoJSON file
2. Creates centerlines that follow corridor shapes without forced clipping
3. Uses parallel processing for better performance
4. Creates a separate line for each tram line in the segment
5. Joins centerlines by tram line reference when endpoints are within a threshold distance
6. Saves the resulting centerlines to a GeoJSON file

Parameters:
    --segments      Input segments file         default: warsaw_tram_segments.geojson
    --output        Output centerlines file     default: warsaw_tram_centerlines.geojson
    --min_width     Minimum corridor width      default: 30m
    --max_width     Maximum corridor width      default: 80m
    --simplify      Simplification tolerance    default: 2m
    --smooth        Smoothing factor            default: 1
    --sample_dist   Sampling distance           default: 5m
    --max_join      Maximum join distance       default: 300m
"""

import argparse
import json
import sys
import time
import math
import multiprocessing
from functools import partial
from typing import Dict, List, Set, Tuple
import os

from shapely.geometry import LineString, Point, MultiLineString, Polygon, mapping, shape
from shapely.ops import transform, unary_union
from pyproj import Transformer

# Default parameters
MIN_WIDTH_M = 30.0     # Minimum corridor width in meters
MAX_WIDTH_M = 80.0     # Maximum corridor width in meters
SIMPLIFY_M = 2.0       # Simplification tolerance in meters
SMOOTH_FACTOR = 1      # Smoothing iterations
SAMPLE_DIST_M = 5.0    # Sampling distance along corridor
MAX_JOIN_DIST_M = 350.0  # Maximum joining distance in meters

DEFAULT_FOLDER = "D:/QGIS/transit_scheme/python"

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


def load_segments(src: str):
    """Load corridor segments from GeoJSON"""
    print(f"Loading segments from {src}...")
    with open(src, encoding="utf-8") as f:
        gj = json.load(f)

    segments = []
    for ft in gj["features"]:
        props = ft["properties"]
        geom_ll = shape(ft["geometry"])
        geom_m = transform(FWD, geom_ll)
        
        if not geom_m.is_empty:
            segment = {
                "id": props.get("id", "unknown"),
                "tram_lines": props.get("tram_lines", []),
                "line_count": props.get("line_count", 0),
                "area_m2": props.get("area_m2", 0),
                "geometry": geom_m
            }
            segments.append(segment)
    
    print(f"Loaded {len(segments)} segments")
    return segments


def estimate_width_at_point(point, polygon):
    """
    Estimate the width of the polygon at the given point.
    Returns the estimated width in the polygon's units.
    """
    if not polygon.contains(point):
        return 0
    
    # Try different angles
    widths = []
    for angle in range(0, 180, 15):  # Check every 15 degrees
        # Convert angle to radians
        radians = math.radians(angle)
        
        # Create a long line through the point
        dx = math.cos(radians)
        dy = math.sin(radians)
        
        # Make it long enough to extend beyond the polygon
        max_dim = max(polygon.bounds[2] - polygon.bounds[0], 
                      polygon.bounds[3] - polygon.bounds[1])
        
        line = LineString([
            (point.x - dx * max_dim, point.y - dy * max_dim),
            (point.x + dx * max_dim, point.y + dy * max_dim)
        ])
        
        # Intersect with the polygon
        intersection = line.intersection(polygon)
        
        # Calculate width
        if not intersection.is_empty:
            if isinstance(intersection, LineString):
                widths.append(intersection.length)
            elif isinstance(intersection, MultiLineString):
                # Sum the lengths of all parts
                widths.append(sum(part.length for part in intersection.geoms))
    
    if widths:
        # The width is the minimum distance across the polygon
        return min(widths)
    return 0


def filter_by_width(points, polygon, min_width, max_width):
    """
    Filter points based on the estimated width of the polygon at each point.
    Returns only points where the width is between min_width and max_width.
    """
    valid_points = []
    for point in points:
        width = estimate_width_at_point(Point(point), polygon)
        if min_width <= width <= max_width:
            valid_points.append(point)
    return valid_points


def smooth_line(coords, iterations=1):
    """
    Apply iterative smoothing to line coordinates.
    Each point is moved to the average position of itself and its neighbors.
    """
    if len(coords) <= 2:
        return coords
        
    for _ in range(iterations):
        new_coords = [coords[0]]  # Keep first point fixed
        
        # Smooth inner points
        for i in range(1, len(coords) - 1):
            # Calculate average with neighbors
            avg_x = (coords[i-1][0] + coords[i][0] + coords[i+1][0]) / 3
            avg_y = (coords[i-1][1] + coords[i][1] + coords[i+1][1]) / 3
            new_coords.append((avg_x, avg_y))
            
        new_coords.append(coords[-1])  # Keep last point fixed
        coords = new_coords
        
    return coords


def follow_skeleton(polygon, min_width, max_width, sample_dist=5.0, simplify_tolerance=3.0, smooth_iterations=2):
    """
    Create a centerline that follows the skeleton of the corridor.
    Does not clip to polygon boundaries.
    """
    # First simplify the polygon slightly to remove noise
    simple_poly = polygon.simplify(simplify_tolerance / 2, preserve_topology=True)
    
    # Generate a dense grid of points inside the polygon
    minx, miny, maxx, maxy = simple_poly.bounds
    grid_points = []
    
    x = minx
    while x <= maxx:
        y = miny
        while y <= maxy:
            point = Point(x, y)
            if simple_poly.contains(point):
                # Calculate distance to boundary
                dist = point.distance(simple_poly.boundary)
                grid_points.append((x, y, dist))
            y += sample_dist
        x += sample_dist
    
    if not grid_points:
        # Fallback to centroid
        centroid = simple_poly.centroid
        return LineString([(centroid.x - 1, centroid.y), (centroid.x + 1, centroid.y)])
    
    # Find local maxima (points furthest from boundary)
    local_maxima = []
    for x, y, dist in grid_points:
        is_local_max = True
        for nx, ny, ndist in grid_points:
            # Check if point is within neighborhood
            if abs(nx - x) <= sample_dist*1.5 and abs(ny - y) <= sample_dist*1.5:
                if (nx, ny) != (x, y) and ndist > dist:
                    is_local_max = False
                    break
        if is_local_max:
            local_maxima.append((x, y, dist))
    
    # If no local maxima, use points with highest distances
    if not local_maxima:
        grid_points.sort(key=lambda p: -p[2])
        local_maxima = grid_points[:max(3, len(grid_points) // 10)]
    
    # Extract coordinates
    points = [(x, y) for x, y, _ in local_maxima]
    
    # Filter by width without considering polygon containment
    valid_points = []
    for x, y in points:
        point = Point(x, y)
        if simple_poly.contains(point):  # Only check containment here
            width = estimate_width_at_point(point, simple_poly)
            if min_width <= width <= max_width:
                valid_points.append((x, y))
    
    if len(valid_points) < 2:
        # Try best fit line if not enough points
        return find_best_fit_line(simple_poly)
    
    # Use path-based ordering instead of principal axis
    
    # First, find the two endpoints that are farthest apart
    # These will likely be at opposite ends of the corridor
    max_distance = 0
    endpoints = None
    
    for i, p1 in enumerate(valid_points):
        for j, p2 in enumerate(valid_points[i+1:], i+1):
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if dist > max_distance:
                max_distance = dist
                endpoints = (p1, p2)
    
    if not endpoints:
        # Fallback to original method if we can't find endpoints
        try:
            # Calculate centroid
            sum_x = sum(p[0] for p in valid_points)
            sum_y = sum(p[1] for p in valid_points)
            center_x = sum_x / len(valid_points)
            center_y = sum_y / len(valid_points)
            
            # Calculate principal direction
            # Calculate covariance matrix
            cov_xx = sum((p[0] - center_x)**2 for p in valid_points) / len(valid_points)
            cov_xy = sum((p[0] - center_x) * (p[1] - center_y) for p in valid_points) / len(valid_points)
            cov_yy = sum((p[1] - center_y)**2 for p in valid_points) / len(valid_points)
            
            # Calculate eigenvalues and eigenvectors
            trace = cov_xx + cov_yy
            det = cov_xx * cov_yy - cov_xy**2
            
            eigenval1 = trace/2 + math.sqrt((trace/2)**2 - det)
            eigenval2 = trace/2 - math.sqrt((trace/2)**2 - det)
            
            # Principal direction
            if abs(eigenval1) > abs(eigenval2):
                if cov_xy == 0:
                    principal_dir = (1, 0)
                else:
                    principal_dir = (eigenval1 - cov_yy, cov_xy)
            else:
                if cov_xy == 0:
                    principal_dir = (0, 1)
                else:
                    principal_dir = (cov_xy, eigenval2 - cov_xx)
            
            # Normalize
            length = math.sqrt(principal_dir[0]**2 + principal_dir[1]**2)
            if length > 0:
                principal_dir = (principal_dir[0]/length, principal_dir[1]/length)
            
            # Project points onto principal direction
            projections = []
            for x, y in valid_points:
                # Projection value
                proj = (x - center_x) * principal_dir[0] + (y - center_y) * principal_dir[1]
                projections.append((x, y, proj))
            
            # Sort by projection value
            projections.sort(key=lambda p: p[2])
            
            # Extract points
            sorted_points = [(p[0], p[1]) for p in projections]
        except Exception:
            # Fallback to sorting by x-coordinate
            sorted_points = sorted(valid_points, key=lambda p: p[0])
    else:
        # Use nearest neighbor algorithm to create a path from one endpoint to the other
        start_point = endpoints[0]
        end_point = endpoints[1]
        
        # Add the start point
        path = [start_point]
        remaining_points = [p for p in valid_points if p != start_point]
        
        current = start_point
        
        # Build path by repeatedly finding the nearest unvisited point
        while remaining_points and len(path) < len(valid_points):
            # Find the nearest point
            nearest = min(remaining_points, key=lambda p: 
                          math.sqrt((p[0] - current[0])**2 + (p[1] - current[1])**2))
            
            # Add to path and remove from remaining
            path.append(nearest)
            remaining_points.remove(nearest)
            current = nearest
        
        # Calculate if the path should be reversed
        # We want the start point to be closest to the original start_point
        dist_to_start = math.sqrt((path[0][0] - start_point[0])**2 + (path[0][1] - start_point[1])**2)
        dist_to_end = math.sqrt((path[-1][0] - start_point[0])**2 + (path[-1][1] - start_point[1])**2)
        
        sorted_points = path if dist_to_start <= dist_to_end else list(reversed(path))
    
    # Create line from sorted points
    line = LineString(sorted_points)
    
    # Simplify to remove noise but preserve shape
    if simplify_tolerance > 0:
        line = line.simplify(simplify_tolerance, preserve_topology=True)
    
    # Apply smoothing
    smoothed_coords = smooth_line(list(line.coords), iterations=smooth_iterations)
    smoothed_line = LineString(smoothed_coords)
    
    # Important: Do NOT clip to polygon boundary
    return smoothed_line


def find_best_fit_line(polygon):
    """Find the best-fitting straight line through the polygon."""
    # Create line through center at various angles
    minx, miny, maxx, maxy = polygon.bounds
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    
    max_dim = max(maxx - minx, maxy - miny) * 2
    
    best_line = None
    max_length = 0
    
    for angle in range(0, 180, 10):
        radians = math.radians(angle)
        dx = math.cos(radians)
        dy = math.sin(radians)
        
        line = LineString([
            (center[0] - dx * max_dim, center[1] - dy * max_dim),
            (center[0] + dx * max_dim, center[1] + dy * max_dim)
        ])
        
        intersection = line.intersection(polygon)
        if not intersection.is_empty:
            if isinstance(intersection, LineString) and intersection.length > max_length:
                max_length = intersection.length
                best_line = intersection
            elif isinstance(intersection, MultiLineString):
                longest = max(intersection.geoms, key=lambda x: x.length)
                if longest.length > max_length:
                    max_length = longest.length
                    best_line = longest
    
    if best_line:
        return best_line
    
    # Last resort - short line through centroid
    centroid = polygon.centroid
    return LineString([(centroid.x - 1, centroid.y), (centroid.x + 1, centroid.y)])


def process_segment(segment, min_width, max_width, sample_dist, simplify_tolerance, smooth_iterations):
    """Process a single segment to create a centerline without clipping to boundaries."""
    try:
        # Get the centerline for this segment
        centerline = follow_skeleton(
            segment["geometry"], 
            min_width, 
            max_width,
            sample_dist,
            simplify_tolerance,
            smooth_iterations
        )
        
        if centerline.is_empty or centerline.length <= 0:
            print(f"\nWarning: Could not create centerline for segment {segment['id']}")
            return []
            
        # Create a separate centerline feature for each tram line
        return [{
            "segment_id": segment["id"],
            "tram_line": tram_line,
            "geometry": centerline
        } for tram_line in segment["tram_lines"]]
            
    except Exception as e:
        print(f"\nError processing segment {segment['id']}: {e}")
        return []


def create_centerlines_parallel(segments, min_width, max_width, sample_dist, simplify_tolerance, smooth_iterations):
    """
    Create centerlines for each segment using parallel processing.
    """
    print("Creating centerlines (parallel processing)...")
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_segment,
        min_width=min_width,
        max_width=max_width,
        sample_dist=sample_dist,
        simplify_tolerance=simplify_tolerance,
        smooth_iterations=smooth_iterations
    )
    
    # Use multiple cores
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    print(f"Using {num_cores} CPU cores")
    
    # Process segments in parallel
    try:
        with multiprocessing.Pool(num_cores) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(process_func, segments)):
                results.extend(result)
                show_progress(i + 1, len(segments), "Processing segments")
        return results
    except Exception as e:
        print(f"Parallel processing error: {e}")
        # Fallback to sequential processing
        results = []
        for i, segment in enumerate(segments):
            segment_results = process_segment(
                segment, min_width, max_width, sample_dist, simplify_tolerance, smooth_iterations
            )
            results.extend(segment_results)
            show_progress(i + 1, len(segments), "Processing segments")
        return results


def join_centerlines_by_ref(centerlines, max_distance=300.0):
    """
    Join centerlines that share the same reference number if their endpoints are within max_distance.
    Uses a more aggressive approach to ensure all possible connections are made.
    
    Args:
        centerlines: List of centerline dictionaries with 'geometry' and 'tram_line' keys
        max_distance: Maximum distance in meters between endpoints to add a connecting segment
        
    Returns:
        List of joined centerlines
    """
    print("Joining centerlines by reference number...")
    
    # Filter out centerlines with invalid geometry
    valid_centerlines = []
    for cl in centerlines:
        if cl["geometry"] and not cl["geometry"].is_empty:
            valid_centerlines.append(cl)
        else:
            print(f"Skipping centerline with invalid geometry: {cl.get('segment_id', 'unknown')}")
    
    # Group centerlines by tram_line
    centerlines_by_ref = {}
    for cl in valid_centerlines:
        tram_line = cl["tram_line"]
        if tram_line not in centerlines_by_ref:
            centerlines_by_ref[tram_line] = []
        centerlines_by_ref[tram_line].append(cl)
    
    joined_centerlines = []
    
    # Process each reference group
    for tram_line, lines in centerlines_by_ref.items():
        print(f"Processing tram line {tram_line} with {len(lines)} segments")
        
        # Skip if only one line in this group
        if len(lines) <= 1:
            joined_centerlines.extend(lines)
            continue
        
        # Create working copies of all lines
        working_lines = []
        for line in lines:
            working_lines.append({
                "segment_id": line["segment_id"],
                "tram_line": line["tram_line"],
                "geometry": LineString(line["geometry"].coords),
                "merged": False  # Track if this line has been merged into another
            })
        
        # Repeat until no more merges are possible
        merged_something = True
        iteration = 0
        max_iterations = len(working_lines) * 2  # Safety limit
        
        while merged_something and iteration < max_iterations:
            iteration += 1
            merged_something = False
            
            # Find all possible connections between unmerged lines
            connections = []
            for i, line1 in enumerate(working_lines):
                if line1["merged"]:
                    continue
                    
                geom1 = line1["geometry"]
                if not geom1 or geom1.is_empty:
                    continue
                    
                # Get start and end points
                start1 = Point(geom1.coords[0])
                end1 = Point(geom1.coords[-1])
                
                for j, line2 in enumerate(working_lines):
                    if i == j or line2["merged"]:
                        continue
                        
                    geom2 = line2["geometry"]
                    if not geom2 or geom2.is_empty:
                        continue
                        
                    start2 = Point(geom2.coords[0])
                    end2 = Point(geom2.coords[-1])
                    
                    # Calculate distances between all endpoint combinations
                    distances = [
                        (start1.distance(start2), 'start-start', i, j),
                        (start1.distance(end2), 'start-end', i, j),
                        (end1.distance(start2), 'end-start', i, j),
                        (end1.distance(end2), 'end-end', i, j)
                    ]
                    
                    # Find the shortest connection that's within our distance threshold
                    for dist, conn_type, from_idx, to_idx in sorted(distances, key=lambda x: x[0]):
                        if dist <= max_distance:
                            connections.append((dist, conn_type, from_idx, to_idx))
                            break  # Just take the best connection between these two lines
            
            # Sort connections by distance (shortest first)
            connections.sort(key=lambda x: x[0])
            
            # Process connections
            for dist, conn_type, i, j in connections:
                # Skip if either line has been merged already (might have changed during this iteration)
                if working_lines[i]["merged"] or working_lines[j]["merged"]:
                    continue
                
                line1 = working_lines[i]
                line2 = working_lines[j]
                geom1 = line1["geometry"]
                geom2 = line2["geometry"]
                
                # Get coordinates for connecting
                coords1 = list(geom1.coords)
                coords2 = list(geom2.coords)
                
                # Create new coordinates for the merged line
                if conn_type == 'start-start':
                    # Connect start points
                    new_coords = list(reversed(coords2)) + coords1
                elif conn_type == 'start-end':
                    # Connect start of line1 to end of line2
                    new_coords = coords2 + coords1
                elif conn_type == 'end-start':
                    # Connect end of line1 to start of line2
                    new_coords = coords1 + coords2
                else:  # 'end-end'
                    # Connect end points
                    new_coords = coords1 + list(reversed(coords2))
                
                # Update line1's geometry to include the merged line
                line1["geometry"] = LineString(new_coords)
                line1["segment_id"] = f"{line1['segment_id']}+{line2['segment_id']}"
                
                # Mark line2 as merged
                line2["merged"] = True
                
                # Record that we did a merge
                merged_something = True
                print(f"  Merged segments {line1['segment_id']} and {line2['segment_id']} (distance: {dist:.2f}m)")
                
                # Break after first merge to rebuild the connection list
                break
        
        print(f"  Completed after {iteration} iterations")
        
        # Add all unmerged lines to the result
        for line in working_lines:
            if not line["merged"]:
                joined_centerlines.append({
                    "segment_id": line["segment_id"],
                    "tram_line": line["tram_line"],
                    "geometry": line["geometry"]
                })
    
    print(f"Final centerline count: {len(joined_centerlines)}")
    return joined_centerlines


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


def main(segments_src, centerlines_dst, min_width, max_width, sample_dist, simplify_tolerance, smooth_iterations, max_join_distance=320.0):
    start_time = time.time()
    
    # Load segments
    segments = load_segments(segments_src)
    
    # Create centerlines
    centerlines = create_centerlines_parallel(
        segments, min_width, max_width, sample_dist, simplify_tolerance, smooth_iterations
    )
    
    # Join centerlines that share the same reference number
    joined_centerlines = join_centerlines_by_ref(centerlines, max_distance=max_join_distance)
    
    # Create GeoJSON features
    print("Creating GeoJSON features...")
    centerline_features = []
    for i, centerline in enumerate(joined_centerlines):
        try:
            geom_ll = transform(BACK, centerline["geometry"])
            centerline_features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": f"line_{i}",
                        "segment_id": centerline["segment_id"],
                        "tram_line": centerline["tram_line"],
                    },
                    "geometry": mapping(geom_ll),
                }
            )
        except Exception as e:
            print(f"\nError processing centerline {i}: {e}")
            continue
    
    # Save to GeoJSON file
    save_geojson(centerline_features, centerlines_dst)
    
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Unclipped tram centerline creation")
    p.add_argument("--segments", default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_segments.geojson"), help="Input segments file")
    p.add_argument("--output", default=os.path.join(DEFAULT_FOLDER, "warsaw_tram_centerlines.geojson"), help="Output centerlines file")
    p.add_argument("--min_width", type=float, default=MIN_WIDTH_M, help="Minimum corridor width (m)")
    p.add_argument("--max_width", type=float, default=MAX_WIDTH_M, help="Maximum corridor width (m)")
    p.add_argument("--sample_dist", type=float, default=SAMPLE_DIST_M, help="Sampling distance (m)")
    p.add_argument("--simplify", type=float, default=SIMPLIFY_M, help="Simplification tolerance (m)")
    p.add_argument("--smooth", type=int, default=SMOOTH_FACTOR, help="Smoothing iterations")
    p.add_argument("--max_join", type=float, default=MAX_JOIN_DIST_M, help="Maximum distance for joining centerlines (m)")
    a = p.parse_args()
    main(a.segments, a.output, a.min_width, a.max_width, a.sample_dist, a.simplify, a.smooth, a.max_join)