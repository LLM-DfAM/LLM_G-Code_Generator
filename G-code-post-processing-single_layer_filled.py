import re
import math
import time
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString, LinearRing
from shapely.ops import unary_union

# ================= Configuration =================
INPUT_FILE = "DeepSeek-G-Coder_generate_gcode_files-complete/GPU0_DeepSeek_module-1.12_teeth_count-16_bore_diameter-5.53.gcode"
OUTPUT_FILE = "DeepSeek-G-Coder_generate_gcode_files-filled/GPU0_DeepSeek_module-1.12_teeth_count-16_bore_diameter-5.53.gcode"


# --- Print config ---
LAYER_HEIGHT = 0.20
LINE_WIDTH = 0.45
TRAVEL_SPEED = 42000
PRINT_SPEED = 1473.915  # refer to OrcaSlicer

# --- Filament config ---
FILAMENT_DIAMETER = 1.75
FLOW_RATIO = 0.95

# --- Infill config ---
INFILL_DENSITY = 1.0
OVERLAP_PERCENT = 0.15

# --- Extrusion ---
RETRACTION_DIST = 0.8
PRIME_EXTRA = 0.0

# --- E ---
FILAMENT_AREA = math.pi * (FILAMENT_DIAMETER / 2) ** 2
E_PER_MM = ((LINE_WIDTH * LAYER_HEIGHT) / FILAMENT_AREA) * FLOW_RATIO


# ================= 1. G-code parsing  =================
def parse_arc(start_pt, end_pt, offset, clockwise, segments=40):
    x0, y0 = start_pt
    x1, y1 = end_pt
    i, j = offset
    xc, yc = x0 + i, y0 + j
    start_angle = math.atan2(y0 - yc, x0 - xc)
    end_angle = math.atan2(y1 - yc, x1 - xc)
    if clockwise:
        if end_angle > start_angle: end_angle -= 2 * math.pi
    else:
        if end_angle < start_angle: end_angle += 2 * math.pi
    points = []
    r = math.sqrt(i ** 2 + j ** 2)
    for step in range(1, segments + 1):
        t = start_angle + (end_angle - start_angle) * step / segments
        px = xc + r * math.cos(t)
        py = yc + r * math.sin(t)
        points.append((px, py))
    return points


def extract_loops_from_gcode(gcode_lines):
    loops = []
    current_loop = []
    current_x, current_y = 0.0, 0.0
    re_g1 = re.compile(r"G1\s+.*X([\d\.-]+).*Y([\d\.-]+)")
    re_g1_x = re.compile(r"G1\s+.*X([\d\.-]+)")
    re_g1_y = re.compile(r"G1\s+.*Y([\d\.-]+)")
    re_arc = re.compile(r"(G2|G3)\s+.*X([\d\.-]+).*Y([\d\.-]+).*I([\d\.-]+).*J([\d\.-]+)")
    re_extrude = re.compile(r"E([\d\.-]+)")

    for line in gcode_lines:
        line = line.strip()
        if not line or line.startswith(";"): continue
        e_match = re_extrude.search(line)
        is_extruding = (e_match is not None) and (float(e_match.group(1)) > 0)

        if (line.startswith("G0") or (line.startswith("G1") and not is_extruding)):
            if len(current_loop) > 2: loops.append(current_loop)
            current_loop = []
            mx = re.search(r"X([\d\.-]+)", line)
            my = re.search(r"Y([\d\.-]+)", line)
            if mx: current_x = float(mx.group(1))
            if my: current_y = float(my.group(1))
            continue

        if line.startswith("G1"):
            nx, ny = current_x, current_y
            found_coord = False
            m_full = re_g1.search(line)
            if m_full:
                nx, ny = float(m_full.group(1)), float(m_full.group(2))
                found_coord = True
            else:
                mx = re_g1_x.search(line)
                if mx: nx, found_coord = float(mx.group(1)), True
                my = re_g1_y.search(line)
                if my: ny, found_coord = float(my.group(1)), True
            if found_coord and is_extruding:
                current_loop.append((nx, ny))
                current_x, current_y = nx, ny
        elif line.startswith("G2") or line.startswith("G3"):
            m_arc = re_arc.search(line)
            if m_arc:
                cmd = m_arc.group(1)
                nx, ny = float(m_arc.group(2)), float(m_arc.group(3))
                i_val, j_val = float(m_arc.group(4)), float(m_arc.group(5))
                if is_extruding:
                    arc_points = parse_arc((current_x, current_y), (nx, ny), (i_val, j_val), cmd == "G2")
                    current_loop.extend(arc_points)
                current_x, current_y = nx, ny
    if len(current_loop) > 2: loops.append(current_loop)
    return loops


# ================= 2. Fill generation =================

def generate_concentric_infill(outer_poly, inner_holes):
    """。
    1. Continuously shrink (buffer) only for outer_poly.
    2. After each contraction, use inner_holes to "trim" off the excess.
    3. This ensures that the path always advances in a one-way direction from the outside inward, until it closely adheres to the central hole.
    """
    if INFILL_DENSITY <= 0: return []

    infill_paths = []

    hole_overlap = 0.1
    safe_buffer = LINE_WIDTH * (1.0 - hole_overlap)

    if inner_holes:
        raw_holes = unary_union(inner_holes)
        holes_keep_out_zone = raw_holes.buffer(safe_buffer, join_style=2)
    else:
        holes_keep_out_zone = None

    current_offset = -LINE_WIDTH * (1.0 - OVERLAP_PERCENT)
    step_offset = -LINE_WIDTH / INFILL_DENSITY

    max_loops = 1000

    for _ in range(max_loops):
        shrunk_poly = outer_poly.buffer(current_offset, join_style=2)

        if shrunk_poly.is_empty:
            break

        if holes_keep_out_zone:
            clipped_poly = shrunk_poly.difference(holes_keep_out_zone)
        else:
            clipped_poly = shrunk_poly

        if clipped_poly.is_empty:
            break

        boundaries = []
        if clipped_poly.geom_type == 'Polygon':
            boundaries.append(clipped_poly.exterior)
        elif clipped_poly.geom_type == 'MultiPolygon':
            for p in clipped_poly.geoms:
                boundaries.append(p.exterior)

        for b in boundaries:
            if b.length > 1.0:
                infill_paths.append(b)

        current_offset += step_offset

    return infill_paths


def lines_to_gcode(paths):
    gcode = ["; FEATURE: Infill (Concentric Safe)", "M83"]

    is_retracted = True

    for path in paths:
        coords = list(path.coords)
        if len(coords) < 2: continue

        start = coords[0]

        # --- 1. Preparation ---
        if not is_retracted:
            gcode.append(f"G1 F1800 E-{RETRACTION_DIST:.5f}")
            is_retracted = True

        # --- 2. Travel ---
        gcode.append(f"G0 X{start[0]:.3f} Y{start[1]:.3f} F{TRAVEL_SPEED}")

        # --- 3. Prime ---
        if is_retracted:
            gcode.append(f"G1 F1800 E{RETRACTION_DIST + PRIME_EXTRA:.5f}")
            is_retracted = False

        # --- 4. Print ---
        gcode.append(f"G1 F{PRINT_SPEED}")
        for i in range(1, len(coords)):
            p1 = coords[i - 1]
            p2 = coords[i]
            dist = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            e_amt = dist * E_PER_MM
            gcode.append(f"G1 X{p2[0]:.3f} Y{p2[1]:.3f} E{e_amt:.5f}")

    # --- 5. End ---
    if not is_retracted:
        gcode.append(f"G1 F1800 E-{RETRACTION_DIST:.5f}")
        is_retracted = True

    return gcode


def process_file():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = "; start printing object, unique label id: 15"
    end_marker = "; stop printing object, unique label id: 15"
    idx_start = content.find(start_marker)
    idx_end = content.find(end_marker)

    if idx_start == -1:
        return

    block = content[idx_start:idx_end]
    lines = block.split('\n')
    raw_loops = extract_loops_from_gcode(lines)

    if not raw_loops: return

    polys = [Polygon(loop) for loop in raw_loops]
    polys.sort(key=lambda p: p.area, reverse=True)
    outer_poly = polys[0]
    inner_holes = polys[1:]

    infill_lines = generate_concentric_infill(outer_poly, inner_holes)
    infill_gcode = lines_to_gcode(infill_lines)

    new_content = content[:idx_end] + "\n" + "\n".join(infill_gcode) + "\n" + content[idx_end:]

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Generation completed: {OUTPUT_FILE}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    process_file()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script running time: {execution_time:.4f} s")