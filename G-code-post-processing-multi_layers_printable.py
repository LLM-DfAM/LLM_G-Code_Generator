import re
import os
import time


def generate_multi_layer_gear(
        source_file_path,
        output_file_path,
        target_layer_count=5,
        layer_height=0.2,
        bed_temp=60
):
    if not os.path.exists(source_file_path):
        print(f"Error: File not found {source_file_path}")
        return

    with open(source_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    start_marker = "; start printing object, unique label id: 15"
    stop_marker = "; stop printing object, unique label id: 15"

    try:
        header_part, rest = content.split(start_marker, 1)
        body_part_raw, footer_part = rest.split(stop_marker, 1)
        body_lines = (start_marker + body_part_raw).strip().splitlines()
    except ValueError:
        return

    # Calculate the final height
    final_print_height = target_layer_count * layer_height
    print(f"Target Height: {final_print_height:.2f}mm ({target_layer_count} layers)")

    # 2. Modify Header
    # Modify bed temperature
    header_part = re.sub(r'M140 S\d+', f'M140 S{bed_temp}', header_part)
    header_part = re.sub(r'M190 S\d+', f'M190 S{bed_temp}', header_part)
    header_part = header_part.replace("total layer number: 1", f"total layer number: {target_layer_count}")

    # 3. Modify Footer
    def sanitize_footer_z(match):
        original_z = float(match.group(1))
        if original_z < final_print_height:
            # + 5mm safety distance
            safe_z = final_print_height + 5.0
            return f"G1 Z{safe_z:.2f}"
        else:
            return match.group(0)

    footer_part_fixed = re.sub(r'G1 Z([\d\.]+)', sanitize_footer_z, footer_part)

    # Lift by 5mm
    safety_footer_prefix = (
        f"\n{stop_marker}\n"
        "M400 ; Wait for moves to finish\n"
        "G91 ; Relative positioning\n"
        "G1 Z5 F3000 ; SAFETY LIFT 5mm\n"  
        "G90 ; Absolute positioning\n"
    )

    with open(output_file_path, 'w', encoding='utf-8') as out:
        # Write Header
        out.write(header_part)

        # --- Make multi layers ---
        for layer_idx in range(1, target_layer_count + 1):
            current_z_offset = (layer_idx - 1) * layer_height

            if layer_idx == 1:
                out.write("; [Fan Logic] Layer 1: All Fans OFF to prevent warping\n")
                out.write("M106 S0    ; Main Part Fan OFF\n")
                out.write("M106 P1 S0 ; Aux Fan OFF (for Bambu/Orca)\n")

            if layer_idx > 1:
                out.write(f"\n;IyL_Script: LAYER {layer_idx} START\n")
                out.write(f"M73 L{layer_idx}\n")

                safe_hop_z = (layer_idx * layer_height) + 0.4
                out.write(f"G1 Z{safe_hop_z:.3f} F18000 ; Layer-Change Hop\n")

                if layer_idx == 2:
                    out.write("M106 P1 S100 ; Low Fan (30%) to prevent warping\n")
                elif layer_idx == 4:
                    out.write("M106 P1 S178 ; Normal Fan (70%)\n")

            for line in body_lines:
                if "M624" in line:
                    continue

                if 'Z' in line:
                    new_line = re.sub(r'Z([\d\.]+)',
                                      lambda m: f"Z{float(m.group(1)) + current_z_offset:.3f}",
                                      line)
                    out.write(new_line + "\n")
                else:
                    out.write(line + "\n")

        # --- Write Footer ---
        out.write(safety_footer_prefix)
        out.write(footer_part_fixed.strip())

    print(f"Generation completed: {output_file_path}")


# ==========================================
# Operation Config
# ==========================================
if __name__ == "__main__":
    start_time = time.perf_counter()

    # 1. Set up the source file (0.2mm single-layer G-code file)
    source_gcode = "DeepSeek-G-Coder_generate_gcode_files-filled/GPU0_DeepSeek_module-1.12_teeth_count-16_bore_diameter-5.53.gcode"

    # 2. Set up the target file
    output_gcode = "DeepSeek-G-Coder_generate_gcode_files-printable/GPU0_DeepSeek_module-1.12_teeth_count-16_bore_diameter-5.53.gcode"

    # 3. Parameters config
    TARGET_LAYERS = 20
    BED_TEMP = 68

    # Generate multi layers
    generate_multi_layer_gear(
        source_gcode,
        output_gcode,
        target_layer_count=TARGET_LAYERS,
        bed_temp=BED_TEMP
    )


    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Script running time: {execution_time:.4f} s")
