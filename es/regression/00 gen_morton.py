import numpy as np

# ================================================================
# 1. Morton Encode (3D Morton Code, 5 bits per dimension)
# ================================================================

def _expand_bits_5(v):
    """Expand 5-bit integer into Morton interleaved bit pattern."""
    v &= 0x1F
    v = (v | (v << 8)) & 0x0000F00F
    v = (v | (v << 4)) & 0x000C30C3
    v = (v | (v << 2)) & 0x00249249
    return v

def morton3D(ix, iy, iz):
    """Compute 3D Morton code from 3× 5-bit coordinates."""
    return (_expand_bits_5(ix) << 2) | (_expand_bits_5(iy) << 1) | _expand_bits_5(iz)

# ================================================================
# 2. Define View Grid (Spherical Parameters)
# ================================================================

VIEW_GRID_SIZE = 32      # 32 levels per dimension

R_LEVELS = np.linspace(2, 8, VIEW_GRID_SIZE, endpoint=False)
YAW_LEVELS = np.linspace(0, 360, VIEW_GRID_SIZE, endpoint=False)
PITCH_LEVELS = np.linspace(-90, 90, VIEW_GRID_SIZE, endpoint=False)

# Convert to radians for completeness

YAW_RAD = np.radians(YAW_LEVELS)
PITCH_RAD = np.radians(PITCH_LEVELS)

print(f"Grid Size: R={len(R_LEVELS)}, Yaw={len(YAW_RAD)}, Pitch={len(PITCH_RAD)}")

# ================================================================
# 3. Build Morton Table Directly Using Quantized Indices
# ================================================================

view_db = []

for ir, r in enumerate(R_LEVELS):
    for iy, yaw in enumerate(YAW_RAD):
        for ip, pitch in enumerate(PITCH_RAD):
        # Directly use integer indices (ir, iy, ip) for Morton
            morton_id = morton3D(ir, iy, ip)

            view_db.append({
                "id": morton_id,
                "r": r,
                "yaw": yaw,
                "pitch": pitch,
                "indices": (ir, iy, ip)
            })


# ================================================================
# 4. Sort Morton Table (LUT building requires ordering)
# ================================================================

view_db.sort(key=lambda e: e["id"])

# ================================================================
# 5. Print Sample
# ================================================================

print("\n--- Sample Data (Sorted by Morton ID) ---")
print(f"{'MortonID':<10} | {'Indices(r,yaw,pitch)':<25} | {'Radius':<8} | {'Yaw(deg)':<10} | {'Pitch(deg)':<10}")
print("-"*80)

for item in view_db[:10]:
    mid = item["id"]
    ir, iy, ip = item["indices"]
    r, yaw, pitch = item["r"], item["yaw"], item["pitch"]
    print(f"{mid:<10} | {(ir, iy, ip)!s:<25} | {r:<8.2f} | {np.degrees(yaw):<10.1f} | {np.degrees(pitch):<10.1f}")

# ================================================================
# 6. Build LUT
# ================================================================

MORTON_BITS_PER_DIM = 5
LUT_SIZE = 1 << (3 * MORTON_BITS_PER_DIM)  # 32^3 = 32768
alpha_LUT = np.zeros(LUT_SIZE, dtype=np.float32)

# Example: fill LUT

# for v in view_db:
#   alpha_LUT[v["id"]] = 0.0025
