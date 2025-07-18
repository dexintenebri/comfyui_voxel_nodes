import numpy as np
from PIL import Image
import noise

# ------------ Biome/Color Palette ---------------
BIOME_PALETTE = {
    (60,120,220): 1,  # water
    (90,190,90): 2,   # grass
    (40,120,40): 3,   # forest
    (160,150,140): 4, # mountain
    (220,210,140): 5, # desert
    (230,230,220): 6, # snow
    (130,130,130): 7, # rock
    (220,200,120): 8, # sand
    (120,120,120): 9, # unknown/fallback
}

def color_to_biome_idx(rgb):
    """Map RGB tuple to nearest biome index."""
    min_dist = 1e9
    best_idx = 9
    for col, idx in BIOME_PALETTE.items():
        dist = np.linalg.norm(np.array(col)-np.array(rgb))
        if dist < min_dist:
            min_dist = dist
            best_idx = idx
    return best_idx

# ------------ Noise and Height Utilities ---------------
def perlin_fractal_noise(shape, scale=0.07, octaves=5, seed=0):
    h, w = shape
    arr = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            n = noise.pnoise2(
                x*scale, y*scale,
                octaves=octaves, base=seed
            )
            arr[y, x] = n
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

# ------------ 3D Grid Generation from Image+Noise ---------------
def img_to_seeded_3d_grid(img, output_shape, noise_strength=0.3, noise_seed=0):
    img = img.resize(output_shape[:2], Image.BILINEAR)
    arr = np.array(img.convert("RGB"))
    h, w, d = output_shape
    grid = np.zeros((h, w, d), dtype=np.uint8)

    brightness = arr.mean(axis=2) / 255.0
    noise_map = perlin_fractal_noise((h, w), scale=0.07, octaves=4, seed=noise_seed)
    heightmap = (brightness * (1-noise_strength) + noise_map * noise_strength)
    for y in range(h):
        for x in range(w):
            rgb = tuple(arr[y, x])
            biome_idx = color_to_biome_idx(rgb)
            height = int(heightmap[y, x] * (d-1))
            for z in range(d):
                if z < height:
                    grid[y, x, z] = biome_idx
                else:
                    grid[y, x, z] = 0  # air/empty
    return grid

# ------------ WFC Pattern Extraction and Synthesis ---------------
class Pattern3D:
    def __init__(self, data):
        self.data = data
        self.hash = hash(data.tobytes())
    def __eq__(self, other):
        return np.array_equal(self.data, other.data)
    def __hash__(self):
        return self.hash
    @staticmethod
    def from_sample(sample, n, channels=1):
        sx, sy, sz = sample.shape
        patterns = []
        pattern_map = {}
        pattern_id_grid = np.zeros((sx-n+1, sy-n+1, sz-n+1), dtype=np.int32)
        for x in range(sx-n+1):
            for y in range(sy-n+1):
                for z in range(sz-n+1):
                    sub = sample[x:x+n, y:y+n, z:z+n]
                    p = Pattern3D(sub.copy())
                    if p not in pattern_map:
                        pattern_map[p] = len(patterns)
                        patterns.append(p)
                    pattern_id_grid[x, y, z] = pattern_map[p]
        return patterns, pattern_id_grid

def pattern_agree(p1, p2, dx, dy, dz):
    n = p1.shape[0]
    x1a, x1b = max(0, dx), n if dx >= 0 else n + dx
    y1a, y1b = max(0, dy), n if dy >= 0 else n + dy
    z1a, z1b = max(0, dz), n if dz >= 0 else n + dz
    x2a, x2b = max(0, -dx), n if dx <= 0 else n - dx
    y2a, y2b = max(0, -dy), n if dy <= 0 else n - dy
    z2a, z2b = max(0, -dz), n if dz <= 0 else n - dz
    return np.all(
        p1[x1a:x1b, y1a:y1b, z1a:z1b] == p2[x2a:x2b, y2a:y2b, z2a:z2b]
    )

class Propagator3D:
    def __init__(self, patterns):
        self.patterns = patterns
        self.num_patterns = len(patterns)
        self.dirs = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
        self.propagator = [
            [set() for _ in range(6)] for _ in range(self.num_patterns)
        ]
        for t in range(self.num_patterns):
            for d, (dx,dy,dz) in enumerate(self.dirs):
                for t2 in range(self.num_patterns):
                    if pattern_agree(
                        patterns[t].data, patterns[t2].data, dx, dy, dz
                    ):
                        self.propagator[t][d].add(t2)
    def compatible_patterns(self, pattern_idx, direction):
        return self.propagator[pattern_idx][direction]

class Cell3D:
    def __init__(self, num_patterns):
        self.allowed = np.ones(num_patterns, dtype=bool)
        self.observed = False
        self.chosen_pattern = None
    def entropy(self):
        if self.observed: return 0
        return np.sum(self.allowed)
    def observe(self, rng):
        choices = np.flatnonzero(self.allowed)
        if len(choices) == 0: return False
        chosen = rng.choice(choices)
        self.allowed[:] = False
        self.allowed[chosen] = True
        self.observed = True
        self.chosen_pattern = chosen
        return True

class Grid3D:
    def __init__(self, shape, num_patterns, seed=None):
        self.shape = shape
        self.nx, self.ny, self.nz = shape
        self.num_patterns = num_patterns
        self.cells = np.empty(shape, dtype=object)
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    self.cells[x, y, z] = Cell3D(num_patterns)
        self.rng = np.random.default_rng(seed)
    def find_lowest_entropy(self):
        min_entropy = np.inf
        min_pos = None
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    c = self.cells[x, y, z]
                    if not c.observed:
                        e = c.entropy()
                        if 1 < e < min_entropy:
                            min_entropy = e
                            min_pos = (x, y, z)
        return min_pos
    def observe(self):
        pos = self.find_lowest_entropy()
        if pos is None: return None
        success = self.cells[pos].observe(self.rng)
        if not success: return "contradiction"
        return pos
    def is_contradiction(self):
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    if not self.cells[x, y, z].observed and np.sum(self.cells[x, y, z].allowed) == 0:
                        return True
        return False
    def is_fully_observed(self):
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    if not self.cells[x, y, z].observed:
                        return False
        return True
    def get_output_grid(self):
        out = np.zeros(self.shape, dtype=int)
        for x in range(self.nx):
            for y in range(self.ny):
                for z in range(self.nz):
                    c = self.cells[x, y, z]
                    idx = (
                        c.chosen_pattern
                        if c.observed and c.chosen_pattern is not None
                        else -1
                    )
                    out[x, y, z] = idx
        return out

class WFC3D:
    def __init__(self, sample, n, out_shape, seed=None, max_propagate_steps=50000):
        self.patterns, self.pattern_id_grid = Pattern3D.from_sample(sample, n)
        self.grid = Grid3D(out_shape, len(self.patterns), seed=seed)
        self.propagator = Propagator3D(self.patterns)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.max_propagate_steps = max_propagate_steps
    def run(self, max_steps=100000):
        steps = 0
        while not self.grid.is_fully_observed():
            if self.grid.is_contradiction():
                print("[WFC3D ERROR] Contradiction detected. Failed.")
                return False
            pos = self.grid.observe()
            if pos is None or pos == "contradiction":
                print("[WFC3D ERROR] Observation failed. Contradiction.")
                return False
            ok = self.propagate(*pos)
            if not ok:
                print("[WFC3D ERROR] Propagation failed or exceeded step limit.")
                return False
            steps += 1
            if steps > max_steps:
                print("[WFC3D ERROR] Too many steps, aborting.")
                return False
        return True
    def propagate(self, x, y, z):
        queue = [(x, y, z)]
        steps = 0
        seen = set()
        while queue:
            if steps > self.max_propagate_steps:
                print("[WFC3D ERROR] Propagation exceeded maximum steps. Aborting propagation.")
                return False
            cx, cy, cz = queue.pop(0)
            ccell = self.grid.cells[cx, cy, cz]
            for d, (dx, dy, dz) in enumerate(self.propagator.dirs):
                nx, ny, nz = cx + dx, cy + dy, cz + dz
                if (
                    0 <= nx < self.grid.nx
                    and 0 <= ny < self.grid.ny
                    and 0 <= nz < self.grid.nz
                ):
                    ncell = self.grid.cells[nx, ny, nz]
                    if ncell.observed:
                        continue
                    changed = False
                    for p in range(self.grid.num_patterns):
                        if not ncell.allowed[p]:
                            continue
                        compatible = False
                        for cp in range(self.grid.num_patterns):
                            if ccell.allowed[cp]:
                                compat_set = self.propagator.compatible_patterns(cp, d ^ 1)
                                if p in compat_set:
                                    compatible = True
                                    break
                        if not compatible:
                            ncell.allowed[p] = False
                            changed = True
                    if np.sum(ncell.allowed) == 0:
                        print(f"[WFC3D ERROR] Contradiction at cell ({nx},{ny},{nz}) during propagation.")
                        return False
                    if changed:
                        cell_hash = (nx, ny, nz)
                        if cell_hash not in seen:
                            queue.append(cell_hash)
                            seen.add(cell_hash)
            steps += 1
        return True
    def get_output_grid(self):
        return self.grid.get_output_grid()
    def decode_to_voxel_grid(self):
        idx_grid = self.grid.get_output_grid()
        nx, ny, nz = idx_grid.shape
        out = np.zeros((nx, ny, nz), dtype=self.patterns[0].data.dtype)
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    idx = idx_grid[x, y, z]
                    if idx < 0:
                        continue
                    out[x, y, z] = self.patterns[idx].data[0, 0, 0]
        return out

# ---- Greedy Meshing (stub) ----
def greedy_mesh(voxel_grid):
    # This is a stub: you can integrate PyMCubes, trimesh, or similar libraries for full greedy meshing.
    # Here, just count nonzero voxels as a proxy.
    return np.count_nonzero(voxel_grid)

# ---- Export to MagicaVoxel .vox using midvoxio ----
def save_vox(voxel_grid, palette, out_path):
    try:
        import midvoxio
        # Build voxels array: (N, 4) with columns x, y, z, color_index
        voxels = []
        sx, sy, sz = voxel_grid.shape
        for x in range(sx):
            for y in range(sy):
                for z in range(sz):
                    color_idx = voxel_grid[x, y, z]
                    if color_idx != 0:
                        voxels.append([x, y, z, color_idx])
        voxels = np.array(voxels, dtype=np.uint8)
        # Save with palette (expects RGBA uint8 array, shape (256,4))
        midvoxio.save(out_path, voxels, palette)
        print(f"Saved VOX to {out_path} using midvoxio.")
    except ImportError:
        print("midvoxio not installed, can't save .vox files.")

# ---- Main Terrain Generation Pipeline ----
def generate_procedural_terrain(
    img_path, output_shape=(64,64,32), tile_size=2, seed=42,
    noise_strength=0.3, out_vox="terrain.vox"
):
    img = Image.open(img_path)
    print("[INFO] Creating seeded grid from image + noise...")
    grid = img_to_seeded_3d_grid(img, output_shape, noise_strength=noise_strength, noise_seed=seed)
    print("[INFO] Extracting patterns and running WFC...")
    wfc = WFC3D(grid, n=tile_size, out_shape=output_shape, seed=seed)
    ok = wfc.run()
    if not ok:
        print("[ERROR] WFC failed. Try a different image, tile size, or seed.")
        return
    print("[INFO] Decoding to voxel grid...")
    vox_grid = wfc.decode_to_voxel_grid()
    palette_arr = np.zeros((256,4), dtype=np.uint8)
    for rgb, idx in BIOME_PALETTE.items():
        palette_arr[idx,0:3] = rgb
        palette_arr[idx,3] = 255
    palette_arr[0] = [0,0,0,0]
    print(f"[INFO] Voxel grid shape: {vox_grid.shape}")
    save_vox(vox_grid, palette_arr, out_vox)
    print("[INFO] Greedy mesh complexity (proxy):", greedy_mesh(vox_grid))
    print("[INFO] Done!")

# ---- Usage Example ----
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python wfc3d.py path/to/image.png")
    else:
        generate_procedural_terrain(
            img_path=sys.argv[1],
            output_shape=(64,64,32),
            tile_size=2,
            seed=42,
            noise_strength=0.35,
            out_vox="output_terrain.vox"
        )