import numpy as np
from PIL import Image
import os
import datetime
import threading

from .wfc3d import WFCTerrain3DNode

from midvoxio.voxio import write_list_to_vox

try:
    import trimesh
except ImportError:
    trimesh = None

# --- Biome color table ---
BIOME_TABLE = [
    {"name": "water",    "rgb": (60,120,220), "color_idx": 1},
    {"name": "grass",    "rgb": (90,190,90),  "color_idx": 2},
    {"name": "forest",   "rgb": (40,120,40),  "color_idx": 3},
    {"name": "mountain", "rgb": (160,150,140),"color_idx": 4},
    {"name": "desert",   "rgb": (220,210,140),"color_idx": 5},
    {"name": "snow",     "rgb": (230,230,220),"color_idx": 6},
    {"name": "rock",     "rgb": (130,130,130),"color_idx": 7},
    {"name": "sand",     "rgb": (220,200,120),"color_idx": 8},
    {"name": "unknown",  "rgb": (120,120,120),"color_idx": 9}
]

def rgb_to_biome(rgb, depth):
    r, g, b = rgb
    r, g, b = int(r), int(g), int(b)    
    if b > 150 and depth < 64: return "water"
    if r > 210 and g > 210 and b > 210 and depth > 180: return "snow"
    if r > 200 and g > 200 and b < 150 and depth < 140: return "desert"
    if g > 130 and b < 150 and r < 160 and depth < 160: return "grass"
    if g > 90 and g > r and g > b and r < 100 and b < 100: return "forest"
    if r > 120 and g > 120 and b < 120 and depth > 180: return "mountain"
    if abs(r-g) < 20 and abs(r-b) < 20 and r > 120 and depth > 180: return "rock"
    if r > 200 and g > 180 and b < 120 and depth < 80: return "sand"
    return "unknown"

def biome_to_coloridx(biome):
    for b in BIOME_TABLE:
        if b["name"] == biome:
            return b["color_idx"]
    return 9

def biome_to_rgb(biome):
    for b in BIOME_TABLE:
        if b["name"] == biome:
            return b["rgb"]
    return (120,120,120)

def build_palette():
    arr = np.zeros((256,4), dtype=np.uint8)
    for i, b in enumerate(BIOME_TABLE, start=1):
        arr[i,0:3] = b["rgb"]
        arr[i,3] = 255
    arr[0] = [0,0,0,0]
    return arr

def img_to_biome_array(img, biome_mask=None, h=None, w=None, d=None):
    arr = np.array(img.convert("RGB"))
    if biome_mask is not None:
        mask = np.array(biome_mask.convert("L"))
    else:
        mask = None
    if h is None: h = arr.shape[0]
    if w is None: w = arr.shape[1]
    if d is None: d = arr.shape[0] // 4
    biome_grid = np.zeros((h, w, d), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rgb = arr[y % arr.shape[0], x % arr.shape[1]]
            biome = rgb_to_biome(rgb, y)
            idx = biome_to_coloridx(biome)
            zmax = int(d * (y / h))
            biome_grid[y, x, :zmax] = idx
    return biome_grid

def extract_edges_3d(grid, tile_size):
    top_face = grid[0:tile_size,:,:].copy()
    left_face = grid[:,0:tile_size,:].copy()
    front_face = grid[:,:,0:tile_size].copy()
    return {'top': top_face, 'left': left_face, 'front': front_face}

def grid_to_vox_rgba(voxel_grid, palette):
    # (X, Y, Z) -> (X, Y, Z, 4)
    shape = voxel_grid.shape
    out = np.zeros(shape + (4,), dtype=np.uint8)
    for idx in np.unique(voxel_grid):
        if idx == 0: continue
        out[voxel_grid == idx] = palette[idx]
    return out

def grid_to_trimesh(voxel_grid, palette, cube_size=1.0):
    if trimesh is None:
        return None
    H, W, D = voxel_grid.shape
    meshes = []
    for x in range(H):
        for y in range(W):
            for z in range(D):
                cidx = int(voxel_grid[x, y, z])
                if cidx > 0:
                    color = palette[cidx][:3] / 255.0
                    box = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
                    box.apply_translation((x * cube_size, y * cube_size, z * cube_size))
                    box.visual.face_colors = np.tile(np.append((color * 255).astype(np.uint8), 255), (box.faces.shape[0], 1))
                    meshes.append(box)
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)

def export_to_glb_from_grid(voxel_grid, palette, glb_path, cube_size=1.0):
    mesh = grid_to_trimesh(voxel_grid, palette, cube_size)
    if mesh is not None:
        mesh.export(glb_path)
        print(f"Saved GLB terrain to {glb_path}")
        return glb_path
    print("No mesh exported.")
    return ""

def wfc_generate_chunk(reference_grid, shape, tile_size, seed, edge_constraints=None):
    output_height, output_width, output_depth = shape
    print(f"[DEBUG] Generating WFC chunk: shape={shape}, tile_size={tile_size}, seed={seed}")
    print(f"[DEBUG] Reference grid shape: {reference_grid.shape}")
    wfc = WFC3D(reference_grid, n=tile_size, out_shape=(output_height, output_width, output_depth), seed=seed)
    print("[DEBUG] WFC3D instance created.")
    try:
        wfc.run()
    except Exception as e:
        print(f"[ERROR] Exception during wfc.run(): {e}")
        raise
    print("[DEBUG] WFC3D run completed.")
    try:
        chunk = wfc.decode_to_voxel_grid()
    except Exception as e:
        print(f"[ERROR] Exception during decode_to_voxel_grid: {e}")
        raise
    print("[DEBUG] Decoded chunk shape:", chunk.shape)
    return chunk

def generate_chunk_threaded(ref_grid, shape, tile_size, seed, edge_constraints, vox_path, glb_path, terrain_chunks, idx, edge_list, palette, cube_size):
    print(f"[THREAD] Starting chunk generation idx={idx}, seed={seed}")
    try:
        chunk = wfc_generate_chunk(ref_grid, shape, tile_size, seed, edge_constraints)
        rgba_grid = grid_to_vox_rgba(chunk, palette)
        write_list_to_vox(rgba_grid, vox_path, palette_arr=palette)
        glb_file = ""
        if glb_path and trimesh is not None:
            glb_file = export_to_glb_from_grid(chunk, palette, glb_path, cube_size)
        terrain_chunks[idx] = chunk
        edge_list[idx] = extract_edges_3d(chunk, tile_size)
        print(f"[THREAD] Chunk idx={idx} completed successfully.")
    except Exception as e:
        print(f"[THREAD ERROR] Exception in chunk idx={idx}: {e}")
        terrain_chunks[idx] = None
        edge_list[idx] = None

class WFC3DTerrainGeneratorNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "biome_mask": ("IMAGE",),
                "output_height": ("INT", {"default": 64, "min": 8, "max": 256}),
                "output_width": ("INT", {"default": 64, "min": 8, "max": 256}),
                "output_depth": ("INT", {"default": 32, "min": 8, "max": 128}),
                "tile_size": ("INT", {"default": 3, "min": 2, "max": 8}),
                "chunk_count": ("INT", {"default": 1, "min": 1, "max": 8}),
                "seed": ("INT", {"default": 42}),
                "export_path": ("STRING", {"default": "wfc_terrain.vox"}),
                "export_glb": ("BOOLEAN", {"default": True}),
                "glb_base_path": ("STRING", {"default": "wfc_terrain.glb"}),
                "cube_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0}),
            }
        }
    RETURN_TYPES = ("NP_ARRAY", "STRING", "STRING")
    RETURN_NAMES = ("terrain_chunks", "vox_paths", "glb_paths")
    FUNCTION = "run"
    CATEGORY = "Procedural"

    def run(self, reference_image, biome_mask, output_height, output_width, output_depth, tile_size, chunk_count, seed, export_path, export_glb, glb_base_path, cube_size):
        print("[NODE] WFCTerrain3DNode.run called")
        ref_img = Image.fromarray((reference_image[0].cpu().numpy()*255).astype(np.uint8))
        if biome_mask is not None and hasattr(biome_mask, "__getitem__"):
            biome_img = Image.fromarray((biome_mask[0].cpu().numpy()*255).astype(np.uint8))
        else:
            biome_img = None
        palette = build_palette()
        ref_grid = img_to_biome_array(ref_img, biome_img, output_height, output_width, output_depth)
        print(f"[NODE] Reference grid shape: {ref_grid.shape}")
        shape = (output_height, output_width, output_depth)
        terrain_chunks = [None]*chunk_count
        vox_paths = [None]*chunk_count
        glb_paths = [None]*chunk_count
        edge_list = [None]*chunk_count
        threads = []
        for idx in range(chunk_count):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            vox_path = export_path.replace(".vox", f"_chunk{idx}_{ts}.vox")
            glb_path = ""
            if export_glb:
                glb_path = glb_base_path.replace(".glb", f"_chunk{idx}_{ts}.glb")
            vox_paths[idx] = vox_path
            glb_paths[idx] = glb_path
            edge_constraints = edge_list[idx-1] if idx > 0 else None
            print(f"[NODE] Launching thread for chunk idx={idx}")
            t = threading.Thread(target=generate_chunk_threaded, args=(
                ref_grid, shape, tile_size, seed+idx, edge_constraints, vox_path, glb_path, terrain_chunks, idx, edge_list, palette, cube_size))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print("[NODE] All threads completed")
        return (terrain_chunks, vox_paths, glb_paths)