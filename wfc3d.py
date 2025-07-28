import numpy as np
from PIL import Image
import os
import struct

# Optional dependencies for enhanced features
try:
    from perlin_noise import PerlinNoise
except ImportError:
    PerlinNoise = None

try:
    import midvoxio
except ImportError:
    midvoxio = None

try:
    import trimesh
    from skimage import measure
except ImportError:
    trimesh = None

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

try:
    import fastwfc
except ImportError:
    fastwfc = None
    print("[WARN] fastwfc is not installed, WFC will fallback to slow Python implementation.")

def get_biome_color(biome, height, BIOME_RGB):
    base = np.array(BIOME_RGB.get(biome, [120, 120, 120]), dtype=np.float32)
    return tuple(base.astype(np.uint8))

def build_palette(colors, max_colors=255):
    """
    Given a list or array of (R, G, B) tuples, quantize to max_colors if needed.
    Returns (palette, color_to_palette_idx).
      - palette: array of palette colors, shape (<=max_colors, 3)
      - color_to_palette_idx: dict mapping color tuple to 1-based palette index
    """
    unique_colors = np.array(list({tuple(int(x) for x in c) for c in colors}))
    if len(unique_colors) > max_colors and KMeans is not None:
        kmeans = KMeans(n_clusters=max_colors, random_state=42, n_init='auto')
        kmeans.fit(unique_colors)
        palette = np.round(kmeans.cluster_centers_).astype(np.uint8)
        color_to_palette_idx = {}
        for color in unique_colors:
            dists = np.sum((palette - color) ** 2, axis=1)
            idx = int(np.argmin(dists))
            color_to_palette_idx[tuple(color)] = idx + 1
    else:
        palette = unique_colors
        color_to_palette_idx = {tuple(color): i + 1 for i, color in enumerate(palette)}
    return palette, color_to_palette_idx

def build_palette_array(BIOME_PALETTE, BIOME_IDX):
    arr = np.zeros((256, 4), dtype=np.uint8)
    items = list(BIOME_PALETTE.items())[:255]
    for i, (rgb, name) in enumerate(items):
        idx = i + 1
        arr[idx, 0:3] = rgb
        arr[idx, 3] = 255
    arr[0] = [0,0,0,0]
    return arr

def quantize_color(color, step=16):
    return tuple(int((c // step) * step) for c in color)

def voxels_to_rgb_array(voxel_grid, palette):
    sx, sy, sz = voxel_grid.shape
    rgb = np.zeros((sx, sy, sz, 3), dtype=np.uint8)
    for idx in np.unique(voxel_grid):
        if idx == 0:
            continue
        mask = (voxel_grid == idx)
        rgb[mask] = palette[idx][:3]
    return rgb

def perlin_noise_2d(shape, scale=0.035, octaves=4, seed=0):
    h, w = shape
    arr = np.zeros((h, w), dtype=np.float32)
    if PerlinNoise is None:
        arr[...] = np.random.default_rng(seed).random((h,w))
        return arr
    noise2d = PerlinNoise(octaves=octaves, seed=seed)
    for y in range(h):
        for x in range(w):
            arr[y,x] = noise2d([x*scale, y*scale])
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def perlin_noise_3d(shape, scale=0.06, octaves=4, seed=0):
    h, w, d = shape
    arr = np.zeros((h, w, d), dtype=np.float32)
    if PerlinNoise is None:
        arr[...] = np.random.default_rng(seed).random((h,w,d))
        return arr
    def fill_slice(z):
        noise3d = PerlinNoise(octaves=octaves, seed=seed + z)
        for y in range(h):
            for x in range(w):
                arr[y,x,z] = noise3d([x*scale, y*scale, z*scale])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        list(executor.map(fill_slice, range(d)))
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def to_pil_image(im):
    import torch
    if isinstance(im, torch.Tensor):
        arr = im.detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        return Image.fromarray(arr)
    elif isinstance(im, np.ndarray):
        if im.ndim == 4 and im.shape[0] == 1:
            im = im[0]
        if im.shape[-1] == 1:
            im = np.repeat(im, 3, axis=-1)
        arr = (im * 255).astype(np.uint8) if im.max() <= 1.0 else im.astype(np.uint8)
        return Image.fromarray(arr)
    elif isinstance(im, Image.Image):
        return im
    else:
        raise ValueError("Unsupported image input type: " + str(type(im)))

def quantize_and_downsample(arr, n_colors=8, scale=1.0):
    if scale < 1.0:
        arr_img = Image.fromarray(arr.astype(np.uint8))
        small = arr_img.resize(
            (int(arr.shape[1]*scale), int(arr.shape[0]*scale)),
            Image.BILINEAR
        )
        arr = np.array(small)
    arr = (arr / 255.0 * n_colors).astype(np.uint8)
    arr = (arr.astype(np.float32) / n_colors * 255).astype(np.uint8)
    return arr

def crop_to_multiple(arr, n):
    h, w = arr.shape[:2]
    if h < n or w < n:
        return None
    new_h = (h // n) * n
    new_w = (w // n) * n
    return arr[:new_h, :new_w]

def remap_image_to_palette(arr, palette):
    palette = np.array(list(palette))
    arr_flat = arr.reshape(-1, 3)
    dists = np.linalg.norm(arr_flat[:, None, :] - palette[None, :, :], axis=2)
    idxs = dists.argmin(axis=1)
    arr_mapped = palette[idxs].reshape(arr.shape)
    return arr_mapped

def image_to_surface_biome(
    img,
    map_shape,
    seed=0,
    depthmap=None,
    depth_sensitivity=1.0,
    quantize=True,
    scale=1.0,
    tile_size=None,
    n_palette_colors=None,
):
    from PIL import Image as PILImage

    def build_dynamic_biome_mappings(palette):
        BIOME_PALETTE = {tuple(palette[i]): f"biome_{i}" for i in range(len(palette))}
        BIOME_IDX = {f"biome_{i}": i+1 for i in range(len(palette))}
        IDX_BIOME = {i+1: f"biome_{i}" for i in range(len(palette))}
        BIOME_RGB = {f"biome_{i}": palette[i] for i in range(len(palette))}
        BIOME_FILL = {f"biome_{i}": f"biome_{i}" for i in range(len(palette))}
        return BIOME_PALETTE, BIOME_IDX, IDX_BIOME, BIOME_RGB, BIOME_FILL

    def color_to_biome_idx(arr, lut):
        arr_flat = arr.reshape(-1, 3)
        biome_idxs = np.array([lut[tuple(int(x) for x in c)] for c in arr_flat])
        return biome_idxs.reshape(arr.shape[:2])

    arr = np.array(to_pil_image(img).resize((map_shape[1], map_shape[0]), PILImage.BILINEAR).convert("RGB"))
    print("[DEBUG] Unique colors in img:", np.unique(arr.reshape(-1, 3), axis=0))
    palette, lut = build_palette(arr.reshape(-1, 3), max_colors=255)
    BIOME_PALETTE, BIOME_IDX, IDX_BIOME, BIOME_RGB, BIOME_FILL = build_dynamic_biome_mappings(palette)
    biome_idx = color_to_biome_idx(arr, lut)

    from scipy.ndimage import generic_filter
    def mode_filter(x):
        counts = np.bincount(x.astype(np.int32))
        return np.argmax(counts)
    biome_idx = generic_filter(biome_idx, mode_filter, size=3)

    if depthmap is not None:
        dimg = to_pil_image(depthmap).resize((arr.shape[1], arr.shape[0]), PILImage.BILINEAR)
        darr = np.array(dimg.convert("L"))
        heightmap = darr / 255.0 * depth_sensitivity
        heightmap = np.clip(heightmap, 0, 1)
    else:
        heightmap = arr.mean(axis=2) / 255.0
    noise = perlin_noise_2d((arr.shape[0], arr.shape[1]), scale=0.035, octaves=4, seed=seed)
    heightmap = (heightmap * 0.85 + noise * 0.15)

    if quantize:
        heightmap_q = quantize_and_downsample((heightmap*255).astype(np.uint8), n_colors=8, scale=scale) / 255.0
        heightmap_q = np.array(PILImage.fromarray((heightmap_q*255).astype(np.uint8)).resize((arr.shape[1], arr.shape[0]), PILImage.BILINEAR)) / 255.0
        if tile_size is not None:
            heightmap_q = crop_to_multiple(heightmap_q, tile_size)
            biome_idx_q = crop_to_multiple(biome_idx, tile_size)
            if heightmap_q is None or biome_idx_q is None:
                return None, None, None
            heightmap = heightmap_q
            biome_idx = biome_idx_q
        else:
            heightmap = heightmap_q

    biome_idx = np.clip(biome_idx, 1, len(palette))

    if heightmap.shape != map_shape:
        from skimage.transform import resize
        heightmap = resize(heightmap, map_shape, order=1, preserve_range=True, anti_aliasing=True).astype(heightmap.dtype)
    if biome_idx.shape != map_shape:
        from skimage.transform import resize
        biome_idx = resize(biome_idx, map_shape, order=0, preserve_range=True, anti_aliasing=False).astype(biome_idx.dtype)

    print("[DEBUG] After image_to_surface_biome:", heightmap.shape, biome_idx.shape)
    print("[DEBUG] biome_map unique values:", np.unique(biome_idx))

    return heightmap, biome_idx, (BIOME_PALETTE, BIOME_IDX, IDX_BIOME, BIOME_RGB, BIOME_FILL)

def fill_internal_holes(vox):
    wx, wy, wz = vox.shape
    filled = vox.copy()
    for x in range(1, wx-1):
        for y in range(1, wy-1):
            for z in range(1, wz-1):
                if vox[x,y,z] == 0:
                    neighbors = vox[x-1:x+2, y-1:y+2, z-1:z+2]
                    if np.all(neighbors != 0):
                        filled[x,y,z] = np.max(neighbors)
    return filled

def build_voxel_chunk(surf, biome, depth, BIOME_RGB, IDX_BIOME, BIOME_FILL, BIOME_IDX, cave_freq=0.06, cave_threshold=0.65, seed=0):
    w, h = surf.shape
    vox = np.zeros((w, h, depth), dtype=np.uint8)
    color_grid = np.zeros((w, h, depth, 3), dtype=np.uint8)
    if 'PerlinNoise' in globals() and PerlinNoise:
        cave_noise = perlin_noise_3d((w, h, depth), scale=cave_freq, octaves=4, seed=seed)
    else:
        cave_noise = np.random.default_rng(seed).random((w, h, depth))
    for x in range(w):
        for y in range(h):
            z_surf = int(surf[x, y] * (depth - 3)) + 2
            biome_idx = int(biome[x, y])
            biome_name = IDX_BIOME.get(biome_idx, f"biome_{biome_idx-1}")
            for z in range(depth):
                rel_height = float(z) / float(depth - 1)
                if z == z_surf:
                    vox[x, y, z] = biome_idx if biome_idx > 0 else 1
                    color_grid[x, y, z] = get_biome_color(biome_name, rel_height, BIOME_RGB)
                elif z < z_surf:
                    fill_block = BIOME_FILL.get(biome_name, biome_name)
                    fill_idx = BIOME_IDX.get(fill_block, 1)
                    fill_name = IDX_BIOME.get(fill_idx, fill_block)
                    vox[x, y, z] = fill_idx
                    color_grid[x, y, z] = get_biome_color(fill_name, rel_height, BIOME_RGB)
                else:
                    vox[x, y, z] = 0
                    color_grid[x, y, z] = [0, 0, 0]
    return vox, color_grid

def save_vox(voxel_grid, color_grid, out_path):
    print("Unique colors in color_grid before quantization:")
    print(np.unique(color_grid.reshape(-1, 3), axis=0))
    voxels = []
    colors = []
    size_x, size_y, size_z = voxel_grid.shape
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                if voxel_grid[x, y, z] == 0:
                    continue
                color = tuple(int(c) for c in color_grid[x, y, z])
                color = quantize_color(color, step=16)
                voxels.append((x, y, z))
                colors.append(color)
    if not voxels:
        print("[WFC3D] Warning: No voxels to save.")
        return

    print("Unique quantized colors to be used in palette:")
    # Use unified palette logic
    palette, color_to_palette_idx = build_palette(colors, max_colors=255)
    while len(palette) < 256:
        palette = np.vstack([palette, [0,0,0]])
    palette = palette[:256]

    print(palette)
    print(f"[WFC3D] Saving {len(voxels)} voxels to: {out_path}")
    try:
        with open(out_path, "wb") as f:
            f.write(b'VOX ')
            f.write(struct.pack('<I', 150))
            f.write(b'MAIN')
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', 0))
            main_start = f.tell()
            f.write(b'SIZE')
            f.write(struct.pack('<I', 12))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<III', int(size_x), int(size_y), int(size_z)))
            f.write(b'XYZI')
            xyzi_content_size = 4 + len(voxels)*4
            f.write(struct.pack('<I', xyzi_content_size))
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', len(voxels)))
            for i, (x, y, z) in enumerate(voxels):
                color = colors[i]
                color_index = color_to_palette_idx[color]
                x = max(0, min(int(x), size_x - 1))
                y = max(0, min(int(y), size_y - 1))
                z = max(0, min(int(z), size_z - 1))
                color_index = max(1, min(255, int(color_index)))
                f.write(struct.pack('<BBBB', x, y, z, color_index))
            f.write(b'RGBA')
            f.write(struct.pack('<I', 1024))
            f.write(struct.pack('<I', 0))
            for r, g, b in palette:
                f.write(struct.pack('<BBBB', int(r), int(g), int(b), 255))
            end_pos = f.tell()
            children_size = end_pos - main_start
            f.seek(main_start - 8)
            f.write(struct.pack('<I', 0))
            f.write(struct.pack('<I', children_size))
        print(f"[WFC3D] Successfully saved {len(voxels)} voxels to {out_path}")
        print(f"[WFC3D] File size: {os.path.getsize(out_path)} bytes")
    except Exception as e:
        print(f"[WFC3D] Error saving file: {e}")
        raise

def get_color_index(color, palette):
    color = np.array(color)
    palette_array = np.array(palette)
    distances = np.sum((palette_array - color) ** 2, axis=1)
    index = np.argmin(distances) + 1
    return max(1, min(255, int(index)))

def combine_chunks(chunks, n_chunks, chunk_size):
    nx, ny = n_chunks
    sx, sy, sz = chunk_size
    big_vox = np.zeros((nx*sx, ny*sy, sz), dtype=np.uint8)
    big_color = np.zeros((nx*sx, ny*sy, sz, 3), dtype=np.uint8)
    for (cx, cy), (vox, color_grid) in chunks:
        x0, x1 = cx*sx, (cx+1)*sx
        y0, y1 = cy*sy, (cy+1)*sy
        big_vox[x0:x1, y0:y1, :] = vox
        big_color[x0:x1, y0:y1, :, :] = color_grid
    return big_vox, big_color

def contiguous_reindex(arr):
    unique = np.unique(arr)
    remap = {v: i for i, v in enumerate(unique)}
    arr2 = np.vectorize(remap.get)(arr)
    return arr2, remap, unique

# --- Main ComfyUI Node ---
class WFCTerrain3DNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depthmap": ("IMAGE",),
                "chunk_size": ("INT", {"default": 32, "min": 4, "max": 256}),
                "n_chunks_x": ("INT", {"default": 2, "min": 1, "max": 8}),
                "n_chunks_y": ("INT", {"default": 2, "min": 1, "max": 8}),
                "tile_size": ("INT", {"default": 3, "min": 2, "max": 8}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 999999}),
                "cave_freq": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0}),
                "cave_threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0}),
                "depth_sensitivity": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0}),
                "smooth_passes": ("INT", {"default": 2, "min": 0, "max": 5}),
                "wfc_enabled": ("BOOLEAN", {"default": True}),
                "wfc_periodic_output": ("BOOLEAN", {"default": True}),
                "wfc_periodic_input": ("BOOLEAN", {"default": True}),
                "wfc_symmetry": ("INT", {"default": 8, "min": 1, "max": 8}),
                "wfc_n": ("INT", {"default": 3, "min": 2, "max": 8}),
                "wfc_retries": ("INT", {"default": 3, "min": 1, "max": 10}),
                "wfc_retry_seed_offset": ("INT", {"default": 1000, "min": 1, "max": 10000}),
                "n_palette_colors": ("INT", {"default": 16, "min": 2, "max": 256}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": "output_chunks"}),
                "export_vox": ("BOOLEAN", {"default": False}),
                "export_glb": ("BOOLEAN", {"default": False}),
                "export_combined_vox": ("BOOLEAN", {"default": False}),
            }
        }
    RETURN_TYPES = ("NP_ARRAY", "STRING", "STRING")
    RETURN_NAMES = ("voxels", "vox_path", "glb_path")
    FUNCTION = "execute"
    CATEGORY = "Geometry/Procedural"

    def execute(
        self,
        image,
        depthmap,
        chunk_size=32,
        n_chunks_x=2,
        n_chunks_y=2,
        tile_size=3,
        seed=42,
        cave_freq=0.06,
        cave_threshold=0.65,
        output_dir="output_chunks",
        export_vox=False,
        export_glb=False,
        depth_sensitivity=1.0,
        export_combined_vox=False,
        smooth_passes=2,
        wfc_enabled=True,
        wfc_periodic_output=True,
        wfc_periodic_input=True,
        wfc_symmetry=8,
        wfc_n=3,
        wfc_retries=3,
        wfc_retry_seed_offset=1000,
        n_palette_colors=16,
    ):
        import numpy as np
        import os

        nx = n_chunks_x
        ny = n_chunks_y

        img = to_pil_image(image)
        dep = to_pil_image(depthmap)

        min_size = chunk_size * max(nx, ny)
        if img.width < min_size or img.height < min_size:
            print(f"[ERROR] Input image too small, resizing to {min_size}x{min_size}")
            img = img.resize((min_size, min_size), Image.BILINEAR)
        if dep.width < min_size or dep.height < min_size:
            print(f"[ERROR] Depthmap too small, resizing to {min_size}x{min_size}")
            dep = dep.resize((min_size, min_size), Image.BILINEAR)

        map_shape = (chunk_size * nx, chunk_size * ny)
        arr = np.array(img.convert("RGB"))
        print("[DEBUG] Unique colors in img:", np.unique(arr.reshape(-1, 3), axis=0))

        heightmap, biome_map, mappings = image_to_surface_biome(
            img, map_shape, seed=seed, depthmap=dep, depth_sensitivity=depth_sensitivity,
            quantize=True, scale=1.0, tile_size=tile_size, n_palette_colors=n_palette_colors
        )
        BIOME_PALETTE, BIOME_IDX, IDX_BIOME, BIOME_RGB, BIOME_FILL = mappings

        print("[DEBUG] After image_to_surface_biome:", heightmap.shape, biome_map.shape)
        print("[DEBUG] heightmap min/max:", np.min(heightmap), np.max(heightmap))
        print("[DEBUG] biome_map min/max:", np.min(biome_map), np.max(biome_map))
        print("[DEBUG] biome_map unique values:", np.unique(biome_map))

        expected_shape = (chunk_size * nx, chunk_size * ny)
        if heightmap.shape != expected_shape:
            print(f"[PATCH] Resizing heightmap from {heightmap.shape} to {expected_shape}")
            from skimage.transform import resize
            heightmap = resize(heightmap, expected_shape, order=1, preserve_range=True, anti_aliasing=True).astype(heightmap.dtype)
        if biome_map.shape != expected_shape:
            print(f"[PATCH] Resizing biome_map from {biome_map.shape} to {expected_shape}")
            from skimage.transform import resize
            biome_map = resize(biome_map, expected_shape, order=0, preserve_range=True, anti_aliasing=False).astype(biome_map.dtype)
        print("[DEBUG] After resizing/cropping:", heightmap.shape, biome_map.shape)

        from scipy.ndimage import generic_filter
        def mode_filter(x):
            counts = np.bincount(x.astype(np.int32))
            return np.argmax(counts)
        for _ in range(smooth_passes):
            biome_map = generic_filter(biome_map, mode_filter, size=3)

        if heightmap is None or biome_map is None:
            raise ValueError("Map too small for tile size!")
        palette_arr = build_palette_array(BIOME_PALETTE, BIOME_IDX)
        surface_chunks, biome_chunks = [], []

        if wfc_enabled and 'fastwfc' in globals() and fastwfc is not None:
            print("[Terrain3D] Using chunked WFC for large map (fastwfc).")
            for cx in range(nx):
                for cy in range(ny):
                    x0 = cx * chunk_size
                    x1 = x0 + chunk_size
                    y0 = cy * chunk_size
                    y1 = y0 + chunk_size
                    surf_patch = heightmap[x0:x1, y0:y1]
                    biome_patch = biome_map[x0:x1, y0:y1]
                    print(f"[DEBUG] CHUNK ({cx},{cy}) surf_patch shape: {surf_patch.shape} biome_patch shape: {biome_patch.shape}")
                    contig_biome_map, remap, unique_biomes = contiguous_reindex(biome_patch)
                    print("[DEBUG] contig_biome_map shape:", contig_biome_map.shape, "remap:", remap, "unique_biomes:", unique_biomes)
                    sample = np.ascontiguousarray(contig_biome_map.astype(np.uint8))
                    print("sample.shape:", sample.shape)
                    print("sample.dtype:", sample.dtype)
                    print("sample min/max:", sample.min(), sample.max())
                    print("sample unique:", np.unique(sample))

                    out_biome = None
                    for retry in range(wfc_retries):
                        try:
                            print(f"[DEBUG] WFC CALL: chunk ({cx},{cy}), retry {retry}")
                            res = fastwfc.apply_wfc(
                                width=sample.shape[0],
                                height=sample.shape[1],
                                input_img=sample,
                                periodic_output=wfc_periodic_output,
                                N=wfc_n,
                                periodic_input=wfc_periodic_input,
                                nb_samples=1,
                                symmetry=wfc_symmetry,
                                seed=seed + cx * 31 + cy * 101 + retry * wfc_retry_seed_offset
                            )
                            print("fastwfc result type:", type(res))
                            if isinstance(res, (tuple, list)):
                                print("fastwfc result[0] shape:", res[0].shape)
                            else:
                                print("fastwfc result shape:", res.shape)
                            if isinstance(res, (tuple, list)) and len(res) > 0:
                                wfc_result = res[0]
                            else:
                                wfc_result = res
                            print("wfc_result shape:", wfc_result.shape, "min/max:", wfc_result.min(), wfc_result.max(), "unique:", np.unique(wfc_result))
                            out_biome = unique_biomes[wfc_result]
                            print("out_biome shape:", out_biome.shape, "min/max:", out_biome.min(), out_biome.max(), "unique:", np.unique(out_biome))
                            if np.any(out_biome < 1):
                                print(f"[fastwfc] Hole detected in chunk ({cx},{cy}), retrying...")
                                continue
                            break
                        except Exception as e:
                            import traceback
                            print(f"[WFCTerrain3DNode][fastwfc] WFC failed for chunk ({cx},{cy}) with error: {e}, using direct patch.")
                            traceback.print_exc()
                            out_biome = None
                    if out_biome is None:
                        biome = biome_patch
                        surface = surf_patch
                        print("[Fallback] Indices:", np.unique(biome_patch))
                    else:
                        biome = out_biome
                        surface = surf_patch
                        print("[WFC] Indices:", np.unique(out_biome))
                    surface_chunks.append(surface)
                    biome_chunks.append(biome)
        else:
            print("[Terrain3D] Using full-map or fallback Python WFC (slow!).")
            for cx in range(nx):
                for cy in range(ny):
                    x0 = cx * chunk_size
                    x1 = x0 + chunk_size
                    y0 = cy * chunk_size
                    y1 = y0 + chunk_size
                    surf_patch = heightmap[x0:x1, y0:y1]
                    biome_patch = biome_map[x0:x1, y0:y1]
                    if surf_patch.shape != (chunk_size, chunk_size) or biome_patch.shape != (chunk_size, chunk_size):
                        print(f"[ERROR] Fallback chunk ({cx},{cy}) has wrong shape: surf_patch {surf_patch.shape}, biome_patch {biome_patch.shape}, expected ({chunk_size}, {chunk_size})")
                        continue
                    surface_chunks.append(surf_patch)
                    biome_chunks.append(biome_patch)

        chunk_size_z = chunk_size // 2
        chunks = []
        vox_paths, glb_paths = [], []
        for idx, (surface, biome) in enumerate(zip(surface_chunks, biome_chunks)):
            cx = idx // ny
            cy = idx % ny
            if surface.shape != (chunk_size, chunk_size) or biome.shape != (chunk_size, chunk_size):
                print(f"[ERROR] Output chunk ({cx},{cy}) has wrong shape: surf {surface.shape}, bio {biome.shape}, expected ({chunk_size}, {chunk_size})")
                continue
            vox, color_grid = build_voxel_chunk(
                surface, biome, chunk_size_z,
                BIOME_RGB, IDX_BIOME, BIOME_FILL, BIOME_IDX,
                cave_freq=cave_freq, cave_threshold=cave_threshold,
                seed=seed + cx * 31 + cy * 101
            )
            if cx == 0 and cy == 0:
                print(f"[DEBUG] Sample color_grid values: {color_grid[0,0,:5]}")
                print(f"[DEBUG] Sample biome values: {biome[0,:5]}")
            chunks.append(((cx, cy), (vox, color_grid)))
            if export_vox:
                vox_path = os.path.join(output_dir, f"chunk_{cx}_{cy}.vox")
                save_vox(vox, color_grid, vox_path)
                vox_paths.append(vox_path)
        if export_combined_vox:
            big_vox, big_color = combine_chunks(chunks, (nx, ny), (chunk_size, chunk_size, chunk_size_z))
            save_vox(big_vox, big_color, os.path.join(output_dir, "big_map.vox"))
        rgb_voxels = voxels_to_rgb_array(chunks[0][1][0], palette_arr)
        vox_path = vox_paths[0] if vox_paths else ""
        glb_path = glb_paths[0] if glb_paths else ""
        import torch
        if not isinstance(rgb_voxels, torch.Tensor):
            rgb_voxels = torch.from_numpy(rgb_voxels).float() / 255.0
            assert rgb_voxels.ndim == 4 and rgb_voxels.shape[-1] == 3, "rgb_voxels must be [X, Y, Z, 3] or similar"

        print("\n[Summary Table]")
        print("Symptom\t\t\t\tRoot Cause\t\t\t\t\t\t\tFix")
        print("All biomes zero\t\t\tInput image not matching BIOME_PALETTE or grayscale only\tUse a proper color image; print unique colors")
        print("WFC fails, shape error\t\tWFC expects 4D or color, not integer labels\t\tPass a color array [H,W,3] or follow fastwfc docs")
        print("Only 1 chunk\t\t\tMap cropped to 64x64 for chunking\t\t\t\tResize inputs before processing")
        print("Only 1 chunk in .vox\t\tOnly one non-empty chunk generated\t\t\t\tFix map/chunk size logic")
        
        return (rgb_voxels, vox_path, glb_path)