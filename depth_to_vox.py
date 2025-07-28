import numpy as np
from PIL import Image
import struct
import os
from sklearn.cluster import KMeans
import datetime

VOXEL_TEMPLATES = [
    # 30+ common game asset categories, each with a fitting preset for size and fusion style
    {
        "category": "Character",
        "description": "Main playable or NPC character.",
        "default_max_depth": 72,
        "surface_mode": "solid_columns",
        "resolution": 96,
        "fusion_mode": "category_adaptive",
        "soft_sigma": 2.5
    },
    {
        "category": "Animal",
        "description": "Animals, wildlife, monsters.",
        "default_max_depth": 64,
        "surface_mode": "solid_columns",
        "resolution": 80,
        "fusion_mode": "soft_blend",
        "soft_sigma": 2.0
    },
    {
        "category": "Enemy",
        "description": "Hostile character or creature.",
        "default_max_depth": 68,
        "surface_mode": "solid_columns",
        "resolution": 88,
        "fusion_mode": "category_adaptive",
        "soft_sigma": 2.5
    },
    {
        "category": "Boss",
        "description": "Large enemy or boss.",
        "default_max_depth": 128,
        "surface_mode": "solid_columns",
        "resolution": 160,
        "fusion_mode": "category_adaptive",
        "soft_sigma": 3.5
    },
    {
        "category": "Weapon",
        "description": "Sword, gun, bow, etc.",
        "default_max_depth": 36,
        "surface_mode": "thick_surface",
        "resolution": 48,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Shield",
        "description": "Protective item.",
        "default_max_depth": 32,
        "surface_mode": "thick_surface",
        "resolution": 44,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Armor",
        "description": "Wearable protection.",
        "default_max_depth": 40,
        "surface_mode": "solid_columns",
        "resolution": 56,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.5
    },
    {
        "category": "Hat",
        "description": "Headwear, helmets.",
        "default_max_depth": 28,
        "surface_mode": "thick_surface",
        "resolution": 32,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.2
    },
    {
        "category": "Potion",
        "description": "Consumable item, bottle.",
        "default_max_depth": 24,
        "surface_mode": "surface_only",
        "resolution": 24,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.0
    },
    {
        "category": "Food",
        "description": "Edible item.",
        "default_max_depth": 20,
        "surface_mode": "surface_only",
        "resolution": 28,
        "fusion_mode": "soft_blend",
        "soft_sigma": 0.8
    },
    {
        "category": "Tool",
        "description": "Pickaxe, axe, hammer, etc.",
        "default_max_depth": 36,
        "surface_mode": "thick_surface",
        "resolution": 40,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Tree",
        "description": "Tree, bush, foliage.",
        "default_max_depth": 56,
        "surface_mode": "solid_columns",
        "resolution": 80,
        "fusion_mode": "soft_blend",
        "soft_sigma": 2.0
    },
    {
        "category": "Rock",
        "description": "Boulder, stone, pebble.",
        "default_max_depth": 32,
        "surface_mode": "solid_columns",
        "resolution": 32,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.5
    },
    {
        "category": "Bush",
        "description": "Small foliage.",
        "default_max_depth": 28,
        "surface_mode": "solid_columns",
        "resolution": 36,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.3
    },
    {
        "category": "Grass",
        "description": "Ground cover.",
        "default_max_depth": 18,
        "surface_mode": "surface_only",
        "resolution": 32,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.1
    },
    {
        "category": "Flower",
        "description": "Floral decoration.",
        "default_max_depth": 18,
        "surface_mode": "surface_only",
        "resolution": 20,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.2
    },
    {
        "category": "Building",
        "description": "Generic building.",
        "default_max_depth": 128,
        "surface_mode": "solid_columns",
        "resolution": 160,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Small House",
        "description": "Small residential structure.",
        "default_max_depth": 72,
        "surface_mode": "solid_columns",
        "resolution": 96,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Medium House",
        "description": "Medium-sized home.",
        "default_max_depth": 112,
        "surface_mode": "solid_columns",
        "resolution": 120,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Large House",
        "description": "Large mansion or villa.",
        "default_max_depth": 160,
        "surface_mode": "solid_columns",
        "resolution": 184,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Church",
        "description": "Church or cathedral.",
        "default_max_depth": 180,
        "surface_mode": "solid_columns",
        "resolution": 200,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Castle",
        "description": "Large fortress, castle.",
        "default_max_depth": 220,
        "surface_mode": "solid_columns",
        "resolution": 256,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Tower",
        "description": "Watchtower, spire.",
        "default_max_depth": 160,
        "surface_mode": "solid_columns",
        "resolution": 120,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Door",
        "description": "Entrances, doors.",
        "default_max_depth": 32,
        "surface_mode": "thick_surface",
        "resolution": 40,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Window",
        "description": "Windows, glass.",
        "default_max_depth": 24,
        "surface_mode": "surface_only",
        "resolution": 32,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Bridge",
        "description": "Structure for crossing gaps.",
        "default_max_depth": 96,
        "surface_mode": "solid_columns",
        "resolution": 140,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Vehicle",
        "description": "Car, truck, ship, plane.",
        "default_max_depth": 120,
        "surface_mode": "solid_columns",
        "resolution": 128,
        "fusion_mode": "soft_blend",
        "soft_sigma": 2.0
    },
    {
        "category": "Road",
        "description": "Path, road, street.",
        "default_max_depth": 24,
        "surface_mode": "surface_only",
        "resolution": 128,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Terrain",
        "description": "Ground, hills, landscape.",
        "default_max_depth": 128,
        "surface_mode": "solid_columns",
        "resolution": 256,
        "fusion_mode": "soft_blend",
        "soft_sigma": 3.2
    },
    {
        "category": "Water",
        "description": "River, lake, ocean.",
        "default_max_depth": 64,
        "surface_mode": "surface_only",
        "resolution": 128,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.8
    },
    {
        "category": "Cloud",
        "description": "Sky cloud, fog.",
        "default_max_depth": 32,
        "surface_mode": "surface_only",
        "resolution": 60,
        "fusion_mode": "soft_blend",
        "soft_sigma": 2.3
    },
    {
        "category": "Fire",
        "description": "Flames, fire, torch.",
        "default_max_depth": 24,
        "surface_mode": "surface_only",
        "resolution": 32,
        "fusion_mode": "soft_blend",
        "soft_sigma": 2.0
    },
    {
        "category": "Furniture",
        "description": "Chair, table, shelf.",
        "default_max_depth": 40,
        "surface_mode": "solid_columns",
        "resolution": 64,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Chest",
        "description": "Storage container.",
        "default_max_depth": 36,
        "surface_mode": "solid_columns",
        "resolution": 48,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Barrel",
        "description": "Barrel, keg.",
        "default_max_depth": 30,
        "surface_mode": "solid_columns",
        "resolution": 36,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.2
    },
    {
        "category": "Crate",
        "description": "Wooden box.",
        "default_max_depth": 32,
        "surface_mode": "solid_columns",
        "resolution": 40,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Sign",
        "description": "Sign, billboard.",
        "default_max_depth": 16,
        "surface_mode": "surface_only",
        "resolution": 32,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Lamp",
        "description": "Lamp, light source.",
        "default_max_depth": 22,
        "surface_mode": "surface_only",
        "resolution": 28,
        "fusion_mode": "soft_blend",
        "soft_sigma": 1.1
    },
    {
        "category": "Statue",
        "description": "Decorative statue.",
        "default_max_depth": 80,
        "surface_mode": "solid_columns",
        "resolution": 96,
        "fusion_mode": "category_adaptive",
        "soft_sigma": 2.4
    },
    {
        "category": "Gate",
        "description": "Large entrance.",
        "default_max_depth": 56,
        "surface_mode": "solid_columns",
        "resolution": 64,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Stairs",
        "description": "Steps, staircase.",
        "default_max_depth": 40,
        "surface_mode": "solid_columns",
        "resolution": 48,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Fence",
        "description": "Fence, wall segment.",
        "default_max_depth": 28,
        "surface_mode": "solid_columns",
        "resolution": 40,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Path",
        "description": "Trail, walkway.",
        "default_max_depth": 16,
        "surface_mode": "surface_only",
        "resolution": 64,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    },
    {
        "category": "Clutter",
        "description": "Small props, books, bottles, trash.",
        "default_max_depth": 20,
        "surface_mode": "surface_only",
        "resolution": 24,
        "fusion_mode": "normal",
        "soft_sigma": 0.0
    }
]

def get_template_by_category(category):
    for t in VOXEL_TEMPLATES:
        if t["category"] == category:
            return t
    return VOXEL_TEMPLATES[0]

def split_into_chunks(img, chunk_size=256):
    h, w = img.shape[:2]
    chunks = []
    for y0 in range(0, h, chunk_size):
        for x0 in range(0, w, chunk_size):
            chunk = img[y0:y0+chunk_size, x0:x0+chunk_size]
            chunks.append((chunk, x0, y0))
    return chunks

class DepthToVox:
    @classmethod
    def INPUT_TYPES(cls):
        categories = [t["category"] for t in VOXEL_TEMPLATES]
        return {
            "required": {
                "rgb_image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "category": (categories, {"default": categories[0]}),
                "custom_tag": ("STRING", {"default": ""}),
                "use_template_settings": ("BOOLEAN", {"default": True}),
                "resolution": ("INT", {"default": 128, "min": 16, "max": 1024}),
                "max_depth": ("INT", {"default": 64, "min": 8, "max": 256}),
                "surface_mode": (["surface_only", "solid_columns", "thick_surface"], {"default": "solid_columns"}),
                "invert_depth": ("BOOLEAN", {"default": True}),
                "chunked_export": ("BOOLEAN", {"default": False}),
                "export_path": ("STRING", {"default": "output.vox"}),
                "fusion_mode": (["normal", "soft_blend", "weighted_blend", "category_adaptive"], {"default": "normal"}),
                "soft_sigma": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0}),
                "weighted_front": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weighted_side": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
                "weighted_top": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("NP_ARRAY", "STRING")
    RETURN_NAMES = ("voxels", "vox_path")
    FUNCTION = "run"
    CATEGORY = "VoxelTools"
    OUTPUT_NODE = True

    def run(self, rgb_image, depth_map, category, custom_tag, use_template_settings, resolution, max_depth,
            surface_mode, invert_depth, chunked_export, export_path,
            fusion_mode, soft_sigma, weighted_front, weighted_side, weighted_top):
        try:
            print("[DepthToVox] Starting voxel generation...")

            template = get_template_by_category(category)
            # Use template settings if requested
            if use_template_settings:
                resolution = template["resolution"]
                max_depth = template["default_max_depth"]
                surface_mode = template["surface_mode"]
                fusion_mode = template.get("fusion_mode", fusion_mode)
                soft_sigma = template.get("soft_sigma", soft_sigma)

            rgb_tensor = rgb_image[0].cpu().numpy()
            depth_tensor = depth_map[0].cpu().numpy()
            rgb, depth = self.prepare_images(rgb_tensor, depth_tensor, resolution)
            print(f"[DepthToVox] RGB shape: {rgb.shape}, Depth shape: {depth.shape}")

            last_path = ""
            all_voxel_arrays = []

            if chunked_export and (rgb.shape[0] > 256 or rgb.shape[1] > 256):
                print("[DepthToVox] Chunked export enabled.")
                chunk_size = 256
                rgb_chunks = split_into_chunks(rgb, chunk_size)
                depth_chunks = split_into_chunks(depth, chunk_size)
                num_chunks = len(rgb_chunks)
                print(f"[DepthToVox] Exporting {num_chunks} chunks.")
                for idx, ((rgb_chunk, x0, y0), (depth_chunk, _, _)) in enumerate(zip(rgb_chunks, depth_chunks)):
                    chunk_path = self.get_safe_output_path(export_path, category, custom_tag, chunk_idx=(x0, y0))
                    voxel_array = self.export_vox(rgb_chunk, depth_chunk, max_depth, chunk_path, surface_mode, invert_depth, fusion_mode,
                                                  soft_sigma, weighted_front, weighted_side, weighted_top, category, return_voxel_array=True)
                    all_voxel_arrays.append(voxel_array)
                    last_path = chunk_path
                    print(f"[DepthToVox] Saved chunk {idx+1}/{num_chunks}: {chunk_path}")
                voxel_array = all_voxel_arrays[0] if all_voxel_arrays else None
            else:
                safe_path = self.get_safe_output_path(export_path, category, custom_tag)
                voxel_array = self.export_vox(rgb, depth, max_depth, safe_path, surface_mode, invert_depth, fusion_mode,
                                              soft_sigma, weighted_front, weighted_side, weighted_top, category, return_voxel_array=True)
                last_path = safe_path

            print("[DepthToVox] Voxel generation complete.")
            return (voxel_array, last_path)
        except Exception as e:
            print(f"[DepthToVox][ERROR] {e}")
            raise

    def prepare_images(self, rgb_tensor, depth_tensor, resolution):
        if rgb_tensor.ndim == 3:
            if rgb_tensor.shape[0] in (1, 3):
                rgb = rgb_tensor.transpose(1, 2, 0)
            elif rgb_tensor.shape[2] in (1, 3):
                rgb = rgb_tensor
            else:
                raise ValueError(f"Unexpected RGB shape: {rgb_tensor.shape}")
        else:
            raise ValueError(f"RGB tensor must have 3 dimensions, got {rgb_tensor.ndim}")

        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        elif rgb.shape[2] != 3:
            raise ValueError(f"RGB image must have 1 or 3 channels, got {rgb.shape[2]}")

        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        if depth_tensor.ndim == 3:
            if depth_tensor.shape[2] == 1:
                depth = depth_tensor[:, :, 0]
            else:
                depth = depth_tensor.mean(axis=2)
        elif depth_tensor.ndim == 2:
            depth = depth_tensor
        else:
            raise ValueError(f"Depth map must be 2D or 3D, got {depth_tensor.ndim}D")

        if depth.max() <= 1.0:
            depth = depth * 255.0
        depth = np.clip(depth, 0, 255).astype(np.uint8)

        rgb_img = Image.fromarray(rgb).resize((resolution, resolution), Image.BILINEAR)
        rgb = np.array(rgb_img)
        depth_img = Image.fromarray(depth).resize((resolution, resolution), Image.BILINEAR)
        depth = np.array(depth_img)
        return rgb, depth

    def export_vox(self, rgb, depth, max_depth, export_path, surface_mode, invert_depth, fusion_mode,
                   soft_sigma, weighted_front, weighted_side, weighted_top, category, return_voxel_array=False):
        h, w = depth.shape
        if invert_depth:
            depth_processed = 255 - depth
        else:
            depth_processed = depth
        depth_arr = np.clip((depth_processed / 255.0 * (max_depth - 1)).astype(int), 0, max_depth - 1)
        background_color_threshold = 0
        background_depth_threshold = 0
        color_mask = np.sum(rgb, axis=2) > background_color_threshold
        depth_mask = depth_arr >= background_depth_threshold
        valid_mask = color_mask & depth_mask
        voxels = []
        colors = []
        voxel_grid = np.zeros((w, h, max_depth, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                if not valid_mask[y, x]:
                    continue
                d = int(depth_arr[y, x])
                color = tuple(int(c) for c in rgb[y, x])
                if surface_mode == "surface_only":
                    if 0 <= x < w and 0 <= y < h and 0 <= d < max_depth:
                        voxels.append((x, y, d))
                        colors.append(color)
                        voxel_grid[x, y, d] = color
                elif surface_mode == "thick_surface":
                    if 0 <= x < w and 0 <= y < h and 0 <= d < max_depth:
                        voxels.append((x, y, d))
                        colors.append(color)
                        voxel_grid[x, y, d] = color
                        if d > 0:
                            voxels.append((x, y, d-1))
                            colors.append(color)
                            voxel_grid[x, y, d-1] = color
                elif surface_mode == "solid_columns":
                    max_col_height = min(d + 1, max_depth)
                    for z in range(max_col_height):
                        if 0 <= x < w and 0 <= y < h and 0 <= z < max_depth:
                            voxels.append((x, y, z))
                            colors.append(color)
                            voxel_grid[x, y, z] = color

        # --- FUSION MODES ---
        if fusion_mode in ["soft_blend", "weighted_blend", "category_adaptive"]:
            if fusion_mode == "soft_blend":
                voxel_grid = self.soft_blend(voxel_grid, sigma=soft_sigma)
            elif fusion_mode == "weighted_blend":
                voxel_grid = self.weighted_blend(voxel_grid, weights=[weighted_front, weighted_side, weighted_top])
            elif fusion_mode == "category_adaptive":
                voxel_grid = self.category_adaptive(voxel_grid, category)

            # Recompute voxels/colors for .vox export
            voxels = []
            colors = []
            w, h, d, _ = voxel_grid.shape
            for x in range(w):
                for y in range(h):
                    for z in range(d):
                        color = tuple(int(c) for c in voxel_grid[x, y, z])
                        if np.any(color):
                            voxels.append((x, y, z))
                            colors.append(color)

        if len(voxels) == 0:
            print("[DepthToVox] No voxels generated, creating test cube.")
            w = h = max_depth = 8
            voxel_grid = np.zeros((w, h, max_depth, 3), dtype=np.uint8)
            for x in range(w):
                for y in range(h):
                    for z in range(max_depth):
                        voxels.append((x, y, z))
                        colors.append((255,255,255))
                        voxel_grid[x, y, z] = (255,255,255)

        palette = self.build_palette(colors)
        self.save_vox(voxels, colors, palette, export_path, w, h, max_depth)
        if return_voxel_array:
            return voxel_grid
        return None

    # --- Fusion/Blending Modes ---
    def soft_blend(self, voxel_grid, sigma=2.0):
        from scipy.ndimage import gaussian_filter
        mask = np.any(voxel_grid > 0, axis=-1).astype(np.float32)
        smooth_mask = gaussian_filter(mask, sigma=sigma)
        softmask = smooth_mask > 0.2
        new_grid = np.zeros_like(voxel_grid)
        for c in range(3):
            color_chan = voxel_grid[..., c]
            blurred_chan = gaussian_filter(color_chan, sigma=sigma)
            new_grid[..., c] = np.where(softmask, blurred_chan, 0)
        new_grid = np.clip(new_grid, 0, 255).astype(np.uint8)
        return new_grid

    def weighted_blend(self, voxel_grid, weights=[1.0,1.0,1.0]):
        # For future multi-view support; for now just scale mask by weight[0]
        weight = weights[0]
        new_grid = (voxel_grid.astype(np.float32) * weight)
        new_grid = np.clip(new_grid, 0, 255).astype(np.uint8)
        return new_grid

    def category_adaptive(self, voxel_grid, category):
        if category in ["Character", "Animal", "Enemy", "Boss", "Tree", "Bush", "Rock", "Vehicle", "Statue", "Barrel"]:
            return self.soft_blend(voxel_grid, sigma=2.5)
        elif category in ["Building", "Small House", "Medium House", "Large House", "Church", "Castle", "Tower", "Door", "Window", "Bridge", "Road", "Terrain", "Furniture", "Chest", "Crate", "Gate", "Stairs", "Fence", "Path", "Clutter", "Sign"]:
            return voxel_grid
        elif category in ["Grass", "Flower", "Lamp", "Fire", "Water", "Cloud"]:
            return self.soft_blend(voxel_grid, sigma=1.2)
        else:
            return voxel_grid

    def get_safe_output_path(self, export_path, category, custom_tag, chunk_idx=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_str = f"_{custom_tag}" if custom_tag else ""
        base_name = f"{category}{custom_str}_{timestamp}"
        if chunk_idx is not None:
            base_name += f"_chunk_{chunk_idx[0]}_{chunk_idx[1]}"
        if not export_path.endswith('.vox'):
            export_path += '.vox'
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            category_dir = os.path.join(output_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            export_path = os.path.join(category_dir, base_name + ".vox")
        except:
            category_dir = os.path.join(".", category)
            os.makedirs(category_dir, exist_ok=True)
            export_path = os.path.join(category_dir, base_name + ".vox")
        print(f"[DepthToVox] Output path: {export_path}")
        return export_path

    def build_palette(self, colors):
        unique_colors = list(set(colors))
        if len(unique_colors) > 255:
            print(f"[DepthToVox] Too many colors ({len(unique_colors)}), reducing to 255 using K-means")
            kmeans = KMeans(n_clusters=255, random_state=42)
            color_array = np.array(unique_colors)
            kmeans.fit(color_array)
            palette = kmeans.cluster_centers_.astype(np.uint8).tolist()
        else:
            palette = unique_colors
        while len(palette) < 256:
            palette.append((0, 0, 0))
        return [tuple(int(c) for c in col) for col in palette[:256]]

    def get_color_index(self, color, palette):
        color = np.array(color)
        palette_array = np.array(palette)
        distances = np.sum((palette_array - color) ** 2, axis=1)
        index = np.argmin(distances) + 1
        return max(1, min(255, int(index)))

    def save_vox(self, voxels, colors, palette, export_path, size_x, size_y, size_z):
        if not voxels:
            print("[DepthToVox] Warning: No voxels to save.")
            return

        print(f"[DepthToVox] Saving {len(voxels)} voxels to: {export_path}")
        try:
            with open(export_path, "wb") as f:
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
                    color_index = self.get_color_index(colors[i], palette)
                    x = max(0, min(int(x), size_x - 1))
                    y = max(0, min(int(y), size_y - 1))
                    z = max(0, min(int(z), size_z - 1))
                    color_index = max(1, min(255, int(color_index)))
                    f.write(struct.pack('<BBBB', x, y, z, color_index))
                f.write(b'RGBA')
                f.write(struct.pack('<I', 1024))
                f.write(struct.pack('<I', 0))
                for r,g,b in palette:
                    f.write(struct.pack('<BBBB', int(r), int(g), int(b), 255))
                end_pos = f.tell()
                children_size = end_pos - main_start
                f.seek(main_start - 8)
                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<I', children_size))
            print(f"[DepthToVox] Successfully saved {len(voxels)} voxels to {export_path}")
            print(f"[DepthToVox] File size: {os.path.getsize(export_path)} bytes")
        except Exception as e:
            print(f"[DepthToVox] Error saving file: {e}")
            raise

def save_minimal_vox(filename):
    with open(filename, "wb") as f:
        f.write(b'VOX ')
        f.write(struct.pack('<I', 150))
        f.write(b'MAIN')
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', 0))
        main_start = f.tell()
        f.write(b'SIZE')
        f.write(struct.pack('<I', 12))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<III', 8, 8, 8))
        f.write(b'XYZI')
        n_voxels = 1
        f.write(struct.pack('<I', 5))
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', n_voxels))
        f.write(struct.pack('<BBBB', 0, 0, 0, 1))
        f.write(b'RGBA')
        f.write(struct.pack('<I', 1024))
        f.write(struct.pack('<I', 0))
        for i in range(256):
            f.write(struct.pack('<BBBB', 255, 255, 255, 255))
        end_pos = f.tell()
        children_size = end_pos - main_start
        f.seek(main_start - 8)
        f.write(struct.pack('<I', 0))
        f.write(struct.pack('<I', children_size))
    print(f"Minimal .vox file saved to {filename}")