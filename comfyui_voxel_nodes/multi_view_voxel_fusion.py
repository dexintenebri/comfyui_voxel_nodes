import numpy as np
from PIL import Image
import struct
import os
from sklearn.cluster import KMeans
import datetime

def prepare_images(rgb_tensor, depth_tensor, resolution):
    # RGB
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

    # Depth
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

def render_voxel_view(voxel_grid, view="front"):
    """
    Render a 2D RGB image and depth map for a given view.
    Supported views: "front", "back", "left", "right", "top", "bottom"
    Returns (rgb_img, depth_img), both np.uint8 arrays.
    """
    # axis: (x, y, z, 3)
    w, h, d, _ = voxel_grid.shape
    if view == "front":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                for z in range(d-1, -1, -1):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[y, x] = color
                        depth[y, x] = int(255 * (z / (d-1)))
                        break
    elif view == "back":
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                for z in range(d):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[y, x] = color
                        depth[y, x] = int(255 * ((d-1-z) / (d-1)))
                        break
    elif view == "left":
        rgb = np.zeros((h, d, 3), dtype=np.uint8)
        depth = np.zeros((h, d), dtype=np.uint8)
        for y in range(h):
            for z in range(d-1, -1, -1):
                for x in range(w):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[y, d-1-z] = color
                        depth[y, d-1-z] = int(255 * (x / (w-1)))
                        break
    elif view == "right":
        rgb = np.zeros((h, d, 3), dtype=np.uint8)
        depth = np.zeros((h, d), dtype=np.uint8)
        for y in range(h):
            for z in range(d):
                for x in range(w-1, -1, -1):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[y, z] = color
                        depth[y, z] = int(255 * ((w-1-x) / (w-1)))
                        break
    elif view == "top":
        rgb = np.zeros((w, d, 3), dtype=np.uint8)
        depth = np.zeros((w, d), dtype=np.uint8)
        for x in range(w):
            for z in range(d-1, -1, -1):
                for y in range(h):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[x, d-1-z] = color
                        depth[x, d-1-z] = int(255 * (y / (h-1)))
                        break
    elif view == "bottom":
        rgb = np.zeros((w, d, 3), dtype=np.uint8)
        depth = np.zeros((w, d), dtype=np.uint8)
        for x in range(w):
            for z in range(d):
                for y in range(h-1, -1, -1):
                    color = voxel_grid[x, y, z]
                    if np.any(color):
                        rgb[x, z] = color
                        depth[x, z] = int(255 * ((h-1-y) / (h-1)))
                        break
    else:
        raise ValueError(f"Unknown view: {view}")
    return rgb, depth

class MultiViewVoxelFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_image": ("IMAGE",),
                "front_depth": ("IMAGE",),
                "back_image": ("IMAGE",),
                "back_depth": ("IMAGE",),
                "left_image": ("IMAGE",),
                "left_depth": ("IMAGE",),
                "right_image": ("IMAGE",),
                "right_depth": ("IMAGE",),
                "resolution": ("INT", {"default": 128, "min": 16, "max": 256}),
                "max_depth": ("INT", {"default": 64, "min": 8, "max": 256}),
                "surface_mode": (["surface_only", "solid_columns", "thick_surface"], {"default": "solid_columns"}),
                "invert_depth": ("BOOLEAN", {"default": True}),
                "export_path": ("STRING", {"default": "fusion_output.vox"}),
            },
            "optional": {
                "top_image": ("IMAGE",),
                "top_depth": ("IMAGE",),
                "bottom_image": ("IMAGE",),
                "bottom_depth": ("IMAGE",),
                "front_image_refined": ("IMAGE",),
                "front_depth_refined": ("IMAGE",),
                "back_image_refined": ("IMAGE",),
                "back_depth_refined": ("IMAGE",),
                "left_image_refined": ("IMAGE",),
                "left_depth_refined": ("IMAGE",),
                "right_image_refined": ("IMAGE",),
                "right_depth_refined": ("IMAGE",),
                "top_image_refined": ("IMAGE",),
                "top_depth_refined": ("IMAGE",),
                "bottom_image_refined": ("IMAGE",),
                "bottom_depth_refined": ("IMAGE",),
                "ai_edit_images": ("LIST",),
                "ai_edit_depths": ("LIST",),
                "ai_edit_views": ("LIST",),
            }
        }

    RETURN_TYPES = ("NP_ARRAY", "STRING")
    RETURN_NAMES = ("voxels", "vox_path")
    FUNCTION = "run"
    CATEGORY = "VoxelTools"
    OUTPUT_NODE = True

    def run(self, front_image, front_depth,
                  back_image, back_depth,
                  left_image, left_depth,
                  right_image, right_depth,
                  resolution, max_depth,
                  surface_mode, invert_depth,
                  export_path,
                  top_image=None, top_depth=None,
                  bottom_image=None, bottom_depth=None,
                  front_image_refined=None, front_depth_refined=None,
                  back_image_refined=None, back_depth_refined=None,
                  left_image_refined=None, left_depth_refined=None,
                  right_image_refined=None, right_depth_refined=None,
                  top_image_refined=None, top_depth_refined=None,
                  bottom_image_refined=None, bottom_depth_refined=None,
                  ai_edit_images=None, ai_edit_depths=None, ai_edit_views=None):

        def pick(raw, refined):
            return refined if refined is not None else raw

        # Compose required views
        view_dict = {
            "front": (pick(front_image, front_image_refined), pick(front_depth, front_depth_refined)),
            "back": (pick(back_image, back_image_refined), pick(back_depth, back_depth_refined)),
            "left": (pick(left_image, left_image_refined), pick(left_depth, left_depth_refined)),
            "right": (pick(right_image, right_image_refined), pick(right_depth, right_depth_refined)),
        }
        # Optionally add top/bottom if present
        if top_image is not None and top_depth is not None:
            view_dict["top"] = (pick(top_image, top_image_refined), pick(top_depth, top_depth_refined))
        if bottom_image is not None and bottom_depth is not None:
            view_dict["bottom"] = (pick(bottom_image, bottom_image_refined), pick(bottom_depth, bottom_depth_refined))

        # Prepare image/depth for each view
        prepared_views = {}
        for key, (img, dmap) in view_dict.items():
            rgb_tensor = img[0].cpu().numpy()
            depth_tensor = dmap[0].cpu().numpy()
            rgb, depth = prepare_images(rgb_tensor, depth_tensor, resolution)
            prepared_views[key] = (rgb, depth)

        # Space Carving
        carved_voxel_grid = self.space_carve_voxels(prepared_views, resolution, max_depth, invert_depth)

        # Assign colors (front view priority, fallback to others)
        colored_grid = self.assign_colors_to_voxels(carved_voxel_grid, prepared_views, resolution, max_depth)

        # (Optional) Post-fusion AI edit step
        if ai_edit_images is not None and ai_edit_depths is not None and ai_edit_views is not None:
            colored_grid = self.apply_ai_edits_to_voxels(
                colored_grid,
                ai_edit_images,
                ai_edit_depths,
                ai_edit_views,
                max_depth,
                surface_mode,
                invert_depth
            )

        safe_path = self.get_safe_output_path(export_path)
        self.save_vox_from_grid(colored_grid, safe_path)

        return (colored_grid, safe_path)

    def space_carve_voxels(self, prepared_views, resolution, max_depth, invert_depth):
        # Start with all voxels empty
        occupancy = np.zeros((resolution, resolution, max_depth), dtype=bool)
        for view, (rgb, depth) in prepared_views.items():
            if invert_depth:
                depth_processed = 255 - depth
            else:
                depth_processed = depth
            depth_indices = np.clip((depth_processed / 255.0 * (max_depth - 1)).astype(int), 0, max_depth - 1)
            if view == "front":
                for y in range(resolution):
                    for x in range(resolution):
                        z_limit = depth_indices[y, x]
                        occupancy[x, y, :z_limit+1] = True
            elif view == "back":
                for y in range(resolution):
                    for x in range(resolution):
                        z_limit = depth_indices[y, x]
                        occupancy[x, y, max_depth-z_limit-1:] = True
            elif view == "left":
                for y in range(resolution):
                    for z in range(max_depth):
                        x_limit = depth_indices[y, z]
                        occupancy[:x_limit+1, y, z] = True
            elif view == "right":
                for y in range(resolution):
                    for z in range(max_depth):
                        x_limit = depth_indices[y, z]
                        occupancy[max_depth-x_limit-1:, y, z] = True
            elif view == "top":
                for x in range(resolution):
                    for z in range(max_depth):
                        y_limit = depth_indices[x, z]
                        occupancy[x, :y_limit+1, z] = True
            elif view == "bottom":
                for x in range(resolution):
                    for z in range(max_depth):
                        y_limit = depth_indices[x, z]
                        occupancy[x, max_depth-y_limit-1:, z] = True
        print("[MultiViewVoxelFusion] Occupied voxels:", np.count_nonzero(occupancy))
        return occupancy
    
    def assign_colors_to_voxels(self, occupancy, prepared_views, resolution, max_depth):
        """
        Assign colors to occupied voxels. Priority: front, then back, left, right, top, bottom.
        """
        color_grid = np.zeros((resolution, resolution, max_depth, 3), dtype=np.uint8)
        # For each occupied voxel, assign color from nearest view if possible
        for x in range(resolution):
            for y in range(resolution):
                for z in range(max_depth):
                    if occupancy[x, y, z]:
                        assigned = False
                        # Try front
                        if "front" in prepared_views:
                            rgb, depth = prepared_views["front"]
                            if y < rgb.shape[0] and x < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[y, x]
                                assigned = True
                        # Try others if front not assigned
                        if not assigned and "back" in prepared_views:
                            rgb, depth = prepared_views["back"]
                            if y < rgb.shape[0] and x < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[y, x]
                                assigned = True
                        if not assigned and "left" in prepared_views:
                            rgb, depth = prepared_views["left"]
                            if y < rgb.shape[0] and z < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[y, z]
                                assigned = True
                        if not assigned and "right" in prepared_views:
                            rgb, depth = prepared_views["right"]
                            if y < rgb.shape[0] and z < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[y, z]
                                assigned = True
                        if not assigned and "top" in prepared_views:
                            rgb, depth = prepared_views["top"]
                            if x < rgb.shape[0] and z < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[x, z]
                                assigned = True
                        if not assigned and "bottom" in prepared_views:
                            rgb, depth = prepared_views["bottom"]
                            if x < rgb.shape[0] and z < rgb.shape[1]:
                                color_grid[x, y, z] = rgb[x, z]
                                assigned = True
        return color_grid

    def apply_ai_edits_to_voxels(self, voxel_grid, ai_images, ai_depths, ai_views, max_depth, surface_mode, invert_depth):
        view_to_rot = {
            "front": lambda v: v,
            "back": lambda v: np.flip(v, axis=0),
            "left": lambda v: np.transpose(v, (2, 1, 0, 3)),
            "right": lambda v: np.flip(np.transpose(v, (2, 1, 0, 3)), axis=0),
            "top": lambda v: np.transpose(v, (0, 2, 1, 3)),
            "bottom": lambda v: np.flip(np.transpose(v, (0, 2, 1, 3)), axis=2),
        }
        for img, dmap, view in zip(ai_images, ai_depths, ai_views):
            rgb, depth = np.array(img), np.array(dmap)
            vgrid = self.image_depth_to_voxel(rgb, depth, max_depth, surface_mode, invert_depth)
            vgrid = view_to_rot[view](vgrid)
            mask = np.any(vgrid > 0, axis=-1)
            voxel_grid[mask] = vgrid[mask]
        return voxel_grid

    def get_safe_output_path(self, export_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"multi_view_{timestamp}"
        if not export_path.endswith('.vox'):
            export_path += '.vox'
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            out_path = os.path.join(output_dir, base_name + ".vox")
        except:
            out_path = os.path.join(".", base_name + ".vox")
        print(f"[MultiViewVoxelFusion] Output path: {out_path}")
        return out_path

    def save_vox_from_grid(self, voxel_grid, export_path):
        w, h, d, _ = voxel_grid.shape
        voxels = []
        colors = []
        for x in range(w):
            for y in range(h):
                for z in range(d):
                    color = tuple(int(c) for c in voxel_grid[x, y, z])
                    if np.any(color):
                        voxels.append((x, y, z))
                        colors.append(color)
        palette = self.build_palette(colors)
        self.save_vox(voxels, colors, palette, export_path, w, h, d)

    def build_palette(self, colors):
        unique_colors = list(set(colors))
        if len(unique_colors) > 255:
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
            print("[MultiViewVoxelFusion] Warning: No voxels to save.")
            return
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
                xyzi_content_size = 4 + len(voxels) * 4
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
                for r, g, b in palette:
                    f.write(struct.pack('<BBBB', int(r), int(g), int(b), 255))
                end_pos = f.tell()
                children_size = end_pos - main_start
                f.seek(main_start - 8)
                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<I', children_size))
            print(f"[MultiViewVoxelFusion] Saved {len(voxels)} voxels to {export_path}")
        except Exception as e:
            print(f"[MultiViewVoxelFusion] Error saving file: {e}")