import numpy as np
from PIL import Image
import struct
import os
from sklearn.cluster import KMeans
import datetime

class MultiViewVoxelFusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front_image": ("IMAGE",),
                "front_depth": ("IMAGE",),
                "side_image": ("IMAGE",),
                "side_depth": ("IMAGE",),
                "top_image": ("IMAGE",),
                "top_depth": ("IMAGE",),
                "resolution": ("INT", {"default": 128, "min": 16, "max": 256}),
                "max_depth": ("INT", {"default": 64, "min": 8, "max": 256}),
                "surface_mode": (["surface_only", "solid_columns", "thick_surface"], {"default": "solid_columns"}),
                "invert_depth": ("BOOLEAN", {"default": True}),
                "export_path": ("STRING", {"default": "fusion_output.vox"}),
                "preview_voxel": ("BOOLEAN", {"default": False}),
                "export_glb": ("BOOLEAN", {"default": False}),
                "glb_cube_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
                "preview_point_size": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ("NP_ARRAY", "STRING", "STRING")
    RETURN_NAMES = ("voxels", "vox_path", "glb_path")
    FUNCTION = "run"
    CATEGORY = "VoxelTools"
    OUTPUT_NODE = True

    def run(self, front_image, front_depth,
                  side_image, side_depth,
                  top_image, top_depth,
                  resolution, max_depth,
                  surface_mode, invert_depth,
                  export_path, preview_voxel,
                  export_glb, glb_cube_size, preview_point_size):

        # Convert each view to its voxel grid
        front_grid = self.image_depth_to_voxel(front_image, front_depth, resolution, max_depth, surface_mode, invert_depth)
        side_grid  = self.image_depth_to_voxel(side_image,  side_depth,  resolution, max_depth, surface_mode, invert_depth)
        top_grid   = self.image_depth_to_voxel(top_image,   top_depth,   resolution, max_depth, surface_mode, invert_depth)

        # Align all grids into common space
        # Front: [X,Y,Z] as is
        # Side: rotate so [Z,Y,X]
        side_grid_rot = np.transpose(side_grid, (2,1,0,3))
        # Top: rotate so [X,Z,Y]
        top_grid_rot = np.transpose(top_grid, (0,2,1,3))

        # Fuse grids: take union (if any voxel is filled, use color from first non-empty)
        fused = self.fuse_voxel_grids([front_grid, side_grid_rot, top_grid_rot])

        # Save .vox file
        safe_path = self.get_safe_output_path(export_path)
        self.save_vox_from_grid(fused, safe_path)

        # Preview
        if preview_voxel:
            self.voxel_preview(fused, preview_point_size)

        # Export to GLB
        glb_path = ""
        if export_glb:
            glb_path = self.voxel_to_glb(fused, glb_cube_size)

        return (fused, safe_path, glb_path)

    def image_depth_to_voxel(self, image, depth_map, resolution, max_depth, surface_mode, invert_depth):
        # Convert ComfyUI tensors to numpy
        rgb_tensor = image[0].cpu().numpy()
        depth_tensor = depth_map[0].cpu().numpy()

        # Resize images
        rgb = self.prepare_image(rgb_tensor, resolution)
        depth = self.prepare_image(depth_tensor, resolution, is_depth=True)

        if invert_depth:
            depth = 255 - depth

        depth_arr = np.clip((depth / 255.0 * (max_depth - 1)).astype(int), 0, max_depth - 1)
        h, w = depth.shape
        voxel_grid = np.zeros((w, h, max_depth, 3), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                d = int(depth_arr[y, x])
                color = tuple(int(c) for c in rgb[y, x])
                if surface_mode == "surface_only":
                    voxel_grid[x, y, d] = color
                elif surface_mode == "thick_surface":
                    voxel_grid[x, y, d] = color
                    if d > 0:
                        voxel_grid[x, y, d-1] = color
                elif surface_mode == "solid_columns":
                    for z in range(d+1):
                        voxel_grid[x, y, z] = color
        return voxel_grid

    def prepare_image(self, arr, resolution, is_depth=False):
        if arr.ndim == 3:
            if arr.shape[0] in (1,3):
                arr = arr.transpose(1,2,0)
            elif arr.shape[2] in (1,3):
                arr = arr
            else:
                raise ValueError("Unexpected shape")
        if is_depth:
            if arr.ndim == 3:
                arr = arr.mean(axis=2)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = img.resize((resolution, resolution), Image.BILINEAR)
        return np.array(img)

    def fuse_voxel_grids(self, grids):
        shape = grids[0].shape
        fused = np.zeros(shape, dtype=np.uint8)
        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    for grid in grids:
                        color = grid[x,y,z]
                        if np.any(color):
                            fused[x,y,z] = color
                            break
        return fused

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
                    color = tuple(int(c) for c in voxel_grid[x,y,z])
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
            palette.append((0,0,0))
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
                xyzi_content_size = 4 + len(voxels)*4
                f.write(struct.pack('<I', xyzi_content_size))
                f.write(struct.pack('<I', 0))
                f.write(struct.pack('<I', len(voxels)))
                for i, (x, y, z) in enumerate(voxels):
                    color_index = self.get_color_index(colors[i], palette)
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
            print(f"[MultiViewVoxelFusion] Saved {len(voxels)} voxels to {export_path}")
        except Exception as e:
            print(f"[MultiViewVoxelFusion] Error saving file: {e}")

    def voxel_preview(self, voxels, point_size=4.0):
        try:
            import open3d as o3d
            voxels = np.array(voxels)
            points = []
            colors = []
            sx, sy, sz, _ = voxels.shape
            for x in range(sx):
                for y in range(sy):
                    for z in range(sz):
                        color = voxels[x, y, z]
                        if np.any(color):
                            points.append([x, y, z])
                            colors.append(color / 255.0)
            if not points:
                print("⚠️ No voxels to preview.")
                return
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(points))
            pc.colors = o3d.utility.Vector3dVector(np.array(colors))
            o3d.visualization.draw_geometries([
                pc
            ], window_name="Multi-View Voxel Preview", width=800, height=600, point_show_normal=False)
        except Exception as e:
            print(f"❌ VoxelPreview failed: {e}")

    def voxel_to_glb(self, voxels, cube_size=1.0):
        try:
            import trimesh
            import tempfile
            voxels = np.array(voxels)
            mesh = trimesh.Scene()
            vx, vy, vz, _ = voxels.shape
            for x in range(vx):
                for y in range(vy):
                    for z in range(vz):
                        color = voxels[x, y, z]
                        if not np.any(color):
                            continue
                        cube = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
                        cube.apply_translation((x * cube_size, y * cube_size, z * cube_size))
                        cube.visual.face_colors = np.tile(np.append(color, 255), (cube.faces.shape[0], 1))
                        mesh.add_geometry(cube)
            if not mesh.geometry:
                raise RuntimeError("No voxel cubes were added to the scene!")
            combined = mesh.dump(concatenate=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp:
                combined.export(tmp.name, file_type="glb")
                glb_path = tmp.name
            print(f"[MultiViewVoxelFusion] GLB exported to {glb_path}")
            return glb_path
        except Exception as e:
            print(f"❌ VoxelToGLB failed: {e}")
            return ""
