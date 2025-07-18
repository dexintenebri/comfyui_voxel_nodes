import numpy as np
import open3d as o3d

class VoxelPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxels": ("NP_ARRAY",),
                "point_size": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "VoxelTools"

    def run(self, voxels, point_size=4.0):
        try:
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
                return ()

            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(points))
            pc.colors = o3d.utility.Vector3dVector(np.array(colors))

            o3d.visualization.draw_geometries([
                pc
            ], window_name="Voxel Preview", width=800, height=600)
            return ()

        except Exception as e:
            print(f"❌ VoxelPreview failed: {e}")
            return ()
