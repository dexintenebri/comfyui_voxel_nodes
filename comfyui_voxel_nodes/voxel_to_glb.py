import numpy as np
import trimesh
import tempfile
import os
from PIL import ImageColor

class VoxelToGLB:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxels": ("NP_ARRAY",),  # Shape (X, Y, Z, 3)
                "cube_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "run"
    CATEGORY = "VoxelTools"

    def run(self, voxels, cube_size):
        if not isinstance(voxels, np.ndarray) or voxels.ndim != 4 or voxels.shape[-1] != 3:
            raise ValueError("Expected voxel input of shape (X, Y, Z, 3)")

        mesh = trimesh.Scene()
        vx, vy, vz, _ = voxels.shape

        # Iterate over voxels
        for x in range(vx):
            for y in range(vy):
                for z in range(vz):
                    color = voxels[x, y, z]
                    if not np.any(color):  # Skip black/empty
                        continue

                    # Create a colored cube at the given position
                    cube = trimesh.creation.box(extents=(cube_size, cube_size, cube_size))
                    cube.apply_translation((x * cube_size, y * cube_size, z * cube_size))
                    cube.visual.face_colors = np.tile(np.append(color, 255), (cube.faces.shape[0], 1))

                    mesh.add_geometry(cube)

        if not mesh.geometry:
            raise RuntimeError("No voxel cubes were added to the scene!")

        combined = mesh.dump(concatenate=True)

        # Save to temp .glb file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as tmp:
            combined.export(tmp.name, file_type="glb")
            glb_path = tmp.name

        return (glb_path,)
