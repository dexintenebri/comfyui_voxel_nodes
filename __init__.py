from .auto_voxel_scaler import AutoVoxelScaler
from .depth_to_vox import DepthToVox
from .voxel_preview import VoxelPreview
from .voxel_to_glb import VoxelToGLB
from .multi_view_voxel_fusion import MultiViewVoxelFusion
from .wfc3d import WFCTerrain3DNode
from .voxel_nodes import VoxelModelLoader
from .voxel_nodes import DepthEstimationNode
from .voxel_nodes import OptimizedVoxelizationNode
from .voxel_nodes import ShapeCompletionNode
from .voxel_nodes import VoxelPreviewNode
import os
import importlib
import time

MODULE_NAME = "comfyui_voxel_nodes"

ascii_art = r"""
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣷⡄⠀⠀⠀⠀⠀⠀⠀⢠⣄⣤⣦⣤⣀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠿⠟⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⠘⢿⣿⠟⠀⢠⡀⠀⠀⠀⠀⠀⠀⠀⣰⡗⠀⠀⠀⠀
⠀⠀⠀⠀⠀⢠⣾⠀⣿⠀⣷⣦⣤⣴⡇⢸⡇⠀⣷⠀⠀⠀⠀⣰⡟⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⣿⠀⣿⣤⣈⣉⣉⣉⣠⣼⡇⠀⣿⡆⠀⠀⣰⡟⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣿⠇⠀⠀⠛⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠛⠛⠀⠛⠛⠛⠛⠛⠛⠛⠛⠃⠀⠛⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⣤⣤⣤⣤⣤⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠈⠙⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⠿⠿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠇⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠘⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠛⠃⠀⠀⠀⠀⠀
"""

print(f"\033[95m{ascii_art}\033[0m")  # Purple color

print(f"[{MODULE_NAME}] 🔧 Initializing...")

start_time = time.time()
module_path = os.path.dirname(__file__)

for fname in os.listdir(module_path):
    if fname.endswith(".py") and fname != "__init__.py":
        mod_name = fname[:-3]
        try:
            importlib.import_module(f"{MODULE_NAME}.{mod_name}")
            print(f"[{MODULE_NAME}] ✅ Loaded module: {mod_name}")
        except Exception as e:
            print(f"[{MODULE_NAME}] ❌ Failed to load {mod_name}: {e}")

elapsed = time.time() - start_time
print(f"[{MODULE_NAME}] 🚀 Ready in {elapsed:.2f}s")





NODE_CLASS_MAPPINGS = {
    "AutoVoxelScaler": AutoVoxelScaler,
    "DepthToVox": DepthToVox,
    "VoxelPreview": VoxelPreview,
    "VoxelToGLB": VoxelToGLB,
    "MultiViewVoxelFusion": MultiViewVoxelFusion,
    "ProceduralChunkedWFC3DTerrain": WFCTerrain3DNode,
    "VoxelModelLoader": VoxelModelLoader,
    "DepthEstimationNode": DepthEstimationNode,
    "OptimizedVoxelizationNode": OptimizedVoxelizationNode,
    "ShapeCompletionNode": ShapeCompletionNode,
    "VoxelPreviewNode": VoxelPreviewNode,    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoVoxelScaler": "Auto Voxel Scaler",
    "DepthToVox": "Depth To Vox",
    "VoxelPreview": "Voxel Preview",
    "VoxelToGLB": "Voxel to .GLB",
    "MultiViewVoxelFusion": "Multi View Voxel Fusion",
    "ProceduralChunkedWFC3DTerrain": "Procedural Chunked WFC3D Terrain",
    "VoxelModelLoader": "Load Voxel Model",
    "DepthEstimationNode": "Depth Estimation",
    "OptimizedVoxelizationNode": "Voxelization (Optimized)",
    "ShapeCompletionNode": "3D Shape Completion",
    "VoxelPreviewNode": "Voxel Preview",
}