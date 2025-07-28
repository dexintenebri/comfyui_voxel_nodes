from .auto_voxel_scaler import AutoVoxelScaler
from .depth_to_vox import DepthToVox
from .voxel_preview import VoxelPreview
from .voxel_to_glb import VoxelToGLB
from .multi_view_voxel_fusion import MultiViewVoxelFusion
from .wfc3d import WFCTerrain3DNode
from .wfc_terrain_generator import WFC3DTerrainGeneratorNode
from .voxel_nodes import (
    VoxelModelLoader,
    DepthEstimationNode,
    OptimizedVoxelizationNode,
    ShapeCompletionNode,
    VoxelPreviewNode,
)

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

print(f"\033[95m{ascii_art}\033[0m")
print(f"[{MODULE_NAME}] 🔧 Initializing...")
start_time = time.time()

print(f"[{MODULE_NAME}] ✅ All modules imported.")
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
    "WFC3DTerrainGenerator": WFC3DTerrainGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoVoxelScaler": "Auto Voxel Scaler WIP",
    "DepthToVox": "Depth To Vox SEMI WORKING",
    "VoxelPreview": "Voxel Preview WIP",
    "VoxelToGLB": "Voxel to .GLB",
    "MultiViewVoxelFusion": "Multi View Voxel Fusion SEMI WORKING",
    "ProceduralChunkedWFC3DTerrain": "Chunk WFC 3D Terrain From Image SEMI WORKING",
    "VoxelModelLoader": "Load Voxel Model",
    "DepthEstimationNode": "Depth Estimation",
    "OptimizedVoxelizationNode": "Voxelization (Optimized)",
    "ShapeCompletionNode": "3D Shape Completion",
    "VoxelPreviewNode": "Voxel Preview WIP",
    "WFC3DTerrainGenerator": "WFC 3D Terrain Generator WIP",
}