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