from .auto_voxel_scaler import AutoVoxelScaler
from .depth_to_vox import DepthToVox
from .voxel_preview import VoxelPreview
from .voxel_to_glb import VoxelToGLB
from .multi_view_voxel_fusion import MultiViewVoxelFusion
from .wfc_terrain_generator import WFCTerrain3DNode
from .wfc3d import WFC3D

NODE_CLASS_MAPPINGS = {
    "AutoVoxelScaler": AutoVoxelScaler,
    "DepthToVox": DepthToVox,
    "VoxelPreview": VoxelPreview,
    "VoxelToGLB": VoxelToGLB,
    "MultiViewVoxelFusion": MultiViewVoxelFusion,
    "WFCTerrain3DNode": WFCTerrain3DNode,
    "WFC3D": WFC3D,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoVoxelScaler": "Auto Voxel Scaler",
    "DepthToVox": "Depth To Vox",
    "VoxelPreview": "Voxel Preview",
    "VoxelToGLB": "Voxel to .GLB",
    "MultiViewVoxelFusion": "Multi View Voxel Fusion",
    "WFCTerrain3DNode": "WFC Terrain 3D Node",
    "WFC3D": "WFC3D Engine",
}