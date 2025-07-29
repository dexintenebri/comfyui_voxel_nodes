import torch
import numpy as np
import comfy.utils
import folder_paths
from PIL import Image
import json
import os
import sys
import time
import trimesh
from torchvision import transforms
import yaml
from types import SimpleNamespace

# Add Pix2Vox++ to path
pix2vox_path = os.path.join(os.path.dirname(__file__), "pix2vox")
sys.path.append(pix2vox_path)

# Import Pix2Vox++ model
from .pix2vox.model.decoder import Decoder
from .pix2vox.model.encoder import Encoder
from .pix2vox.model.refiner import Refiner
from .pix2vox.model.merger import Merger

# Constants for memory optimization
MAX_RESOLUTION = 128  # Safe limit for VRAM
SPARSE_THRESHOLD = 0.05  # Skip voxels with low depth

def add_voxel_models_path():
    """
    Ensure voxel_models directory is registered in folder_paths.
    Works for all users cloning the repo without modifying global folder_paths.py.
    """
    models_dir = os.path.join(os.path.dirname(folder_paths.__file__), "..", "models")
    voxel_models_path = os.path.normpath(os.path.join(models_dir, "voxel_models"))

    if not os.path.exists(voxel_models_path):
        os.makedirs(voxel_models_path, exist_ok=True)

    if "voxel_models" in folder_paths.folder_names_and_paths:
        if voxel_models_path not in folder_paths.folder_names_and_paths["voxel_models"][0]:
            folder_paths.folder_names_and_paths["voxel_models"][0].insert(0, voxel_models_path)
    else:
        folder_paths.folder_names_and_paths["voxel_models"] = ([voxel_models_path], {".pth", ".safetensors"})

add_voxel_models_path()

def dict_to_namespace(d):
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys if present."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def load_pix2vox_weights(model_dir, model_type, encoder, decoder, refiner, merger):
    """
    Load weights for Pix2Vox++ models from model_dir.
    """
    model_name_map = {
        "pix2vox++": "pix2vox++-shapenet.pth",
        "shapenet": "pix2vox++-shapenet.pth",
        "custom": "custom.pth"
    }

    # Try combined weights file first
    filename = model_name_map.get(model_type, f"{model_type}.pth")
    combined_path = os.path.join(model_dir, filename)
    
    if os.path.exists(combined_path):
        print(f"Loading combined weights from {combined_path}")
        checkpoint = torch.load(combined_path, map_location="cpu")
        
        # Handle different checkpoint formats
        if all(key in checkpoint for key in ["encoder_state_dict", "decoder_state_dict", "refiner_state_dict", "merger_state_dict"]):
            encoder.load_state_dict(remove_module_prefix(checkpoint["encoder_state_dict"]), strict=False)
            decoder.load_state_dict(remove_module_prefix(checkpoint["decoder_state_dict"]), strict=False)
            refiner.load_state_dict(remove_module_prefix(checkpoint["refiner_state_dict"]), strict=False)
            merger.load_state_dict(remove_module_prefix(checkpoint["merger_state_dict"]), strict=False)
            return True
        elif all(key in checkpoint for key in ["encoder", "decoder", "refiner", "merger"]):
            encoder.load_state_dict(remove_module_prefix(checkpoint["encoder"]), strict=False)
            decoder.load_state_dict(remove_module_prefix(checkpoint["decoder"]), strict=False)
            refiner.load_state_dict(remove_module_prefix(checkpoint["refiner"]), strict=False)
            merger.load_state_dict(remove_module_prefix(checkpoint["merger"]), strict=False)
            return True
        else:
            # Fallback to flat checkpoint (encoder only)
            print("Loading flat checkpoint (encoder only)")
            encoder.load_state_dict(remove_module_prefix(checkpoint), strict=False)
            return True

    print(f"⚠ No model weights found at {combined_path}")
    return False

class VoxelModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["pix2vox++", "shapenet", "custom"], {"default": "pix2vox++"}),
            }
        }

    RETURN_TYPES = ("VOXEL_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "VoxelNodes"

    def load_model(self, model_type):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        config_path = os.path.join(os.path.dirname(__file__), "pix2vox", "configs", "pix2vox.yml")
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        cfg = dict_to_namespace(cfg_dict)

        # Initialize models in FP32
        encoder = Encoder(cfg).float().to(device)
        decoder = Decoder(cfg).float().to(device)
        refiner = Refiner(cfg).float().to(device)
        merger = Merger(cfg).float().to(device)

        # Load weights and enforce FP32
        model_path = os.path.join(os.path.dirname(__file__), "pix2vox", "model", "pix2vox++-shapenet.pth")
        checkpoint = torch.load(model_path, map_location=device)
        
        encoder.load_state_dict(remove_module_prefix(checkpoint['encoder_state_dict']), strict=False)
        decoder.load_state_dict(remove_module_prefix(checkpoint['decoder_state_dict']), strict=False)
        refiner.load_state_dict(remove_module_prefix(checkpoint['refiner_state_dict']), strict=False)
        merger.load_state_dict(remove_module_prefix(checkpoint['merger_state_dict']), strict=False)

        # Explicitly set to FP32 again (in case weights were FP16)
        for model in [encoder, decoder, refiner, merger]:
            model.float()

        voxel_model = {
            "type": model_type,
            "encoder": encoder,
            "decoder": decoder,
            "refiner": refiner,
            "merger": merger,
            "config": cfg,
            "threshold": 0.4,
            "device": device
        }

        return (voxel_model,)

class DepthEstimationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_size": (["small", "medium", "large"], {"default": "small"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "estimate_depth"
    CATEGORY = "VoxelNodes"

    def __init__(self):
        self.models = {}
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_midas_model(self, model_size):
        original_sys_path = sys.path.copy()
        filtered_paths = [p for p in sys.path if "comfyui_controlnet_aux" not in p]
        sys.path = filtered_paths
        
        try:
            model = torch.hub.load("intel-isl/MiDaS", f"MiDaS_{model_size}")
        finally:
            sys.path = original_sys_path
            
        return model

    def estimate_depth(self, image, model_size):
        if model_size not in self.models:
            try:
                model = self.load_midas_model(model_size)
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                self.models[model_size] = model
            except Exception as e:
                print(f"Error loading MiDaS model: {e}")
                self.models[model_size] = None
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = image[0].permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        if self.models[model_size] is None:
            depth_map = img.mean(dim=1, keepdim=True)
        else:
            with torch.no_grad():
                prediction = self.models[model_size](img.to(device))
                depth_min = prediction.min()
                depth_max = prediction.max()
                depth_map = (prediction - depth_min) / (depth_max - depth_min)
        
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = Image.fromarray(depth_map).resize((image.shape[2], image.shape[1]))
        depth_map = torch.from_numpy(np.array(depth_map)).float() / 255.0
        depth_map = depth_map.unsqueeze(0).unsqueeze(-1)
        
        return (depth_map,)

class OptimizedVoxelizationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "voxel_resolution": ("INT", {"default": 64, "min": 32, "max": MAX_RESOLUTION, "step": 16}),
                "depth_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("VOXEL_DATA",)
    FUNCTION = "voxelize"
    CATEGORY = "VoxelNodes"

    def voxelize(self, image, depth_map, voxel_resolution, depth_scale):
        voxel_resolution = min(voxel_resolution, MAX_RESOLUTION)
        img_data = image[0].numpy()
        depth_data = depth_map[0].numpy().squeeze()
        
        if depth_data.ndim == 3 and depth_data.shape[-1] == 3:
            depth_data = depth_data.mean(axis=-1)
        elif depth_data.ndim > 2:
            depth_data = depth_data[..., 0]
            
        depth_data = depth_data * depth_scale
        H, W = depth_data.shape
        
        min_depth = depth_data.min()
        max_depth = depth_data.max()
        depth_range = max(max_depth - min_depth, 1e-6)
        
        voxels_set = set()
        color_dict = {}
        
        y_coords, x_coords = np.indices((H, W))
        nx = x_coords / W
        ny = y_coords / H
        nz = (depth_data - min_depth) / depth_range
        
        vx = np.clip((nx * voxel_resolution).astype(int), 0, voxel_resolution-1)
        vy = np.clip(((1 - ny) * voxel_resolution).astype(int), 0, voxel_resolution-1)
        vz = np.clip((nz * voxel_resolution).astype(int), 0, voxel_resolution-1)
        
        valid_mask = (depth_data > SPARSE_THRESHOLD)
        vx = vx[valid_mask]
        vy = vy[valid_mask]
        vz = vz[valid_mask]
        colors = img_data[valid_mask]
        
        for i in range(len(vx)):
            coord = (vy[i], vx[i], vz[i])
            voxels_set.add(coord)
            color_dict[coord] = colors[i]
        
        voxel_data = {
            "voxels": voxels_set,
            "colors": color_dict,
            "resolution": voxel_resolution,
            "depth_map": depth_data,
            "image": img_data
        }
        return (voxel_data,)

class ShapeCompletionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel_data": ("VOXEL_DATA",),
                "voxel_model": ("VOXEL_MODEL",),
                "completion_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("VOXEL_DATA",)
    FUNCTION = "complete_shape"
    CATEGORY = "VoxelNodes"
    
    def complete_shape(self, voxel_data, voxel_model, completion_strength):
        encoder = voxel_model["encoder"]
        decoder = voxel_model["decoder"]
        refiner = voxel_model["refiner"]
        merger = voxel_model["merger"]
        threshold = voxel_model["threshold"]
        device = voxel_model["device"]

        # Prepare input tensor with proper dimensions
        with torch.no_grad():
            img = Image.fromarray((voxel_data["image"] * 255).astype(np.uint8))
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Create 5D tensor: [batch=1, views=1, channels=3, height=224, width=224]
            img_tensor = transform(img).unsqueeze(0).unsqueeze(0).to(device)
            print(f"Input tensor shape: {img_tensor.shape}")  # Should be [1,1,3,224,224]

            # Run encoder
            image_features = encoder(img_tensor)
            print(f"Encoder output shape: {image_features.shape}")

            # Prepare for decoder - ensure 5D input
            if image_features.dim() == 4:
                # Add view dimension if missing
                image_features = image_features.unsqueeze(1)
                print(f"Added view dimension: {image_features.shape}")

            # Proper permutation for decoder
            batch_size, num_views, channels, height, width = image_features.shape
            image_features = image_features.permute(0, 1, 3, 4, 2).contiguous()  # [B,V,H,W,C]
            image_features = image_features.view(batch_size * num_views, height, width, channels)
            print(f"Decoder input shape: {image_features.shape}")

            # Run decoder
            raw_features, generated_volume = decoder(image_features)
            print(f"Decoder outputs - raw: {raw_features.shape}, generated: {generated_volume.shape}")

            # Run merger
            merged_volume = merger(raw_features, generated_volume)
            print(f"Merged volume shape: {merged_volume.shape}")

            # Run refiner
            completed_voxels = refiner(merged_volume)
            completed_voxels = (torch.sigmoid(completed_voxels) > threshold).float()
            print(f"Final voxels shape: {completed_voxels.shape}")
        
        # Convert to numpy
        completed_voxels = completed_voxels.squeeze().cpu().numpy()
        resolution = completed_voxels.shape[0]
        
        # Create completed voxel set
        vy, vx, vz = np.where(completed_voxels)
        completed_set = set(zip(vy, vx, vz))
        completed_colors = {coord: [0.8, 0.8, 1.0] for coord in completed_set}
        
        # Blend with original voxels
        orig_res = voxel_data["resolution"]
        scale_factor = resolution / orig_res
        orig_voxels = voxel_data["voxels"]
        orig_colors = voxel_data["colors"]
        
        for coord in orig_voxels:
            y, x, z = coord
            ny = min(int(y * scale_factor), resolution-1)
            nx = min(int(x * scale_factor), resolution-1)
            nz = min(int(z * scale_factor), resolution-1)
            new_coord = (ny, nx, nz)
            
            if new_coord not in completed_set or np.random.random() >= completion_strength:
                completed_set.add(new_coord)
                if coord in orig_colors:
                    completed_colors[new_coord] = orig_colors[coord]
        
        voxel_data["voxels"] = completed_set
        voxel_data["colors"] = completed_colors
        voxel_data["resolution"] = resolution
        
        return (voxel_data,)

class VoxelPreviewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel_data": ("VOXEL_DATA",),
                "preview_size": ("INT", {"default": 256, "min": 128, "max": 1024, "step": 64}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "VoxelNodes"
    
    def preview(self, voxel_data, preview_size):
        resolution = voxel_data["resolution"]
        voxels = voxel_data["voxels"]
        colors = voxel_data["colors"]
        
        voxel_coords = []
        voxel_colors = []
        
        for coord in voxels:
            y, x, z = coord
            voxel_coords.append([x, y, z])
            if coord in colors:
                color = colors[coord]
                voxel_colors.append(f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}")
            else:
                voxel_colors.append("#8888ff")
        
        preview_id = f"voxel-preview-{int(time.time()*1000)}"
        
        viewer_html = f"""
        <div id="{preview_id}" style="width:100%; height:400px;"></div>
        <script>
            function initVoxelPreview() {{
                const container = document.getElementById('{preview_id}');
                if(!container) return;
                
                while(container.firstChild) {{
                    container.removeChild(container.firstChild);
                }}
                
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setClearColor(0x222222, 1);
                container.appendChild(renderer.domElement);
                
                const light1 = new THREE.DirectionalLight(0xffffff, 1);
                light1.position.set(1, 1, 1);
                scene.add(light1);
                const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
                light2.position.set(-1, -1, -1);
                scene.add(light2);
                const ambient = new THREE.AmbientLight(0x404040);
                scene.add(ambient);
                
                const geometry = new THREE.BoxGeometry(0.9, 0.9, 0.9);
                const materialCache = {{}};
                const group = new THREE.Group();
                
                const voxelCoords = {json.dumps(voxel_coords)};
                const voxelColors = {json.dumps(voxel_colors)};
                
                for (let i = 0; i < voxelCoords.length; i++) {{
                    const [x, y, z] = voxelCoords[i];
                    const color = voxelColors[i];
                    
                    if (!materialCache[color]) {{
                        materialCache[color] = new THREE.MeshLambertMaterial({{
                            color: color,
                            transparent: true,
                            opacity: 0.9
                        }});
                    }}
                    
                    const cube = new THREE.Mesh(geometry, materialCache[color]);
                    cube.position.set(
                        x - {resolution/2}, 
                        y - {resolution/2}, 
                        z - {resolution/2}
                    );
                    group.add(cube);
                }}
                
                scene.add(group);
                camera.position.z = {resolution * 1.5};
                camera.position.y = {resolution * 0.5};
                
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.rotateSpeed = 0.5;
                
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                
                function onWindowResize() {{
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                }}
                
                window.addEventListener('resize', onWindowResize);
            }}
            
            if(typeof THREE !== 'undefined') {{
                initVoxelPreview();
            }} else {{
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js';
                script.onload = () => {{
                    const controlsScript = document.createElement('script');
                    controlsScript.src = 'https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js';
                    controlsScript.onload = initVoxelPreview;
                    document.head.appendChild(controlsScript);
                }};
                document.head.appendChild(script);
            }}
        </script>
        """
        
        preview_img = Image.new("RGB", (preview_size, preview_size), (40, 40, 40))
        preview_tensor = torch.from_numpy(np.array(preview_img)).float() / 255.0
        preview_tensor = preview_tensor.unsqueeze(0)
        
        return {
            "ui": {
                "images": [preview_tensor],
                "html": [viewer_html]
            }
        }

class VoxelExportNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel_data": ("VOXEL_DATA",),
                "filename_prefix": ("STRING", {"default": "voxel_model"}),
                "smooth_mesh": (["true", "false"], {"default": "true"}),
                "decimation_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "export_obj"
    OUTPUT_NODE = True
    CATEGORY = "VoxelNodes"
    
    def export_obj(self, voxel_data, filename_prefix, smooth_mesh, decimation_ratio):
        resolution = voxel_data["resolution"]
        voxels = voxel_data["voxels"]
        colors = voxel_data["colors"]
        
        # Convert to dense grid
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        color_grid = np.zeros((resolution, resolution, resolution, 3))
        
        for coord in voxels:
            y, x, z = coord
            voxel_grid[y, x, z] = True
            if coord in colors:
                color_grid[y, x, z] = colors[coord]
        
        # Create mesh
        try:
            from skimage.measure import marching_cubes
            verts, faces, _, _ = marching_cubes(voxel_grid, level=0.5)
            vertex_colors = []
            for vert in verts:
                x, y, z = [int(v) for v in vert]
                x = max(0, min(x, resolution-1))
                y = max(0, min(y, resolution-1))
                z = max(0, min(z, resolution-1))
                vertex_colors.append(color_grid[x, y, z] if voxel_grid[x, y, z] else [0.8, 0.8, 1.0])
        except ImportError:
            # Fallback to simple voxel meshing
            verts = []
            faces = []
            vertex_colors = []
            
            for z in range(resolution):
                for y in range(resolution):
                    for x in range(resolution):
                        if voxel_grid[z, y, x]:
                            v = [
                                [x-0.5, y-0.5, z-0.5],
                                [x+0.5, y-0.5, z-0.5],
                                [x+0.5, y+0.5, z-0.5],
                                [x-0.5, y+0.5, z-0.5],
                                [x-0.5, y-0.5, z+0.5],
                                [x+0.5, y-0.5, z+0.5],
                                [x+0.5, y+0.5, z+0.5],
                                [x-0.5, y+0.5, z+0.5]
                            ]
                            f = [
                                [0, 1, 2], [2, 3, 0],
                                [4, 5, 6], [6, 7, 4],
                                [0, 1, 5], [5, 4, 0],
                                [2, 3, 7], [7, 6, 2],
                                [0, 3, 7], [7, 4, 0],
                                [1, 2, 6], [6, 5, 1]
                            ]
                            base_idx = len(verts)
                            verts.extend(v)
                            faces.extend([[base_idx + idx for idx in face] for face in f])
                            vertex_colors.extend([color_grid[z, y, x]] * 8)
        
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=vertex_colors)
        
        if smooth_mesh == "true":
            try:
                trimesh.smoothing.filter_laplacian(mesh, iterations=3)
                target_faces = int(len(mesh.faces) * decimation_ratio)
                if target_faces > 100:
                    mesh = mesh.simplify_quadratic_decimation(target_faces)
            except Exception as e:
                print(f"Mesh processing error: {e}")
        
        # Export OBJ
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{filename_prefix}_{int(time.time())}.obj"
        file_path = os.path.join(output_dir, filename)
        
        with open(file_path, 'w') as f:
            mesh.export(f, file_type='obj', include_color=True)
        
        # Create preview
        try:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axis('off')
            
            vertices = mesh.vertices[:, :2]
            for face in mesh.faces:
                points = vertices[face]
                for i in range(3):
                    start = points[i]
                    end = points[(i + 1) % 3]
                    plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', alpha=0.3)
            
            fig.canvas.draw()
            preview = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            plt.close(fig)
            preview_path = os.path.join(output_dir, filename.replace('.obj', '.png'))
            preview.save(preview_path)
        except Exception as e:
            print(f"Couldn't create preview: {e}")
            preview = Image.new("RGB", (256, 256), (40, 40, 40))
            preview_path = os.path.join(output_dir, filename.replace('.obj', '.png'))
            preview.save(preview_path)
        
        return {
            "ui": {
                "text": [f"Exported OBJ: {filename}"],
                "images": [{"filename": filename.replace('.obj', '.png'), "subfolder": "", "type": "output"}]
            }
        }

NODE_CLASS_MAPPINGS = {
    "VoxelModelLoader": VoxelModelLoader,
    "DepthEstimationNode": DepthEstimationNode,
    "OptimizedVoxelizationNode": OptimizedVoxelizationNode,
    "ShapeCompletionNode": ShapeCompletionNode,
    "VoxelPreviewNode": VoxelPreviewNode,
    "VoxelExportNode": VoxelExportNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxelModelLoader": "Load Voxel Model",
    "DepthEstimationNode": "Depth Estimation",
    "OptimizedVoxelizationNode": "Voxelization (Optimized)",
    "ShapeCompletionNode": "3D Shape Completion",
    "VoxelPreviewNode": "Voxel Preview",
    "VoxelExportNode": "Export OBJ"
}