import torch
import numpy as np
import comfy.utils
import folder_paths
from PIL import Image
import json
import scipy.sparse
import os
import sys
import time
from torchvision import transforms

# Add Pix2Vox++ to path
pix2vox_path = os.path.join(os.path.dirname(__file__), "pix2vox")
sys.path.append(pix2vox_path)

# Import Pix2Vox++ model
from comfyui_voxel_nodes.pix2vox.model.decoder import Decoder
from comfyui_voxel_nodes.pix2vox.model.encoder import Encoder
from comfyui_voxel_nodes.pix2vox.model.refiner import Refiner
from comfyui_voxel_nodes.pix2vox.model.merger import Merger

# Constants for memory optimization
MAX_RESOLUTION = 128  # Safe limit for VRAM
SPARSE_THRESHOLD = 0.05  # Skip voxels with low depth

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
        model_path = folder_paths.get_full_path("voxel_models", 
                                              f"{model_type}.pth" if model_type != "custom" else "custom.pth")
        
        # Load model configuration
        config = {
            "encoder_dim": 256,
            "decoder_dim": 256,
            "refiner_dim": 512,
            "threshold": 0.4
        }
        
        return ({"config": config, "type": model_type, "path": model_path},)

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

    def load_model(self, model_size):
        if model_size not in self.models:
            model = torch.hub.load("intel-isl/MiDaS", f"MiDaS_{model_size}")
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            self.models[model_size] = model
        return self.models[model_size]

    def estimate_depth(self, image, model_size):
        model = self.load_model(model_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process first image in batch
        img = image[0].permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
        img = img * 255.0
        img = img / 255.0
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        with torch.no_grad():
            img = img.to(device)
            prediction = model(img)

        # Normalize depth map
        depth_min = prediction.min()
        depth_max = prediction.max()
        depth_map = (prediction - depth_min) / (depth_max - depth_min)
        depth_map = depth_map.squeeze().cpu().numpy()
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((image.shape[2], image.shape[1]))
        
        # Convert to tensor
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
        # Limit resolution for VRAM safety
        voxel_resolution = min(voxel_resolution, MAX_RESOLUTION)
        
        img_data = image[0].numpy()  # (H, W, 3)
        depth_data = depth_map[0].numpy().squeeze() * depth_scale  # (H, W)
        
        H, W = depth_data.shape
        min_depth = depth_data.min()
        max_depth = depth_data.max()
        depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
        
        # Create sparse matrices
        voxel_grid = scipy.sparse.dok_matrix((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32)
        color_dict = {}  # Store colors in dict for efficiency
        
        # Vectorized processing
        y_coords, x_coords = np.indices((H, W))
        nx = x_coords / W
        ny = y_coords / H
        nz = (depth_data - min_depth) / depth_range
        
        vx = np.clip((nx * voxel_resolution).astype(int), 0, voxel_resolution-1)
        vy = np.clip(((1 - ny) * voxel_resolution).astype(int), 0, voxel_resolution-1)
        vz = np.clip((nz * voxel_resolution).astype(int), 0, voxel_resolution-1)
        
        # Filter valid points
        valid_mask = (depth_data > SPARSE_THRESHOLD)
        vx = vx[valid_mask]
        vy = vy[valid_mask]
        vz = vz[valid_mask]
        colors = img_data[valid_mask]
        
        # Store voxels and colors
        for i in range(len(vx)):
            voxel_grid[vy[i], vx[i], vz[i]] = 1.0
            color_dict[(vy[i], vx[i], vz[i])] = colors[i]
        
        voxel_data = {
            "voxels": voxel_grid,
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
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_info):
        model_key = model_info["type"]
        if model_key not in self.models:
            config = model_info["config"]
            
            # Initialize model
            encoder = Encoder(
                dim=config["encoder_dim"],
                in_channels=3
            )
            decoder = Decoder(
                dim=config["decoder_dim"],
                out_channels=1
            )
            refiner = Refiner(
                dim=config["refiner_dim"],
                out_channels=1
            )
            merger = Merger()
            
            # Load weights
            state_dict = torch.load(model_info["path"], map_location="cpu")
            encoder.load_state_dict(state_dict["encoder"])
            decoder.load_state_dict(state_dict["decoder"])
            refiner.load_state_dict(state_dict["refiner"])
            merger.load_state_dict(state_dict["merger"])
            
            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            refiner = refiner.to(device)
            merger = merger.to(device)
            
            model = {
                "encoder": encoder,
                "decoder": decoder,
                "refiner": refiner,
                "merger": merger,
                "threshold": config["threshold"],
                "device": device
            }
            self.models[model_key] = model
        
        return self.models[model_key]
    
    def complete_shape(self, voxel_data, voxel_model, completion_strength):
        model_info = self.load_model(voxel_model)
        device = model_info["device"]
        
        # Prepare input image
        img = Image.fromarray((voxel_data["image"] * 255).astype(np.uint8))
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Run through Pix2Vox++
        with torch.no_grad():
            # Encoder
            image_features = model_info["encoder"](img_tensor)
            image_features = image_features.permute(0, 2, 3, 4, 1)
            
            # Decoder
            raw_features = model_info["decoder"](image_features)
            
            # Merger
            raw_features = model_info["merger"](raw_features)
            
            # Refiner
            completed_voxels = model_info["refiner"](raw_features)
            completed_voxels = torch.sigmoid(completed_voxels)
            completed_voxels = completed_voxels > model_info["threshold"]
        
        # Convert to numpy
        completed_voxels = completed_voxels.squeeze().cpu().numpy()
        resolution = completed_voxels.shape[0]
        
        # Create sparse representation
        completed_sparse = scipy.sparse.dok_matrix(completed_voxels.shape, dtype=bool)
        completed_colors = {}
        
        # Find coordinates of completed voxels
        vy, vx, vz = np.where(completed_voxels)
        
        # Create color dictionary
        for y, x, z in zip(vy, vx, vz):
            completed_sparse[y, x, z] = True
            completed_colors[(y, x, z)] = [0.8, 0.8, 1.0]  # Default bluish color for predicted voxels
        
        # Blend with original partial voxels
        orig_res = voxel_data["resolution"]
        scale_factor = resolution / orig_res
        orig_voxels = voxel_data["voxels"]
        orig_colors = voxel_data["colors"]
        
        # Process original voxels
        for (y, x, z), _ in orig_voxels.items():
            ny, nx, nz = int(y * scale_factor), int(x * scale_factor), int(z * scale_factor)
            if 0 <= ny < resolution and 0 <= nx < resolution and 0 <= nz < resolution:
                # Skip if already filled by AI and random check passes
                if (ny, nx, nz) in completed_colors and np.random.random() < completion_strength:
                    continue
                
                completed_sparse[ny, nx, nz] = True
                if (y, x, z) in orig_colors:
                    completed_colors[(ny, nx, nz)] = orig_colors[(y, x, z)]
        
        # Update voxel data
        voxel_data["voxels"] = completed_sparse
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
        
        # Prepare Three.js data
        voxel_coords = []
        voxel_colors = []
        
        for (y, x, z), _ in voxels.items():
            voxel_coords.append([x, y, z])
            if (y, x, z) in colors:
                color = colors[(y, x, z)]
                voxel_colors.append(f"#{int(color[0]*255):02x}{int(color[1]*255):02x}{int(color[2]*255):02x}")
            else:
                voxel_colors.append("#8888ff")  # Default color
        
        # Generate unique ID for each preview
        preview_id = f"voxel-preview-{int(time.time()*1000)}"
        
        viewer_html = f"""
        <div id="{preview_id}" style="width:100%; height:400px;"></div>
        <script>
            function initVoxelPreview() {{
                const container = document.getElementById('{preview_id}');
                if(!container) return;
                
                // Remove existing canvas if any
                while(container.firstChild) {{
                    container.removeChild(container.firstChild);
                }}
                
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, container.clientWidth/container.clientHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(container.clientWidth, container.clientHeight);
                renderer.setClearColor(0x222222, 1);
                container.appendChild(renderer.domElement);
                
                // Add lights
                const light1 = new THREE.DirectionalLight(0xffffff, 1);
                light1.position.set(1, 1, 1);
                scene.add(light1);
                const light2 = new THREE.DirectionalLight(0xffffff, 0.5);
                light2.position.set(-1, -1, -1);
                scene.add(light2);
                const ambient = new THREE.AmbientLight(0x404040);
                scene.add(ambient);
                
                // Create voxels
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
                
                // Center camera
                camera.position.z = {resolution * 1.5};
                camera.position.y = {resolution * 0.5};
                
                // Add controls
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.enableDamping = true;
                controls.dampingFactor = 0.05;
                controls.rotateSpeed = 0.5;
                
                // Animation loop
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                
                // Handle resize
                function onWindowResize() {{
                    camera.aspect = container.clientWidth / container.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(container.clientWidth, container.clientHeight);
                }}
                
                window.addEventListener('resize', onWindowResize);
            }}
            
            // Initialize when Three.js is loaded
            if(typeof THREE !== 'undefined') {{
                initVoxelPreview();
            }} else {{
                // Load Three.js if not available
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
        
        # Create simple 2D preview
        preview_img = Image.new("RGB", (preview_size, preview_size), (40, 40, 40))
        preview_tensor = torch.from_numpy(np.array(preview_img)).float() / 255.0
        preview_tensor = preview_tensor.unsqueeze(0)
        
        return {
            "ui": {
                "images": [preview_tensor],
                "html": [viewer_html]
            }
        }

# Node mappings
NODE_CLASS_MAPPINGS = {
    "VoxelModelLoader": VoxelModelLoader,
    "DepthEstimationNode": DepthEstimationNode,
    "OptimizedVoxelizationNode": OptimizedVoxelizationNode,
    "ShapeCompletionNode": ShapeCompletionNode,
    "VoxelPreviewNode": VoxelPreviewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VoxelModelLoader": "Load Voxel Model",
    "DepthEstimationNode": "Depth Estimation",
    "OptimizedVoxelizationNode": "Voxelization (Optimized)",
    "ShapeCompletionNode": "3D Shape Completion",
    "VoxelPreviewNode": "Voxel Preview"
}