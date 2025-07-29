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
from types import SimpleNamespace

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
            "threshold": 0.4,
            "in_channels": 3
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

    def load_midas_model(self, model_size):
        """Load MiDaS model with conflict resolution"""
        # Backup original sys.path
        original_sys_path = sys.path.copy()
        
        # Filter out problematic paths
        filtered_paths = [p for p in sys.path if "comfyui_controlnet_aux" not in p]
        sys.path = filtered_paths
        
        try:
            model = torch.hub.load("intel-isl/MiDaS", f"MiDaS_{model_size}")
        finally:
            # Restore original sys.path
            sys.path = original_sys_path
            
        return model

    def estimate_depth(self, image, model_size):
        if model_size not in self.models:
            try:
                model = self.load_midas_model(model_size)
            except Exception as e:
                print(f"Error loading MiDaS model: {e}")
                print("Falling back to simple depth estimation")
                model = None
                
            if model:
                model.eval()
                if torch.cuda.is_available():
                    model = model.cuda()
                self.models[model_size] = model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process first image in batch
        img = image[0].permute(2, 0, 1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=(384, 384), mode='bilinear', align_corners=False)
        img = img * 255.0
        img = img / 255.0
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        # If model failed to load, use simple grayscale conversion
        if model_size not in self.models or self.models[model_size] is None:
            depth_map = img.mean(dim=1, keepdim=True)  # Simple grayscale
        else:
            with torch.no_grad():
                img = img.to(device)
                prediction = self.models[model_size](img)

            # Normalize depth map
            depth_min = prediction.min()
            depth_max = prediction.max()
            depth_map = (prediction - depth_min) / (depth_max - depth_min)
        
        # Process depth map
        depth_map = depth_map.squeeze().cpu().numpy()
        
        # Convert to uint8 image
        depth_map = (depth_map * 255).astype(np.uint8)
        depth_map = Image.fromarray(depth_map)
        depth_map = depth_map.resize((image.shape[2], image.shape[1]))
        
        # Convert to tensor (H, W, 1)
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
        
        # Handle depth map shape - ensure it's 2D
        depth_data = depth_map[0].numpy()
        
        # Remove any extra dimensions
        while depth_data.ndim > 2 and depth_data.shape[-1] == 1:
            depth_data = depth_data[..., 0]
            
        # If it's still not 2D, try to convert to grayscale
        if depth_data.ndim != 2:
            # If it has 3 channels, convert to grayscale
            if depth_data.ndim == 3 and depth_data.shape[-1] == 3:
                depth_data = depth_data.mean(axis=-1)
            else:
                # For other cases, take the first channel
                depth_data = depth_data[..., 0]
        
        # Now ensure it's 2D
        depth_data = depth_data.squeeze()
        depth_data = depth_data * depth_scale
        
        # Get height and width
        H, W = depth_data.shape
        
        min_depth = depth_data.min()
        max_depth = depth_data.max()
        depth_range = max_depth - min_depth if max_depth > min_depth else 1.0
        
        # Use dictionary-based storage
        voxels_set = set()
        color_dict = {}
        
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
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_info):
        model_key = model_info["type"]
        if model_key not in self.models:
            config = model_info["config"]
            
            # Create configuration namespace expected by Pix2Vox
            cfg = SimpleNamespace()
            cfg.ENCODER = SimpleNamespace()
            cfg.ENCODER.DIM = config["encoder_dim"]
            cfg.ENCODER.IN_CHANNELS = config["in_channels"]  # Fix for input channels
            
            cfg.DECODER = SimpleNamespace()
            cfg.DECODER.DIM = config["decoder_dim"]
            
            cfg.REFINER = SimpleNamespace()
            cfg.REFINER.DIM = config["refiner_dim"]
            
            # Initialize model with configuration
            encoder = Encoder(cfg)  # Removed in_channels parameter
            decoder = Decoder(cfg)
            refiner = Refiner(cfg)
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
        
        # Create new voxel set
        completed_set = set()
        completed_colors = {}
        
        # Find coordinates of completed voxels
        vy, vx, vz = np.where(completed_voxels)
        for y, x, z in zip(vy, vx, vz):
            coord = (y, x, z)
            completed_set.add(coord)
            completed_colors[coord] = [0.8, 0.8, 1.0]  # Default bluish color
        
        # Blend with original partial voxels
        orig_res = voxel_data["resolution"]
        scale_factor = resolution / orig_res
        orig_voxels = voxel_data["voxels"]
        orig_colors = voxel_data["colors"]
        
        # Process original voxels
        for coord in orig_voxels:
            y, x, z = coord
            ny, nx, nz = int(y * scale_factor), int(x * scale_factor), int(z * scale_factor)
            new_coord = (ny, nx, nz)
            
            if (0 <= ny < resolution and 0 <= nx < resolution and 0 <= nz < resolution):
                # Skip if already filled by AI and random check passes
                if new_coord in completed_set and np.random.random() < completion_strength:
                    continue
                
                completed_set.add(new_coord)
                if coord in orig_colors:
                    completed_colors[new_coord] = orig_colors[coord]
        
        # Update voxel data
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
        
        # Prepare Three.js data
        voxel_coords = []
        voxel_colors = []
        
        for coord in voxels:
            y, x, z = coord
            voxel_coords.append([x, y, z])
            if coord in colors:
                color = colors[coord]
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
        
        # Convert to dense grid for meshing
        voxel_grid = np.zeros((resolution, resolution, resolution), dtype=bool)
        color_grid = np.zeros((resolution, resolution, resolution, 3))
        
        for coord in voxels:
            y, x, z = coord
            voxel_grid[y, x, z] = True
            if coord in colors:
                color_grid[y, x, z] = colors[coord]
        
        # Create mesh from voxels
        try:
            vertices, faces, vertex_colors = self.voxels_to_mesh(voxel_grid, color_grid)
        except ImportError:
            print("scikit-image not installed, using simple voxel meshing")
            vertices, faces, vertex_colors = self.simple_voxels_to_mesh(voxel_grid, color_grid)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
        
        # Apply mesh processing
        if smooth_mesh == "true":
            mesh = self.process_mesh(mesh, decimation_ratio)
        
        # Export to OBJ
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename
        filename = f"{filename_prefix}_{int(time.time())}.obj"
        file_path = os.path.join(output_dir, filename)
        
        # Export with vertex colors
        with open(file_path, 'w') as f:
            mesh.export(f, file_type='obj', include_color=True)
        
        # Create thumbnail preview
        preview = self.create_mesh_preview(mesh)
        preview_path = os.path.join(output_dir, filename.replace('.obj', '.png'))
        preview.save(preview_path)
        
        return {
            "ui": {
                "text": [f"Exported OBJ: {filename}"],
                "images": [{"filename": filename.replace('.obj', '.png'), "subfolder": "", "type": "output"}]
            }
        }
    
    def voxels_to_mesh(self, voxels, colors):
        """Convert voxel grid to mesh with vertex colors using marching cubes"""
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(voxels, level=0.5, spacing=(1,1,1))
        
        # Calculate vertex colors
        vertex_colors = []
        for vert in verts:
            x, y, z = int(vert[0]), int(vert[1]), int(vert[2])
            # Clip coordinates to grid bounds
            x = max(0, min(x, voxels.shape[0]-1))
            y = max(0, min(y, voxels.shape[1]-1))
            z = max(0, min(z, voxels.shape[2]-1))
            
            if voxels[x, y, z]:
                vertex_colors.append(colors[x, y, z])
            else:
                # If not found, use average of neighbors
                vertex_colors.append([0.8, 0.8, 1.0])
        
        return verts, faces, np.array(vertex_colors)
    
    def simple_voxels_to_mesh(self, voxels, colors):
        """Fallback meshing without marching cubes"""
        vertices = []
        faces = []
        vertex_colors = []
        
        # Create a cube for each voxel
        for z in range(voxels.shape[0]):
            for y in range(voxels.shape[1]):
                for x in range(voxels.shape[2]):
                    if voxels[z, y, x]:
                        # Define cube vertices
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
                        
                        # Define cube faces (triangles)
                        f = [
                            [0, 1, 2], [2, 3, 0],  # bottom
                            [4, 5, 6], [6, 7, 4],  # top
                            [0, 1, 5], [5, 4, 0],  # front
                            [2, 3, 7], [7, 6, 2],  # back
                            [0, 3, 7], [7, 4, 0],  # left
                            [1, 2, 6], [6, 5, 1]   # right
                        ]
                        
                        # Add to mesh
                        base_idx = len(vertices)
                        vertices.extend(v)
                        faces.extend([[base_idx + idx for idx in face] for face in f])
                        
                        # Add colors for each vertex
                        color = colors[z, y, x]
                        vertex_colors.extend([color] * 8)
        
        return np.array(vertices), np.array(faces), np.array(vertex_colors)
    
    def process_mesh(self, mesh, decimation_ratio):
        """Apply mesh smoothing and simplification"""
        try:
            # Smooth mesh
            trimesh.smoothing.filter_laplacian(mesh, iterations=3)
            
            # Simplify mesh
            target_faces = int(len(mesh.faces) * decimation_ratio)
            if target_faces > 100:  # Don't simplify too much
                mesh = mesh.simplify_quadratic_decimation(target_faces)
        except Exception as e:
            print(f"Mesh processing error: {e}")
        
        return mesh
    
    def create_mesh_preview(self, mesh):
        """Create 2D preview image of the mesh"""
        from matplotlib import pyplot as plt
        
        # Create a simple render without 3D projection
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Get bounding box
        min_x, min_y = mesh.bounds[0][0], mesh.bounds[0][1]
        max_x, max_y = mesh.bounds[1][0], mesh.bounds[1][1]
        size_x = max_x - min_x
        size_y = max_y - min_y
        max_size = max(size_x, size_y)
        
        # Create a blank image
        img_size = 256
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Project vertices to 2D
        vertices = mesh.vertices
        projected = vertices[:, :2]  # Just use X and Y coordinates
        
        # Normalize to image coordinates
        projected[:, 0] = (projected[:, 0] - min_x) / max_size * img_size
        projected[:, 1] = (projected[:, 1] - min_y) / max_size * img_size
        
        # Draw mesh edges
        for face in mesh.faces:
            points = projected[face]
            for i in range(3):
                start = points[i]
                end = points[(i + 1) % 3]
                start_x = int(start[0])
                start_y = img_size - int(start[1])  # Flip Y
                end_x = int(end[0])
                end_y = img_size - int(end[1])  # Flip Y
                
                # Draw line (this is simplified, use proper line drawing in production)
                plt.plot([start_x, end_x], [start_y, end_y], 'b-', alpha=0.3)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Create image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return Image.fromarray(img)

# Node mappings
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