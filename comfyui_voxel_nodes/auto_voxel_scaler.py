import torch
import numpy as np
from PIL import Image
import os

# Define target resolution per category
CATEGORY_RESOLUTIONS = {
    "terrain": (64, 64),
    "building": (48, 64),
    "house": (48, 64),
    "tree": (24, 48),
    "plant": (24, 32),
    "flower": (16, 16),
    "character": (24, 32),
    "animal": (24, 32),
    "item": (16, 16),
    "sword": (16, 32),
    "shield": (24, 24),
    "armor": (24, 32),
    "gem": (8, 8),
    "coin": (8, 8),
    "decoration": (12, 12),
    "clutter": (12, 12),
    "grass": (16, 16),
}

def parse_prompt_category(prompt):
    prompt_lower = prompt.lower()
    for category in CATEGORY_RESOLUTIONS:
        if category in prompt_lower:
            return category
    return "item"  # default

def tensor_to_pil(tensor):
    if isinstance(tensor, torch.Tensor):
        array = tensor.squeeze().detach().cpu().numpy()
    else:
        array = tensor[0]
    if array.ndim == 2:
        return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))
    if array.shape[0] in (1, 3):
        array = np.moveaxis(array, 0, -1)
    return Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))

def resize_nearest(image, size):
    return image.resize(size, Image.Resampling.NEAREST)

def upscale_to_1024(image):
    return image.resize((1024, 1024), Image.Resampling.BICUBIC)

def save_images(pil_color, pil_depth, prompt, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    category = parse_prompt_category(prompt)
    base_name = category.capitalize()

    # Find next number
    n = 1
    while True:
        filename = f"{base_name}{n}"
        if not os.path.exists(os.path.join(output_dir, f"{filename}_color.png")):
            break
        n += 1

    pil_color.save(os.path.join(output_dir, f"{filename}_color.png"))
    pil_depth.save(os.path.join(output_dir, f"{filename}_depth.png"))

    # Save RGB composite for preview
    combined = Image.merge("RGB", (
        pil_color.convert("L"),
        pil_depth.convert("L"),
        pil_color.convert("L")
    ))
    combined.save(os.path.join(output_dir, f"{filename}_combined.png"))

class AutoVoxelScaler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth": ("IMAGE",),
                "prompt": ("STRING",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "scale"
    CATEGORY = "custom/voxel"

    def scale(self, image, depth, prompt):
        category = parse_prompt_category(prompt)
        target_res = CATEGORY_RESOLUTIONS.get(category, (16, 16))

        # Convert tensors to PIL and upscale to 1024x1024
        pil_color = upscale_to_1024(tensor_to_pil(image))
        pil_depth = upscale_to_1024(tensor_to_pil(depth))

        # Downscale both using NEAREST to preserve voxel edges
        resized_color = resize_nearest(pil_color, target_res)
        resized_depth = resize_nearest(pil_depth, target_res)

        # Save files
        save_images(resized_color, resized_depth, prompt)

        # Convert back to tensor format for output
        color_tensor = torch.from_numpy(np.array(resized_color).astype(np.float32) / 255.0).unsqueeze(0)
        depth_tensor = torch.from_numpy(np.array(resized_depth).astype(np.float32) / 255.0).unsqueeze(0)

        return (color_tensor, depth_tensor)
