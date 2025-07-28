import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class VoxelPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxels": ("NP_ARRAY",),
                "view": (["isometric", "top", "front", "side"], {"default": "isometric"}),
                "point_size": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 10.0}),
                "bg_color": ("STRING", {"default": "#222222"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview",)
    FUNCTION = "run"
    CATEGORY = "VoxelTools"

    def run(self, voxels, view="isometric", point_size=4.0, bg_color="#222222"):
        try:
            arr = np.array(voxels)
            assert arr.ndim == 4 and arr.shape[3] == 3, "Voxels should be (X,Y,Z,3) RGB"

            filled = np.any(arr != 0, axis=3)
            xs, ys, zs = np.where(filled)
            if len(xs) == 0:
                print("⚠️ No voxels to preview.")
                # Return a blank image
                img = Image.new("RGB", (256, 256), color=bg_color)
                return (np.array(img),)

            colors = arr[xs, ys, zs] / 255.0

            # Project 3D to 2D for preview
            if view == "isometric":
                px = xs - zs
                py = xs + zs - 2*ys
            elif view == "top":
                px, py = xs, zs
            elif view == "front":
                px, py = xs, ys
            elif view == "side":
                px, py = zs, ys
            else:
                px, py = xs, ys

            px, py = px.astype(np.float32), py.astype(np.float32)
            px -= px.min()
            py -= py.min()
            s = max(px.max(), py.max())
            if s > 0:
                px = px / s * 240 + 8
                py = py / s * 240 + 8

            fig, ax = plt.subplots(figsize=(3, 3), dpi=85)
            ax.set_facecolor(bg_color)
            ax.scatter(px, py, c=colors, s=point_size)
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=False)
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            arr_img = np.array(img)
            return (arr_img,)

        except Exception as e:
            print(f"❌ VoxelPreview failed: {e}")
            img = Image.new("RGB", (256, 256), color="#ff0000")
            return (np.array(img),)