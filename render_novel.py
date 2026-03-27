import torch
import torchvision
import numpy as np
import json
import os
import math
from argparse import ArgumentParser
from gaussian_renderer import render, GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov

OPENCV_CONVENTION = False

def load_cameras_from_json(cameras_json_path):
    with open(cameras_json_path) as f:
        data = json.load(f)

    c2w_list = data["camera_to_world"]
    intrinsics = data["camera_to_pixel"]
    image_sizes = data["image_size_xy"]
    cameras = []

    for idx in range(len(c2w_list)):
        c2w = np.array(c2w_list[idx])

        if OPENCV_CONVENTION:
            c2w[:3, 1:3] *= -1

        # Same transform as the Blender loader: OpenGL -> COLMAP
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        fx = intrinsics[idx][0][0]
        fy = intrinsics[idx][1][1]
        width = int(image_sizes[idx][0])
        height = int(image_sizes[idx][1])

        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)

        dummy_image = torch.zeros((3, height, width))

        cam = Camera(
            colmap_id=idx,
            R=R, T=T,
            FoVx=FoVx, FoVy=FoVy,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name=f"novel_{idx}",
            uid=idx,
        )
        cameras.append(cam)

    return cameras


def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True, type=str)
    parser.add_argument("--cameras_json", required=True, type=str)
    parser.add_argument("--output_dir", "-o", default=None, type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--kernel_size", default=0.1, type=float)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--opencv", action="store_true",
                        help="Set if cameras.json uses OpenCV/COLMAP convention")
    args = parser.parse_args()

    global OPENCV_CONVENTION
    OPENCV_CONVENTION = args.opencv

    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "novel_renders")
    os.makedirs(args.output_dir, exist_ok=True)

    if args.iteration == -1:
        from utils.system_utils import searchForMaxIteration
        iteration = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    else:
        iteration = args.iteration
    print(f"Loading model from iteration {iteration}")

    gaussians = GaussianModel(args.sh_degree)
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    cameras = load_cameras_from_json(args.cameras_json)
    print(f"Rendering {len(cameras)} novel views...")

    class FakePipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    with torch.no_grad():
        for idx, cam in enumerate(cameras):
            rendering = render(cam, gaussians, FakePipe(), background, kernel_size=args.kernel_size)["render"]
            rendering = torch.clamp(rendering, 0.0, 1.0)
            out_path = os.path.join(args.output_dir, f"{idx:05d}.png")
            torchvision.utils.save_image(rendering, out_path)
            print(f"  Saved {out_path} ({cam.image_width}x{cam.image_height})")

    print(f"Done. {len(cameras)} images saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
