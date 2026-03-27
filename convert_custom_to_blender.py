import json
import math
import os
import shutil
import numpy as np

CUSTOM_DIR = os.path.join(os.path.dirname(__file__), "custom_Dataset", "inputs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "custom_Dataset_blender")

OPENCV_CONVENTION = False

def main():
    with open(os.path.join(CUSTOM_DIR, "metadata.json"), "r") as f:
        meta = json.load(f)

    camera = meta["camera"]
    c2w_list = camera["camera_to_world"]
    intrinsics = camera["camera_to_pixel"]
    image_sizes = camera["image_size_xy"]
    num_views = len(c2w_list)

    fx = intrinsics[0][0][0]
    width = image_sizes[0][0]
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames = []
    for i in range(num_views):
        c2w = np.array(c2w_list[i])

        if OPENCV_CONVENTION:
            # Custom dataset uses OpenCV/COLMAP convention (Y down, Z forward).
            # Mip-splatting's Blender loader will flip Y,Z columns internally,
            # so pre-flip here so the double-flip produces the correct result.
            c2w[:3, 1:3] *= -1

        src = os.path.join(CUSTOM_DIR, f"rgb_{i}.png")
        dst = os.path.join(OUTPUT_DIR, f"rgb_{i}.png")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

        frames.append({
            "file_path": f"rgb_{i}",
            "transform_matrix": c2w.tolist()
        })

    train_frames = frames[:26]   # rgb_0 .. rgb_25
    test_frames  = frames[26:]   # rgb_26 .. rgb_28

    transforms_train = {
        "camera_angle_x": camera_angle_x,
        "frames": train_frames
    }
    transforms_test = {
        "camera_angle_x": camera_angle_x,
        "frames": test_frames
    }

    with open(os.path.join(OUTPUT_DIR, "transforms_train.json"), "w") as f:
        json.dump(transforms_train, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "transforms_test.json"), "w") as f:
        json.dump(transforms_test, f, indent=2)

    print(f"Converted {num_views} views -> {OUTPUT_DIR}/")
    print(f"  train: {len(train_frames)} views (rgb_0 .. rgb_25)")
    print(f"  test:  {len(test_frames)} views (rgb_26 .. rgb_28)")
    print(f"  camera_angle_x = {camera_angle_x:.6f} rad ({math.degrees(camera_angle_x):.1f} deg)")
    print(f"  fx={fx}, image_width={width}")
    print(f"  OPENCV_CONVENTION={OPENCV_CONVENTION}")

if __name__ == "__main__":
    main()
