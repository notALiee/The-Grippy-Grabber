import os

# Must be set before importing mujoco
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    from PIL import Image, ImageDraw
    HAS_CV2 = False


# -----------------------------
# YOLO class mapping
# -----------------------------
CLASS_MAP = {
    "PuzzleBase": 0,
    "PuzzleCircle": 1,
    "PuzzleSquare": 2,
    "PuzzleTriangle": 3,
    "cup1": 4,
    "cup2": 4,
    "cup3": 4,
    "book1": 5,
    "book2": 5,
    "book3": 5,
    "pen1": 6,
    "pen2": 6,
    "pen3": 6,
}

CLASS_NAMES = {
    0: "PuzzleBase",
    1: "PuzzleCircle",
    2: "PuzzleSquare",
    3: "PuzzleTriangle",
    4: "cup",
    5: "book",
    6: "pen",
}


# -----------------------------
# Table bounds in meters
# Your mm range:
# x: 250 to 250 - 925 = 250 to -675 mm
# y: -250 to 1350 - 250 = -250 to 1100 mm
# -----------------------------
DEFAULT_Y_MIN = -0.675
DEFAULT_Y_MAX = 0.250
DEFAULT_X_MIN = -0.250
DEFAULT_X_MAX = 1.100

DEFAULT_TABLE_Z = 0.0
DEFAULT_OBJECT_Z = 0.05


ARM_HIDE_KEYWORDS = (
    "panda",
    "franka",
    "robot",
    "arm",
    "gripper",
    "finger",
    "hand",
    "link",
    "joint",
)

TABLE_KEYWORDS = (
    "table",
    "desk",
    "surface",
    "platform",
)


@dataclass
class Target:
    name: str
    class_id: int
    qpos_adr: int
    geom_ids: np.ndarray


# -----------------------------
# Utility
# -----------------------------
def safe_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name.lower() if name else ""


def save_jpg(path: Path, rgb_img: np.ndarray, quality: int = 90):
    if HAS_CV2:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    else:
        Image.fromarray(rgb_img).save(path, quality=quality)


def yaw_quat(theta):
    """
    MuJoCo quaternion format: [w, x, y, z]
    This rotates around vertical z-axis only.
    Good for top-down table images.
    """
    half = theta / 2.0
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def random_color(rng, low=0.05, high=1.0):
    return rng.uniform(low, high, size=3)


# -----------------------------
# Sampling positions
# -----------------------------
def sample_positions_on_table(
    rng,
    n,
    x_min,
    x_max,
    y_min,
    y_max,
    min_spacing,
    edge_offset=0.1,
    max_attempts=5000,
):
    """
    Random non-overlapping XY positions on the table.
    """
    points = []
    
    x_min_eff = x_min + edge_offset
    x_max_eff = x_max - edge_offset
    y_min_eff = y_min + edge_offset
    y_max_eff = y_max - edge_offset

    for _ in range(n):
        accepted = False

        for _attempt in range(max_attempts):
            x = rng.uniform(x_min_eff, x_max_eff)
            y = rng.uniform(y_min_eff, y_max_eff)
            p = np.array([x, y], dtype=np.float64)

            if len(points) == 0:
                points.append(p)
                accepted = True
                break

            dists = [np.linalg.norm(p - q) for q in points]

            if min(dists) >= min_spacing:
                points.append(p)
                accepted = True
                break

        if not accepted:
            # Skip adding this object if we can't find a non-overlapping spot
            continue

    return np.array(points, dtype=np.float64)


# -----------------------------
# Model setup
# -----------------------------
def load_model_and_targets(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    targets = []

    for body_name, class_id in CLASS_MAP.items():
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id == -1:
            print(f"[WARN] Body not found: {body_name}")
            continue

        if model.body_jntnum[body_id] < 1:
            print(f"[WARN] Body has no joint/freejoint, skipping: {body_name}")
            continue

        if model.body_geomnum[body_id] < 1:
            print(f"[WARN] Body has no geom, skipping: {body_name}")
            continue

        jnt_id = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]

        geom_start = model.body_geomadr[body_id]
        geom_count = model.body_geomnum[body_id]
        geom_ids = np.arange(geom_start, geom_start + geom_count, dtype=np.int32)

        # Force direct geom color instead of shared material color
        model.geom_matid[geom_ids] = -1

        targets.append(
            Target(
                name=body_name,
                class_id=class_id,
                qpos_adr=qpos_adr,
                geom_ids=geom_ids,
            )
        )

    if len(targets) == 0:
        raise RuntimeError("No target objects found. Check CLASS_MAP body names.")

    geom_to_target = np.full(model.ngeom, -1, dtype=np.int32)
    target_class_ids = np.array([t.class_id for t in targets], dtype=np.int32)

    for target_index, target in enumerate(targets):
        geom_to_target[target.geom_ids] = target_index

    return model, data, targets, geom_to_target, target_class_ids


def hide_arm_and_extra_unlabeled_geoms(model, target_geom_ids):
    """
    Hides the Panda arm and also hides the unlabeled free cube if present.
    Your XML includes panda.xml and an extra unlabeled box body.
    Those are bad for clean YOLO data.
    """
    target_geom_ids = set(int(g) for g in target_geom_ids)
    hidden = []

    for geom_id in range(model.ngeom):
        if geom_id in target_geom_ids:
            continue

        body_id = model.geom_bodyid[geom_id]

        geom_name = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        combined = f"{geom_name} {body_name}"

        should_hide = False

        if any(k in combined for k in ARM_HIDE_KEYWORDS):
            should_hide = True

        # Hide anonymous/unlabeled small cube-like body from your XML
        if body_name == "" and geom_name == "":
            should_hide = True

        if should_hide:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id] = [0.0, 0.0, 0.0, 0.0]
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            hidden.append(geom_id)

    print(f"Hidden arm/unlabeled geoms: {len(hidden)}")


def find_table_geoms(model, target_geom_ids):
    target_geom_ids = set(int(g) for g in target_geom_ids)
    table_geoms = []

    for geom_id in range(model.ngeom):
        if geom_id in target_geom_ids:
            continue

        body_id = model.geom_bodyid[geom_id]
        geom_name = safe_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = safe_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        combined = f"{geom_name} {body_name}"

        if any(k in combined for k in TABLE_KEYWORDS):
            table_geoms.append(geom_id)

    table_geoms = np.array(table_geoms, dtype=np.int32)
    print(f"Detected table geoms: {table_geoms.tolist()}")

    return table_geoms


# -----------------------------
# Camera
# -----------------------------
def init_gimbal_camera(args):
    """
    Initializes a free camera looking at the center of the table.
    We will randomize its azimuth, elevation, and distance per frame.
    """
    center_x = (args.x_min + args.x_max) / 2.0
    center_y = (args.y_min + args.y_max) / 2.0

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    cam.lookat[:] = np.array(
        [center_x, center_y, args.camera_lookat_z],
        dtype=np.float64,
    )

    return cam


# -----------------------------
# Randomize scene
# -----------------------------
def randomize_scene(
    model,
    data,
    targets,
    table_geoms,
    rng,
    args,
):
    mujoco.mj_resetData(model, data)

    positions = sample_positions_on_table(
        rng=rng,
        n=len(targets),
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        min_spacing=args.min_spacing,
    )

    # Random table color
    if len(table_geoms) > 0:
        table_color = random_color(rng, low=0.15, high=0.95)

        for geom_id in table_geoms:
            model.geom_matid[geom_id] = -1
            model.geom_rgba[geom_id, :3] = table_color
            model.geom_rgba[geom_id, 3] = 1.0

    # Random object positions, z height, yaw rotation, colors
    for idx, target in enumerate(targets):
        adr = target.qpos_adr

        if idx >= len(positions):
            # Hide object that didn't fit (put far away under the table)
            data.qpos[adr] = 100.0
            data.qpos[adr + 1] = 100.0
            data.qpos[adr + 2] = -10.0
            continue

        x, y = positions[idx]

        data.qpos[adr] = x
        data.qpos[adr + 1] = y

        # Force object z to table spawn height.
        # This is what prevents the objects from appearing like they are falling.
        data.qpos[adr + 2] = args.object_z

        if target.name in ("PuzzleCircle", "PuzzleSquare", "PuzzleTriangle"):
            # 90 degrees around the X-axis (w=cos(45deg), x=sin(45deg), y=0, z=0)
            data.qpos[adr + 3:adr + 7] = [0.70710678, 0.70710678, 0.0, 0.0]
        else:
            theta = rng.uniform(0.0, 2.0 * np.pi)
            data.qpos[adr + 3:adr + 7] = yaw_quat(theta)

        obj_color = random_color(rng, low=0.05, high=1.0)
        model.geom_rgba[target.geom_ids, :3] = obj_color
        model.geom_rgba[target.geom_ids, 3] = 1.0

    # Random lighting
    if model.nlight > 0:
        model.light_diffuse[:] = rng.uniform(0.35, 1.0, size=(model.nlight, 3))

    mujoco.mj_forward(model, data)


# -----------------------------
# Segmentation -> YOLO labels
# -----------------------------
def labels_from_segmentation(
    seg_img,
    geom_to_target,
    target_class_ids,
    width,
    height,
    min_area,
):
    """
    MuJoCo segmentation image:
    channel 0 = object id
    channel 1 = object type
    """
    seg_ids = seg_img[:, :, 0]
    seg_types = seg_img[:, :, 1]

    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)

    valid = (
        (seg_types == geom_type)
        & (seg_ids >= 0)
        & (seg_ids < len(geom_to_target))
    )

    ys, xs = np.nonzero(valid)

    if len(xs) == 0:
        return []

    target_ids = geom_to_target[seg_ids[ys, xs]]
    keep = target_ids >= 0

    if not np.any(keep):
        return []

    xs = xs[keep]
    ys = ys[keep]
    target_ids = target_ids[keep]

    n_targets = len(target_class_ids)

    x_min = np.full(n_targets, width, dtype=np.int32)
    y_min = np.full(n_targets, height, dtype=np.int32)
    x_max = np.full(n_targets, -1, dtype=np.int32)
    y_max = np.full(n_targets, -1, dtype=np.int32)
    area = np.zeros(n_targets, dtype=np.int32)

    np.minimum.at(x_min, target_ids, xs)
    np.minimum.at(y_min, target_ids, ys)
    np.maximum.at(x_max, target_ids, xs)
    np.maximum.at(y_max, target_ids, ys)
    np.add.at(area, target_ids, 1)

    labels = []

    for target_id in np.where(area >= min_area)[0]:
        bw_px = x_max[target_id] - x_min[target_id] + 1
        bh_px = y_max[target_id] - y_min[target_id] + 1

        if bw_px <= 1 or bh_px <= 1:
            continue

        x_center = (x_min[target_id] + x_max[target_id] + 1) / 2.0 / width
        y_center = (y_min[target_id] + y_max[target_id] + 1) / 2.0 / height
        bbox_width = bw_px / width
        bbox_height = bh_px / height

        class_id = int(target_class_ids[target_id])

        labels.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} "
            f"{bbox_width:.6f} {bbox_height:.6f}"
        )

    return labels


# -----------------------------
# Draw debug boxes
# -----------------------------
def draw_yolo_boxes(rgb_img, labels, width, height):
    img = rgb_img.copy()

    if HAS_CV2:
        for line in labels:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            label = CLASS_NAMES.get(class_id, str(class_id))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return img

    else:
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)

        for line in labels:
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            label = CLASS_NAMES.get(class_id, str(class_id))

            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, max(0, y1 - 12)), label, fill=(0, 255, 0))

        return np.array(pil_img)


# -----------------------------
# Main generation
# -----------------------------
def generate_dataset(args):
    rng = np.random.default_rng(args.seed)

    output_dir = Path(args.output_dir)
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    debug_dir = output_dir / "debug_boxes"

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model_path)
    model, data, targets, geom_to_target, target_class_ids = load_model_and_targets(
        args.model_path
    )

    target_geom_ids = np.concatenate([t.geom_ids for t in targets])

    hide_arm_and_extra_unlabeled_geoms(model, target_geom_ids)
    table_geoms = find_table_geoms(model, target_geom_ids)

    renderer = mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
    )

    render_camera = init_gimbal_camera(args)

    saved = 0
    start_time = time.time()

    for i in range(args.num_images):
        randomize_scene(
            model=model,
            data=data,
            targets=targets,
            table_geoms=table_geoms,
            rng=rng,
            args=args,
        )

        # Randomize camera gimbal to mimic arm perspective across the table
        render_camera.azimuth = rng.uniform(0.0, 360.0)
        render_camera.elevation = rng.uniform(-89.0, -30.0) # -89 is top-down, -30 is low angle
        render_camera.distance = rng.uniform(0.8, args.camera_height) # Scale between 0.8m and max height

        # RGB render
        renderer.disable_segmentation_rendering()
        renderer.update_scene(data, camera=render_camera)
        rgb_img = renderer.render()

        # Segmentation render
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=render_camera)
        seg_img = renderer.render()

        labels = labels_from_segmentation(
            seg_img=seg_img,
            geom_to_target=geom_to_target,
            target_class_ids=target_class_ids,
            width=args.width,
            height=args.height,
            min_area=args.min_area,
        )

        if labels:
            img_path = img_dir / f"sim_{i:06d}.jpg"
            lbl_path = lbl_dir / f"sim_{i:06d}.txt"

            save_jpg(img_path, rgb_img, quality=args.jpg_quality)
            lbl_path.write_text("\n".join(labels) + "\n")

            if saved < args.debug_boxes:
                boxed = draw_yolo_boxes(
                    rgb_img=rgb_img,
                    labels=labels,
                    width=args.width,
                    height=args.height,
                )
                debug_path = debug_dir / f"sim_{i:06d}_boxed.jpg"
                save_jpg(debug_path, boxed, quality=args.jpg_quality)

            saved += 1

        if i % 100 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / max(elapsed, 1e-9)
            print(f"{i + 1}/{args.num_images} | {fps:.2f} img/s | saved={saved}")

    elapsed = time.time() - start_time
    print()
    print("Dataset generation complete.")
    print(f"Saved images: {saved}/{args.num_images}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Saved img/s: {saved / max(elapsed, 1e-9):.2f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", default="../panda_mujoco/world.xml")
    parser.add_argument("--output-dir", default="dataset")
    parser.add_argument("--num-images", type=int, default=5000)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--jpg-quality", type=int, default=90)
    parser.add_argument("--min-area", type=int, default=20)

    # Table placement
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN)
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX)
    parser.add_argument("--y-min", type=float, default=DEFAULT_Y_MIN)
    parser.add_argument("--y-max", type=float, default=DEFAULT_Y_MAX)

    # Object spawn height
    parser.add_argument("--object-z", type=float, default=DEFAULT_OBJECT_Z)
    parser.add_argument("--min-spacing", type=float, default=0.075)

    # Camera looking down at table center
    parser.add_argument("--camera-height", type=float, default=1.70)
    parser.add_argument("--camera-lookat-z", type=float, default=DEFAULT_TABLE_Z)
    parser.add_argument("--camera-azimuth", type=float, default=0.0)

    # -89 is almost directly top-down.
    # Use -75 for a more angled view.
    parser.add_argument("--camera-elevation", type=float, default=-89.0)

    # Debug visualizations
    parser.add_argument("--debug-boxes", type=int, default=100)

    args = parser.parse_args()

    print("MuJoCo GL:", os.environ.get("MUJOCO_GL"))
    print("OpenCV saving:", HAS_CV2)
    print("Table center:", ((args.x_min + args.x_max) / 2.0, (args.y_min + args.y_max) / 2.0))
    print("Object z:", args.object_z)
    print("Camera height:", args.camera_height)
    print("Camera elevation:", args.camera_elevation)

    generate_dataset(args)


if __name__ == "__main__":
    main()