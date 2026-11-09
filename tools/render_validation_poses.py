"""Render the exact saved validation inputs, historical poses, and predictions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
from pathlib import Path
import textwrap

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def yaw_align_prediction(predicted, reference):
    """Choose the free global-Z yaw that best overlays corresponding vertices."""
    predicted_center = predicted.mean(axis=0)
    reference_center = reference.mean(axis=0)
    predicted_xy = predicted[:, :2] - predicted_center[:2]
    reference_xy = reference[:, :2] - reference_center[:2]
    cosine = float(np.sum(predicted_xy * reference_xy))
    sine = float(np.sum(
        predicted_xy[:, 0] * reference_xy[:, 1]
        - predicted_xy[:, 1] * reference_xy[:, 0]
    ))
    angle = np.arctan2(sine, cosine)
    c, s = np.cos(angle), np.sin(angle)
    yaw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return (predicted - predicted_center) @ yaw.T + reference_center, yaw


def render_part(job):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    font_path = Path("/mnt/c/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    mesh = trimesh.load(job["stl_path"], force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    faces = np.asarray(mesh.faces)
    augmentation = np.asarray(job["augmentation_rotation"])
    correction = np.asarray(job["prediction_rotation"])
    inputs = vertices @ augmentation.T
    predicted_pose, display_yaw = yaw_align_prediction(inputs @ correction.T, vertices)
    poses = [inputs, vertices, predicted_pose]
    input_up = np.array([0., 0., 1.])
    marker_directions = [input_up, augmentation.T @ input_up, display_yaw @ correction @ input_up]
    historical_xy_center = (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0)) / 2
    for index, pose in enumerate(poses):
        xy_center = (
            (pose[:, :2].min(axis=0) + pose[:, :2].max(axis=0)) / 2
            if index == 0 else historical_xy_center
        )
        pose[:, :2] -= xy_center
        pose[:, 2] -= pose[:, 2].min()
    side = max(float(np.ptp(pose, axis=0).max()) for pose in poses) * 1.18
    if np.min(poses[1][:, 2]) < -1e-10 or np.min(poses[2][:, 2]) < -1e-10:
        raise ValueError("Grounded GT/prediction contains vertices below the z=0 platform")
    figure = plt.figure(figsize=(16, 6.1), facecolor="white")
    titles = ["输入：随机初始位姿", "GT：数据集历史打印位姿", "预测：模型输出后的位姿"]
    title_colors = ["#325a91", "#27704c", "#9b531b"]
    light = np.array([-0.4, -0.6, 0.8])
    light /= np.linalg.norm(light)
    for column, (pose, title, title_color) in enumerate(zip(poses, titles, title_colors)):
        axis = figure.add_subplot(1, 3, column + 1, projection="3d", computed_zorder=False)
        triangles = pose[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        brightness = 0.43 + 0.57 * np.clip(normals @ light, 0, 1)
        colors = brightness[:, None] * np.array([0.75, 0.81, 0.89])[None, :]
        # The input view intentionally has no platform. GT and prediction are
        # shown on the same semi-transparent build plane for direct comparison.
        if column > 0:
            plane = np.array([[-side / 2, -side / 2, 0.0],
                              [side / 2, -side / 2, 0.0],
                              [side / 2, side / 2, 0.0],
                              [-side / 2, side / 2, 0.0]])
            axis.add_collection3d(Poly3DCollection(
                [plane], facecolor="#e6eaf0", edgecolor="#9da8b7",
                linewidth=0.7, alpha=0.60, zorder=0,
            ))
            for tick in np.linspace(-side / 2, side / 2, 7):
                axis.plot([-side / 2, side / 2], [tick, tick], [0.0] * 2,
                          color="#aeb8c5", linewidth=.4, alpha=.55, zorder=1)
                axis.plot([tick, tick], [-side / 2, side / 2], [0.0] * 2,
                          color="#aeb8c5", linewidth=.4, alpha=.55, zorder=1)
        axis.add_collection3d(Poly3DCollection(
            triangles, facecolors=colors, edgecolors="none", linewidths=0,
            antialiased=False, zsort="average", zorder=2,
        ))
        marker = marker_directions[column]
        marker = marker / np.linalg.norm(marker)
        arrow_origin = pose.mean(axis=0)
        arrow_length = side * .32
        arrow_end = arrow_origin + marker * arrow_length
        axis.quiver(
            *arrow_origin, *(marker * arrow_length),
            color="white", linewidth=5.5, arrow_length_ratio=.24,
            normalize=False, zorder=5,
        )
        axis.quiver(
            *arrow_origin, *(marker * arrow_length),
            color="#f04432", linewidth=3.0, arrow_length_ratio=.24,
            normalize=False, zorder=6,
        )
        axis.text(
            *arrow_end, "初始 Z 标记", fontproperties=font, fontsize=9,
            color="#c52e22", weight="bold",
            bbox={"facecolor": "white", "edgecolor": "#f04432", "alpha": .9, "pad": 2.5},
            zorder=7,
        )
        origin = np.array([-side * .44, -side * .44, 0])
        for direction, color, label in zip(np.eye(3), ["#b65252", "#4a8659", "#4266a4"], ["X", "Y", "Z"]):
            endpoint = origin + direction * side * .15
            axis.quiver(*origin, *(direction * side * .15), color=color, linewidth=1.2, arrow_length_ratio=.15)
            axis.text(*endpoint, label, color=color, fontsize=8)
        axis.set(xlim=(-side / 2, side / 2), ylim=(-side / 2, side / 2), zlim=(-side * .04, side))
        axis.set_box_aspect((1, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=38, azim=-58)
        axis.set_axis_off()
        axis.set_title(title, fontproperties=font, color=title_color, fontsize=13, pad=-8)
        subtitle = ["固定验证输入：第 1 / 8 个随机朝向（无平台）", "历史 GT：最低点落到平台", f"Pred：自由 yaw 对齐 GT；倾斜差 {job['tilt_error_deg']:.2f}°"]
        axis.text2D(.5, .015, subtitle[column], transform=axis.transAxes,
                    ha="center", fontproperties=font, fontsize=10, color="#444444")
    figure.text(.025, .965, f"{job['number']:02d} / 20  ·  record {job['record_id']}  ·  part {job['source_part_id']}",
                fontsize=14, weight="bold", va="top", color="#202b3b")
    figure.text(.025, .914, "\n".join(textwrap.wrap(job["source_part_name"], 112)),
                fontproperties=font, fontsize=9, va="top", color="#555555")
    figure.text(.5, .045, "红箭头 = 输入坐标系的初始 +Z，固定在零件上并随零件旋转；它是方向标记，不是模型预测方向。",
                ha="center", fontproperties=font, fontsize=10, color="#c52e22")
    figure.text(.5, .012, "三栏相同相机、相同比例；GT/Pred 使用 40% 透明平台。Pred 仅为可视比较绕 Z 对齐 GT；误差只计打印方向倾斜，不计 yaw。",
                ha="center", fontproperties=font, fontsize=9, color="#555555")
    figure.subplots_adjust(left=.005, right=.995, bottom=.08, top=.85, wspace=-.04)
    figure.savefig(job["image_path"], dpi=120, facecolor="white")
    plt.close(figure)
    return job["number"], job["image_path"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/utonia_tiny_direction_ablation/with_normals_r64")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation_pose_comparison")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    configuration = json.loads((arguments.run / "config.json").read_text())
    split = json.loads((arguments.run / "split.json").read_text())
    records = json.loads((ROOT / "outputs/pointclouds/manifest.json").read_text())["records"]
    mesh_records = json.loads((ROOT / "outputs/meshes_stl/manifest.json").read_text())["records"]
    meshes = {(record["source_project"], record["source_part_id"]): record for record in mesh_records}
    with np.load(arguments.run / "validation_predictions.npz") as saved:
        predicted, targets, ids = saved["predicted"], saved["target"], saved["record_ids"]
    # The model is judged only on its chosen build-up vector. An output-space
    # yaw changes neither this vector error nor the part's tilt to the plate.
    angles = np.rad2deg(np.arccos(np.clip(np.sum(predicted * targets, axis=1), -1.0, 1.0)))
    validation_metrics = {
        "mean_tilt_error_deg": float(angles.mean()),
        "median_tilt_error_deg": float(np.median(angles)),
        "p90_tilt_error_deg": float(np.percentile(angles, 90)),
        "within_30_deg": int(np.sum(angles <= 30)),
        "validation_directions": int(len(angles)),
        "yaw_is_ignored": True,
    }
    # Import only in the parent: rendering workers do not need PyTorch.
    from orientation_model import align_up_rotation

    jobs = []
    for number, record_id in enumerate(split["validation_indices"], 1):
        positions = np.flatnonzero(ids == record_id)
        assert len(positions) == configuration["validation_rotations"]
        position = int(positions[0])
        rotation = Rotation.random(random_state=np.random.default_rng(
            configuration["seed"] + 1000000007 + record_id * 1000003
        )).as_matrix()
        np.testing.assert_allclose(targets[position], rotation @ np.array([0., 0., 1.]), atol=1e-6)
        direction = predicted[position]
        correction = align_up_rotation(direction)
        np.testing.assert_allclose(correction @ (direction / np.linalg.norm(direction)), [0, 0, 1], atol=1e-6)
        error = float(np.rad2deg(np.arccos(np.clip(np.dot(direction, targets[position]), -1.0, 1.0))))
        record = records[record_id]
        mesh = meshes[(record["source_project"], record["source_part_id"])]
        filename = f"{number:02d}_record_{record_id:03d}_part_{record['source_part_id']}.png"
        jobs.append({
            "number": number, "record_id": record_id, "validation_sample_index": position,
            "validation_rotation_index": 0, "source_project": record["source_project"],
            "source_part_id": record["source_part_id"], "source_part_name": record["source_part_name"],
            "stl_path": str(ROOT / "outputs/meshes_stl" / mesh["stl_file"]),
            "image_path": str(arguments.output / filename), "image_file": filename,
            "augmentation_rotation": rotation.tolist(), "prediction_rotation": correction.tolist(),
            "target_direction_input_frame": targets[position].tolist(),
            "predicted_direction_input_frame": direction.tolist(), "tilt_error_deg": error,
        })
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        for number, path in executor.map(render_part, jobs):
            print(f"Rendered {number:02d}/{len(jobs)}: {Path(path).name}", flush=True)
    document = {"run": str(arguments.run), "selection": "First fixed validation rotation of every held-out part; saved best-checkpoint predictions.",
                "direction_marker": "Red arrow tracks the input-frame +Z as an object-attached marker through GT restoration and prediction, including display yaw; it is not a predicted or GT build-up axis.",
                "validation_direction_metrics": validation_metrics,
                "views": ["Random validation input (no platform)", "Historical dataset orientation on a 40% transparent platform", "Predicted orientation yaw-aligned to GT for visualization, on a 40% transparent platform"],
                "samples": jobs}
    (arguments.output / "manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = []
    for job in jobs:
        cards.append(f'<article id="part-{job["number"]}"><a href="{job["image_file"]}" target="_blank"><img loading="lazy" src="{job["image_file"]}" alt="{html.escape(job["source_part_name"])}"></a></article>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>验证集 20 个零件 · 位姿对比</title><style>body{margin:24px;background:#eef1f5;color:#233044;font:16px system-ui}main{max-width:1600px;margin:auto}h1{font-size:24px}p{line-height:1.6}article{background:white;margin:20px 0;box-shadow:0 2px 10px #0001}img{display:block;width:100%;height:auto}</style><main><h1>验证集 20 个零件：输入 / 历史 GT / 预测</h1>'
             f'<p>模型：法向 + 64 朝向（with_normals_r64），最佳 checkpoint。20 个验证零件，每个取固定的第一个验证朝向；点击图片可打开原图。输入栏无平台；GT 和 Pred 栏为 40% 透明平台。为公平看倾斜误差，Pred 仅在图上绕平台法向 yaw 对齐 GT（不是模型输出）。夹角指标只比较打印方向，绕 Z 的 yaw 不计：160 个验证方向平均 {validation_metrics["mean_tilt_error_deg"]:.1f}°、中位数 {validation_metrics["median_tilt_error_deg"]:.1f}°、90 分位 {validation_metrics["p90_tilt_error_deg"]:.1f}°，30° 内 {validation_metrics["within_30_deg"]}/{validation_metrics["validation_directions"]}。</p>'
             + '<p style="color:#c52e22">红箭头表示输入姿态的初始 +Z 方向，同一标记随零件一起旋转，用于比较旋转前后的变化；它不是模型预测的打印方向。角落的小 XYZ 轴表示固定的世界坐标。</p>'
             + '\n'.join(cards) + '</main></html>')
    (arguments.output / "index.html").write_text(page, encoding="utf-8")
    from PIL import Image, ImageOps

    thumbnails = []
    for job in jobs:
        with Image.open(job["image_path"]) as image:
            thumbnails.append(ImageOps.contain(image.convert("RGB"), (960, 366)))
    overview = Image.new("RGB", (1920, 366 * 10), "#e9edf2")
    for index, thumbnail in enumerate(thumbnails):
        overview.paste(thumbnail, ((index % 2) * 960, (index // 2) * 366))
    overview.save(arguments.output / "overview.jpg", quality=90)
    print(f"Complete: {len(jobs)} triptychs; gallery={arguments.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
