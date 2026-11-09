"""Render examples from the public LPBF part-orientation dataset.

The paper's embedded Figure 1 is extracted from the local paper PDF. Actual
STLs in the downloaded pyramid archive are rendered in both supplied random
input pose and the labeled reference pose (row vertices @ target matrix).
"""
from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "datasets/public_pbflbm/pyramids_10000.zip"
PAPER = Path(r"C:\Users\panxin\Desktop\3d_bench\s40964-025-00960-6.pdf")
OUT = ROOT / "outputs/public_dataset_preview"

SAMPLES = [
    "pyramids_0003258",  # tall, thin
    "pyramids_0003995",  # short, broad
    "pyramids_0000636",  # high triangle count
    "pyramids_0000000",  # moderate example
]


def find_chinese_font() -> str | None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in available), None)


def parse_binary_stl(blob: bytes) -> np.ndarray:
    """Return triangle vertex array, shape (n, 3, 3)."""
    n = struct.unpack_from("<I", blob, 80)[0]
    expected = 84 + n * 50
    if expected != len(blob):
        raise ValueError(f"Expected binary STL of {expected} bytes; got {len(blob)}")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(blob, dtype=dtype, count=n, offset=84)["vertices"].astype(np.float64)


def platform_panel(ax, tris: np.ndarray, title: str, color: str) -> None:
    vertices = tris.reshape(-1, 3)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    center = (lo + hi) / 2
    extent = hi - lo
    scale = max(float(extent.max()), 1e-9)
    # Fit each model in its panel while retaining the true proportions.
    tris = (tris - center) / scale
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)
    tris[:, :, 2] -= lo[2]
    lo = tris.reshape(-1, 3).min(axis=0)
    hi = tris.reshape(-1, 3).max(axis=0)

    mesh = Poly3DCollection(
        tris,
        facecolors=color,
        edgecolors=(0.13, 0.22, 0.30, 0.23),
        linewidths=0.20,
        antialiased=True,
    )
    ax.add_collection3d(mesh)
    pad = 0.11
    span = max(float(max(hi[0] - lo[0], hi[1] - lo[1])), 0.5)
    zfloor = -0.025
    ax.plot(
        [lo[0] - pad, hi[0] + pad],
        [lo[1] - pad, lo[1] - pad],
        [0, 0], color="#364654", linewidth=1.1,
    )
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_zlim(zfloor, hi[2] + pad)
    ax.set_box_aspect((hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, hi[2] + pad - zfloor))
    ax.set_proj_type("ortho")
    ax.view_init(elev=23, azim=-53)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=1)


def extract_paper_figure() -> None:
    reader = PdfReader(str(PAPER))
    page = next(page for page in reader.pages if "Example geometries" in (page.extract_text() or ""))
    image = next(image for image in page.images if image.image.width > 1000)
    image.image.convert("RGB").save(OUT / "paper_figure1_four_families.png", quality=95)


def render_pyramids() -> None:
    with zipfile.ZipFile(ARCHIVE) as archive:
        fig = plt.figure(figsize=(14, 10), facecolor="#f7f9fb")
        fig.suptitle("真实数据集 STL 样本：随机输入姿态 → 标签参考姿态", fontsize=16, y=0.985)
        for row, sample in enumerate(SAMPLES):
            info = json.loads(archive.read(f"info/{sample}.json"))
            mesh_path = next(
                name for name in archive.namelist()
                if name.startswith("stl/") and name.endswith(f"/{sample}.stl")
            )
            tris = parse_binary_stl(archive.read(mesh_path))
            matrix = np.asarray(info["target_rotational_matrix"], dtype=float)
            # The archive labels row-vector vertices: V_ref = V_input @ M.
            corrected = tris @ matrix

            ax0 = fig.add_subplot(4, 2, row * 2 + 1, projection="3d")
            ax1 = fig.add_subplot(4, 2, row * 2 + 2, projection="3d")
            params = (
                f"h={info['height']}, r={info['radius']}; "
                f"三角面={info['num_triangles']}, 顶点={info['num_vertices']}"
            )
            platform_panel(ax0, tris, f"{sample} · 原始随机姿态\n{params}", "#77a9d1")
            platform_panel(ax1, corrected, "标签参考姿态 · 尖端朝平台", "#df9563")

        fig.text(
            0.5, 0.012,
            "蓝色为下载的原始 STL；橙色为应用样本自带旋转标签后的同一网格。每个小图独立缩放以便观察形状；比例不变。",
            ha="center", fontsize=9, color="#394754",
        )
        fig.tight_layout(rect=(0.02, 0.035, 0.98, 0.96), h_pad=0.5, w_pad=0.15)
        fig.savefig(OUT / "pyramid_stl_random_vs_reference.png", dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    font = find_chinese_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    extract_paper_figure()
    render_pyramids()
    print(f"Rendered to {OUT}")


if __name__ == "__main__":
    main()
