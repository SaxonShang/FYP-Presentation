#!/usr/bin/env python3
"""Generate matched DAVE scene and NAS-GS point-cloud preview views.

The slide image uses the same top, side, and oblique projections for the
simulated DAVE target geometry and the learned NAS-GS Gaussian centres. This
keeps the visual comparison one-to-one instead of mixing unrelated screenshots.
"""

from __future__ import annotations

import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SCENE_SDF = ROOT / "uuv_ws/src/fyp_nasgs_slam/models/nasgs_151216_tall_scene/model.sdf"
PLY_PATH = (
    ROOT
    / "fyp_runs/ply_previews/passA_nasgs_map_iteration1000_downsampled_xyz_60k.ply"
)
OUT_DIR = ROOT / "fyp_runs/ply_previews"
PRESENTATION_DIR = ROOT / "FYP-Presentation/Images/fyp"

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

BG = "#f8fafc"
INK = "#111827"
MUTED = "#334155"
GRID = "#dbe6f2"
CARD = "#ffffff"
CARD_BORDER = "#b8c7d9"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


TITLE = font(36, True)
SUBTITLE = font(22)
CARD_TITLE = font(22, True)
SMALL = font(16)
TINY = font(13)


@dataclass(frozen=True)
class Primitive:
    name: str
    kind: str
    pose: tuple[float, float, float, float, float, float]
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.0
    length: float = 0.0


Point = tuple[float, float, float]
Rgb = tuple[int, int, int]


def parse_pose(text: str | None) -> tuple[float, float, float, float, float, float]:
    values = [float(part) for part in (text or "").split()]
    while len(values) < 6:
        values.append(0.0)
    return tuple(values[:6])  # type: ignore[return-value]


def load_primitives(path: Path) -> list[Primitive]:
    root = ET.parse(path).getroot()
    primitives: list[Primitive] = []
    for link in root.findall(".//link"):
        name = link.attrib.get("name", "unnamed")
        pose = parse_pose(link.findtext("pose"))
        geometry = link.find("./collision/geometry") or link.find("./visual/geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        if box is not None:
            size_text = box.findtext("size")
            if not size_text:
                continue
            sx, sy, sz = (float(part) for part in size_text.split())
            primitives.append(Primitive(name=name, kind="box", pose=pose, size=(sx, sy, sz)))
        elif cylinder is not None:
            radius = float(cylinder.findtext("radius") or "0")
            length = float(cylinder.findtext("length") or "0")
            primitives.append(
                Primitive(name=name, kind="cylinder", pose=pose, radius=radius, length=length)
            )
    return primitives


def load_ascii_ply_xyz(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open("r", encoding="utf-8") as stream:
        in_header = True
        for line in stream:
            if in_header:
                if line.strip() == "end_header":
                    in_header = False
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not points:
        raise RuntimeError(f"No XYZ points found in {path}")
    return points


def matmul_vec(m: tuple[tuple[float, float, float], ...], v: Point) -> Point:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def rotation_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def transformed(local: Point, pose: tuple[float, float, float, float, float, float]) -> Point:
    x, y, z, roll, pitch, yaw = pose
    rx, ry, rz = matmul_vec(rotation_matrix(roll, pitch, yaw), local)
    return x + rx, y + ry, z + rz


def classify_color(name: str) -> Rgb:
    if "panel" in name:
        return (148, 163, 184)
    if "ledge" in name or "cap" in name:
        return (245, 178, 44)
    if "brace" in name:
        return (34, 197, 94)
    if "scatter" in name or "marker" in name or "facet" in name:
        return (249, 115, 22)
    if "riser" in name:
        return (110, 118, 128)
    return (203, 213, 225)


def sample_box(primitive: Primitive, step: float = 0.45) -> list[tuple[Point, Rgb]]:
    sx, sy, sz = primitive.size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    color = classify_color(primitive.name)
    samples: list[tuple[Point, Rgb]] = []

    def axis_values(low: float, high: float, target_step: float) -> list[float]:
        n = max(2, min(80, int(math.ceil((high - low) / max(target_step, 1e-6))) + 1))
        if n == 1:
            return [(low + high) / 2]
        return [low + (high - low) * i / (n - 1) for i in range(n)]

    xs = axis_values(-hx, hx, step)
    ys = axis_values(-hy, hy, step)
    zs = axis_values(-hz, hz, step)

    for x in xs:
        for y in ys:
            samples.append((transformed((x, y, -hz), primitive.pose), color))
            samples.append((transformed((x, y, hz), primitive.pose), color))
    for x in xs:
        for z in zs:
            samples.append((transformed((x, -hy, z), primitive.pose), color))
            samples.append((transformed((x, hy, z), primitive.pose), color))
    for y in ys:
        for z in zs:
            samples.append((transformed((-hx, y, z), primitive.pose), color))
            samples.append((transformed((hx, y, z), primitive.pose), color))
    return samples


def sample_cylinder(primitive: Primitive) -> list[tuple[Point, Rgb]]:
    radius = primitive.radius
    hz = primitive.length / 2.0
    color = classify_color(primitive.name)
    samples: list[tuple[Point, Rgb]] = []
    rings = 36
    levels = max(8, int(primitive.length / 0.65))
    for level in range(levels + 1):
        z = -hz + primitive.length * level / levels
        for i in range(rings):
            a = 2 * math.pi * i / rings
            p = (radius * math.cos(a), radius * math.sin(a), z)
            samples.append((transformed(p, primitive.pose), color))
    for z in (-hz, hz):
        for r_i in range(3):
            r = radius * r_i / 2
            for i in range(rings):
                a = 2 * math.pi * i / rings
                samples.append((transformed((r * math.cos(a), r * math.sin(a), z), primitive.pose), color))
    return samples


def sample_scene(primitives: list[Primitive]) -> list[tuple[Point, Rgb]]:
    samples: list[tuple[Point, Rgb]] = []
    for primitive in primitives:
        if primitive.kind == "box":
            step = 0.35 if max(primitive.size) < 1.0 else 0.55
            samples.extend(sample_box(primitive, step=step))
        elif primitive.kind == "cylinder":
            samples.extend(sample_cylinder(primitive))
    return samples


def z_color(z: float, z_min: float, z_max: float) -> Rgb:
    t = max(0.0, min(1.0, (z - z_min) / max(1e-9, z_max - z_min)))
    stops = [
        (0.00, (37, 99, 235)),
        (0.35, (34, 211, 238)),
        (0.62, (132, 204, 22)),
        (0.82, (245, 158, 11)),
        (1.00, (239, 68, 68)),
    ]
    for (ta, ca), (tb, cb) in zip(stops, stops[1:]):
        if ta <= t <= tb:
            u = (t - ta) / (tb - ta)
            return tuple(round(ca[i] + (cb[i] - ca[i]) * u) for i in range(3))  # type: ignore[return-value]
    return stops[-1][1]


def project_point(point: Point, view: str) -> tuple[float, float]:
    x, y, z = point
    if view == "top":
        return x, y
    if view == "side":
        return x, z
    angle = math.radians(-24.0)
    ca, sa = math.cos(angle), math.sin(angle)
    rx = ca * x - sa * y
    ry = sa * x + ca * y
    return rx + 0.42 * ry, 0.62 * ry + 0.34 * z


def view_bounds(points: list[Point], view: str, pad: float = 0.08) -> tuple[float, float, float, float]:
    coords = [project_point(point, view) for point in points]
    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1.0)
    dy = max(max_y - min_y, 1.0)
    return min_x - dx * pad, max_x + dx * pad, min_y - dy * pad, max_y + dy * pad


def to_screen(
    point: Point,
    view: str,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    px, py = project_point(point, view)
    min_x, max_x, min_y, max_y = bounds
    x0, y0, x1, y1 = rect
    scale = min((x1 - x0) / (max_x - min_x), (y1 - y0) / (max_y - min_y))
    used_w = scale * (max_x - min_x)
    used_h = scale * (max_y - min_y)
    ox = x0 + (x1 - x0 - used_w) / 2
    oy = y0 + (y1 - y0 - used_h) / 2
    sx = ox + (px - min_x) * scale
    sy = oy + used_h - (py - min_y) * scale
    return int(sx), int(sy)


def rounded_card(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(rect, radius=8, fill=CARD, outline=CARD_BORDER, width=2)


def draw_grid(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    for i in range(1, 5):
        x = x0 + (x1 - x0) * i // 5
        y = y0 + (y1 - y0) * i // 5
        draw.line((x, y0, x, y1), fill=GRID, width=1)
        draw.line((x0, y, x1, y), fill=GRID, width=1)


def brighten(color: Rgb, amount: float = 0.22) -> Rgb:
    return tuple(round(c + (255 - c) * amount) for c in color)  # type: ignore[return-value]


def darken(color: Rgb, amount: float = 0.30) -> Rgb:
    return tuple(round(c * (1.0 - amount)) for c in color)  # type: ignore[return-value]


def box_face_points(primitive: Primitive) -> list[tuple[list[Point], Rgb]]:
    sx, sy, sz = primitive.size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    corners = {
        "000": (-hx, -hy, -hz),
        "100": (hx, -hy, -hz),
        "110": (hx, hy, -hz),
        "010": (-hx, hy, -hz),
        "001": (-hx, -hy, hz),
        "101": (hx, -hy, hz),
        "111": (hx, hy, hz),
        "011": (-hx, hy, hz),
    }
    faces = [
        ("bottom", ["000", "100", "110", "010"], 0.05),
        ("top", ["001", "101", "111", "011"], 0.25),
        ("front", ["000", "100", "101", "001"], 0.14),
        ("right", ["100", "110", "111", "101"], 0.10),
        ("back", ["110", "010", "011", "111"], 0.18),
        ("left", ["010", "000", "001", "011"], 0.08),
    ]
    base = classify_color(primitive.name)
    out: list[tuple[list[Point], Rgb]] = []
    for _, keys, shade in faces:
        color = brighten(base, shade)
        out.append(([transformed(corners[key], primitive.pose) for key in keys], color))
    return out


def draw_scene_primitives(
    draw: ImageDraw.ImageDraw,
    primitives: list[Primitive],
    view: str,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
) -> None:
    faces: list[tuple[float, list[tuple[int, int]], Rgb]] = []
    cylinder_samples: list[tuple[Point, Rgb]] = []
    for primitive in primitives:
        if primitive.kind == "cylinder":
            cylinder_samples.extend(sample_cylinder(primitive))
            continue
        for face, color in box_face_points(primitive):
            screen = [to_screen(point, view, bounds, rect) for point in face]
            depth = sum(project_point(point, view)[1] for point in face) / len(face)
            faces.append((depth, screen, color))
    for _, screen, color in sorted(faces, key=lambda item: item[0]):
        draw.polygon(screen, fill=color, outline=darken(color, 0.45))
    draw_samples(draw, cylinder_samples, view, bounds, rect, radius=1)


def draw_samples(
    draw: ImageDraw.ImageDraw,
    samples: list[tuple[Point, Rgb]],
    view: str,
    bounds: tuple[float, float, float, float],
    rect: tuple[int, int, int, int],
    radius: int,
) -> None:
    for point, color in sorted(samples, key=lambda item: project_point(item[0], view)[1]):
        x, y = to_screen(point, view, bounds, rect)
        if radius <= 1:
            draw.point((x, y), fill=color)
        else:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def write_image(
    primitives: list[Primitive],
    scene_samples: list[tuple[Point, Rgb]],
    cloud_points: list[Point],
) -> Path:
    all_points = [point for point, _ in scene_samples] + cloud_points
    z_min = min(point[2] for point in all_points)
    z_max = max(point[2] for point in all_points)
    cloud_samples = [(point, z_color(point[2], z_min, z_max)) for point in cloud_points]

    views = [
        ("top", "Top view  x-y"),
        ("side", "Side view  x-z"),
        ("oblique", "Oblique view"),
    ]
    bounds_by_view = {view: view_bounds(all_points, view) for view, _ in views}

    width, height = 1800, 1320
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    draw.text((46, 34), "DAVE Scene and NAS-GS Map: matched views", font=TITLE, fill=INK)
    draw.text(
        (46, 84),
        "Same top, side, and oblique projections for the simulated target and learned Gaussian centres",
        font=SUBTITLE,
        fill=MUTED,
    )

    card_w, card_h = 530, 450
    plot_pad = 24
    x_positions = [46, 635, 1224]
    y_positions = [150, 705]
    row_titles = ["DAVE target geometry from SDF", "NAS-GS map centres from point_cloud.ply"]

    for row, y in enumerate(y_positions):
        draw.text((46, y - 34), row_titles[row], font=CARD_TITLE, fill=INK)
        for col, (view, label) in enumerate(views):
            x = x_positions[col]
            card = (x, y, x + card_w, y + card_h)
            plot = (x + plot_pad, y + 58, x + card_w - plot_pad, y + card_h - plot_pad)
            rounded_card(draw, card)
            draw.text((x + 22, y + 17), label, font=CARD_TITLE, fill=INK)
            draw.rectangle(plot, fill="#fbfdff", outline="#d7e2ef", width=1)
            draw_grid(draw, plot)
            if row == 0:
                draw_scene_primitives(draw, primitives, view, bounds_by_view[view], plot)
            else:
                draw_samples(draw, cloud_samples, view, bounds_by_view[view], plot, radius=0)

    draw.text(
        (46, 1225),
        (
            "Bounds are shared between rows for each view. "
            f"Scene samples: {len(scene_samples):,}; NAS-GS centres: {len(cloud_points):,}; "
            f"z range: {z_min:.2f} to {z_max:.2f} m."
        ),
        font=SMALL,
        fill=MUTED,
    )
    draw.text(
        (46, 1260),
        "The point cloud shows Gaussian centres only; the online render server uses the trained sonar splats.",
        font=TINY,
        fill="#52657c",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "passA_dave_vs_nasgs_matched_three_view.png"
    img.save(out)
    shutil.copy2(out, PRESENTATION_DIR / out.name)
    return out


def main() -> None:
    primitives = load_primitives(SCENE_SDF)
    scene_samples = sample_scene(primitives)
    cloud_points = load_ascii_ply_xyz(PLY_PATH)
    out = write_image(primitives, scene_samples, cloud_points)
    print(out)
    print(PRESENTATION_DIR / out.name)


if __name__ == "__main__":
    main()
