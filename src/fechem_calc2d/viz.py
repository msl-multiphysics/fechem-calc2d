"""Gmsh mesh snapshot extraction and PyVista Dom/Bnd/Itf rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# meshio cell type -> topological dimension
_CELL_DIM = {
    "vertex": 0,
    "line": 1,
    "line3": 1,
    "triangle": 2,
    "triangle6": 2,
    "quad": 2,
    "quad4": 2,
    "quad8": 2,
    "quad9": 2,
    "quadrilateral": 2,
    "tetra": 3,
    "tetra10": 3,
    "hexahedron": 3,
    "hexahedron20": 3,
    "hexahedron27": 3,
    "wedge": 3,
    "pyramid": 3,
}

# meshio cell type -> (vtk cell type name used by pyvista, nodes per cell)
_CELL_VTK = {
    "line": ("LINE", 2),
    "line3": ("QUADRATIC_EDGE", 3),
    "triangle": ("TRIANGLE", 3),
    "triangle6": ("QUADRATIC_TRIANGLE", 6),
    "quad": ("QUAD", 4),
    "quad4": ("QUAD", 4),
    "quadrilateral": ("QUAD", 4),
    "quad8": ("QUADRATIC_QUAD", 8),
    "quad9": ("BIQUADRATIC_QUAD", 9),
    "tetra": ("TETRA", 4),
    "tetra10": ("QUADRATIC_TETRA", 10),
    "hexahedron": ("HEXAHEDRON", 8),
    "hexahedron20": ("QUADRATIC_HEXAHEDRON", 20),
    "hexahedron27": ("TRIQUADRATIC_HEXAHEDRON", 27),
    "wedge": ("WEDGE", 6),
    "pyramid": ("PYRAMID", 5),
}


@dataclass
class RegionMesh:
    tag: int
    name: str
    dim: int
    cells: np.ndarray  # (n_cells, n_nodes)
    cell_type: str
    label_points: np.ndarray  # (n_labels, 3); on-curve for edges


@dataclass
class MeshSnapshot:
    points: np.ndarray  # (N, 3)
    topo_dim: int
    domains: list[RegionMesh] = field(default_factory=list)
    boundaries: list[RegionMesh] = field(default_factory=list)
    interfaces: list[RegionMesh] = field(default_factory=list)


def _as_points3(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must be a 2D array")
    if pts.shape[1] == 3:
        return pts
    if pts.shape[1] == 2:
        return np.column_stack([pts, np.zeros(len(pts))])
    raise ValueError(f"points must have 2 or 3 columns, got {pts.shape[1]}")


def _cell_dim(cell_type: str) -> int:
    if cell_type in _CELL_DIM:
        return _CELL_DIM[cell_type]
    if cell_type.startswith("line"):
        return 1
    if cell_type.startswith("triangle") or cell_type.startswith("quad"):
        return 2
    return 3


def _invert_field_data(field_data: dict) -> dict[tuple[int, int], str]:
    mapping: dict[tuple[int, int], str] = {}
    for name, info in field_data.items():
        tag, dim = int(info[0]), int(info[1])
        mapping[(tag, dim)] = name
    return mapping


def _region_centroid(points: np.ndarray, cells: np.ndarray, cell_type: str) -> np.ndarray:
    """Geometric centroid of a region (length-/area-/volume-weighted)."""
    pts = points
    cells = np.asarray(cells)
    dim = _cell_dim(cell_type)

    if dim == 1:
        # length-weighted midpoints of line segments
        p0 = pts[cells[:, 0]]
        p1 = pts[cells[:, 1]]
        mids = 0.5 * (p0 + p1)
        weights = np.linalg.norm(p1 - p0, axis=1)
    elif cell_type.startswith("triangle"):
        p0, p1, p2 = pts[cells[:, 0]], pts[cells[:, 1]], pts[cells[:, 2]]
        mids = (p0 + p1 + p2) / 3.0
        cross = np.cross(p1 - p0, p2 - p0)
        weights = 0.5 * np.linalg.norm(cross, axis=1 if cross.ndim > 1 else None)
        weights = np.atleast_1d(weights)
    elif cell_type.startswith("quad"):
        # split each quad into two triangles
        p0, p1, p2, p3 = (
            pts[cells[:, 0]],
            pts[cells[:, 1]],
            pts[cells[:, 2]],
            pts[cells[:, 3]],
        )
        c0 = (p0 + p1 + p2) / 3.0
        c1 = (p0 + p2 + p3) / 3.0
        w0 = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        w1 = 0.5 * np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1)
        mids = np.vstack([c0, c1])
        weights = np.concatenate([w0, w1])
    elif cell_type.startswith("tetra"):
        p0, p1, p2, p3 = (
            pts[cells[:, 0]],
            pts[cells[:, 1]],
            pts[cells[:, 2]],
            pts[cells[:, 3]],
        )
        mids = (p0 + p1 + p2 + p3) / 4.0
        weights = np.abs(np.einsum("ij,ij->i", p3 - p0, np.cross(p1 - p0, p2 - p0))) / 6.0
    else:
        # fallback: mean of unique nodes
        return pts[np.unique(cells.ravel())].mean(axis=0)

    total = float(np.sum(weights))
    if total <= 0.0:
        return pts[np.unique(cells.ravel())].mean(axis=0)
    return (mids * weights[:, None]).sum(axis=0) / total


def _on_curve_label_points(
    points: np.ndarray,
    cells: np.ndarray,
    geo_tags: np.ndarray | None,
) -> np.ndarray:
    """One on-curve label per CAD edge (via gmsh:geometrical), else one mesh sample."""
    cells = np.asarray(cells)
    if geo_tags is not None and len(geo_tags) == len(cells):
        label_pts = []
        for g in np.unique(geo_tags):
            sub = cells[geo_tags == g]
            i = len(sub) // 2
            label_pts.append(0.5 * (points[sub[i, 0]] + points[sub[i, 1]]))
        return np.asarray(label_pts, dtype=float)
    i = len(cells) // 2
    return np.asarray([0.5 * (points[cells[i, 0]] + points[cells[i, 1]])], dtype=float)


def extract_snapshot_from_meshio(mesh, topo_dim: int) -> MeshSnapshot:
    """Build a MeshSnapshot from a meshio.Mesh using physical group names."""
    points = _as_points3(mesh.points)
    name_of = _invert_field_data(getattr(mesh, "field_data", {}) or {})

    # Accumulate cells per (tag, dim, cell_type)
    buckets: dict[tuple[int, int, str], list[np.ndarray]] = {}
    geo_buckets: dict[tuple[int, int, str], list[np.ndarray]] = {}

    phys_blocks = mesh.cell_data.get("gmsh:physical", None)
    if phys_blocks is None:
        raise ValueError("Mesh has no gmsh:physical cell data")
    geo_blocks = mesh.cell_data.get("gmsh:geometrical", None)

    for idx, (block, tags) in enumerate(zip(mesh.cells, phys_blocks)):
        dim = _cell_dim(block.type)
        data = np.asarray(block.data)
        tags = np.asarray(tags, dtype=int)
        geos = None
        if geo_blocks is not None:
            geos = np.asarray(geo_blocks[idx], dtype=int)
        for tag in np.unique(tags):
            mask = tags == tag
            key = (int(tag), dim, block.type)
            buckets.setdefault(key, []).append(data[mask])
            if geos is not None:
                geo_buckets.setdefault(key, []).append(geos[mask])

    domains: list[RegionMesh] = []
    boundaries: list[RegionMesh] = []
    interfaces: list[RegionMesh] = []
    bnd_dim = topo_dim - 1

    for (tag, dim, cell_type), parts in buckets.items():
        cells = np.vstack(parts)
        name = name_of.get((tag, dim), f"Physical {tag}")
        if dim == topo_dim:
            label_points = np.asarray(
                [_region_centroid(points, cells, cell_type)], dtype=float
            )
        else:
            geos = None
            if (tag, dim, cell_type) in geo_buckets:
                geos = np.concatenate(geo_buckets[(tag, dim, cell_type)])
            label_points = _on_curve_label_points(points, cells, geos)
        region = RegionMesh(
            tag=tag,
            name=name,
            dim=dim,
            cells=cells,
            cell_type=cell_type,
            label_points=label_points,
        )
        if dim == topo_dim:
            domains.append(region)
        elif dim == bnd_dim:
            if name.startswith(("Boundary", "Bnd")):
                boundaries.append(region)
            elif name.startswith(("Interface", "Itf")):
                interfaces.append(region)
            else:
                # unnamed lower-dim groups: treat as boundary by default
                boundaries.append(region)

    domains.sort(key=lambda r: r.tag)
    boundaries.sort(key=lambda r: r.tag)
    interfaces.sort(key=lambda r: r.tag)

    return MeshSnapshot(
        points=points,
        topo_dim=topo_dim,
        domains=domains,
        boundaries=boundaries,
        interfaces=interfaces,
    )


def extract_snapshot_from_file(file_path: str, topo_dim: int) -> MeshSnapshot:
    """Extract a MeshSnapshot from an existing .msh file."""
    import meshio

    mesh = meshio.read(file_path)
    return extract_snapshot_from_meshio(mesh, topo_dim=topo_dim)


def _region_to_grid(points: np.ndarray, region: RegionMesh):
    import pyvista as pv

    if region.cell_type not in _CELL_VTK:
        raise ValueError(f"Unsupported cell type for plotting: {region.cell_type}")
    vtk_name, n_nodes = _CELL_VTK[region.cell_type]
    cells = np.asarray(region.cells)
    if cells.ndim != 2 or cells.shape[1] != n_nodes:
        raise ValueError(
            f"Expected cells of shape (n, {n_nodes}) for {region.cell_type}, "
            f"got {cells.shape}"
        )
    n = len(cells)
    cell_arr = np.hstack(
        [np.full((n, 1), n_nodes, dtype=np.int64), cells.astype(np.int64)]
    )
    celltypes = np.full(n, getattr(pv.CellType, vtk_name), dtype=np.uint8)
    return pv.UnstructuredGrid(cell_arr.ravel(), celltypes, points)


def _region_colors(n: int, cmap_name: str) -> list[tuple]:
    """n discrete RGB colors from a categorical colormap (stable mesh+label match)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    return [cmap(i % cmap.N)[:3] for i in range(n)]


def _contrast_text_color(rgb) -> str:
    r, g, b = rgb[:3]
    # Rec. 601 luma; light backgrounds get black text 
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def _text_button_texture(label: str, font_path: str | None = None, font_size: int = 16):
    """Build a PyVista ImageData texture with a labeled button appearance."""
    import pyvista as pv
    from PIL import Image, ImageDraw, ImageFont

    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    probe = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(probe)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x, pad_y = 14, 8
    width = text_w + 2 * pad_x
    height = text_h + 2 * pad_y

    img = Image.new("RGB", (width, height), (235, 235, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width - 1, height - 1], outline=(60, 60, 60), width=2)
    draw.text((pad_x - bbox[0], pad_y - bbox[1]), label, fill=(20, 20, 20), font=font)

    arr = np.asarray(img, dtype=np.uint8)  # (height, width, 3), y=0 at top
    # VTK ImageData is x-fastest with y=0 at bottom
    arr = np.flipud(arr)
    button = pv.ImageData(dimensions=(width, height, 1))
    button.point_data["texture"] = arr.reshape(width * height, 3)
    return button, width, height


def _add_reset_camera_button(plotter, *, topo_dim: int, margin: float = 10.0):
    """Add a clickable 'Reset Camera' text button at the bottom-right."""
    from vtkmodules.vtkCommonCore import vtkCommand
    from vtkmodules.vtkInteractionWidgets import (
        vtkButtonWidget,
        vtkTexturedButtonRepresentation2D,
    )

    texture, width, height = _text_button_texture(
        "Reset Camera",
        font_path=_resolve_label_font("IBM Plex Sans"),
    )
    win_w, win_h = plotter.window_size
    position = (win_w - width - margin, margin)

    # Same texture for both states so the control does not look like a toggle
    button_rep = vtkTexturedButtonRepresentation2D()
    button_rep.SetNumberOfStates(2)
    button_rep.SetButtonTexture(0, texture)
    button_rep.SetButtonTexture(1, texture)
    button_rep.SetState(0)
    button_rep.SetPlaceFactor(1)
    button_rep.PlaceWidget(
        [position[0], position[0] + width, position[1], position[1] + height, 0.0, 0.0]
    )

    button_widget = vtkButtonWidget()
    button_widget.SetInteractor(plotter.iren.interactor)
    button_widget.SetRepresentation(button_rep)
    button_widget.SetCurrentRenderer(plotter.renderer)
    button_widget.On()

    def _on_click(_widget, _event):
        if topo_dim == 2:
            plotter.view_xy()
        plotter.reset_camera()
        button_rep.SetState(0)

    button_widget.AddObserver(vtkCommand.StateChangedEvent, _on_click)
    plotter.widgets.button_widgets.append(button_widget)
    return button_widget


def _resolve_label_font(family: str = "IBM Plex Sans") -> str | None:
    """Return a TTF path for the label font, or None if unavailable."""
    try:
        from matplotlib import font_manager
    except ImportError:
        return None
    try:
        path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=False,
        )
    except (ValueError, RuntimeError):
        return None
    if path and "DejaVu" not in path:  # matplotlib's silent fallback name
        return path
    # Explicit scan in case findfont fell back quietly on some versions
    for entry in font_manager.fontManager.ttflist:
        if entry.name == family and entry.style == "normal" and int(entry.weight) == 400:
            return entry.fname
    for entry in font_manager.fontManager.ttflist:
        if entry.name == family and entry.style == "normal":
            return entry.fname
    return None


def render_snapshot(
    snap: MeshSnapshot,
    *,
    show_domains: bool = True,
    show_boundaries: bool = True,
    show_interfaces: bool = True,
    show_labels: bool = True,
):
    """Open an interactive PyVista window for the snapshot."""
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "show_mesh() requires pyvista. Install with: pip install pyvista"
        ) from exc

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.show_axes()

    label_items: list[tuple[RegionMesh, tuple]] = []

    def _add_group(regions: list[RegionMesh], cmap_name: str, **mesh_kwargs):
        colors = _region_colors(len(regions), cmap_name)
        for region, color in zip(regions, colors):
            grid = _region_to_grid(snap.points, region)
            plotter.add_mesh(grid, color=color, **mesh_kwargs)
            label_items.append((region, color))

    if show_domains and snap.domains:
        _add_group(
            snap.domains,
            "Pastel1",
            show_edges=True,
        )

    if show_boundaries and snap.boundaries:
        line_width = 4 if snap.topo_dim == 2 else 1
        _add_group(
            snap.boundaries,
            "Set1",
            line_width=line_width,
        )

    if show_interfaces and snap.interfaces:
        line_width = 5 if snap.topo_dim == 2 else 1
        style = "wireframe" if snap.topo_dim == 3 else "surface"
        _add_group(
            snap.interfaces,
            "Set2",
            line_width=line_width,
            style=style,
        )

    if show_labels:
        font_file = _resolve_label_font("IBM Plex Sans")
        for region, rgb in label_items:
            pts = np.asarray(region.label_points, dtype=float).reshape(-1, 3)
            names = [region.name] * len(pts)
            plotter.add_point_labels(
                pts,
                names,
                point_size=5,
                font_size=14,
                bold=False,
                font_file=font_file,
                text_color=_contrast_text_color(rgb),
                shape_color=rgb,
                shape_opacity=0.8,
                always_visible=True,
            )

    if snap.topo_dim == 2:
        plotter.view_xy()

    _add_reset_camera_button(plotter, topo_dim=snap.topo_dim)

    plotter.show()
