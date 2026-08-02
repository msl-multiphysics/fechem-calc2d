"""Calc2D: Gmsh / structured mesh + FEChem VTU quadrature integrals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import read_vtu_values, remap_vtu_to_domain
from .integrate import (
    build_edge_adjacency,
    lineint_scl_adv,
    lineint_scl_diff,
    lineint_vec_adv,
    lineint_vec_pres,
    lineint_vec_visc,
    surfint_scl_src,
    surfint_vec_src,
)
from .quadrature import build_dom_quad_tables
from .viz import MeshSnapshot, RegionMesh


def _snapshot_from_bounds(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    num_elem_x: int,
    num_elem_y: int,
) -> MeshSnapshot:
    """Structured Quad4 mesh matching Rust ``Mesh::new_from_bounds``."""
    if x_min >= x_max:
        raise ValueError(f"Invalid x bounds: {x_min} >= {x_max}")
    if y_min >= y_max:
        raise ValueError(f"Invalid y bounds: {y_min} >= {y_max}")
    if num_elem_x < 1 or num_elem_y < 1:
        raise ValueError("num_elem_x and num_elem_y must be >= 1")

    dx = (x_max - x_min) / float(num_elem_x)
    dy = (y_max - y_min) / float(num_elem_y)
    stride = num_elem_x + 1

    xs = [x_min + i * dx for i in range(num_elem_x + 1)]
    ys = [y_min + j * dy for j in range(num_elem_y + 1)]
    points = np.array(
        [[xs[i], ys[j], 0.0] for j in range(num_elem_y + 1) for i in range(num_elem_x + 1)],
        dtype=float,
    )

    quads = []
    for j in range(num_elem_y):
        for i in range(num_elem_x):
            nid0 = i + j * stride
            nid1 = nid0 + 1
            nid2 = nid0 + stride + 1
            nid3 = nid0 + stride
            quads.append([nid0, nid1, nid2, nid3])
    quads_arr = np.asarray(quads, dtype=int)

    # 1D edges: left, right, bottom, top (same order as Rust)
    left, right, bottom, top = [], [], [], []
    for j in range(num_elem_y - 1, -1, -1):
        nid0 = (j + 1) * stride
        nid1 = j * stride
        left.append([nid0, nid1])
    for j in range(num_elem_y):
        nid0 = num_elem_x + j * stride
        nid1 = num_elem_x + (j + 1) * stride
        right.append([nid0, nid1])
    for i in range(num_elem_x):
        bottom.append([i, i + 1])
    for i in range(num_elem_x - 1, -1, -1):
        nid0 = num_elem_y * stride + i + 1
        nid1 = num_elem_y * stride + i
        top.append([nid0, nid1])

    # Gmsh-style tags start at 1 → FEChem indices = tag - 1
    dom = RegionMesh(
        tag=1,
        name="Domain_0",
        dim=2,
        cells=quads_arr,
        cell_type="quad",
        label_points=np.array([[0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.0]]),
    )
    boundaries = [
        RegionMesh(
            tag=1,
            name="Boundary_0",
            dim=1,
            cells=np.asarray(left, dtype=int),
            cell_type="line",
            label_points=np.array([[x_min, 0.5 * (y_min + y_max), 0.0]]),
        ),
        RegionMesh(
            tag=2,
            name="Boundary_1",
            dim=1,
            cells=np.asarray(right, dtype=int),
            cell_type="line",
            label_points=np.array([[x_max, 0.5 * (y_min + y_max), 0.0]]),
        ),
        RegionMesh(
            tag=3,
            name="Boundary_2",
            dim=1,
            cells=np.asarray(bottom, dtype=int),
            cell_type="line",
            label_points=np.array([[0.5 * (x_min + x_max), y_min, 0.0]]),
        ),
        RegionMesh(
            tag=4,
            name="Boundary_3",
            dim=1,
            cells=np.asarray(top, dtype=int),
            cell_type="line",
            label_points=np.array([[0.5 * (x_min + x_max), y_max, 0.0]]),
        ),
    ]
    return MeshSnapshot(
        points=points,
        topo_dim=2,
        domains=[dom],
        boundaries=boundaries,
        interfaces=[],
    )


class Calc2D:
    """Load a FEChem mesh and evaluate Dom / Bnd / Itf integrals on VTU fields."""

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], str):
            from .viz import extract_snapshot_from_file

            self._msh_path: str | None = args[0]
            self._snap = extract_snapshot_from_file(args[0], topo_dim=2)
        elif len(args) == 6:
            x_min, y_min, x_max, y_max, nx, ny = args
            self._msh_path = None
            self._snap = _snapshot_from_bounds(
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
                int(nx),
                int(ny),
            )
        else:
            raise TypeError(
                "Calc2D(msh_path) or Calc2D(x_min, y_min, x_max, y_max, nx, ny)"
            )

        self._points = np.asarray(self._snap.points, dtype=float)

        # FEChem region id = Gmsh physical tag - 1
        self._domains: dict[int, Any] = {}
        for region in self._snap.domains:
            self._domains[int(region.tag) - 1] = region

        self._boundaries: dict[int, Any] = {}
        for region in self._snap.boundaries:
            self._boundaries[int(region.tag) - 1] = region

        self._interfaces: dict[int, Any] = {}
        for region in self._snap.interfaces:
            self._interfaces[int(region.tag) - 1] = region

        self._dom_node_ids: dict[int, np.ndarray] = {}
        self._dom_global_to_local: dict[int, dict[int, int]] = {}
        self._dom_cells_local: dict[int, np.ndarray] = {}
        self._dom_points_xy: dict[int, np.ndarray] = {}
        self._dom_cell_type: dict[int, str] = {}
        self._dom_quad: dict[int, Any] = {}

        self._vtu_match_tol = 5e-6

        for dom_id, region in self._domains.items():
            cells = np.asarray(region.cells, dtype=int)
            node_ids = np.unique(cells.ravel())
            node_ids.sort()
            g2l = {int(gid): loc for loc, gid in enumerate(node_ids)}
            cells_local = np.vectorize(g2l.__getitem__, otypes=[int])(cells)
            self._dom_node_ids[dom_id] = node_ids
            self._dom_global_to_local[dom_id] = g2l
            self._dom_cells_local[dom_id] = cells_local
            self._dom_points_xy[dom_id] = self._points[node_ids, :2]
            self._dom_cell_type[dom_id] = region.cell_type
            self._dom_quad[dom_id] = build_dom_quad_tables(
                self._dom_points_xy[dom_id], cells_local, region.cell_type
            )

        self._edge_adj = build_edge_adjacency(self._dom_cells_local)
        self._field_domain: dict[int, int] = {}

    def show_mesh(self) -> None:
        """Display Dom / Bnd / Itf indices."""
        from .viz import render_snapshot

        render_snapshot(self._snap)

    def read_scl(self, dom: int, path: str) -> np.ndarray:
        """Load a nodal scalar VTU PointData ``value`` onto Dom-local order."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 1:
            raise ValueError(
                f"Expected scalar VTU for read_scl, got shape {arr.shape}"
            )
        return arr

    def read_vec(self, dom: int, path: str) -> np.ndarray:
        """Load a nodal vector VTU PointData ``value`` onto Dom-local order."""
        arr = self._read_vtu(dom, path)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected vector VTU shape (n, 2) for read_vec, got {arr.shape}"
            )
        return arr

    def _read_vtu(self, dom: int, path: str) -> np.ndarray:
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        vtu_points, vtu_values = read_vtu_values(path)
        remapped = remap_vtu_to_domain(
            vtu_points,
            vtu_values,
            self._dom_points_xy[dom],
            self._vtu_match_tol,
        )
        out = np.array(remapped, dtype=float, copy=True)
        self._field_domain[id(out)] = dom
        return out

    def _resolve_field_domain(self, *fields: Any, require: bool = True) -> int | None:
        found: set[int] = set()
        for field in fields:
            if np.isscalar(field) or callable(field):
                continue
            if isinstance(field, (tuple, list)) and len(field) == 2 and all(
                np.isscalar(c) for c in field
            ):
                continue
            arr = np.asarray(field)
            if id(field) in self._field_domain:
                found.add(self._field_domain[id(field)])
                continue
            if arr.ndim >= 1:
                n = arr.shape[0]
                matches = [
                    d for d, ids in self._dom_node_ids.items() if len(ids) == n
                ]
                if len(matches) == 1:
                    found.add(matches[0])
        if not found:
            if require:
                raise ValueError(
                    "Cannot determine domain for field; pass arrays from read_scl/read_vec"
                )
            return None
        if len(found) > 1:
            raise ValueError(
                f"Field arrays belong to multiple domains: {sorted(found)}"
            )
        return next(iter(found))

    def _line_region(self, reg: int):
        if reg in self._boundaries:
            return self._boundaries[reg]
        if reg in self._interfaces:
            return self._interfaces[reg]
        raise ValueError(f"Unknown boundary/interface index {reg}")

    def _check_unk(self, dom: int, unk: np.ndarray, kind: str) -> np.ndarray:
        arr = np.asarray(unk, dtype=float)
        n = len(self._dom_node_ids[dom])
        if kind == "scl":
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"Scalar field must have shape ({n},), got {arr.shape}"
                )
        else:
            if arr.ndim != 2 or arr.shape != (n, 2):
                raise ValueError(
                    f"Vector field must have shape ({n}, 2), got {arr.shape}"
                )
        return arr

    # --- scalar integrals -------------------------------------------------

    def surfint_scl_src(self, dom: int, src: Any, unk: Any) -> float:
        """Surface integral ∫ src(u) dΩ over domain ``dom``."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        u = self._check_unk(dom, unk, "scl")
        return surfint_scl_src(
            self._dom_quad[dom], self._dom_cells_local[dom], src, u
        )

    def lineint_scl_diff(self, bnd: int, diff: Any, unk: Any) -> float:
        """Line integral ∫ -diff(u) (∇u · n) dl on boundary/interface ``bnd``."""
        region = self._line_region(bnd)
        dom = self._resolve_field_domain(unk)
        assert dom is not None
        u = self._check_unk(dom, unk, "scl")
        return lineint_scl_diff(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xy[dom],
            self._dom_cell_type[dom],
            self._edge_adj,
            diff,
            u,
        )

    def lineint_scl_adv(self, bnd: int, wgt: Any, vel: Any, unk: Any) -> float:
        """Line integral ∫ wgt(u) u (v · n) dl on boundary/interface ``bnd``."""
        region = self._line_region(bnd)
        dom = self._resolve_field_domain(unk, vel)
        assert dom is not None
        u = self._check_unk(dom, unk, "scl")
        v = self._check_unk(dom, vel, "vec")
        return lineint_scl_adv(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xy[dom],
            self._dom_cell_type[dom],
            self._edge_adj,
            wgt,
            v,
            u,
        )

    # --- vector integrals -------------------------------------------------

    def surfint_vec_src(self, dom: int, src: Any, fce: Any) -> np.ndarray:
        """Surface integral ∫ src(f) dΩ over domain ``dom``; returns (2,)."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        f = self._check_unk(dom, fce, "vec")
        return surfint_vec_src(
            self._dom_quad[dom], self._dom_cells_local[dom], src, f
        )

    def lineint_vec_visc(self, bnd: int, visc: Any, vel: Any) -> np.ndarray:
        """Line integral ∫ -(τ · n) dl, viscous force of fluid on ``bnd``."""
        region = self._line_region(bnd)
        dom = self._resolve_field_domain(vel)
        assert dom is not None
        v = self._check_unk(dom, vel, "vec")
        return lineint_vec_visc(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xy[dom],
            self._dom_cell_type[dom],
            self._edge_adj,
            visc,
            v,
        )

    def lineint_vec_adv(self, bnd: int, den: Any, vel: Any) -> np.ndarray:
        """Line integral ∫ ρ (v · n) v dl on ``bnd``."""
        region = self._line_region(bnd)
        dom = self._resolve_field_domain(vel)
        assert dom is not None
        v = self._check_unk(dom, vel, "vec")
        return lineint_vec_adv(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xy[dom],
            self._dom_cell_type[dom],
            self._edge_adj,
            den,
            v,
        )

    def lineint_vec_pres(self, bnd: int, pres: Any) -> np.ndarray:
        """Line integral ∫ p n dl, pressure force of fluid on ``bnd``."""
        region = self._line_region(bnd)
        dom = self._resolve_field_domain(pres)
        assert dom is not None
        p = self._check_unk(dom, pres, "scl")
        return lineint_vec_pres(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            self._dom_points_xy[dom],
            self._dom_cell_type[dom],
            self._edge_adj,
            p,
        )
