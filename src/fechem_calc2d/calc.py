"""Calc2D: Gmsh mesh + FEChem VTU post-processing."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import (
    as_scalar_field,
    as_vector_field,
    read_vtu_values,
    remap_vtu_to_domain,
)
from .integrate import (
    build_edge_adjacency,
    line_integral_adv,
    line_integral_diff,
    surface_integral,
)


class Calc2D:
    """Load a FEChem Gmsh mesh and evaluate domain / boundary integrals on VTU fields."""

    def __init__(self, msh_path: str) -> None:
        from .viz import extract_snapshot_from_file

        self._msh_path = msh_path
        self._snap = extract_snapshot_from_file(msh_path, topo_dim=2)
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

        # Dom-local node numbering (sorted global ids for stability)
        self._dom_node_ids: dict[int, np.ndarray] = {}
        self._dom_global_to_local: dict[int, dict[int, int]] = {}
        self._dom_cells_local: dict[int, np.ndarray] = {}
        self._dom_points_xy: dict[int, np.ndarray] = {}

        # FEChem VTU writer rounds coords to 6 decimals (~5e-7); allow slack
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

        self._edge_adj = build_edge_adjacency(self._dom_cells_local)

        # id(array) → domain index for arrays returned by load_vtu
        self._field_domain: dict[int, int] = {}

    def show_mesh(self) -> None:
        """Display Dom / Bnd / Itf indices."""
        from .viz import render_snapshot

        render_snapshot(self._snap)

    def load_vtu(self, dom: int, path: str) -> np.ndarray:
        """Load a domain VTU PointData ``value`` onto Dom-local node order."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        vtu_points, vtu_values = read_vtu_values(path)
        remapped = remap_vtu_to_domain(
            vtu_points,
            vtu_values,
            self._dom_points_xy[dom],
            self._vtu_match_tol,
        )
        # Ensure a fresh contiguous array so id() is unique to this load
        out = np.array(remapped, dtype=float, copy=True)
        self._field_domain[id(out)] = dom
        return out

    def _resolve_field_domain(self, *fields: Any, require: bool = True) -> int | None:
        """Infer owning domain from registered load_vtu arrays."""
        found: set[int] = set()
        for field in fields:
            if np.isscalar(field):
                continue
            if isinstance(field, (tuple, list)) and len(field) == 2 and all(
                np.isscalar(c) for c in field
            ):
                continue
            arr = np.asarray(field)
            # Prefer exact object identity when still the same array
            if id(field) in self._field_domain:
                found.add(self._field_domain[id(field)])
                continue
            # Fallback: match length to a unique domain
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
                    "Cannot determine domain for field; pass arrays from load_vtu"
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

    def surfint(self, dom: int, field: Any) -> float:
        """Surface integral of a scalar field over domain ``dom``."""
        if dom not in self._domains:
            raise ValueError(f"Unknown domain index {dom}")
        n_nodes = len(self._dom_node_ids[dom])
        if not np.isscalar(field):
            owned = self._resolve_field_domain(field, require=False)
            if owned is not None and owned != dom:
                raise ValueError(
                    f"Field belongs to domain {owned}, but surfint requested domain {dom}"
                )
        q = as_scalar_field(field, n_nodes)
        # cells_local indexes into Dom-local field; surface_integral expects
        # points indexed by the same ids as cells. Build Dom-local point table.
        node_ids = self._dom_node_ids[dom]
        local_points = self._points[node_ids]
        return surface_integral(local_points, self._dom_cells_local[dom], q)

    def lineint_diff(self, reg: int, k: Any, temp: Any) -> float:
        """Line integral of k * grad(T) · n on boundary/interface ``reg``."""
        region = self._line_region(reg)
        dom = self._resolve_field_domain(temp, k)
        assert dom is not None
        n_nodes = len(self._dom_node_ids[dom])
        k_local = as_scalar_field(k, n_nodes)
        t_local = as_scalar_field(temp, n_nodes)
        return line_integral_diff(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            k_local,
            t_local,
            self._edge_adj,
            self._dom_node_ids[dom],
        )

    def lineint_adv(self, reg: int, weight: Any, vel: Any, temp: Any) -> float:
        """Line integral of weight * (v · n) * T on boundary/interface ``reg``."""
        region = self._line_region(reg)
        dom = self._resolve_field_domain(temp, weight, vel)
        assert dom is not None
        n_nodes = len(self._dom_node_ids[dom])
        w_local = as_scalar_field(weight, n_nodes)
        v_local = as_vector_field(vel, n_nodes)
        t_local = as_scalar_field(temp, n_nodes)
        return line_integral_adv(
            self._points,
            np.asarray(region.cells, dtype=int),
            self._dom_global_to_local[dom],
            dom,
            w_local,
            v_local,
            t_local,
            self._edge_adj,
            self._dom_node_ids[dom],
        )
