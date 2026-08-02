"""P1 surface and line integrals on Dom / Bnd / Itf regions."""

from __future__ import annotations

import numpy as np


def triangle_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    return 0.5 * float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))


def quad_area(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return abs(triangle_area(p0, p1, p2)) + abs(triangle_area(p0, p2, p3))


def cell_area(pts: np.ndarray) -> float:
    if len(pts) == 3:
        return abs(triangle_area(pts[0], pts[1], pts[2]))
    if len(pts) == 4:
        return quad_area(pts[0], pts[1], pts[2], pts[3])
    raise ValueError(f"Unsupported cell with {len(pts)} nodes")


def triangle_grad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, v0: float, v1: float, v2: float) -> np.ndarray:
    """Constant gradient of a P1 triangle field."""
    # v = a + b x + c y; solve for b, c from nodal values
    a00 = p1[0] - p0[0]
    a01 = p1[1] - p0[1]
    a10 = p2[0] - p0[0]
    a11 = p2[1] - p0[1]
    det = a00 * a11 - a01 * a10
    if abs(det) < 1e-30:
        return np.zeros(2, dtype=float)
    rhs0 = v1 - v0
    rhs1 = v2 - v0
    gx = (rhs0 * a11 - a01 * rhs1) / det
    gy = (a00 * rhs1 - rhs0 * a10) / det
    return np.array([gx, gy], dtype=float)


def quad_grad_at_point(
    pts: np.ndarray, vals: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """Approximate P1-quad gradient by area-weighted average of two triangles."""
    # Split 0-1-2 and 0-2-3
    g1 = triangle_grad(pts[0], pts[1], pts[2], vals[0], vals[1], vals[2])
    g2 = triangle_grad(pts[0], pts[2], pts[3], vals[0], vals[2], vals[3])
    a1 = abs(triangle_area(pts[0], pts[1], pts[2]))
    a2 = abs(triangle_area(pts[0], pts[2], pts[3]))
    denom = a1 + a2
    if denom < 1e-30:
        return np.zeros(2, dtype=float)
    # Weight by inverse distance to triangle centroid (mild preference for nearer split)
    c1 = (pts[0] + pts[1] + pts[2]) / 3.0
    c2 = (pts[0] + pts[2] + pts[3]) / 3.0
    d1 = np.linalg.norm(x - c1) + 1e-30
    d2 = np.linalg.norm(x - c2) + 1e-30
    w1 = a1 / d1
    w2 = a2 / d2
    return (w1 * g1 + w2 * g2) / (w1 + w2)


def surface_integral(
    points: np.ndarray,
    cells: np.ndarray,
    field_local: np.ndarray,
) -> float:
    """∫ Q dS over Dom cells; ``field_local`` indexed by Dom-local node ids in ``cells``."""
    total = 0.0
    for cell in cells:
        pts = points[cell, :2]
        vals = field_local[cell]
        area = cell_area(pts)
        total += float(np.mean(vals)) * area
    return total


def edge_length(p0: np.ndarray, p1: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p0))


def outward_unit_normal(
    p0: np.ndarray, p1: np.ndarray, elem_centroid: np.ndarray
) -> np.ndarray:
    """Unit normal of edge p0→p1 pointing away from the element centroid."""
    t = p1 - p0
    length = float(np.linalg.norm(t))
    if length < 1e-30:
        return np.zeros(2, dtype=float)
    # Left normal of directed edge
    n = np.array([-t[1], t[0]], dtype=float) / length
    mid = 0.5 * (p0 + p1)
    if np.dot(n, mid - elem_centroid) < 0.0:
        n = -n
    return n


def build_edge_adjacency(
    dom_cells: dict[int, np.ndarray],
) -> dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]]:
    """
    Map unordered edge {n0, n1} (Dom-local ids) → list of
    (dom_id, cell_node_ids, edge_ordered_as_in_cell).
    """
    adj: dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for dom_id, cells in dom_cells.items():
        for cell in cells:
            n = len(cell)
            for i in range(n):
                a = int(cell[i])
                b = int(cell[(i + 1) % n])
                key = frozenset((a, b))
                ordered = np.array([a, b], dtype=int)
                adj.setdefault(key, []).append((dom_id, cell.copy(), ordered))
    return adj


def line_integral_diff(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    k_local: np.ndarray,
    t_local: np.ndarray,
    edge_adj: dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]],
    local_to_global: np.ndarray,
) -> float:
    """∫ k ∇T · n dl on Bnd/Itf edges using Dom ``field_dom`` side."""
    total = 0.0
    for edge in edge_cells:
        g0, g1 = int(edge[0]), int(edge[1])
        if g0 not in global_to_local or g1 not in global_to_local:
            raise ValueError(
                f"Boundary edge nodes {g0}, {g1} are not in domain {field_dom}"
            )
        l0 = global_to_local[g0]
        l1 = global_to_local[g1]
        key = frozenset((l0, l1))
        candidates = [
            item for item in edge_adj.get(key, []) if item[0] == field_dom
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one adjacent cell in domain {field_dom} for edge "
                f"{g0}-{g1}, found {len(candidates)}"
            )
        _, cell, _ = candidates[0]
        pts = points[local_to_global[cell], :2]
        tvals = t_local[cell]
        centroid = np.mean(pts, axis=0)
        p0 = points[g0, :2]
        p1 = points[g1, :2]
        mid = 0.5 * (p0 + p1)
        if len(cell) == 3:
            grad = triangle_grad(pts[0], pts[1], pts[2], tvals[0], tvals[1], tvals[2])
        else:
            grad = quad_grad_at_point(pts, tvals, mid)
        n = outward_unit_normal(p0, p1, centroid)
        k_mid = 0.5 * (k_local[l0] + k_local[l1])
        total += float(k_mid * np.dot(grad, n) * edge_length(p0, p1))
    return total


def line_integral_adv(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    w_local: np.ndarray,
    v_local: np.ndarray,
    t_local: np.ndarray,
    edge_adj: dict[frozenset[int], list[tuple[int, np.ndarray, np.ndarray]]],
    local_to_global: np.ndarray,
) -> float:
    """∫ w (v · n) T dl on Bnd/Itf edges using Dom ``field_dom`` side."""
    total = 0.0
    for edge in edge_cells:
        g0, g1 = int(edge[0]), int(edge[1])
        if g0 not in global_to_local or g1 not in global_to_local:
            raise ValueError(
                f"Boundary edge nodes {g0}, {g1} are not in domain {field_dom}"
            )
        l0 = global_to_local[g0]
        l1 = global_to_local[g1]
        key = frozenset((l0, l1))
        candidates = [
            item for item in edge_adj.get(key, []) if item[0] == field_dom
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one adjacent cell in domain {field_dom} for edge "
                f"{g0}-{g1}, found {len(candidates)}"
            )
        _, cell, _ = candidates[0]
        pts = points[local_to_global[cell], :2]
        centroid = np.mean(pts, axis=0)
        p0 = points[g0, :2]
        p1 = points[g1, :2]
        n = outward_unit_normal(p0, p1, centroid)
        w_mid = 0.5 * (w_local[l0] + w_local[l1])
        v_mid = 0.5 * (v_local[l0] + v_local[l1])
        t_mid = 0.5 * (t_local[l0] + t_local[l1])
        total += float(w_mid * np.dot(v_mid, n) * t_mid * edge_length(p0, p1))
    return total
