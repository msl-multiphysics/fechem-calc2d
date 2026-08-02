"""Tri3 / Quad4 / Lin2 Gauss rules and precomputed Dom / Bnd integral tables."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_G = 1.0 / np.sqrt(3.0)  # 0.5773502691896257


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

def tri3_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b) for the 3-point Tri3 rule."""
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=float)
    a = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)
    b = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0], dtype=float)
    return w, a, b


def quad4_quad_rule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (w, a, b) for the 2x2 Quad4 Gauss rule."""
    w = np.ones(4, dtype=float)
    a = np.array([-_G, -_G, _G, _G], dtype=float)
    b = np.array([-_G, _G, -_G, _G], dtype=float)
    return w, a, b


def lin2_quad_rule() -> tuple[np.ndarray, np.ndarray]:
    """Return (w, a) for the 2-point Lin2 Gauss rule."""
    w = np.ones(2, dtype=float)
    a = np.array([-_G, _G], dtype=float)
    return w, a


def tri3_eval(a: float, b: float) -> np.ndarray:
    return np.array([1.0 - a - b, a, b], dtype=float)


def tri3_grad_ref(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    dn_da = np.array([-1.0, 1.0, 0.0], dtype=float)
    dn_db = np.array([-1.0, 0.0, 1.0], dtype=float)
    return dn_da, dn_db


def quad4_eval(a: float, b: float) -> np.ndarray:
    return np.array(
        [
            0.25 * (1.0 - a) * (1.0 - b),
            0.25 * (1.0 + a) * (1.0 - b),
            0.25 * (1.0 + a) * (1.0 + b),
            0.25 * (1.0 - a) * (1.0 + b),
        ],
        dtype=float,
    )


def quad4_grad_ref(a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    dn_da = np.array(
        [
            -0.25 * (1.0 - b),
            0.25 * (1.0 - b),
            0.25 * (1.0 + b),
            -0.25 * (1.0 + b),
        ],
        dtype=float,
    )
    dn_db = np.array(
        [
            -0.25 * (1.0 - a),
            -0.25 * (1.0 + a),
            0.25 * (1.0 + a),
            0.25 * (1.0 - a),
        ],
        dtype=float,
    )
    return dn_da, dn_db


def lin2_eval(a: float) -> np.ndarray:
    return np.array([0.5 * (1.0 - a), 0.5 * (1.0 + a)], dtype=float)


def lin2_grad_ref(a: float) -> np.ndarray:
    return np.array([-0.5, 0.5], dtype=float)


def _is_triangle(cell_type: str) -> bool:
    return cell_type.startswith("triangle")


def _is_quad(cell_type: str) -> bool:
    return cell_type.startswith("quad")


def physical_grad(
    node_xy: np.ndarray,
    dn_da: np.ndarray,
    dn_db: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Map reference gradients to physical (gnx, gny) and return |jac_det| signed det."""
    dxda = float(np.dot(dn_da, node_xy[:, 0]))
    dxdb = float(np.dot(dn_db, node_xy[:, 0]))
    dyda = float(np.dot(dn_da, node_xy[:, 1]))
    dydb = float(np.dot(dn_db, node_xy[:, 1]))
    det = dxda * dydb - dxdb * dyda
    if abs(det) < 1e-30:
        n = len(dn_da)
        return np.zeros(n), np.zeros(n), det
    inv00, inv01 = dydb / det, -dxdb / det
    inv10, inv11 = -dyda / det, dxda / det
    gnx = inv00 * dn_da + inv10 * dn_db
    gny = inv01 * dn_da + inv11 * dn_db
    return gnx, gny, det


# ---------------------------------------------------------------------------
# Parent-face mapping: Lin2 parameter a ∈ [-1,1] → Tri3/Quad4 (ξ, η)
# ---------------------------------------------------------------------------

def edge_directed_index(cell: np.ndarray, l0: int, l1: int) -> tuple[int, bool]:
    """Return (face_index, same_orientation) for edge l0→l1 on cell."""
    n = len(cell)
    for i in range(n):
        a = int(cell[i])
        b = int(cell[(i + 1) % n])
        if a == l0 and b == l1:
            return i, True
        if a == l1 and b == l0:
            return i, False
    raise ValueError(f"Edge {l0}-{l1} not found on cell {cell}")


def parent_ref_on_face(
    cell_type: str, face: int, lin_a: float, same_orientation: bool
) -> tuple[float, float]:
    """Reference (a, b) on a Tri3/Quad4 face at Lin2 parameter ``lin_a``."""
    # If boundary edge is opposite to cell edge order, flip parameter
    a = lin_a if same_orientation else -lin_a

    if _is_triangle(cell_type):
        # Tri3: N = [1-ξ-η, ξ, η]; edges 0-1, 1-2, 2-0
        s = 0.5 * (1.0 + a)  # [0, 1] along directed cell edge
        if face == 0:  # 0→1, η=0, ξ: 0→1
            return s, 0.0
        if face == 1:  # 1→2, ξ+η=1, ξ: 1→0
            return 1.0 - s, s
        if face == 2:  # 2→0, ξ=0, η: 1→0
            return 0.0, 1.0 - s
        raise ValueError(f"Invalid triangle face {face}")

    if _is_quad(cell_type):
        # Quad4 edges: 0-1 (η=-1), 1-2 (ξ=+1), 2-3 (η=+1), 3-0 (ξ=-1)
        if face == 0:
            return a, -1.0
        if face == 1:
            return 1.0, a
        if face == 2:
            return -a, 1.0
        if face == 3:
            return -1.0, -a
        raise ValueError(f"Invalid quad face {face}")

    raise ValueError(f"Unsupported cell type {cell_type!r}")


def shape_at_ref(
    cell_type: str, xi: float, eta: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (N, dN/dξ, dN/dη) at reference coords."""
    if _is_triangle(cell_type):
        return tri3_eval(xi, eta), *tri3_grad_ref(xi, eta)
    if _is_quad(cell_type):
        return quad4_eval(xi, eta), *quad4_grad_ref(xi, eta)
    raise ValueError(f"Unsupported cell type {cell_type!r}")


# ---------------------------------------------------------------------------
# Precomputed tables
# ---------------------------------------------------------------------------

@dataclass
class DomQuadTables:
    """Per-element domain quadrature data (Dom-local indexing)."""

    cell_type: str
    num_quad: list[int] = field(default_factory=list)
    quad_w: list[np.ndarray] = field(default_factory=list)
    quad_n: list[np.ndarray] = field(default_factory=list)  # [e](q, v)
    quad_gnx: list[np.ndarray] = field(default_factory=list)
    quad_gny: list[np.ndarray] = field(default_factory=list)
    jac_det: list[np.ndarray] = field(default_factory=list)
    quad_xy: list[np.ndarray] = field(default_factory=list)  # [e](q, 2)


def build_dom_quad_tables(
    points_xy: np.ndarray,
    cells_local: np.ndarray,
    cell_type: str,
) -> DomQuadTables:
    """Precompute Dom quadrature tables for a uniform Tri3 or Quad4 domain."""
    tables = DomQuadTables(cell_type=cell_type)
    if _is_triangle(cell_type):
        w_ref, a_ref, b_ref = tri3_quad_rule()
    elif _is_quad(cell_type):
        w_ref, a_ref, b_ref = quad4_quad_rule()
    else:
        raise ValueError(f"Unsupported domain cell type {cell_type!r}")

    nq = len(w_ref)
    for cell in cells_local:
        node_xy = points_xy[cell]
        qw = np.empty(nq, dtype=float)
        qn = np.empty((nq, len(cell)), dtype=float)
        gnx = np.empty((nq, len(cell)), dtype=float)
        gny = np.empty((nq, len(cell)), dtype=float)
        jdet = np.empty(nq, dtype=float)
        qxy = np.empty((nq, 2), dtype=float)
        for q in range(nq):
            n, dna, dnb = shape_at_ref(cell_type, float(a_ref[q]), float(b_ref[q]))
            gx, gy, det = physical_grad(node_xy, dna, dnb)
            qw[q] = w_ref[q]
            qn[q] = n
            gnx[q] = gx
            gny[q] = gy
            jdet[q] = det
            qxy[q] = n @ node_xy
        tables.num_quad.append(nq)
        tables.quad_w.append(qw)
        tables.quad_n.append(qn)
        tables.quad_gnx.append(gnx)
        tables.quad_gny.append(gny)
        tables.jac_det.append(jdet)
        tables.quad_xy.append(qxy)
    return tables


def edge_unit_n_and_jac(
    p0: np.ndarray,
    p1: np.ndarray,
    lin_a: float,
    cell_centroid: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Unit outward normal, edge jacobian |τ|, and physical qpt for Lin2 edge."""
    n1d = lin2_eval(lin_a)
    dna = lin2_grad_ref(lin_a)
    dxda = float(dna[0] * p0[0] + dna[1] * p1[0])
    dyda = float(dna[0] * p0[1] + dna[1] * p1[1])
    # outward candidate matching Rust (-inward): (dyda, -dxda)
    n_out = np.array([dyda, -dxda], dtype=float)
    mid = n1d[0] * p0 + n1d[1] * p1
    if np.dot(n_out, mid - cell_centroid) < 0.0:
        n_out = -n_out
    jac = float(np.linalg.norm(n_out))
    if jac < 1e-30:
        return np.zeros(2, dtype=float), 0.0, mid
    return n_out / jac, jac, mid


def eval_dom_at_edge_qpt(
    cell: np.ndarray,
    cell_type: str,
    points_xy: np.ndarray,
    l0: int,
    l1: int,
    lin_a: float,
    scl_nodal: np.ndarray | None = None,
    vec_nodal: np.ndarray | None = None,
) -> dict:
    """Evaluate Dom fields / gradients at a Lin2 edge quadrature parameter.

    Returns dict with keys: u, grad (2,), vel (2,), grad_v (2,2) as available.
    ``grad_v[i, j]`` = ∂v_i / ∂x_j.
    """
    face, same = edge_directed_index(cell, l0, l1)
    xi, eta = parent_ref_on_face(cell_type, face, lin_a, same)
    n, dna, dnb = shape_at_ref(cell_type, xi, eta)
    node_xy = points_xy[cell]
    gnx, gny, _ = physical_grad(node_xy, dna, dnb)

    out: dict = {}
    if scl_nodal is not None:
        vals = scl_nodal[cell]
        out["u"] = float(n @ vals)
        out["grad"] = np.array([float(gnx @ vals), float(gny @ vals)], dtype=float)
    if vec_nodal is not None:
        vx = vec_nodal[cell, 0]
        vy = vec_nodal[cell, 1]
        out["vel"] = np.array([float(n @ vx), float(n @ vy)], dtype=float)
        # grad_v[i,j] = ∂v_i/∂x_j
        out["grad_v"] = np.array(
            [
                [float(gnx @ vx), float(gny @ vx)],
                [float(gnx @ vy), float(gny @ vy)],
            ],
            dtype=float,
        )
    return out
