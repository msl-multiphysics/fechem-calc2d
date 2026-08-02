"""Gauss quadrature domain and boundary integrals."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fields import eval_prop_scl, eval_prop_scl_of_vec, eval_prop_vec
from .quadrature import (
    DomQuadTables,
    edge_unit_n_and_jac,
    eval_dom_at_edge_qpt,
    lin2_quad_rule,
)


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


def _resolve_edge_cell(
    edge: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    edge_adj: dict,
) -> tuple[int, int, np.ndarray]:
    g0, g1 = int(edge[0]), int(edge[1])
    if g0 not in global_to_local or g1 not in global_to_local:
        raise ValueError(
            f"Boundary edge nodes {g0}, {g1} are not in domain {field_dom}"
        )
    l0 = global_to_local[g0]
    l1 = global_to_local[g1]
    key = frozenset((l0, l1))
    candidates = [item for item in edge_adj.get(key, []) if item[0] == field_dom]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one adjacent cell in domain {field_dom} for edge "
            f"{g0}-{g1}, found {len(candidates)}"
        )
    _, cell, _ = candidates[0]
    return l0, l1, cell


def _iter_edge_qpts(
    points: np.ndarray,
    edge: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    edge_adj: dict,
):
    """Yield (l0, l1, cell, lin_a, unit_n, jac) for each Lin2 quadrature point."""
    l0, l1, cell = _resolve_edge_cell(edge, global_to_local, field_dom, edge_adj)
    g0, g1 = int(edge[0]), int(edge[1])
    p0 = points[g0, :2]
    p1 = points[g1, :2]
    centroid = np.mean(dom_points_xy[cell], axis=0)
    _, a_ref = lin2_quad_rule()
    for a in a_ref:
        a = float(a)
        unit_n, jac, _ = edge_unit_n_and_jac(p0, p1, a, centroid)
        if jac < 1e-30:
            continue
        yield l0, l1, cell, a, unit_n, jac


def surfint_scl_src(
    tables: DomQuadTables,
    cells_local: np.ndarray,
    src: Any,
    unk: np.ndarray,
) -> float:
    """∫ src(u) dΩ over Dom cells."""
    total = 0.0
    for eid, cell in enumerate(cells_local):
        nq = tables.num_quad[eid]
        qw = tables.quad_w[eid]
        qn = tables.quad_n[eid]
        jdet = tables.jac_det[eid]
        u_nodes = unk[cell]
        for q in range(nq):
            u_q = float(qn[q] @ u_nodes)
            s = eval_prop_scl(src, u_q)
            # abs(jac_det): some Gmsh regions have consistently reversed orientation
            total += s * float(qw[q]) * abs(float(jdet[q]))
    return total


def surfint_vec_src(
    tables: DomQuadTables,
    cells_local: np.ndarray,
    src: Any,
    fce: np.ndarray,
) -> np.ndarray:
    """∫ src(f) dΩ over Dom cells; returns length-2 vector."""
    total = np.zeros(2, dtype=float)
    for eid, cell in enumerate(cells_local):
        nq = tables.num_quad[eid]
        qw = tables.quad_w[eid]
        qn = tables.quad_n[eid]
        jdet = tables.jac_det[eid]
        f_nodes = fce[cell]
        for q in range(nq):
            f_q = qn[q] @ f_nodes
            s = eval_prop_vec(src, f_q)
            total += s * float(qw[q]) * abs(float(jdet[q]))
    return total


def lineint_scl_diff(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    cell_type: str,
    edge_adj: dict,
    diff: Any,
    unk: np.ndarray,
) -> float:
    """∫ -diff(u) (∇u · n) dl on Bnd/Itf edges."""
    total = 0.0
    for edge in edge_cells:
        for l0, l1, cell, a, unit_n, jac in _iter_edge_qpts(
            points, edge, global_to_local, field_dom, dom_points_xy, edge_adj
        ):
            ev = eval_dom_at_edge_qpt(
                cell, cell_type, dom_points_xy, l0, l1, a, scl_nodal=unk
            )
            d = eval_prop_scl(diff, ev["u"])
            total += -d * float(np.dot(ev["grad"], unit_n)) * jac
    return total


def lineint_scl_adv(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    cell_type: str,
    edge_adj: dict,
    wgt: Any,
    vel: np.ndarray,
    unk: np.ndarray,
) -> float:
    """∫ wgt(u) u (v · n) dl on Bnd/Itf edges."""
    total = 0.0
    for edge in edge_cells:
        for l0, l1, cell, a, unit_n, jac in _iter_edge_qpts(
            points, edge, global_to_local, field_dom, dom_points_xy, edge_adj
        ):
            ev = eval_dom_at_edge_qpt(
                cell,
                cell_type,
                dom_points_xy,
                l0,
                l1,
                a,
                scl_nodal=unk,
                vec_nodal=vel,
            )
            w = eval_prop_scl(wgt, ev["u"])
            vn = float(np.dot(ev["vel"], unit_n))
            total += w * ev["u"] * vn * jac
    return total


def lineint_vec_visc(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    cell_type: str,
    edge_adj: dict,
    visc: Any,
    vel: np.ndarray,
) -> np.ndarray:
    """∫ -(τ · n) dl, the viscous force of fluid on the boundary."""
    total = np.zeros(2, dtype=float)
    for edge in edge_cells:
        for l0, l1, cell, a, unit_n, jac in _iter_edge_qpts(
            points, edge, global_to_local, field_dom, dom_points_xy, edge_adj
        ):
            ev = eval_dom_at_edge_qpt(
                cell, cell_type, dom_points_xy, l0, l1, a, vec_nodal=vel
            )
            mu = eval_prop_scl_of_vec(visc, ev["vel"])
            gv = ev["grad_v"]
            div = float(gv[0, 0] + gv[1, 1])
            tau = np.zeros((2, 2), dtype=float)
            for i in range(2):
                for j in range(2):
                    tau[i, j] = mu * (gv[i, j] + gv[j, i])
                tau[i, i] -= (2.0 / 3.0) * mu * div
            total += -(tau @ unit_n) * jac
    return total


def lineint_vec_adv(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    cell_type: str,
    edge_adj: dict,
    den: Any,
    vel: np.ndarray,
) -> np.ndarray:
    """∫ ρ (v · n) v dl on Bnd/Itf edges."""
    total = np.zeros(2, dtype=float)
    for edge in edge_cells:
        for l0, l1, cell, a, unit_n, jac in _iter_edge_qpts(
            points, edge, global_to_local, field_dom, dom_points_xy, edge_adj
        ):
            ev = eval_dom_at_edge_qpt(
                cell, cell_type, dom_points_xy, l0, l1, a, vec_nodal=vel
            )
            rho = eval_prop_scl_of_vec(den, ev["vel"])
            vn = float(np.dot(ev["vel"], unit_n))
            total += rho * vn * ev["vel"] * jac
    return total


def lineint_vec_pres(
    points: np.ndarray,
    edge_cells: np.ndarray,
    global_to_local: dict[int, int],
    field_dom: int,
    dom_points_xy: np.ndarray,
    cell_type: str,
    edge_adj: dict,
    pres: np.ndarray,
) -> np.ndarray:
    """∫ p n dl on Bnd/Itf edges."""
    total = np.zeros(2, dtype=float)
    for edge in edge_cells:
        for l0, l1, cell, a, unit_n, jac in _iter_edge_qpts(
            points, edge, global_to_local, field_dom, dom_points_xy, edge_adj
        ):
            ev = eval_dom_at_edge_qpt(
                cell, cell_type, dom_points_xy, l0, l1, a, scl_nodal=pres
            )
            total += ev["u"] * unit_n * jac
    return total
