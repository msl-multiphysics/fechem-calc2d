"""VTU loading and property / field argument helpers."""

from __future__ import annotations

from typing import Any

import meshio
import numpy as np


def read_vtu_values(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read VTU points and PointData ``value`` (scalar or 2D vector)."""
    mesh = meshio.read(path)
    points = np.asarray(mesh.points, dtype=float)
    if "value" not in mesh.point_data:
        raise ValueError(f"VTU {path!r} has no PointData array named 'value'")
    values = np.asarray(mesh.point_data["value"], dtype=float)
    if values.ndim == 1:
        pass
    elif values.ndim == 2 and values.shape[1] >= 2:
        values = values[:, :2]
    else:
        raise ValueError(
            f"Unsupported PointData 'value' shape {values.shape} in {path!r}"
        )
    return points, values


def remap_vtu_to_domain(
    vtu_points: np.ndarray,
    vtu_values: np.ndarray,
    dom_points_xy: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Place VTU nodal values onto Dom-local ordering via nearest coordinates.

    FEChem VTUs round coordinates to 6 decimals, so exact equality with the
    Gmsh mesh fails; match each VTU node to the nearest Dom node within ``tol``.
    """
    n_dom_nodes = len(dom_points_xy)
    if len(vtu_points) != n_dom_nodes:
        raise ValueError(
            f"VTU has {len(vtu_points)} nodes but domain has {n_dom_nodes}"
        )
    if vtu_values.ndim == 1:
        out = np.empty(n_dom_nodes, dtype=float)
    else:
        out = np.empty((n_dom_nodes, vtu_values.shape[1]), dtype=float)

    seen = np.zeros(n_dom_nodes, dtype=bool)
    for i, pt in enumerate(vtu_points):
        d2 = np.sum((dom_points_xy - pt[:2]) ** 2, axis=1)
        loc = int(np.argmin(d2))
        dist = float(np.sqrt(d2[loc]))
        if dist > tol:
            raise ValueError(
                f"VTU node at ({pt[0]}, {pt[1]}) is {dist:g} from nearest "
                f"domain node (tol={tol:g})"
            )
        if seen[loc]:
            raise ValueError(
                f"Multiple VTU nodes map to the same domain node (local {loc})"
            )
        out[loc] = vtu_values[i]
        seen[loc] = True
    if not np.all(seen):
        missing = int(np.count_nonzero(~seen))
        raise ValueError(f"VTU did not cover {missing} domain node(s)")
    return out


def eval_prop_scl(prop: Any, unk_q: float) -> float:
    """Evaluate a scalar property: constant or ``Callable[[float], float]``."""
    if callable(prop) and not np.isscalar(prop):
        return float(prop(unk_q))
    return float(prop)


def eval_prop_scl_of_vec(prop: Any, vel_q: np.ndarray) -> float:
    """Evaluate a scalar property of a vector: constant or ``Callable[[ndarray], float]``."""
    if callable(prop) and not np.isscalar(prop):
        return float(prop(vel_q))
    return float(prop)


def eval_prop_vec(prop: Any, vec_q: np.ndarray) -> np.ndarray:
    """Evaluate a vector property: float scale, or ``Callable[[ndarray], ndarray]``."""
    if callable(prop) and not np.isscalar(prop):
        out = np.asarray(prop(vec_q), dtype=float).reshape(2)
        return out
    return float(prop) * np.asarray(vec_q, dtype=float)
