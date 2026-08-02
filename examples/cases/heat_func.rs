use fechem_fem2d::*;
use std::fs::create_dir_all;

/// Steady-state heat equation with non-constant properties.
/// Run with: `cargo run --release --example heat_func`
///
/// Geometry:
/// Square domain defined by lengths along each axis.
/// 
///          bnd_3
///       +---------+
///       |         |
/// bnd_0 |  dom_0  | bnd_1
///       |         |
///       +---------+
///          bnd_2
///
/// Domains:
/// 0 - main domain (1.0 m x 1.0 m)
/// 
/// Boundaries:
/// 0 - left boundary (1.0 m)
/// 1 - right boundary (1.0 m)
/// 2 - bottom boundary (1.0 m)
/// 3 - top boundary (1.0 m)
///
/// Properties:
/// [non-constant properties in SI units]
/// dom_0 - thermal conductivity (k = 0.1 + 0.3 * T)
/// dom_0 - heat source (Q = -200.0 - 0.5 * T)
///
/// Boundary conditions:
/// [non-constant properties in SI units]
/// bnd_0 - outward flux (q = 10.0 + 0.1 * T)
/// bnd_1 - temperature (T = 300 K)
/// bnd_2 - no flux (q = 0.0 W m-2)
/// bnd_3 - temperature (T = 400 K)
///
fn main() -> Result<(), FEChemError> {
    // output directory
    create_dir_all("examples/output_heat_func").unwrap();

    // mesh and variables
    // new_from_bounds - create rectangular mesh from bounding box
    // arguments: x_min, y_min, x_max, y_max, num_elem_x, num_elem_y
    let mut vars = Variables::new_from_bounds(0.0, 0.0, 1.0, 1.0, 20, 20)?;

    // geometry
    // if using new_from_bounds, the domain is region 0
    // the left, right, bottom, and top boundaries are regions 0, 1, 2, and 3
    let dom = vars.add_dom(0)?;
    let bnd_l = vars.add_bnd(dom, 0)?;  // left
    let bnd_r = vars.add_bnd(dom, 1)?;  // right
    let bnd_b = vars.add_bnd(dom, 2)?;  // bottom
    let bnd_t = vars.add_bnd(dom, 3)?;  // top

    // unknown domain scalars
    // arguments: domain, initial_value, output_file
    // initial_value - initial guess for steady-state problems; initial_value for transient problems
    // output_file - can be .csv or .vtu; if empty string, no file is written
    let temp = vars.add_scldom_unk(dom, 0.0, "examples/output_heat_func/temp.vtu".to_string())?;

    // non-constant domain scalars
    // arguments: domain, value_func, scalar_ids, output_file
    // value_func - returns value of scalar as a function of time and *unknown* scalars
    // - time is zero for steady-state problems
    // - scalar values are given in the same order as in scalar_ids
    let cond_func = |_t: f64, scl: &[f64]| 0.1 + 0.3 * scl[0];  // scl[0] is T
    let hsrc_func = |_t: f64, scl: &[f64]| -(200.0 + 0.5 * scl[0]);
    let cond = vars.add_scldom_fun(dom, Box::new(cond_func), vec![temp], "".to_string())?;  // scalar_ids[0] is also T
    let hsrc = vars.add_scldom_fun(dom, Box::new(hsrc_func), vec![temp], "".to_string())?;

    // non-constant boundary scalars
    // arguments: boundary, value_func, scalar_ids, output_file
    let hflx_func = |_t: f64, scl: &[f64]| 10.0 + 0.1 * scl[0];  // scl[0] is T
    let hflx_l = vars.add_sclbnd_fun(bnd_l, Box::new(hflx_func), vec![temp], "".to_string())?;  // scalar_ids[0] is also T
    
    // constant boundary scalars
    // arguments: boundary, value, output_file
    let temp_r = vars.add_sclbnd_con(bnd_r, 300.0, "".to_string())?;  // temperature
    let hflx_b = vars.add_sclbnd_con(bnd_b, 0.0, "".to_string())?;  // heat flux
    let temp_t = vars.add_sclbnd_con(bnd_t, 400.0, "".to_string())?;  // temperature

    // steady-state heat transfer solver
    // add_heat_dom - register domain with heat transfer
    // add_hflx_bnd - register boundary with heat flux
    // add_temp_bnd - register boundary with temperature
    let mut phys = SteadyHeat::new();
    phys.add_heat_dom(dom, temp, cond, hsrc);  // arguments: domain, T, k, Q
    phys.add_hflx_bnd(bnd_l, hflx_l);  // arguments: boundary, q
    phys.add_temp_bnd(bnd_r, temp_r);  // arguments: boundary, T
    phys.add_hflx_bnd(bnd_b, hflx_b);  // arguments: boundary, q
    phys.add_temp_bnd(bnd_t, temp_t);  // arguments: boundary, T

    // physics solver
    // arguments: max_iter, tol, damping_factor
    // damping_factor - between 0.0 and 1.0; lower for stability and higher for speed (if linear or nearly linear)
    // for highly non-linear problems, using a lower damping factor (e.g., 0.8-0.9) may be faster
    let linsolve = SolverLu::new(1)?;
    phys.solve(&mut vars, Box::new(linsolve), 10, 1e-3, 1.0)?;

    Ok(())
}
