use fechem_fem2d::*;
use std::fs::create_dir_all;

/// Steady-state heat equation with multiple domains.
/// Run with: `cargo run --release --example heat_multi`
/// 
/// Geometry:
/// Square domain with bottom-left and top-right corner rectangles.
/// 
///        bnd_3  bnd_2
///       +-------------+
/// bnd_3 |     | dom_2 | bnd_2
///       |     +-------+
///       |    dom_0    |
///       +-------+     | bnd_1
/// bnd_0 | dom_1 |     |
///       +-------------+
///         bnd_0  bnd_1
///
/// - itf_4 - between dom_0 and dom_1
/// - itf_5 - between dom_0 and dom_2
///
/// Domains:
/// 0 - middle domain
/// 1 - bottom-left rectangle (0.75 m x 0.25 m)
/// 2 - top-right rectangle (0.75 m x 0.25 m)
/// 
/// Boundaries:
/// 0 - bottom-left boundary
/// 1 - bottom-right boundary
/// 2 - top-right boundary
/// 3 - top-left boundary
///
/// Interfaces:
/// 4 - bottom-middle interface
/// 5 - top-middle interface
/// 
/// Properties:
/// dom_0 - thermal conductivity (k = 0.5 W m-1 K-1)
/// dom_1 - thermal conductivity (k = 1.0 W m-1 K-1)
/// dom_2 - thermal conductivity (k = 1.0 W m-1 K-1)
/// dom_0 - heat source (Q = 0 W m-3)
/// dom_1 - heat source (Q = -500.0 W m-3)
/// dom_2 - heat source (Q = +500.0 W m-3)
///
/// Boundary conditions:
/// bnd_0 - temperature (T = 300 K)
/// bnd_1 - temperature (T = 300 K)
/// bnd_2 - temperature (T = 300 K)
/// bnd_3 - temperature (T = 300 K)
///
/// Interface conditions:
/// itf_4 - temperature and flux continuity
/// itf_5 - contact resistance (0.1 W m-2 K-1)
///
fn main() -> Result<(), FEChemError> {
    // output directory
    create_dir_all("examples/output_heat_multi").unwrap();

    // mesh and variables
    // new - import mesh from gmsh file
    // arguments: input_file
    let mut vars = Variables::new("examples/gmsh/gmsh_threereg.msh".to_string())?;

    // geometry
    // gmsh counts 1D and 2D physical groups from 1
    // subtract 1 to get FEChem domain and boundary indices
    // for interfaces, the order of the domains does not matter
    let dom_m = vars.add_dom(0)?;  // middle
    let dom_b = vars.add_dom(1)?;  // bottom
    let dom_t = vars.add_dom(2)?;  // top
    let bnd_lb = vars.add_bnd(dom_b, 0)?;  // bottom-left
    let bnd_br = vars.add_bnd(dom_m, 1)?;  // bottom-right
    let bnd_rt = vars.add_bnd(dom_t, 2)?;  // top-right
    let bnd_tb = vars.add_bnd(dom_m, 3)?;  // top-left
    let itf_bm = vars.add_itf(dom_b, dom_m, 4)?;  // bottom-middle
    let itf_tm = vars.add_itf(dom_t, dom_m, 5)?;  // top-middle

    // unknown domain scalars
    // arguments: domain, initial_value, output_file
    // initial_value - initial guess for steady-state problems; initial_value for transient problems
    // output_file - can be .csv or .vtu; if empty string, no file is written
    let temp_m = vars.add_scldom_unk(dom_m, 0.0, "examples/output_heat_multi/temp_m.vtu".to_string())?;
    let temp_b = vars.add_scldom_unk(dom_b, 0.0, "examples/output_heat_multi/temp_b.vtu".to_string())?;
    let temp_t = vars.add_scldom_unk(dom_t, 0.0, "examples/output_heat_multi/temp_t.vtu".to_string())?;

    // constant domain scalars
    // arguments: domain, value, output_file
    let cond_m = vars.add_scldom_con(dom_m, 0.5, "".to_string())?;  // thermal conductivity
    let cond_b = vars.add_scldom_con(dom_b, 1.0, "".to_string())?;  // thermal conductivity
    let cond_t = vars.add_scldom_con(dom_t, 1.0, "".to_string())?;  // thermal conductivity
    let hsrc_m = vars.add_scldom_con(dom_m, 0.0, "".to_string())?;  // heat source (positive if source; negative if sink)
    let hsrc_b = vars.add_scldom_con(dom_b, -500.0, "".to_string())?;  // heat source (positive if source; negative if sink)
    let hsrc_t = vars.add_scldom_con(dom_t, 500.0, "".to_string())?;  // heat source (positive if source; negative if sink)

    // constant boundary scalars
    // arguments: boundary, value, output_file
    let temp_lb = vars.add_sclbnd_con(bnd_lb, 300.0, "".to_string())?;  // temperature
    let temp_br = vars.add_sclbnd_con(bnd_br, 300.0, "".to_string())?;  // temperature
    let temp_rt = vars.add_sclbnd_con(bnd_rt, 300.0, "".to_string())?;  // temperature
    let temp_tb = vars.add_sclbnd_con(bnd_tb, 300.0, "".to_string())?;  // temperature

    // unknown interface scalars
    // arguments: interface, initial_value, output_file
    // lagrange multipliers are needed for continuity interfaces
    let lmd_bm = vars.add_sclitf_unk(itf_bm, 0.0, "".to_string())?;  // lagrange multiplier
    
    // constant interface scalars
    // arguments: interface, value, output_file
    // contact resistance is needed for contact resistance interfaces
    let hres_tm = vars.add_sclitf_con(itf_tm, 0.1, "".to_string())?;  // contact resistance

    // steady-state heat transfer solver
    // add_heat_dom - register domain with heat transfer
    // add_temp_bnd - register boundary with temperature
    // add_cont_itf - register continuity interface
    // add_hres_itf - register contact resistance interface
    let mut phys = SteadyHeat::new();
    phys.add_heat_dom(dom_m, temp_m, cond_m, hsrc_m);  // arguments: domain, T, k, Q
    phys.add_heat_dom(dom_b, temp_b, cond_b, hsrc_b);  // arguments: domain, T, k, Q
    phys.add_heat_dom(dom_t, temp_t, cond_t, hsrc_t);  // arguments: domain, T, k, Q
    phys.add_temp_bnd(bnd_lb, temp_lb);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_br, temp_br);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_rt, temp_rt);  // arguments: boundary, T
    phys.add_temp_bnd(bnd_tb, temp_tb);  // arguments: boundary, T
    phys.add_cont_itf(itf_bm, lmd_bm);  // arguments: interface, lagrange multiplier
    phys.add_hres_itf(itf_tm, hres_tm);  // arguments: interface, contact resistance

    // physics solver
    // arguments: max_iter, tol, damping_factor
    // damping_factor - between 0.0 and 1.0; lower for stability and higher for speed (if linear or nearly linear)
    // for highly non-linear problems, using a lower damping factor (e.g., 0.8-0.9) may be faster
    let linsolve = SolverLu::new(1)?;
    phys.solve(&mut vars, Box::new(linsolve), 10, 1e-3, 1.0)?;

    Ok(())
}
