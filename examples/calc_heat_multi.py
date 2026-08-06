from fechem_calc2d import Calc2D

# load gmsh mesh
calc = Calc2D("examples/cases/gmsh_threereg.msh")

# option to view the mesh
# calc.show_mesh()

# load a nodal scalar VTU into one of the domains (dependent variable only)
# arguments: domain index, vtu file path
T_v1 = calc.read_scl(1, "examples/cases/output_heat_multi/temp_b_0.vtu")  # returns np.array with temperature (T)

# surface integral (Q dS); Q may be a constant or a function of T
# arguments: domain index, source (float or callable), scalar field
# should work even if the heat source is constant
heat_v1 = calc.surfint_scl_src(1, -500.0, T_v1)  # returns float64
# heat_v1 = calc.surfint_scl_src(1, lambda T: -500.0, T_v1)  # returns float64
print(heat_v1)

# diffusive line integral (-k * grad(T) . n dl); n is outward unit normal
# arguments: boundary or interface index, diffusivity (float or callable), scalar field
# should work even if the diffusivity is constant
# WARNING: diffusive fluxes are not necessarily conservative (see readme.txt).
q_s4 = calc.lineint_scl_diff(4, 1.0, T_v1)  # returns float64
# q_s4 = calc.lineint_scl_diff(4, lambda T: 1.0, T_v1)  # returns float64
print(q_s4)
