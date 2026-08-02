from fechem_calc2d import Calc2D

# load gmsh mesh
calc = Calc2D("examples/cases/gmsh_threereg.msh")

# option to view the mesh
# calc.show_mesh()

# load vtu files into one of the domains
# arguments: domain index, vtu file path
temp_v0 = calc.load_vtu(0, "examples/cases/output_heat_multi/temp_m_0.vtu")  # returns np.array with temperature (T)
cond_v0 = calc.load_vtu(0, "examples/cases/output_heat_multi/cond_m_0.vtu")  # returns np.array with thermal conductivity (k)
hsrc_v0 = calc.load_vtu(0, "examples/cases/output_heat_multi/hsrc_m_0.vtu")  # returns np.array with volumetric heat source (Q)

# surface integral (Q dS)
# arguments: domain index, scalar field
# should work even if heat source is constant
heat_v0 = calc.surfint(0, hsrc_v0)  # returns float64
# heat_v0 = calc.surfint(0, 0.0)  # returns float64
print(heat_v0)

# diffusive line integral (k * grad(T) . n dl); n is outward unit normal
# arguments: boundary or interface index, diffusion coefficient, scalar field
# should work even if diffusion is constant
diff_s4 = calc.lineint_diff(4, cond_v0, temp_v0)  # returns float64
# diff_s4 = calc.lineint_diff(4, 0.5, temp_v0)  # returns float64
print(diff_s4)
