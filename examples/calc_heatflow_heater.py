from fechem_calc2d import Calc2D

# load gmsh mesh
calc = Calc2D("examples/cases/gmsh_heater.msh")

# option to view the mesh
# calc.show_mesh()

# load vtu files into one of the domains
# arguments: domain index, vtu file path
temp_v0 = calc.load_vtu(0, "examples/cases/output_heatflow_heater/temp_c_0.vtu")  # np.array with temperature (T)
vel_v0 = calc.load_vtu(0, "examples/cases/output_heatflow_heater/vel_0.vtu")  # 2D np.array with velocity (v)
cond_v0 = calc.load_vtu(0, "examples/cases/output_heatflow_heater/cond_c_0.vtu")  # np.array with thermal conductivity (k)
hsrc_v0 = calc.load_vtu(0, "examples/cases/output_heatflow_heater/hsrc_c_0.vtu")  # np.array with volumetric heat source (Q)
vlcp_v0 = calc.load_vtu(0, "examples/cases/output_heatflow_heater/vlcp_c_0.vtu")  # np.array with volumetric heat capacity (rho * cp)

# diffusive line integral (k * grad(T) . n dl); n is outward unit normal
# arguments: boundary or interface index, diffusion coefficient, scalar field
# should work even if diffusion is constant
diff_s1 = calc.lineint_diff(1, cond_v0, temp_v0)  # float64
# diff_s1 = calc.lineint_diff(1, 0.5, temp_v0)  # float64
print(diff_s1)

# advective line integral ((rho * cp) * v * T dl); n is outward unit normal
# arguments: boundary or interface index, weight (rho * cp), velocity, scalar field
adv_s1 = calc.lineint_adv(1, vlcp_v0, vel_v0, temp_v0)  # float64
# adv_s1 = calc.lineint_adv(1, 100.0, vel_v0, temp_v0)  # float64
# adv_s1 = calc.lineint_adv(1, 100.0, (0.002, 0.0), temp_v0)  # float64
print(adv_s1)
