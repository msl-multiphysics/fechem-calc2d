from fechem_calc2d import Calc2D

# load gmsh mesh
calc = Calc2D("examples/cases/gmsh_lid.msh")

# option to view the mesh
# calc.show_mesh()

# load nodal VTU files into one of the domains (dependent variables only)
# arguments: domain index, vtu file path
vel = calc.read_vec(0, "examples/cases/output_flow_lid/vel_0.vtu")  # returns (n, 2) np.array with velocity (v)
pres = calc.read_scl(0, "examples/cases/output_flow_lid/pres_0.vtu")  # returns np.array with pressure (p)

# force on the top lid (boundary 3)
# viscous contribution (-tau . n dl); returns force of fluid on the lid
# arguments: boundary or interface index, viscosity (float or callable), velocity
# should work even if the viscosity is constant
F_visc = calc.lineint_vec_visc(3, 0.001, vel)  # returns np.array shape (2,)
# F_visc = calc.lineint_vec_visc(3, lambda v: 0.001, vel)  # returns np.array shape (2,)
print(F_visc)

# pressure contribution (p * n dl); returns force of fluid on the lid
# arguments: boundary or interface index, pressure field
F_pres = calc.lineint_vec_pres(3, pres)  # returns np.array shape (2,)
print(F_pres)

# convective contribution (rho * (v . n) * v dl); returns a 2-vector flux
# arguments: boundary or interface index, density (float or callable), velocity
# should work even if the density is constant
F_adv = calc.lineint_vec_adv(3, 1000.0, vel)  # returns np.array shape (2,)
# F_adv = calc.lineint_vec_adv(3, lambda v: 1000.0, vel)  # returns np.array shape (2,)
print(F_adv)

# total hydrodynamic force of the fluid on the top lid
F_lid = F_visc + F_pres + F_adv  # returns np.array shape (2,)
print(F_lid)
