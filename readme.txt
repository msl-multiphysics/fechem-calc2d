 _____ _____ ____ _                    
|  ___| ____/ ___| |__   ___ _ __ ___  
| |_  |  _|| |   | '_ \ / _ \ '_ ` _ \ 
|  _| | |__| |___| | | |  __/ | | | | |
|_|   |_____\____|_| |_|\___|_| |_| |_|

Finite Element Method Solver for Chemical Engineering Applications
2D Finite Element Method Post-Processing Utility
Copyright (c) 2026 FEChem Development Team

Overview
--------
This is a post-processing utility for FEChem, a finite element method (FEM) solver for chemical engineering applications.
It loads Gmsh meshes and FEChem VTU field files, and evaluates domain and boundary integrals.

Note: The project is under active development. The public API and numerical models may change between releases.

Features
--------
- Loads Gmsh `.msh` files with domain, boundary, and integral physical groups
- Loads domain VTU PointData (`value`) as NumPy arrays
- Interactive mesh visualization with PyVista, including group indices
- Surface integrals over a 2D domain
- Diffusive line integrals on 1D boundaries or interfaces (see warning below)
- Advective line integrals on 1D boundaries or interfaces

Warning: Diffusive fluxes are estimated by reconstructing gradients from adjacent elements' nodal values.
The computed diffusive fluxes are NOT NECESSARILY CONSERVATIVE given the nature of P1 FEM.

Requirements
------------
- Python 3.9 or newer
- Dependencies: numpy, meshio, pyvista, matplotlib, and pillow (installed with the package)

Quick Start
-----------
Clone or download the project, then install it in editable mode from the project root:

    pip install -e .

Run an example from the `examples` folder, for instance:

    python examples/calc_heat_multi.py

This loads the three-region heat example mesh and VTUs and prints surface and line integrals.

Project Structure
-----------------
- `src/fechem_calc2d/` — `Calc2D` API, VTU loading, integrals, and visualization
- `examples/` — example scripts
- `examples/cases/` — sample Gmsh meshes, FEChem case sources, and VTU outputs

Contributing
------------
Due to limited maintenance capacity, we are not currently accepting external contributions.
However, users are welcome to fork and modify the repository in accordance with the MIT License.

License
-------
FEChem calc2d is available under the MIT License. See license.txt for details.
