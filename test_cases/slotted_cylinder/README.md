## Solid body rotation test case:
"High-resolution conservative algorithms for advection in incompressible flow" (1996), R. LeVeque

### Original problem

* Unsteady advection diffusion
* Unit square domain
* BCs: Dirichlet zero on inflow
* ICs: three solid bodies
* Zero diffusivity
* Time duration: `2*pi`
* Time step: `pi/300`
* Solver: `'CrankNicolson'`

### This implementation

* P1DG with upwinding (see Colin's Firedrake demo)

### Remarks

* Adjoint based error esimators not particularly useful.
* Metric advection could be very useful
