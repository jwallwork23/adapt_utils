## Solid body rotation test case:

Original problem, as described in [1], concerns the advection of three solid bodies in a rotating
fluid: a Gaussian bell, a cone and a slotted cylinder. In this test case, we seek to advect the
slotted cylinder conservatively and are not interested in the advection of the other bodies.

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


[1] R. LeVeque, "High-resolution conservative algorithms for advection in incompressible flow" (1996)
