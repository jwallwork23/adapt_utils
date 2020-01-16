## Solid body rotation test case:

Original problem, as described in [LeVeque], concerns the advection of three solid bodies in a
rotating fluid: a Gaussian bell, a cone and a slotted cylinder. In this test case, we seek to
advect the slotted cylinder conservatively and are not interested in the advection of the other
bodies.

A fixed mesh version of this test case was coded in Firedrake as [demo][1].

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

* P1DG with upwinding (see [demo][1]).

### Remarks

* Adjoint based error esimators not particularly useful.
* Metric advection could be very useful

### References

[LeVeque] R. LeVeque, "High-resolution conservative algorithms for advection in incompressible flow"
(1996)

[1]: https://firedrakeproject.org/demos/DG_advection.py.html "this demo"
