## Bubble shear test case

Original problem described in [McManus et al.].

### Original problem

* Unsteady advection
* Unit square domain
* BCs: Dirichlet zero on inflow
* ICs: hat function
* Zero diffusivity
* Time duration: `1.5`
* Monitor function: m(x)=sqrt(1+grad(u(x))^2), using method of Cenicerous & Hou

### References

[McManus et al.] T.M. McManus, J.R. Percival, B.A. Yeager, N. Barral, G.J. Gorman, M.D. Piggott,
"Moving mesh methods in Fluidity and Firedrake" (2015).
