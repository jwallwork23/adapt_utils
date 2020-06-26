### Box Discharge with Diffusion

Test case as described in [Power et al.].


### Parameters:
  * Domain: [0, 4]x[0, 4]
  * Fluid velocity: (15.0, 0.0)
  * Diffusivity coefficient: 1.0
  * Source location:
    * Aligned case: (1.0, 2.0)
    * Offset case: (1.0, 1.5)
  * Box diameter: 0.2

### Boundary conditions:
  * Dirichlet zero on inflow x=0
  * Outflow condition on x=50
  * Neumann zero elsewhere

### Quantity of interest:
Integral of tracer concentration over region of interest, given by a box of radius 0.2, centred at
  * Centred case: (3.0, 2.0)
  * Offset case: (3.0, 2.5)

### References

[Power et al.] P.W. Power et al. "Adjoint a posteriori error measures for anisotropic mesh
optimisation", Computers & Mathematics with Applications 52.8-9 (2006): 1213-1242.
