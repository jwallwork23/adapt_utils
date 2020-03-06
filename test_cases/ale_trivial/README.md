### Trivial Arbitrary Lagrangian-Eulerian advection

This simple example illustrates a case where the Lagrangian interpretation yields a lower relative
L2 error norm than the Eulerian interpretation.

### Parameters

* Domain: [0,10]x[0,10] square with periodicity in x-direction
* Spatial discretisation: P1 elements with SUPG stabilisation
* Temporal discretisation: implicit midpoint rule
* BCs: free-slip on non-periodic boundaries
* IC: Gaussian bump exp(-((x-0.5)^2 + (y-0.5)^2))
* Zero diffusivity
* Time duration: 10
* Time step: 0.2
* Interpretation switch: 'eulerian' vs. 'lagrangian'
