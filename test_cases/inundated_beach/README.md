### Inundated beach test case

"1D uniformly sloping basin" test case as described on p.92 of [1].

Many thanks to [Mariana Clare][1] for coding the fixed mesh version of this test case.

### Original problem:
  * Domain length: 13,800m.
  * Uniformly sloping bathymetry: 5m at the inflow and 0m at the dry end.
  * Initial mesh: grid step of 1,200m.
  * Strickler bed roughness coefficient: 50 m^{1/3} s^{-1}.
  * Timestep: 600s.
  * Wetting and drying alpha: 0.43m.

### Differences in this implementation:
  * Uniformly sloping bathymetry: 5m at the inflow and -2.5m at the dry end.
  * Time period: 24h.

### Boundary conditions:
Sinusoidal variation of elevation at the open boundary, with amplitude 0.5m and period 12h.

### Quantity of interest:
Integral of modified surface elevation over the originally dry region in both space and time.

[1] A. Balzano, "Evaluation of methods for numerical simulation of wetting and drying in shallow
    water flow models." Coastal Engineering 34.1-2 (1998): 83-107.

[1]: http://www.imperial.ac.uk/people/m.clare17 "Mariana Clare"
