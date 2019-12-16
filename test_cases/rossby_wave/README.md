## Equatorial Rossby wave test case
"Rossby equatorial soliton", Huang et al., FVCOM validation experiments (2008), pp. 3-6.

### Original problem

* Unsteady shallow water: "propagation of a small amplitude Rossby soliton on an equatorial beta
  plane"
* Zonal equatorial channel, [-24, 24]x[-12, 12].
* BCs: rigid walls to N and S; periodic E-W.
* ICs: "modon with two sea level peaks of equal size and strength decaying exponentially with
  distance away from their centers"
* Zero diffusivity.
* Time duration: 120.
* Time steps considered: 0.01, 0.005, 0.002
* Solver: finite volume.

* Beta-plane Coriolis approximation `f = f0 + beta*y` with `f0 = 0`, `beta = 1`.
* Non-dimensionalised, constant bathymetry `b = 1`.
* Non-dimensionalised, constant gravitational acceleration `g = 1`.
* Zeroth order asymptotic solution derived by Boyd (1980).
* First order asymptotic solution derived by Boyd (1985).

### This implementation

* P1DG-P1DG mixed finite element space for velocity and free surface.
* Periodic BC ignored for mesh adaptive case.

### Remarks

* "This is a good test case for examining the dispersion and numerical damping of a given model
  because the shape preservation and constant translation speed of the soliton wave are achieved
  through a delicate balance between nonlinearity and dispersion."
* Illustrates a situation where adjoint based error estimators are not useful.
* Flow is constrained to a band... suitable candidate for mesh optimisation?

