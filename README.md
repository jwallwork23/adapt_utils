## Mesh Adaptation and Adjoint Methods for Coastal Ocean Modelling in Firedrake

In this code, adjoint methods and mesh adaptation are applied to solving a variety of 2D coastal ocean
modelling problems using the coastal, estuarine and ocean modelling solvers provided by [Thetis][2].
Thetis is built upon the [Firedrake][1] project, which enables the efficient solution of FEM problems
in Python by automatic generation of [PETSc][3] code.

Currently supported model components:
  * Shallow water equations (i.e. depth-averaged hydrodynamics);
  * Passive tracer transport (both non-conservative and conservative options available);
  * Sediment transport;
  * Exner equation.

Anisotropic mesh adaptation based on Riemannian metric fields is achieved using [PRAgMaTIc][4]. This
code provides a wide range of utilities for metric-based adaptation, including Hessian recovery,
metric combination, Lp normalisation and (isotropic and anisotropic) goal-oriented error estimators
and metrics. Monitor-based r-adaptation methods based on solutions of Monge-Ampere type equations is
also supported, as well as limited Lagrangian FEM functionality.

Continuous adjoint solvers are provided for shallow water and advection-diffusion problems. The
discrete adjoint code [dolfin-adjoint][5] can also be used to generate adjoint solutions for more general
problems.

This is research of the Applied Modelling and Computation Group ([AMCG][6]) at Imperial College
London.


### Publications

Listed below are publications which use `adapt_utils`, along with the corresponding code versions
and the test cases considered in those works. BibTeX formatted references are also available in
`docs/publications.bib`. The specific versions of Firedrake and Thetis used for each publication
are also shown below and can be downloaded from the corresponding links. To install a specific
version of Firedrake, please follow the instructions [here][7].

  * J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, <em>'Anisotropic Goal-Oriented Mesh Adaptation in Firedrake'</em>, In: 28th International Meshing Roundtable, pp.83-100, (2020).
    * [Paper][9]: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3653101.svg)](https://doi.org/10.5281/zenodo.3653101).
    * Code:
      * `adapt_utils`: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3358565.svg)](https://doi.org/10.5281/zenodo.3358565) (`v1.0`).
      * Firedrake: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3250888.svg)](https://doi.org/10.5281/zenodo.3250888).
    * Test cases:
      * `steady/test_cases/point_discharge2d` (Section 5.3);
      * `steady/test_cases/point_discharge3d` (Section 5.4).

  * J. G. Wallwork, N. Barral, S. C. Kramer, D. A. Ham, M. D. Piggott, <em>'Goal-Oriented Error Estimation and Mesh Adaptation for Shallow Water Modelling'</em>, Springer Nature Applied Sciences, volume 2, pp.1053--1063 (2020).
    * [Paper][10]: DOI: 10.1007/s42452-020-2745-9.
    * Code:
      * `adapt_utils`: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3695801.svg)](https://doi.org/10.5281/zenodo.3695801) (`v1.3`).
      * Firedrake: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3568997.svg)](https://doi.org/10.5281/zenodo.3568997).
      * Thetis: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3689727.svg)](https://doi.org/10.5281/zenodo.3689727).
    * Test case:
      * `steady/test_cases/turbine_array` (Section 4).
    
 * M. C. A. Clare, J. G. Wallwork, S. C. Kramer, H. Weller, C. J. Cotter, M. D. Piggott, <em> 'Multi-scale hydro-morphodynamic modelling using mesh movement methods'</em>, Submitted to International Journal on Geomathematics.
    * [Preprint][11]: DOI: 10.31223/osf.io/tpqvy.
    * Code:
      * `adapt_utils` [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4110708.svg)](https://doi.org/10.5281/zenodo.4110708) (`v2.2`).
      * Firedrake: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4110204.svg)](https://doi.org/10.5281/zenodo.4110204).
      * Thetis: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4778411.svg)](https://doi.org/10.5281/zenodo.4778411).
    * Test cases: `unsteady/test_cases/trench_1d` (Section 4.1); `unsteady/test_cases/trench_slant` (Section 4.1); `unsteady/test_cases/beach_slope` (Section 4.2); `unsteady/test_cases/tsunami_bump` (Section 4.3). For the versions in the paper use `hydro_morpho_paper` branch.

  * J. G. Wallwork, N. Barral, D. A. Ham, M. D. Piggott, <em>'Goal-Oriented Error Estimation and Mesh Adaptation for Tracer Transport Modelling'</em>, Submitted to Computer Aided Design (2021).
    * [Preprint][12]: DOI: 10.31223/X56021.
    * Code:
      * `adapt_utils`: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4468992.svg)](https://doi.org/10.5281/zenodo.4468992) (`adapt_utils_20210126`).
      * Firedrake: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4293614.svg)](https://doi.org/10.5281/zenodo.4293614).
      * Thetis: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4288261.svg)](https://doi.org/10.5281/zenodo.4288261).
    * Test cases: `steady/test_cases/point_discharge2d` (Section 4.2), `steady/test_cases/point_discharge3d` (Section 4.3), `unsteady/test_cases/idealised_desalination` (Section 5.1).

  * J. G. Wallwork, <em>'Mesh Adaptation and Adjoint Methods for Finite Element Coastal Ocean Modelling'</em>, PhD thesis, Imperial College London (2021).
    * Code:
      * `adapt_utils`: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4692664.svg)](https://doi.org/10.5281/zenodo.4692664) (`adapt_utils_20210415`). 
      * Firedrake: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4561836.svg)](https://doi.org/10.5281/zenodo.4561836).
      * Thetis: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4560054.svg)](https://doi.org/10.5281/zenodo.4560054).
    * Test cases:
      * `test/interp` (Sections 2.8 and 3.4);
      * `steady/test_cases/point_discharge2d` (Sections 2.7, 3.6, 7.3 and 7.5);
      * `case_studies/tohoku/inversion/1d` (Section 4.4);
      * `case_studies/tohoku/inversion/okada` (Sections 4.4 and 4.5);
      * `test/adapt` (Sections 5.8 and 6.4);
      * `unsteady/test_cases/bubble_shear` (Sections 5.8 and 6.1);
      * `unsteady/test_cases/trench_1d` (Section 6.5);
      * `steady/test_cases/turbine_array` (Sections 7.3 and 7.5);
      * `steady/test_cases/idealised_desalination` (Section 7.7);
      * `case_studies/tohoku/hazard` (Section 7.8).


### User instructions

  * Clone this repository and make it accessible to the `$PYTHONPATH` environment variable.
  * Set the `$SOFTWARE` environment variable to where you would like your PETSc and Firedrake installations to exist.
  * Copy the contents of the `install` directory into `$SOFTWARE` and enter that directory.
  * Install Firedrake:
      * For an installation with Pragmatic, call `source install_petsc.sh` and then `source install_firedrake.sh`, modifying these scripts, if desired.
      * For an installation without Pragmatic, call `source install_firedrake_no_adapt.sh`.
      * If installing on a fresh Ubuntu OS then you will need to call `source install_compilers.sh` beforehand.
  * Test your installation by calling `make test_all` from the root directory.
  * Once you have a working Firedrake installation, get to grips with `adapt_utils` by looking at the test cases in `steady/test_cases` and `unsteady/test_cases`, as well as the notebooks hosted [here][8].


#### For feedback, comments and questions, please email j.wallwork16@imperial.ac.uk or m.clare17@imperial.ac.uk (for feedback related to Clare et al.).

[1]: http://firedrakeproject.org/ "Firedrake"
[2]: http://thetisproject.org/index.html "Thetis"
[3]: https://www.mcs.anl.gov/petsc/ "PETSc"
[4]: https://github.com/meshadaptation/pragmatic "PRAgMaTIc"
[5]: https://www.dolfin-adjoint.org/ "dolfin-adjoint"
[6]: http://www.imperial.ac.uk/earth-science/research/research-groups/amcg/ "AMCG"
[7]: https://www.firedrakeproject.org/zenodo.html "firedrake_zenodo"
[8]: https://github.com/jwallwork23/adapt_utils_notebooks "adapt_utils_notebooks"
[9]: https://doi.org/10.5281/zenodo.3653101 "imr_paper"
[10]: https://rdcu.be/b35wZ "snas_paper"
[11]: https://doi.org/10.31223/osf.io/tpqvy "mesh_movement_paper"
[12]: https://doi.org/10.31223/X56021 "cad_paper"
