.. adapt_utils documentation master file, created by
   sphinx-quickstart on Thu Aug 27 19:28:50 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. title:: adapt_utils: Mesh adaptation for coastal ocean modelling in Firedrake and Thetis


``adapt_utils``: Mesh adaptation for coastal ocean modelling in Firedrake and Thetis
====================================================================================

``adapt_utils`` is a Python package which implements various mesh adaptation strategies
which may be used when solving PDEs using the `Firedrake <http://www.firedrakeproject.org/>`__
finite element framework.
Two main implementations are considered:

* (Riemannian) metric-based mesh adaptation using `Parallel AnisotRopic Adaptive Mesh ToolkIt (PRAgMaTIc) <https://github.com/meshadaptation/pragmatic>`__;
* Mesh movement methods, hand-coded in Firedrake.
  
The software was designed for use within the 2D model of the unstructured mesh discontinuous
Galerkin (DG) coastal ocean modelling using `Thetis <http://www.thetisproject.org/>`__.
The associated equation sets are:

* Shallow water equations (i.e. depth-averaged hydrodynamics);
* Passive tracer transport (conservative or non-conservative);
* Sediment transport;
* Exner equation (which models bed level changes).

There is some support for other equation sets, such as 3D advection-diffusion.


Goal-oriented error estimation is implemented for shallow water and tracer transport equations and
a variety of finite element discretisations thereof. The metric-based mesh adaptation functionality
includes goal-oriented mesh adaptation in isotropic/anisotropic cases and time-independent/dependent.

**Main contents**

.. toctree::
   :maxdepth: 2

   publications
   authors
   contact


**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
