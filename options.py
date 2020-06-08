from thetis import *
from thetis.configuration import *

import os


__all__ = ["Options"]


# TODO: Improve doc
class Options(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # Spatial discretisation
    family = Unicode('dg-dg', help="""
    Mixed finite element pair to use for the shallow water system. Choose from:
      'cg-cg': Taylor-Hood                    (P2-P1);
      'dg-dg': Equal order DG                 (PpDG-PpDG);
      'dg-cg': Mixed continuous-discontinuous (P1DG-P2),
    where p is the polynomial order specified by :attr:`degree`.
    """).tag(config=True)
    tracer_family = Unicode('dg', help="""
    Finite element pair to use for the tracer transport model. Choose from:
      'cg': Continuous Galerkin    (Pp);
      'dg': Discontinuous Galerkin (PpDG),
    where p is the polynomial order specified by :attr:`degree_tracer`.
    """).tag(config=True)
    degree = NonNegativeInteger(1, help="""
    Polynomial order for shallow water finite element pair :attr:`family'.
    """).tag(config=True)
    degree_tracer = NonNegativeInteger(1, help="""
    Polynomial order for tracer finite element pair :attr:`tracer_family'.
    """).tag(config=True)
    degree_increase = NonNegativeInteger(0, help="""
    When defining an enriched shallow water finite element space, how much should the
    polynomial order of the finite element space by incremented? (NOTE: zero is an option)
    """).tag(config=True)
    degree_increase_tracer = NonNegativeInteger(1, help="""
    When defining an enriched tracer finite element space, how much should the
    polynomial order of the finite element space by incremented? (NOTE: zero is an option)
    """).tag(config=True)

    # Time discretisation
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest.").tag(config=True)
    end_time = PositiveFloat(60., help="End of time window of interest.").tag(config=True)
    num_meshes = PositiveInteger(1, help="Number of meshes in :class:`AdaptiveProblem` solver").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export.").tag(config=True)
    use_semi_implicit_linearisation = Bool(False).tag(config=True)  # TODO: doc
    implicitness_theta = NonNegativeFloat(0.5).tag(config=True)  # TODO: doc

    # Boundary conditions
    boundary_conditions = PETScSolverParameters({}, help="Boundary conditions expressed as a dictionary.").tag(config=True)
    adjoint_boundary_conditions = PETScSolverParameters({}, help="Boundary conditions for adjoint problem expressed as a dictionary.").tag(config=True)

    # Stabilisation
    stabilisation = Unicode(None, allow_none=True, help="Stabilisation approach, chosen from {'SU', 'SUPG', 'lax_friedrichs'}, if not None.").tag(config=True)
    use_automatic_sipg_parameter = Bool(True, help="Toggle automatic generation of symmetric interior penalty method.").tag(config=True)

    # Solver parameters
    params = PETScSolverParameters({}).tag(config=True)
    adjoint_params = PETScSolverParameters({}).tag(config=True)

    # Outputs
    debug = Bool(False, help="Toggle debugging mode for more verbose screen output.").tag(config=True)
    plot_pvd = Bool(True, help="Toggle plotting of fields.").tag(config=True)
    save_hdf5 = Bool(False, help="Toggle saving fields to HDF5.").tag(config=True)

    # Adaptation
    approach = Unicode('fixed_mesh', help="Mesh adaptive approach.").tag(config=True)
    num_adapt = NonNegativeInteger(4, help="Number of mesh adaptations per remesh.").tag(config=True)
    rescaling = PositiveFloat(0.85, help="Scaling parameter for target number of vertices.").tag(config=True)
    convergence_rate = PositiveInteger(6, help="Convergence rate parameter used in approach of [Carpio et al. 2013].").tag(config=True)
    h_min = PositiveFloat(1e-10, help="Minimum tolerated element size.").tag(config=True)
    h_max = PositiveFloat(5., help="Maximum tolerated element size.").tag(config=True)
    pseudo_dt = PositiveFloat(0.1, help="Pseudo-timstep used in r-adaptation.").tag(config=True)
    r_adapt_maxit = PositiveInteger(1000, help="Maximum number of iterations in r-adaptation loop.").tag(config=True)
    r_adapt_rtol = PositiveFloat(1.0e-8, help="Relative tolerance for residual in r-adaptation loop.").tag(config=True)
    nonlinear_method = Unicode('quasi_newton', help="Method for solving nonlinear system under r-adaptation.").tag(config=True)
    prescribed_velocity = Unicode('fluid', allow_none=True, help="Prescribed velocity to use in ALE adaptation, if any.").tag(config=True)
    prescribed_velocity_bc = Unicode(None, allow_none=True, help="Boundary conditions to apply to prescribed velocity (if any).").tag(config=True)

    # Metric
    max_anisotropy = PositiveFloat(1000., help="Maximum tolerated anisotropy.").tag(config=True)
    normalisation = Unicode('complexity', help="Metric normalisation approach, from {'complexity', 'error'}.").tag(config=True)
    target = PositiveFloat(1.0e+2, help="Target complexity / inverse desired error for normalisation, as appropriate.").tag(config=True)
    norm_order = NonNegativeFloat(None, allow_none=True, help="Degree p of Lp norm used in spatial normalisation. Use 'None' to specify infinity norm.").tag(config=True)
    intersect_boundary = Bool(False, help="Intersect with initial boundary metric.").tag(config=True)

    # Hessian
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    hessian_solver_parameters = PETScSolverParameters({'snes_rtol': 1e8,
                                                       'ksp_rtol': 1e-5,
                                                       'ksp_gmres_restart': 20,
                                                       'pc_type': 'sor'}).tag(config=True)
    hessian_time_combination = Unicode('integrate', help="Method used to combine Hessians over timesteps, from {'integrate', 'intersect'}.").tag(config=True)
    hessian_timestep_lag = PositiveFloat(1, help="Allow lagged Hessian computation by setting greater than one.").tag(config=True)

    # Goal-oriented adaptation
    region_of_interest = List(default_value=[], help="Spatial region related to quantity of interest").tag(config=True)
    estimate_error = Bool(False, help="For use in Thetis solver object.").tag(config=True)

    # Adaptation loop
    element_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in mesh element count").tag(config=True)
    qoi_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in quantity of interest.").tag(config=True)
    estimator_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in error estimator.").tag(config=True)
    target_base = PositiveFloat(10.0, help="Base for exponential increase/decay of target complexity/error within outer mesh adaptation loop.").tag(config=True)
    outer_iterations = PositiveInteger(1, help="Number of iterations in outer adaptation loop.").tag(config=True)
    indent = Unicode('', help="Indent used in nested print statements.").tag(config=True)

    # Mesh
    periodic = Bool(False, help="Is mesh periodic?").tag(config=True)

    def __init__(self, mesh=None, **kwargs):
        self.default_mesh = mesh
        self.update(kwargs)
        self.di = os.path.join('outputs', self.approach)
        self.end_time -= 0.5*self.dt
        if self.debug:
            # set_log_level(DEBUG)
            set_log_level(INFO)

    def copy(self):
        copy = type(self)()
        copy.update(self)
        return copy

    def set_all_rtols(self, tol):
        """Set all relative tolerances to a single value, `tol`."""
        self.element_rtol = tol
        self.qoi_rtol = tol
        self.estimator_rtol = tol

    def box(self, mesh, scale=1.0, source=False):
        r"""
        Rectangular indicator function associated with region(s) of interest.

        Takes the value `scale` in the region

      ..math::
            (x0 - r_x < x < x0 + r_x) && (y0 - r_y < y < y0 + r_y)

        centred about (x0, y0) and zero elsewhere. Similarly for other dimensions.

        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        dim = mesh.topological_dimension()
        x = SpatialCoordinate(mesh)
        locs = self.source_loc if source else self.region_of_interest
        for j in range(len(locs)):
            expr = And(gt(x[0], locs[j][0]-locs[j][dim]), lt(x[0], locs[j][0]+locs[j][dim]))
            for i in range(1, dim):
                r0 = locs[j][dim] if len(locs) == dim else locs[j][dim+i]
                expr = And(expr, And(gt(x[i], locs[j][i]-r0), lt(x[i], locs[j][i]+r0)))
            b = expr if j == 0 else Or(b, expr)
        return conditional(b, scale, 0.0)

    def ball(self, mesh, scale=1.0, source=False):
        r"""
        Ball indicator function associated with region(s) of interest.

        Takes the value `scale` in the region

      ..math::
            (x - x_0)^2 + (y - y_0)^2 < r_0^2

        and zero elsewhere. Similarly for other dimensions.

        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        dim = mesh.topological_dimension()
        x = SpatialCoordinate(mesh)
        locs = self.source_loc if source else self.region_of_interest
        for j in range(len(locs)):
            r0 = locs[j][dim]
            expr = lt(sum((x[i]-locs[j][i])**2 for i in range(dim)), r0**2 + 1e-10)
            b = expr if j == 0 else Or(b, expr)
        return conditional(b, scale, 0.0)

    def bump(self, mesh, scale=1.0, source=False):
        r"""
        Rectangular bump function associated with region(s) of interest. (A smooth approximation
        to the box function.)

        Takes the form

      ..math::
            \exp\left(1 - \frac1{\left1 - \left(\frac{x - x_0}{r_x}\right)^2\right)}\right)
            * \exp\left(1 - \frac1{\left1 - \left(\frac{y - y_0}{r_y}\right)^2\right)}\right)

        scaled by `scale` inside the box region. Similarly for other dimensions.

        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        dim = mesh.topological_dimension()
        x = SpatialCoordinate(mesh)
        locs = self.source_loc if source else self.region_of_interest
        b = 0
        for j in range(len(locs)):
            vol = 1.0
            expr = scale
            for i in range(dim):
                ri = locs[j][dim] if len(locs) == dim else locs[j][dim+i]
                vol *= ri
                expr = expr*exp(1 - 1/(1 - ((x[0]-locs[j][0])/ri)**2))
            b += conditional(lt(sum((x[i]-locs[j][i])**2 for i in range(dim)), vol), expr, 0.0)
        return b

    def circular_bump(self, mesh, scale=1.0, source=False):
        r"""
        Circular bump function associated with region(s) of interest. (A smooth approximation to
        the ball function.)

        Defining the radius :math:`r^2 := (x - x_0)^2 + (y - y_0)^2`, the circular bump takes the
        form

      ..math::
            \exp\left(1 - \frac1{\left1 - \frac{r^2}{r_0^2}\right)}\right)

        scaled by `scale` inside the ball region. Similarly for other dimensions.

        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        dim = mesh.topological_dimension()
        x = SpatialCoordinate(mesh)
        locs = self.source_loc if source else self.region_of_interest
        b = 0
        for j in range(len(locs)):
            r0_sq = locs[j][dim]**2
            r_sq = sum((x[i]-locs[j][i])**2 for i in range(dim))
            b += conditional(lt(r_sq, r0_sq + 1e-10), scale*exp(1 - 1/(1 - r_sq/r0_sq)), 0)
        return b

    def gaussian(self, mesh, scale=1.0, source=False):
        r"""
        Gaussian bell associated with region(s) of interest.

        Takes the form

      ..math::
            \exp\left(1 - \frac1{1 - \frac{x^2 + y^2}{r_0^2}}\right)

        scaled by `scale` inside the ball region. Similarly for other dimensions.

        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        dim = mesh.topological_dimension()
        x = SpatialCoordinate(mesh)
        locs = self.source_loc if source else self.region_of_interest
        b = 0
        for j in range(len(locs)):
            r0_sq = locs[j][dim]**2
            r_sq = sum((x[i]-locs[j][i])**2 for i in range(dim))
            b += conditional(lt(r_sq, r0_sq), scale*exp(1 - 1/(1 - r_sq/r0_sq)), 0)
        return b

    def set_start_condition(self, fs, adjoint=False):
        return self.set_final_condition(fs) if adjoint else self.set_initial_condition(fs)

    def set_initial_condition(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_final_condition(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_boundary_conditions(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_boundary_surface(self):
        """Should be implemented in derived class."""
        pass

    def set_source(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_qoi_kernel(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def exact_solution(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def exact_qoi(self):
        raise NotImplementedError("Should be implemented in derived class.")

    def get_update_forcings(self, prob, i):
        """Should be implemented in derived class."""
        def update_forcings(t):
            return
        return update_forcings

    def get_export_func(self, prob, i):
        """Should be implemented in derived class."""
        def export_func():
            return
        return export_func

    def print_debug(self, msg):
        if self.debug:
            print_output(msg)

    # TODO: USEME
    def get_mesh_velocity(self):
        """
        Prescribed a mesh velocity.
        """
        if self.prescribed_velocity == "zero":

            # Eulerian case (fixed mesh)
            self.mesh_velocity = lambda mesh: Constant(as_vector([0.0, 0.0]))

        elif self.prescribed_velocity == "fluid":

            # Lagrangian case (move mesh with fluid)
            velocity_file = File(os.path.join(self.di, 'fluid_velocity.pvd'))

            def mesh_velocity(mesh):  # TODO: Make these available as options

                # Get fluid velocity
                coord_space = mesh.coordinates.function_space()
                self.set_velocity(coord_space)
                v = Function(coord_space, name="Mesh velocity")
                v.assign(self.fluid_velocity)

                # No constraints on boundary
                bc = None
                bbc = None

                if self.prescribed_velocity_bc is None:
                    velocity_file.write(v)
                    return self.fluid_velocity

                # Use fluid velocity in domain interior
                n = FacetNormal(mesh)
                trial, test = TrialFunction(coord_space), TestFunction(coord_space)
                a = dot(test, trial)*dx
                L = dot(test, self.fluid_velocity)*dx

                if self.prescribed_velocity_bc == 'noslip':

                    # Enforce no boundary movement
                    bc = DirichletBC(coord_space, Constant([0.0, 0.0]), 'on_boundary')

                elif self.prescribed_velocity_bc == 'freeslip':

                    # Enforce no velocity normal to boundaries
                    a_bc = dot(test, n)*dot(trial, n)*ds
                    L_bc = dot(test, n)*Constant(0.0)*ds
                    bc = [EquationBC(a == L, v, 'on_boundary')]

                    # Allow tangential movement ...
                    s = as_vector([n[1], -n[0]])
                    a_bc = dot(test, s)*dot(trial, s)*ds
                    L_bc = dot(test, s)*dot(self.fluid_velocity, s)*ds
                    edges = set(mesh.exterior_facets.unique_markers)
                    if len(edges) > 1:  # ... but only up until the end of boundary segments
                        corners = [(i, j) for i in edges for j in edges.difference([i])]
                        bbc = DirichletBC(coord_space, 0, corners)
                    bc.append(EquationBC(a_bc == L_bc, v, 'on_boundary', bcs=bbc))

                elif self.prescribed_velocity_bc == 'sponge':  # TODO: Generalise

                    # Sponge out boundary movement
                    x, y = SpatialCoordinate(mesh)
                    alpha = 100
                    L = dot(test, exp(-alpha*((x-0.5)**2+(y-0.5)**2))*self.fluid_velocity)*dx

                else:
                    raise ValueError("Prescribed boundary method {:s} not recognised.".format(self.prescribed_velocity_bc))

                solve(a == L, v, bcs=bc)
                self.fluid_velocity.assign(v)
                velocity_file.write(v)
                return self.fluid_velocity

            self.mesh_velocity = mesh_velocity

        elif self.prescribed_velocity == "rezoning":
            raise NotImplementedError  # TODO
        else:
            raise ValueError("Mesh velocity {:s} not recognised.".format(self.prescribed_velocity))
        return self.mesh_velocity
