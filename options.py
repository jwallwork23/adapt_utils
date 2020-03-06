from thetis import *
from thetis.configuration import *

import os


__all__ = ["Options"]


class Options(FrozenConfigurable):
    name = 'Common parameters for mesh adaptive simulations'

    # Spatial discretisation  # TODO: use a notation which is more general
    family = Unicode('dg-dg', help="Mixed finite element family, from {'dg-dg', 'dg-cg'}.").tag(config=True)
    degree_increase = NonNegativeInteger(1, help="Polynomial degree increase in enriched space").tag(config=True)
    degree = NonNegativeInteger(1, help="Order of function space").tag(config=True)

    # Time discretisation
    timestepper = Unicode('CrankNicolson', help="Time integration scheme used.").tag(config=True)
    dt = PositiveFloat(0.1, help="Timestep").tag(config=True)
    start_time = NonNegativeFloat(0., help="Start of time window of interest.").tag(config=True)
    end_time = PositiveFloat(60., help="End of time window of interest.").tag(config=True)
    dt_per_export = PositiveFloat(10, help="Number of timesteps per export.").tag(config=True)
    dt_per_remesh = PositiveFloat(20, help="Number of timesteps per mesh adaptation.").tag(config=True)

    # Boundary conditions
    boundary_conditions = PETScSolverParameters({}, help="Boundary conditions expressed as a dictionary.").tag(config=True)
    adjoint_boundary_conditions = PETScSolverParameters({}, help="Boundary conditions for adjoint problem expressed as a dictionary.").tag(config=True)

    # Stabilisation
    stabilisation = Unicode(None, allow_none=True, help="Stabilisation approach, chosen from {'SU', 'SUPG', 'lax_friedrichs'}, if not None.").tag(config=True)
    stabilisation_parameter = FiredrakeScalarExpression(Constant(1.0), help="Scalar stabilisation parameter.").tag(config=True)
    sipg_parameter = FiredrakeScalarExpression(None, allow_none=True, help="Value for parameter used in symmetric interior penalty method. Chosen automatically if set to None.").tag(config=True)

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

    # Metric
    max_anisotropy = PositiveFloat(1000., help="Maximum tolerated anisotropy.").tag(config=True)
    normalisation = Unicode('complexity', help="Metric normalisation approach, from {'complexity', 'error'}.").tag(config=True)
    target = PositiveFloat(1.0e+2, help="Target complexity / inverse desired error for normalisation, as appropriate.").tag(config=True)
    norm_order = NonNegativeInteger(None, allow_none=True, help="Degree p of Lp norm used in 'error' normalisation approach. Use 'None' to specify infinity norm.").tag(config=True)
    intersect_boundary = Bool(False, help="Intersect with initial boundary metric.").tag(config=True)

    # Hessian
    hessian_recovery = Unicode('dL2', help="Hessian recovery technique, from {'dL2', 'parts'}.").tag(config=True)
    hessian_solver_parameters = PETScSolverParameters({'snes_rtol': 1e8,
                                                       'ksp_rtol': 1e-5,
                                                       'ksp_gmres_restart': 20,
                                                       'pc_type': 'sor'}).tag(config=True)

    # Goal-oriented adaptation
    region_of_interest = List(default_value=[], help="Spatial region related to quantity of interest").tag(config=True)

    # Adaptation loop
    element_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in mesh element count").tag(config=True)
    qoi_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in quantity of interest.").tag(config=True)
    estimator_rtol = PositiveFloat(0.005, help="Relative tolerance for convergence in error estimator.").tag(config=True)
    target_base = PositiveFloat(10.0, help="Base for exponential increase/decay of target complexity/error within outer mesh adaptation loop.").tag(config=True)
    outer_iterations = PositiveInteger(1, help="Number of iterations in outer adaptation loop.").tag(config=True)
    indent = Unicode('', help="Indent used in nested print statements.").tag(config=True)

    # Mesh
    periodic = Bool(False, help="Is mesh periodic?").tag(config=True)

    def __init__(self, **kwargs):
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

    def ball(self, fs, scale=1., source=False):
        """
        Ball indicator function associated with region(s) of interest

        :arg fs: Desired `FunctionSpace`.
        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        eps = 1e-10
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            if dim == 3:
                z0 = locs[j][2]
                expr = lt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0), r*r + eps)
            else:
                expr = lt((x-x0)*(x-x0) + (y-y0)*(y-y0), r*r + eps)
            b = expr if j == 0 else Or(b, expr)
        expr = conditional(b, scale, 0.)
        indi = Function(fs)
        indi.interpolate(expr)
        return indi

    def bump(self, fs, scale=1., source=False):
        """
        Rectangular bump function associated with region(s) of interest

        :arg fs: Desired `FunctionSpace`.
        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r0 = locs[j][2] if dim == 2 else locs[j][3]
            if dim == 2 and len(locs) == 4:
                r1 = locs[j][3]
            elif dim == 3 and len(locs) > 4:
                r1 = locs[j][4]
            else:
                r1 = r0
            expr1 = (x-x0)*(x-x0) + (y-y0)*(y-y0)
            expr2 = scale*exp(1 -1/(1 - (x-x0)*(x-x0)/r0**2))*exp(1 - 1/(1 - (y-y0)*(y-y0)/r1**2))
            vol = r0*r1
            if dim == 3:
                z0 = locs[j][2]
                r2 = r0 if len(locs) < 6 else locs[j][5]
                expr1 += (z-z0)*(z-z0)
                expr2 *= exp(1 - 1/(1 - (z-z0)*(z-z0)/r2**2))
                vol *= r2
            i += conditional(lt(expr1, vol), expr2, 0)
        bump = Function(fs)
        bump.interpolate(i)
        return bump

    def gaussian(self, fs, scale=1., source=False):
        """
        Gaussian function associated with region(s) of interest

        :arg fs: Desired `FunctionSpace`.
        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        i = 0
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r = locs[j][2] if dim == 2 else locs[j][3]
            expr = (x-x0)*(x-x0) + (y-y0)*(y-y0)
            if dim == 3:
                z0 = locs[j][2]
                expr += (z-z0)*(z-z0)
            i += conditional(lt(expr, r*r), scale*exp(1 - 1/(1 - expr/r**2)), 0)
        bump = Function(fs)
        bump.interpolate(i)
        return bump

    def box(self, fs, scale=1., source=False):
        """
        Rectangular indicator function associated with region(s) of interest

        :arg fs: Desired `FunctionSpace`.
        :kwarg scale: Scale factor for indicator.
        :kwarg source: Toggle source term or region of interest location.
        """
        mesh = fs.mesh()
        dim = mesh.topological_dimension()
        assert dim in (2, 3)
        if dim == 2:
            x, y = SpatialCoordinate(fs)
        else:
            x, y, z = SpatialCoordinate(fs)
        locs = self.source_loc if source else self.region_of_interest
        for j in range(len(locs)):
            x0 = locs[j][0]
            y0 = locs[j][1]
            r0 = locs[j][2] if dim == 2 else locs[j][3]
            if dim == 2 and len(locs) == 4:
                r1 = locs[j][3]
            elif dim == 3 and len(locs) > 4:
                r1 = locs[j][4]
            else:
                r1 = r0
            expr = And(And(gt(x, x0-r0), lt(x, x0+r0)), And(gt(y, y0-r1), lt(y, y0+r1)))
            if dim == 3:
                r2 = r0 if len(locs) < 6 else locs[j][5]
                z0 = locs[j][2]
                expr = And(expr, And(gt(z, z0-r2), lt(z, z0+r2)))
            b = expr if j == 0 else Or(b, expr)
        expr = conditional(b, scale, 0.)
        box = Function(fs)
        box.interpolate(expr)
        return box

    def set_start_condition(self, fs, adjoint=False):
        if adjoint:
            return self.set_final_condition(fs)
        else:
            return self.set_initial_condition(fs)

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

    def get_update_forcings(self, solver_obj):
        """Should be implemented in derived class."""
        def update_forcings(t):
            return
        return update_forcings

    def get_export_func(self, solver_obj):
        """Should be implemented in derived class."""
        def export_func():
            return
        return export_func

    def print_debug(self, msg):
        if self.debug:
            print_output(msg)

    def get_mesh_velocity(self):
        """
        Prescribed a mesh velocity.
        """
        if self.prescribed_velocity == "constant":  # Fixed mesh
            self.mesh_velocity = lambda mesh: Constant(as_vector([0.0, 0.0]))
        elif self.prescribed_velocity == "fluid":  # Move mesh with fluid

            velocity_file = File(os.path.join(self.di, 'fluid_velocity.pvd'))

            def mesh_velocity(mesh):  # FIXME

                # Get fluid velocity
                P1_vec = VectorFunctionSpace(mesh, "CG", 1)
                self.set_velocity(P1_vec)

                # Use fluid velocity in domain interior
                n = FacetNormal(mesh)
                v = Function(P1_vec, name="Mesh velocity")
                trial, test = TrialFunction(P1_vec), TestFunction(P1_vec)
                a = dot(test, trial)*dx
                L = dot(test, self.fluid_velocity)*dx

                # TODO: Sponge condition?
                # x, y = SpatialCoordinate(mesh)
                # alpha = 100
                # L = dot(test, exp(-alpha*((x-0.5)**2+(y-0.5)**2))*self.fluid_velocity)*dx

                # # Enforce no velocity normal to boundaries
                # a_bc = dot(test, n)*dot(trial, n)*ds
                # L_bc = dot(test, n)*Constant(0.0)*ds
                # bc = [EquationBC(a == L, v, 'on_boundary')]

                # # Allow tangential movement, but only up until the end of boundary segments
                # s = as_vector([n[1], -n[0]])
                # a_bc = dot(test, s)*dot(trial, s)*ds
                # L_bc = dot(test, s)*dot(self.fluid_velocity, s)*ds
                # edges = set(mesh.exterior_facets.unique_markers)
                # corners = [(i, j) for i in edges for j in edges.difference([i])]
                # bbc = DirichletBC(P1_vec, 0, corners)
                # bc.append(EquationBC(a_bc == L_bc, v, 'on_boundary', bcs=bbc))
                # # self.fluid_velocity /= norm(self.fluid_velocity)

                # Zero mesh velocity on boundary
                bc = DirichletBC(P1_vec, Constant([0.0, 0.0]), 'on_boundary')

                # # No constraints on boundary
                # bc = None

                solve(a == L, v, bcs=bc)
                self.fluid_velocity.assign(v)

                velocity_file.write(v)
                return self.fluid_velocity

            self.mesh_velocity = mesh_velocity
        else:
            raise NotImplementedError
        return self.mesh_velocity
