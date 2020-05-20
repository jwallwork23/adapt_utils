from thetis import *

import os

from adapt_utils.swe.adapt_solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveShallowWaterProblem):
    """General solver object for adaptive tsunami propagation problems."""

    # -- Setup

    def __init__(self, *args, extension=None, **kwargs):
        self.extension = extension
        super(AdaptiveTsunamiProblem, self).__init__(*args, **kwargs)
        self.callbacks = [{} for mesh in self.meshes]

        # Use linearised equations
        self.shallow_water_options['use_nonlinear_equations'] = False

        # # Don't bother plotting velocity
        # self.io_options['fields_to_export'] = ['elev_2d'] if self.op.plot_pvd else []
        # self.io_options['fields_to_export_hdf5'] = ['elev_2d'] if self.op.save_hdf5 else []

    def set_fields(self):
        self.fields = []
        for P1 in self.P1:
            self.fields.append({
                'horizontal_viscosity': self.op.set_viscosity(P1),
                'coriolis_frequency': self.op.set_coriolis(P1),
                'quadratic_drag_coefficient': self.op.set_quadratic_drag_coefficient(P1),
                'manning_drag_coefficient': self.op.set_manning_drag_coefficient(P1),
            })
        self.bathymetry = [self.op.set_bathymetry(P1) for P1 in self.P1]

    def add_callbacks(self, i):
        # super(AdaptiveTsunamiProblem, self).add_callbacks(i)
        op = self.op

        # --- Gauge timeseries

        names = [g for g in op.gauges]
        locs = [op.gauges[g]["coords"] for g in names]
        fname = "gauges"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(i)])
        self.callbacks[i]['gauges'] = callback.DetectorsCallback(
            self.fwd_solvers[i], locs, ['elev_2d'], fname, names)
        self.fwd_solvers[i].add_callback(self.callbacks[i]['gauges'], 'export')
        # for g in names:
        #     x, y = op.gauges[g]["coords"]
        #     self.callbacks[i][g] = callback.TimeSeriesCallback2D(
        #         self.fwd_solvers[i], ['elev_2d'], x, y, g, self.di)
        #     self.fwd_solvers[i].add_callback(self.callbacks[i][g], 'export')

        # --- Quantity of interest

        self.get_qoi_kernels(i)
        self.kernel_file._topology = None
        kernel_proj = project(self.kernels[i].split()[1], self.P1[i])
        kernel_proj.rename("QoI kernel")
        self.kernel_file.write(kernel_proj)
        kt = Constant(0.0)  # Kernel in time

        def qoi(sol):
            t = self.fwd_solvers[i].simulation_time
            kt.assign(1.0 if t >= op.start_time else 0.0)
            return assemble(kt*inner(self.kernels[i], sol)*dx)

        self.callbacks[i]["qoi"] = callback.TimeIntegralCallback(
            qoi, self.fwd_solvers[i], self.fwd_solvers[i].timestepper,
            # name="qoi", append_to_log=op.debug
            name="qoi", append_to_log=False
        )
        self.fwd_solvers[i].add_callback(self.callbacks[i]["qoi"], 'timestep')

    def quantity_of_interest(self):
        self.qoi = sum(c["qoi"].get_value() for c in self.callbacks)
        return self.qoi

    def setup_solver_adjoint(self, i, **kwargs):
        """Setup continuous adjoint solver on mesh `i`."""
        op = self.op
        dtc = Constant(op.dt)
        g = Constant(op.g)
        b = self.bathymetry[i]
        f = self.fields[i]['coriolis_frequency']
        n = FacetNormal(self.meshes[i])

        # Mixed function space
        z, zeta = TrialFunctions(self.V[i])
        z_test, zeta_test = TestFunctions(self.V[i])
        self.adj_solutions_old[i].assign(self.adj_solutions[i])  # Assign previous value
        z_old, zeta_old = split(self.adj_solutions_old[i])

        def TaylorHood(f0, f1):
            F = -inner(z_test, b*grad(f1))*dx                     # - ∇ (b ζ)
            F += -inner(z_test, f*as_vector((-f0[1], f0[0])))*dx  # - f perp(z)
            F += g*inner(grad(zeta_test), f0)*dx                  # - g ∇ . z
            return F

        def Mixed(f0, f1):
            F = -inner(z_test, b*grad(f1))*dx                     # - ∇ (b ζ)
            F += -inner(z_test, f*as_vector((-f0[1], f0[0])))*dx  # - f perp(z)
            F += g*inner(grad(zeta_test), f0)*dx                  # - g ∇ . z
            # F += -g*inner(avg(zeta_test*n), avg(f0))*dS           # flux term
            return F

        family = self.shallow_water_options['element_family']
        try:
            G = {'taylor-hood': TaylorHood, 'dg-cg': Mixed}[family]
        except KeyError:
            raise ValueError("Mixed discretisation {:s} not supported.".format(family))

        # Time derivative
        a = inner(z_test, z)*dx + inner(zeta_test, zeta)*dx
        L = inner(z_test, z_old)*dx + inner(zeta_test, zeta_old)*dx

        # Crank-Nicolson timestepping
        try:
            assert self.timestepping_options['timestepper_type'] == 'CrankNicolson'
        except AssertionError:
            raise NotImplementedError  # TODO
        print_output("### TODO: Implement adjoint ts other than Crank-Nicolson")
        a += 0.5*dtc*G(z, zeta)
        L -= 0.5*dtc*G(z_old, zeta_old)

        # Boundary conditions
        # ===================
        #   In the forward, we permit only free-slip conditions for the velocity and Dirichlet
        #   conditions for the elevation. Suppose these are imposed on Gamma_1 and Gamma_2, which
        #   are not necessarily disjoint. Then the adjoint has a free-slip condition for the adjoint
        #   velocity on the complement of Gamma_2 and Dirichlet conditions for the elevation on the
        #   complement of Gamma_1.
        boundary_markers = self.meshes[i].exterior_facets.unique_markers
        if self.boundary_conditions[i] == {}:  # Default Thetis boundary conditions are free-slip
            self.boundary_conditions[i] = {j: {'un': Constant(0.0)} for j in boundary_markers}
        for j in boundary_markers:
            for bc in self.boundary_conditions[i].get(j):
                if bc not in ('un', 'elev'):
                    msg = "Have not considered continuous adjoint for boundary condition {:s}"
                    raise NotImplementedError(msg.format(bc))
        dbcs = []
        for j in boundary_markers:
            bcs = self.boundary_conditions[i].get(j)
            if 'un' in bcs and 'elev' not in bcs:
                L += g*inner(zeta_test, bcs['un'])*ds(j)
            if 'elev' in bcs and 'un' not in bcs:
                if zeta.ufl_element().family() == 'Lagrange':
                    dbcs.append(DirichletBC(zeta.function_space(), 0, j))
                else:
                    raise NotImplementedError("Weak boundary conditions not yet implemented")  # TODO

        # dJdq forcing term
        self.get_qoi_kernels(i)
        t = op.dt*(i+1)*self.dt_per_mesh
        self.time_kernel = Constant(1.0 if t >= op.start_time else 0.0)
        dJdu, dJdeta = self.kernels[i].split()
        L += dtc*inner(z_test, self.time_kernel*dJdu)*dx
        L += dtc*inner(zeta_test, self.time_kernel*dJdeta)*dx

        # Solver object
        problem = LinearVariationalProblem(a, L, self.adj_solutions[i], bcs=dbcs)
        self.adj_solvers[i] = LinearVariationalSolver(problem, solver_parameters=op.adjoint_params)

    def solve_adjoint_step(self, i, export_func=None, update_forcings=None):
        """
        Solve adjoint PDE on mesh `i`.

        :kwarg update_forcings: a function which takes simulation time as an argument and is
            evaluated at the start of every timestep.
        :kwarg export_func: a function with no arguments which is evaluated at every export step.
        """
        op = self.op
        t = op.dt*(i+1)*self.dt_per_mesh
        end_time = op.dt*i*self.dt_per_mesh
        op.print_debug("Entering adjoint time loop...")
        j = 0

        # Need to project to P1 for plotting
        self.adjoint_solution_file._topology = None  # Account for mesh adaptations
        z, zeta = self.adj_solutions[i].split()
        z_out = Function(self.P1_vec[i], name="Projected adjoint velocity")
        zeta_out = Function(self.P1[i], name="Projected adjoint elevation")

        while t > end_time:
            if update_forcings is not None:
                update_forcings(t)
            if j % op.dt_per_export == 0:
                print_output("t = {:6.1f}".format(t))
                if export_func is not None:
                    export_func()
                z, zeta = self.adj_solutions[i].split()
                zeta_out.project(zeta)
                z_out.project(z)
                self.adjoint_solution_file.write(z_out, zeta_out)
            self.time_kernel.assign(1.0 if t >= op.start_time else 0.0)
            self.adj_solvers[i].solve()
            self.adj_solutions_old[i].assign(self.adj_solutions[i])
            t -= op.dt
            j += 1
        assert j % op.dt_per_export == 0
        if export_func is not None:
            export_func()
        op.print_debug("Done!")
