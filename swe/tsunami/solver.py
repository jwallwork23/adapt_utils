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
        kernel_file = File(os.path.join(self.di, 'kernel_mesh{:d}.pvd'.format(i)))
        kernel_file.write(self.kernels[i].split()[1])
        kt = Constant(0.0)  # Kernel in time

        def qoi(sol):
            t = self.fwd_solvers[i].simulation_time
            kt.assign(1.0 if t >= op.start_time else 0.0)
            return assemble(kt*inner(self.kernels[i], sol)*dx)

        self.callbacks[i]["qoi"] = callback.TimeIntegralCallback(
            qoi, self.fwd_solvers[i], self.fwd_solvers[i].timestepper,
            name="qoi", append_to_log=op.debug
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

            # - ∇ (b ζ)
            F = -inner(z_test, grad(b*f1))*dx

            # - f perp(z)
            F += -inner(z_test, f*as_vector((-f0[1], f0[0])))*dx

            # - g ∇ . z
            F += g*inner(grad(zeta_test), f0)*dx
            # F += -g*inner(zeta_test, f0)*ds
            return F

        def Mixed(f0, f1):

            # - ∇ (b ζ)
            F = -inner(z_test, grad(b*f1))*dx

            # - f perp(z)
            F += -inner(z_test, f*as_vector((-f0[1], f0[0])))*dx

            # - g ∇ . z
            F += g*inner(grad(zeta_test), f0)*dx
            # F += -g*inner(zeta_test*n, f0)*ds
            F += -g*inner(avg(zeta_test*n), avg(f0))*dS  # TODO: Check
            return F

        def EqualOrder(f0, f1):  # TODO: test

            # - ∇ (b ζ)
            F = inner(div(z_test), b*f1)*dx
            F += -inner(z_test, b*f1*n)*ds
            zeta_star = avg(f1) + sqrt(avg(b)/g)*jump(f0, n)
            F += zeta_star*jump(z_test, n)*dS

            # - f perp(z)
            F += -inner(z_test, f*as_vector((-f0[1], f0[0])))*dx

            # - g ∇ . z
            F += g*inner(grad(zeta_test), f0)*dx
            z_rie = avg(f0) + sqrt(g/avg(b))*jump(f1, n)
            F += g*inner(jump(zeta_test, n), avg(b)*z_rie)*dS
            # F += -g*inner(zeta_test, f0)*ds
            return F

        family = self.shallow_water_options['element_family']
        if family == 'taylor-hood':
            G = TaylorHood
        elif family == 'dg-cg':
            G = Mixed
        elif family == 'dg-dg':
            G = EqualOrder
        else:
            raise ValueError("Mixed discretisation {:s} not supported.".format(family))
        print_output("### TODO: Implement adjoint bcs other than freeslip")

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

        # dJdq forcing term
        self.get_qoi_kernels(i)
        t = op.dt*(i+1)*self.dt_per_mesh
        self.time_kernel = Constant(1.0 if t >= op.start_time else 0.0)
        dJdu, dJdeta = self.kernels[i].split()
        L += dtc*inner(z_test, self.time_kernel*dJdu)*dx
        L += dtc*inner(zeta_test, self.time_kernel*dJdeta)*dx

        # Solver object
        problem = LinearVariationalProblem(a, L, self.adj_solutions[i])
        self.adj_solvers[i] = LinearVariationalSolver(problem, solver_parameters=op.adjoint_params)

    def solve_adjoint_step(self, i, **kwargs):
        """Solve adjoint PDE on mesh `i`."""
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
        z_out.project(z)
        zeta_out.project(zeta)
        self.adjoint_solution_file.write(z_out, zeta_out)

        while t > end_time:
            self.time_kernel.assign(1.0 if t >= op.start_time else 0.0)
            self.adj_solvers[i].solve()
            self.adj_solutions_old[i].assign(self.adj_solutions[i])
            if j > 0 and j % op.dt_per_export == 0:
                z, zeta = self.adj_solutions[i].split()
                zeta_out.project(zeta)
                z_out.project(z)
                self.adjoint_solution_file.write(z_out, zeta_out)
                op.print_debug("t = {:6.1f}".format(t))
            t -= op.dt
            j += 1
        op.print_debug("Done!")
