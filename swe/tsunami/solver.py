from thetis import *

import os
import h5py

from adapt_utils.swe.adapt_solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveShallowWaterProblem):
    """General solver object for adaptive tsunami propagation problems."""

    # -- Setup

    def __init__(self, *args, extension=None, nonlinear=False, **kwargs):
        self.extension = extension
        super(AdaptiveTsunamiProblem, self).__init__(*args, **kwargs)
        self.callbacks = [{} for mesh in self.meshes]

        # Use linearised equations
        self.shallow_water_options['use_nonlinear_equations'] = nonlinear

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

        # --- Mesh data

        fname = "meshdata"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(i)])
        with h5py.File(os.path.join(self.di, fname+'.hdf5'), 'w') as f:
            f.create_dataset('num_cells', data=self.num_cells[-1][i])

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
            name="qoi", append_to_log=op.debug
            # name="qoi", append_to_log=False
        )
        self.fwd_solvers[i].add_callback(self.callbacks[i]["qoi"], 'timestep')

    def quantity_of_interest(self):
        self.qoi = sum(c["qoi"].get_value() for c in self.callbacks)
        return self.qoi

    def setup_solver_forward(self, i, **kwargs):
        nonlinear = self.shallow_water_options['use_nonlinear_equations']

        # For DG cases use Thetis
        family = self.shallow_water_options['element_family']
        if family != 'taylor-hood':
            super(AdaptiveTsunamiProblem, self).setup_solver_forward(i, **kwargs)
            return

        # Collect parameters and fields
        op = self.op
        dtc = Constant(op.dt)
        g = Constant(op.g)
        b = self.bathymetry[i]
        f = self.fields[i]['coriolis_frequency']
        n = FacetNormal(self.meshes[i])

        # Mixed function space
        if nonlinear:
            u, eta = split(self.fwd_solutions[i])
        else:
            u, eta = TrialFunctions(self.V[i])
        u_test, eta_test = TestFunctions(self.V[i])
        self.fwd_solutions_old[i].assign(self.fwd_solutions[i])  # Assign previous value
        u_old, eta_old = split(self.fwd_solutions_old[i])

        def TaylorHood(f0, f1):
            F = inner(u_test, g*grad(f1))*dx                     # g∇ η
            F += inner(u_test, f*as_vector((-f0[1], f0[0])))*dx  # f perp(u)
            F += -inner(grad(eta_test), b*f0)*dx                 # ∇ . bu
            if nonlinear:
                F += inner(u_test, dot(f0, nabla_grad(f0)))*dx   # u . ∇ u
            return F

        # Time derivative
        a = inner(u_test, u)*dx + inner(eta_test, eta)*dx
        L = inner(u_test, u_old)*dx + inner(eta_test, eta_old)*dx

        # Crank-Nicolson timestepping
        try:
            assert self.timestepping_options['timestepper_type'] == 'CrankNicolson'
        except AssertionError:
            raise NotImplementedError  # TODO
        print_output("### TODO: Implement forward ts other than Crank-Nicolson")
        a += 0.5*dtc*TaylorHood(u, eta)
        L -= 0.5*dtc*TaylorHood(u_old, eta_old)

        # Boundary conditions
        boundary_markers = self.meshes[i].exterior_facets.unique_markers
        if self.boundary_conditions[i] == {}:  # Default Thetis boundary conditions are free-slip
            self.boundary_conditions[i] = {j: {'un': Constant(0.0)} for j in boundary_markers}
        dbcs = []
        for j in boundary_markers:
            bcs = self.boundary_conditions[i].get(j)
            if 'un' in bcs:
                L += dtc*b*inner(eta_test, bcs['un'])*ds(j)
            else:
                a += -0.5*dtc*b*inner(eta_test, dot(u, n))*ds(j)
                L += 0.5*dtc*b*inner(eta_test, dot(u_old, n))*ds(j)
            if 'elev' in bcs:
                dbcs.append(DirichletBC(self.V[i].sub(1), 0, j))

        # Solver object
        kwargs = {
            'solver_parameters': op.params,
            'options_prefix': 'forward',
        }
        if nonlinear:
            problem = NonlinearVariationalProblem(L-a, self.fwd_solutions[i], bcs=dbcs)
            self.fwd_solvers[i] = NonlinearVariationalSolver(problem, **kwargs)
        else:
            problem = LinearVariationalProblem(a, L, self.fwd_solutions[i], bcs=dbcs)
            self.fwd_solvers[i] = LinearVariationalSolver(problem, **kwargs)

    def solve_forward_step(self, i, update_forcings=None, export_func=None):
        family = self.shallow_water_options['element_family']
        if family != 'taylor-hood':
            super(AdaptiveTsunamiProblem, self).solve_forward_step(i, **kwargs)
            return
        op = self.op
        t = op.dt*i*self.dt_per_mesh
        end_time = op.dt*(i+1)*self.dt_per_mesh

        # Need to project to P1 for plotting
        self.solution_file._topology = None  # Account for mesh adaptations
        u, eta = self.fwd_solutions[i].split()
        u_out = Function(self.P1_vec[i], name="Projected velocity")
        eta_out = Function(self.P1[i], name="Projected elevation")
        kernel_out = Function(self.P1[i], name="QoI kernel")

        class QoICallback(object):
            """Simple callback class to time integrate a quantity of interest."""
            def __init__(self, kernel):
                self.timeseries = []
                self.ks = kernel         # Kernel in space
                self.kt = Constant(0.0)  # Kernel in time

            def append(self, sol, t=0.0):
                self.kt.assign(1.0 if t >= op.start_time else 0.0)
                self.timeseries.append(assemble(self.kt*inner(self.ks, sol)*dx))

            def get_value(self):
                N = len(self.timeseries)
                assert N >= 2
                val = 0.5*op.dt*(self.timeseries[0] + self.timeseries[-1])
                for i in range(1, N-1):
                    val += op.dt*self.timeseries[i]
                return val

        # Setup callbacks
        self.callbacks[i]['gauges'] = {gauge: [] for gauge in op.gauges}
        self.callbacks[i]['gauges']['time'] = []
        fname = "meshdata"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(i)])
        with h5py.File(os.path.join(self.di, fname+'.hdf5'), 'w') as f:
            f.create_dataset('num_cells', data=self.num_cells[-1][i])
        self.get_qoi_kernels(i)
        self.callbacks[i]['qoi'] = QoICallback(self.kernels[i])
        self.callbacks[i]['qoi'].append(self.fwd_solutions[i])

        # --- Time integrate

        j = 0
        op.print_debug("Entering adjoint time loop...")
        while t < end_time:
            if update_forcings is not None:
                update_forcings(t)
            if j % op.dt_per_export == 0:
                print_output("t = {:6.1f}".format(t))
                if export_func is not None:
                    export_func()
                u, eta = self.fwd_solutions[i].split()

                # Evaluate free surface at gauges
                self.callbacks[i]["gauges"]["time"].append(t)
                for gauge in op.gauges:
                    self.callbacks[i]["gauges"][gauge].append(eta.at(op.gauges[gauge]["coords"]))

                # Plot to pvd
                self.solution_file._topology = None
                self.kernel_file._topology = None
                eta_out.project(eta)
                u_out.project(u)
                kernel_out.project(self.kernels[i].split()[1])
                self.solution_file.write(u_out, eta_out)
                self.kernel_file.write(kernel_out)

            # Solve
            self.fwd_solvers[i].solve()
            self.fwd_solutions_old[i].assign(self.fwd_solutions[i])

            # Evaluate quantity of interest
            self.callbacks[i]["qoi"].append(self.fwd_solutions[i], t=t)

            # Increment
            t += op.dt
            j += 1
        assert j % op.dt_per_export == 0
        if export_func is not None:
            export_func()
        self.callbacks[i]["gauges"]["time"].append(t)
        for gauge in op.gauges:
            self.callbacks[i]["gauges"][gauge].append(eta.at(op.gauges[gauge]["coords"]))
        op.print_debug("Done!")

        # Save to HDF5
        fname = "diagnostic_gauges"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(i)])
        with h5py.File(os.path.join(self.di, fname+'.hdf5'), 'w') as f:
            for gauge in op.gauges:
                f.create_dataset(gauge, data=self.callbacks[i]["gauges"][gauge])
            f.create_dataset("time", data=self.callbacks[i]["gauges"]["time"])

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
        dbcs = []
        for j in boundary_markers:
            bcs = self.boundary_conditions[i].get(j)
            if 'un' in bcs:
                if 'elev' in bcs:
                    a += -0.5*dtc*g*inner(zeta_test, dot(z, n))*ds(j)
                    L += 0.5*dtc*g*inner(zeta_test, dot(z_old, n))*ds(j)
                else:
                    L += dtc*g*inner(zeta_test, bcs['un'])*ds(j)
            elif 'elev' in bcs:
                if zeta.ufl_element().family() == 'Lagrange':
                    dbcs.append(DirichletBC(zeta.function_space(), 0, j))
                else:
                    raise NotImplementedError("Weak boundary conditions not yet implemented")  # TODO
            else:
                msg = "Have not considered continuous adjoint for boundary condition {:s}"
                raise NotImplementedError(msg.format(bc))

        # dJdq forcing term
        self.get_qoi_kernels(i)
        t = op.dt*(i+1)*self.dt_per_mesh
        self.time_kernel = Constant(1.0 if t >= op.start_time else 0.0)
        dJdu, dJdeta = self.kernels[i].split()
        L += dtc*inner(z_test, self.time_kernel*dJdu)*dx
        L += dtc*inner(zeta_test, self.time_kernel*dJdeta)*dx

        # Solver object
        kwargs = {
            'solver_parameters': op.adjoint_params,
            'options_prefix': 'adjoint',
        }
        problem = LinearVariationalProblem(a, L, self.adj_solutions[i], bcs=dbcs)
        self.adj_solvers[i] = LinearVariationalSolver(problem, **kwargs)

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

        # Need to project to P1 for plotting
        self.adjoint_solution_file._topology = None  # Account for mesh adaptations
        z, zeta = self.adj_solutions[i].split()
        z_out = Function(self.P1_vec[i], name="Projected adjoint velocity")
        zeta_out = Function(self.P1[i], name="Projected adjoint elevation")

        j = 0
        op.print_debug("Entering adjoint time loop...")
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
