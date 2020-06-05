from thetis import *

import os
import h5py

from adapt_utils.callback import *
from adapt_utils.swe.adapt_solver import AdaptiveShallowWaterProblem


__all__ = ["AdaptiveTsunamiProblem"]


class AdaptiveTsunamiProblem(AdaptiveShallowWaterProblem):
    """General solver object for adaptive tsunami propagation problems."""

    # -- Setup

    def __init__(self, *args, extension=None, nonlinear=False, **kwargs):
        self.extension = extension
        super(AdaptiveTsunamiProblem, self).__init__(*args, nonlinear=nonlinear, **kwargs)

    def add_callbacks(self, i):
        op = self.op

        # Gauge timeseries
        for gauge in op.gauges:
            self.callbacks[i].add(GaugeCallback(self, i, gauge), 'export')

        # Quantity of interest
        self.get_qoi_kernels(i)
        self.callbacks[i].add(QoICallback(self, i), 'timestep')

    def quantity_of_interest(self):
        self.qoi = sum(c['timestep']['qoi'].time_integrate() for c in self.callbacks)
        return self.qoi

    def save_gauge_data(self, fname):
        fname = "diagnostic_gauges_{:s}.hdf5".format(fname)
        with h5py.File(os.path.join(self.di, fname), 'w') as f:
            for gauge in self.op.gauges:
                timeseries = []
                for i in range(self.num_meshes):
                    timeseries.extend(self.callbacks[i]['export'][gauge].timeseries)
                f.create_dataset(gauge, data=np.array(timeseries))
            f.create_dataset("num_cells", data=np.array(self.num_cells[-1]))

    # # TODO: Use Thetis equation / timeintegrator
    # def setup_solver_adjoint(self, i, **kwargs):
    #     """Setup continuous adjoint solver on mesh `i`."""
    #     op = self.op
    #     dtc = Constant(op.dt)
    #     g = Constant(op.g)
    #     b = self.bathymetry[i]
    #     f = self.fields[i]['coriolis_frequency']
    #     n = FacetNormal(self.meshes[i])

    #     # Mixed function space
    #     z, zeta = TrialFunctions(self.V[i])
    #     z_test, zeta_test = TestFunctions(self.V[i])
    #     self.adj_solutions_old[i].assign(self.adj_solutions[i])  # Assign previous value
    #     z_old, zeta_old = split(self.adj_solutions_old[i])

    #     def TaylorHood(uv, elev):
    #         F = -inner(z_test, b*grad(elev))*dx                   # - ∇ (b ζ)
    #         F += -inner(z_test, f*as_vector((-uv[1], uv[0])))*dx  # - f perp(z)
    #         F += g*inner(grad(zeta_test), uv)*dx                  # - g ∇ . z
    #         return F

    #     def Mixed(uv, elev):
    #         F = -inner(z_test, b*grad(elev))*dx                   # - ∇ (b ζ)
    #         F += -inner(z_test, f*as_vector((-uv[1], uv[0])))*dx  # - f perp(z)
    #         F += g*inner(grad(zeta_test), uv)*dx                  # - g ∇ . z
    #         # F += -g*inner(avg(zeta_test*n), avg(uv))*dS         # flux term
    #         return F

    #     family = self.shallow_water_options['element_family']
    #     try:
    #         G = {'cg-cg': TaylorHood, 'dg-cg': Mixed}[family]
    #     except KeyError:
    #         raise ValueError("Mixed discretisation {:s} not supported.".format(family))

    #     # Time derivative
    #     a = inner(z_test, z)*dx + inner(zeta_test, zeta)*dx
    #     L = inner(z_test, z_old)*dx + inner(zeta_test, zeta_old)*dx

    #     # Crank-Nicolson timestepping
    #     try:
    #         assert self.timestepping_options['timestepper_type'] == 'CrankNicolson'
    #     except AssertionError:
    #         raise NotImplementedError  # TODO
    #     print_output("### TODO: Implement adjoint ts other than Crank-Nicolson")
    #     a += 0.5*dtc*G(z, zeta)
    #     L -= 0.5*dtc*G(z_old, zeta_old)

    #     # Boundary conditions
    #     # ===================
    #     #   In the forward, we permit only free-slip conditions for the velocity and Dirichlet
    #     #   conditions for the elevation. Suppose these are imposed on Gamma_1 and Gamma_2, which
    #     #   are not necessarily disjoint. Then the adjoint has a free-slip condition for the adjoint
    #     #   velocity on the complement of Gamma_2 and Dirichlet conditions for the elevation on the
    #     #   complement of Gamma_1.
    #     boundary_markers = self.meshes[i].exterior_facets.unique_markers
    #     BCs = self.boundary_conditions[i]['shallow_water']
    #     if BCs == {}:  # Default Thetis boundary conditions are free-slip
    #         BCs = {j: {'un': Constant(0.0)} for j in boundary_markers}
    #     dbcs = []
    #     for j in boundary_markers:
    #         bcs = BCs.get(j)
    #         if 'un' in bcs:
    #             if 'elev' in bcs:
    #                 a += -0.5*dtc*g*inner(zeta_test, dot(z, n))*ds(j)
    #                 L += 0.5*dtc*g*inner(zeta_test, dot(z_old, n))*ds(j)
    #             else:
    #                 L += dtc*g*inner(zeta_test, bcs['un'])*ds(j)
    #         elif 'elev' in bcs:
    #             if zeta.ufl_element().family() == 'Lagrange':
    #                 dbcs.append(DirichletBC(zeta.function_space(), 0, j))
    #             else:
    #                 raise NotImplementedError("Weak boundary conditions not yet implemented")  # TODO
    #         else:
    #             msg = "Have not considered continuous adjoint for boundary condition {:s}"
    #             raise NotImplementedError(msg.format(bc))

    #     # dJdq forcing term
    #     self.get_qoi_kernels(i)
    #     t = op.dt*(i+1)*self.dt_per_mesh
    #     self.time_kernel = Constant(1.0 if t >= op.start_time else 0.0)
    #     dJdu, dJdeta = self.kernels[i].split()
    #     L += dtc*inner(z_test, self.time_kernel*dJdu)*dx
    #     L += dtc*inner(zeta_test, self.time_kernel*dJdeta)*dx

    #     # Solver object
    #     kwargs = {
    #         'solver_parameters': op.adjoint_params,
    #         'options_prefix': 'adjoint',
    #     }
    #     problem = LinearVariationalProblem(a, L, self.adj_solutions[i], bcs=dbcs)
    #     self.adj_solvers[i] = LinearVariationalSolver(problem, **kwargs)

    # # TODO: Use Thetis equation / timeintegrator
    # def solve_adjoint_step(self, i, export_func=None, update_forcings=None):
    #     """
    #     Solve adjoint PDE on mesh `i`.

    #     :kwarg update_forcings: a function which takes simulation time as an argument and is
    #         evaluated at the start of every timestep.
    #     :kwarg export_func: a function with no arguments which is evaluated at every export step.
    #     """
    #     op = self.op
    #     t = op.dt*(i+1)*self.dt_per_mesh
    #     end_time = op.dt*i*self.dt_per_mesh

    #     # Need to project to P1 for plotting
    #     self.adjoint_solution_file._topology = None  # Account for mesh adaptations
    #     z, zeta = self.adj_solutions[i].split()
    #     z_out = Function(self.P1_vec[i], name="Projected adjoint velocity")
    #     zeta_out = Function(self.P1[i], name="Projected adjoint elevation")

    #     j = 0
    #     op.print_debug("Entering adjoint time loop...")
    #     while t > end_time:
    #         if update_forcings is not None:
    #             update_forcings(t)
    #         if j % op.dt_per_export == 0:
    #             print_output("t = {:6.1f}".format(t))
    #             if export_func is not None:
    #                 export_func()
    #             z, zeta = self.adj_solutions[i].split()
    #             zeta_out.project(zeta)
    #             z_out.project(z)
    #             self.adjoint_solution_file.write(z_out, zeta_out)
    #         self.time_kernel.assign(1.0 if t >= op.start_time else 0.0)
    #         self.adj_solvers[i].solve()
    #         self.adj_solutions_old[i].assign(self.adj_solutions[i])
    #         t -= op.dt
    #         j += 1
    #     assert j % op.dt_per_export == 0
    #     if export_func is not None:
    #         export_func()
    #     op.print_debug("Done!")
