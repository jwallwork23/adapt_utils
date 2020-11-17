from thetis import *
from thetis.configuration import *

import numpy as np
import os

from adapt_utils.tracer.desalination.options import DesalinationOutfallOptions


__all__ = ["IdealisedDesalinationOutfallOptions"]


class IdealisedDesalinationOutfallOptions(DesalinationOutfallOptions):
    """
    Parameters for an idealised desalination plant outfall scenario.
    """
    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    # Domain specification
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    def __init__(self, level=0, aligned=False, spun=False, **kwargs):
        super(IdealisedDesalinationOutfallOptions, self).__init__(spun=spun, **kwargs)

        # Domain
        self.default_mesh = Mesh(os.path.join(self.resource_dir, 'channel_{:d}.msh'.format(level)))

        # Hydrodynamics
        self.base_viscosity = 3.0
        self.base_diffusivity = 10.0
        self.friction_coeff = 0.0025  # TODO: Increased drag at pipes?
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        self.characteristic_speed = Constant(1.15)  # Max observed fluid speed
        self.characteristic_diffusion = Constant(self.base_diffusivity)

        # Time integration
        self.timestepper = 'CrankNicolson'
        self.start_time = 0.0
        # self.T_tide = 0.1*self.M2_tide_period
        self.T_tide = 0.05*self.M2_tide_period
        # self.T_ramp = 3.855*self.T_tide
        self.T_ramp = self.T_tide
        self.end_time = 2*self.T_tide if spun else self.T_ramp
        self.dt = 2.232
        self.dt_per_export = 10

        # Tracer FEM
        self.degree_tracer = 1
        self.tracer_family = 'cg'
        self.stabilisation_tracer = 'supg'
        self.use_limiter_for_tracers = False

        # Hydrodynamics FEM
        self.degree = 1
        self.family = 'dg-cg'
        self.use_automatic_sipg_parameter = True
        self.stabilisation = 'lax_friedrichs'
        self.lax_friedrichs_velocity_scaling_factor = Constant(1.0)

        # Source (outlet pipe)
        self.source_value = 2.0  # Discharge rate
        outlet_x = 0.0
        outlet_y = 100.0
        self.source_loc = [(outlet_x, outlet_y, 25.0)]  # Outlet

        # Receiver (inlet pipe)
        inlet_x = 0.0 if aligned else 400.0
        inlet_y = -100.0
        self.region_of_interest = [(inlet_x, inlet_y, 25.0)]  # Inlet

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.T_tide
        self.elev_in = [None for i in range(self.num_meshes)]
        self.elev_out = [None for i in range(self.num_meshes)]

    def set_tracer_source(self, fs):
        return self.ball(fs.mesh(), source=True, scale=self.source_value)

    def set_qoi_kernel_tracer(self, prob, i):
        return self.set_qoi_kernel(prob.meshes[i])

    def set_qoi_kernel(self, mesh):
        b = self.ball(mesh, source=False)
        area = assemble(b*dx)
        area_analytical = pi*self.region_of_interest[0][2]**2
        rescaling = 1.0 if np.allclose(area, 0.0) else area_analytical/area
        return rescaling*b

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(fs.mesh())
        bathymetry = Function(fs)
        # W = self.domain_width
        # bathymetry.interpolate(50.0 + 100.0*(W - y)/W)  # TODO: use, accounting for BC
        bathymetry.assign(50.0)
        return bathymetry

    def set_quadratic_drag_coefficient(self, fs):
        return Constant(self.friction_coeff)

    def set_boundary_conditions(self, prob, i):
        """
        Domain
        ======
                3
            ---------
          4 |       | 2
            ---------
                1

        We interpret segment 1 as open ocean, {2, 4} as tidally forced and 3 as coast.
        """
        self.elev_in[i] = Function(prob.V[i].sub(1))
        self.elev_out[i] = Function(prob.V[i].sub(1))
        bottom_tag = 1
        outflow_tag = 2
        top_tag = 3
        inflow_tag = 4
        zero = Constant(0.0)
        bg = Constant(self.background_salinity)
        boundary_conditions = {
            'shallow_water': {
                # bottom_tag: {???},
                outflow_tag: {'elev': self.elev_out[i]},  # forced
                # top_tag: {},                              # free-slip
                inflow_tag: {'elev': self.elev_in[i]},    # forced
            },
            'tracer': {
                # bottom_tag: {},                   # open
                bottom_tag: {'diff_flux': zero},  # Neumann
                outflow_tag: {'value': bg},       # Dirichlet
                top_tag: {'diff_flux': zero},     # Neumann
                inflow_tag: {'value': bg},        # Dirichlet
            },
        }
        return boundary_conditions

    def get_update_forcings(self, prob, i, **kwargs):
        """
        Simple tidal forcing with frequency :attr:`omega` and amplitude :attr:`max_amplitude`.

        In addition, write data for the x-component of velocity to file.

        :arg prob: :class:`AdaptiveDesalinationProblem` object.
        :arg i: mesh index.
        """
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)
        offset = self.T_ramp if self.spun else 0.0
        if self.spun:
            u, eta = prob.fwd_solutions[i].split()
            velocity = Function(prob.P1[i])
            fname = os.path.join(os.path.dirname(__file__), "data", "x-velocity.log")
            with open(fname, 'w') as outfile:
                outfile.write("t      min    max    mean   sd\n")

        def update_forcings(t):
            tt = t + offset

            # Update tidal forcing
            tc.assign(tt)
            self.elev_in[i].assign(hmax*cos(self.omega*tc))
            self.elev_out[i].assign(hmax*cos(self.omega*tc + pi))

            # Write x-velocity to file
            if self.spun:
                velocity.interpolate(u[0])
                data = velocity.vector().gather()
                msg = "{:7.1f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}\n"
                with open(fname, 'a') as outfile:
                    outfile.write(msg.format(tt, data.min(), data.max(), np.mean(data), np.sqrt(np.var(data))))

        return update_forcings

    def get_export_func(self, prob, i, **kwargs):

        def export_func():
            return

        return export_func

    def set_initial_condition(self, prob):
        """
        Specify elevation at the start of the spin-up period so that it satisfies the boundary
        forcing and set an arbitrary small velocity.

        :arg prob: :class:`AdaptiveDesalinationProblem` object.
        """
        assert not self.spun
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])

        # Set an arbitrary, small, non-zero velocity which satisfies the free-slip conditions
        u.interpolate(as_vector([1e-8, 0.0]))

        # Set the initial surface so that it satisfies the forced boundary conditions
        X = 2*x/self.domain_length  # Non-dimensionalised x
        eta.interpolate(-self.max_amplitude*X)
