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

    def __init__(self, level=0, aligned=False, **kwargs):
        self.timestepper = 'CrankNicolson'
        super(IdealisedDesalinationOutfallOptions, self).__init__(**kwargs)
        self.solve_swe = False
        self.solve_tracer = True
        self.adapt_field = 'tracer'

        # Domain
        mesh_file = os.path.join(self.resource_dir, 'channel_{:d}.msh'.format(level))
        if os.path.exists(mesh_file):
            self.default_mesh = Mesh(mesh_file)

        # Hydrodynamics
        self.base_diffusivity = Constant(10.0)
        self.base_bathymetry = 50.0
        self.characteristic_speed = Constant(1.15)  # Max fluid speed
        self.characteristic_diffusion = self.base_diffusivity

        # Time integration
        self.start_time = 0.0
        # self.T_tide = 0.1*self.M2_tide_period
        self.T_tide = 0.05*self.M2_tide_period
        self.end_time = 2*self.T_tide
        self.dt = 2.232

        # FEM
        self.degree_tracer = 1
        self.tracer_family = 'cg'
        self.stabilisation_tracer = 'supg'
        self.use_limiter_for_tracers = False

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
        self.omega = 2*pi/self.T_tide
        self.tc = Constant(0.0)

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

    def set_boundary_conditions(self, prob, i):
        """
        Domain
        ======
                3
            ---------
          4 |       | 2
            ---------
                1
        """
        bottom_tag = 1
        outflow_tag = 2
        top_tag = 3
        inflow_tag = 4
        zero = Constant(0.0)
        bg = Constant(self.background_salinity)
        boundary_conditions = {
            'tracer': {
                bottom_tag: {'diff_flux': zero},  # Neumann
                outflow_tag: {'value': bg},       # Dirichlet
                top_tag: {'diff_flux': zero},     # Neumann
                inflow_tag: {'value': bg},        # Dirichlet
            },
        }
        return boundary_conditions

    def get_velocity(self, t):
        self.tc.assign(t)
        return as_vector([self.characteristic_speed*sin(self.omega*self.tc), 0.0])

    def set_initial_condition(self, prob, i=0, t=0):
        u, eta = prob.fwd_solutions[i].split()
        u.interpolate(self.get_velocity(t))

    def get_update_forcings(self, prob, i, **kwargs):
        """
        Simple tidal forcing with frequency :attr:`omega` which is
        enforced via the velocity, using amplitude :attr:`characteristic_speed`.

        :arg prob: :class:`AdaptiveDesalinationProblem` object.
        :arg i: mesh index.
        """

        def update_forcings(t):
            self.set_initial_condition(prob, i=i, t=t)

        return update_forcings

    def set_initial_condition_tracer(self, prob):
        """
        Initialise salinity with the background value.

        :arg prob: :class:`AdaptiveDesalinationProblem` object.
        """
        prob.fwd_solution_tracer.assign(self.background_salinity)
