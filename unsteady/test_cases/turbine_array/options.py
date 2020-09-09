from thetis import *
from thetis.configuration import *

import os

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions


__all__ = ["TurbineArrayOptions"]


class TurbineArrayOptions(TurbineOptions):
    """Parameters for the unsteady 15 turbine array test case."""

    # Turbine parameters
    turbine_length = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(5.0).tag(config=False)
    array_length = PositiveInteger(5).tag(config=False)
    array_width = PositiveInteger(3).tag(config=False)
    num_turbines = PositiveInteger(15).tag(config=False)

    # Domain specification
    mesh_file = os.path.join(os.path.dirname(__file__), 'channel.msh')
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    def __init__(self, spun=False, **kwargs):
        super(TurbineArrayOptions, self).__init__(**kwargs)
        self.array_ids = np.array([[2, 5, 8, 11, 14],
                                   [3, 6, 9, 12, 15],
                                   [4, 7, 10, 13, 16]])
        self.farm_ids = tuple(self.array_ids.reshape((self.num_turbines, )))

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.default_mesh = Mesh(self.mesh_file)
        else:
            raise OSError("Need to make mesh before initialising TurbineArrayOptions object.")

        # Physics
        self.base_viscosity = 3.0
        self.base_bathymetry = 50.0
        self.max_depth = 50.0
        self.friction_coeff = 0.0025

        # Timestepping
        self.dt = 2.232
        self.T_tide = 0.1*self.M2_tide_period
        # self.T_ramp = 1.06*3600
        self.T_ramp = 5*self.T_tide
        self.end_time = 3*self.T_tide
        self.dt_per_export = 10

        # Tidal farm
        D = self.turbine_length
        d = self.turbine_width
        self.turbine_diameter = max(D, d)
        deltax = 10.0*D
        deltay = 7.5*D
        self.region_of_interest = []
        for i in range(-2, 3):
            for j in range(1, -2, -1):
                self.region_of_interest.append((i*deltax, j*deltay, d, D))
        assert len(self.region_of_interest) == self.num_turbines
        self.turbine_tags = list(range(2, 2+self.num_turbines))

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.T_tide
        self.elev_in = [None for i in range(self.num_meshes)]
        self.elev_out = [None for i in range(self.num_meshes)]
        self.spun = spun

        # Solver parameters and discretisation
        self.stabilisation = 'lax_friedrichs'
        self.use_automatic_sipg_parameter = True
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        self.family = 'dg-cg'

    def set_viscosity(self, fs):
        sponge = False
        if sponge:
            x, y = SpatialCoordinate(fs.mesh())
            xmax = 1000.0
            ramp = 0.5
            eps = 20.0
            return interpolate(self.base_viscosity + exp(ramp*(x - xmax + eps)), fs)
        else:
            return Constant(self.base_viscosity)

    def set_boundary_conditions(self, prob, i):
        self.elev_in[i] = Function(prob.V[i].sub(1))
        self.elev_out[i] = Function(prob.V[i].sub(1))
        inflow_tag = 4
        outflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in[i]},
                outflow_tag: {'elev': self.elev_out[i]},
            }
        }
        return boundary_conditions

    def get_update_forcings(self, prob, i, **kwargs):
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)
        offset = self.T_ramp if self.spun else 0.0

        def update_forcings(t):
            tc.assign(t - offset)
            self.elev_in[i].assign(hmax*cos(self.omega*tc))
            self.elev_out[i].assign(hmax*cos(self.omega*tc + pi))

        return update_forcings

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])

        # Set an arbitrary, small, non-zero velocity which satisfies the free-slip conditions
        u.interpolate(as_vector([1e-8, 0.0]))

        # Set the initial surface so that it satisfies the forced boundary conditions
        X = 2*x/self.domain_length  # Non-dimensionalised x
        eta.interpolate(-self.max_amplitude*X)
