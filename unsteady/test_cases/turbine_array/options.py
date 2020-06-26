from thetis import *
from thetis.configuration import *
import os

from adapt_utils.unsteady.swe.turbine.options import TurbineOptions


__all__ = ["TurbineArrayOptions"]


class TurbineArrayOptions(TurbineOptions):
    """Parameters for the unsteady 15 turbine array test case"""

    # Turbine parameters
    turbine_length = PositiveFloat(20.0).tag(config=True)
    turbine_width = PositiveFloat(5.0).tag(config=True)
    array_length = PositiveInteger(5).tag(config=True)
    array_width = PositiveInteger(3).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    meshfile = os.path.join(os.path.dirname(__file__), 'channel.msh')

    def __init__(self, **kwargs):
        super(TurbineArrayOptions, self).__init__(**kwargs)
        self.domain_length = 3000.0
        self.domain_width = 1000.0
        try:
            os.path.exists(self.meshfile)
        except OSError:
            raise ValueError("Mesh fine not generated.")
        self.default_mesh = Mesh(self.meshfile)

        # Physics
        self.base_viscosity = 3.0
        self.base_bathymetry = 50.0

        # Timestepping
        self.dt = 3.0
        self.T_tide = 1.24*3600.0
        self.T_ramp = 1.0*self.T_tide
        self.end_time = self.T_ramp + 2.0*self.T_tide
        self.dt_per_export = 10
        self.dt_per_remesh = 10  # FIXME: solver seems to go out of sync if this != dt_per_export

        # Tidal farm
        D = self.turbine_length
        d = self.turbine_width
        self.turbine_diameter = max(D, d)
        deltax = 10.0*D
        deltay = 7.5*D
        self.region_of_interest = []
        for i in range(-2, 3):
            for j in range(-1, 2):
                self.region_of_interest.append((i*deltax, j*deltay, D, d))
        self.num_turbines = len(self.region_of_interest)
        self.turbine_tags = range(2, 2+self.num_turbines)
        self.thrust_coefficient_correction()

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.T_tide

        # Solver parameters and discretisation
        self.stabilisation = None
        # self.stabilisation = 'lax_friedrichs'
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
            return interpolate(self.base_viscosity + exp(ramp*(x-xmax+eps)), fs)
        else:
            return Constant(self.base_viscosity)

    def set_boundary_conditions(self, prob, i):
        uv_in, self.elev_in = Function(prob.V[i])
        uv_out, self.elev_out = Function(prob.V[i])
        inflow_tag = 4
        outflow_tag = 2
        boundary_conditions = {
            'shallow_water': {
                inflow_tag: {'elev': self.elev_in},
                outflow_tag: {'elev': self.elev_out},
            }
        }
        return boundary_conditions

    # TODO: update_forcings for elev_in and elev_out

    def set_initial_condition(self, prob):
        u, eta = prob.fwd_solutions[0].split()
        x, y = SpatialCoordinate(prob.meshes[0])
        u.interpolate(as_vector([1e-8, 0.0]))
        eta.interpolate(-1/3000*x)
