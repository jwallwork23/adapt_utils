from thetis import *
from thetis.configuration import *
import os

from adapt_utils.turbine.options import UnsteadyTurbineOptions


__all__ = ["Unsteady15TurbineOptions"]


rootdir = os.environ.get('ADAPT_UTILS_HOME')


class Unsteady15TurbineOptions(UnsteadyTurbineOptions):
    """Parameters for the unsteady 15 turbine array test case"""

    # Turbine parameters
    turbine_length = PositiveFloat(20.0).tag(config=True)
    turbine_width = PositiveFloat(5.0).tag(config=True)
    array_length = PositiveInteger(5).tag(config=True)
    array_width = PositiveInteger(3).tag(config=True)
    thrust_coefficient = NonNegativeFloat(7.6).tag(config=True)
    meshfile = Unicode(os.path.join(rootdir, 'test_cases', 'unsteady_turbine', 'channel.msh')).tag(config=True)
    params = PETScSolverParameters({}).tag(config=True)  # TODO

    def __init__(self, approach='fixed_mesh'):
        self.base_viscosity = 3.0
        super(Unsteady15TurbineOptions, self).__init__(approach)
        self.domain_length = 3000.0
        self.domain_width = 1000.0
        try:
            os.path.exists(self.meshfile)
        except OSError:
            raise ValueError("Mesh fine not generated.")
        self.default_mesh = Mesh(self.meshfile)
        self.bathymetry.assign(50.0)

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
        L = self.domain_length
        W = self.domain_width
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
        self.lax_friedrichs = False  # TODO: temp
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = True
        self.family = 'dg-cg'

    def set_viscosity(self, fs):
        sponge = False
        self.viscosity = Function(fs)
        if sponge:
            x, y = SpatialCoordinate(fs.mesh())
            xmin = 0.0
            xmax = 1000.0
            ramp = 0.5
            eps = 20.0
            self.viscosity.interpolate(self.base_viscosity + exp(ramp*(x-xmax+eps)))
        else:
            self.viscosity.assign(self.base_viscosity)

    def set_boundary_conditions(self, fs):
        self.set_boundary_surface(fs.sub(1))
        inflow_tag = 4
        outflow_tag = 2
        if not hasattr(self, 'boundary_conditions'):
            self.boundary_conditions = {}
        self.boundary_conditions[inflow_tag] = {'elev': self.elev_in}
        self.boundary_conditions[outflow_tag] = {'elev': self.elev_out}
        return self.boundary_conditions

    def set_initial_condition(self, fs):
        self.initial_condition = Function(fs)
        u, eta = self.initial_condition.split()
        x, y = SpatialCoordinate(fs.mesh())
        u.interpolate(as_vector([1e-8, 0.0]))
        eta.interpolate(-1/3000*x)
        return self.initial_condition
