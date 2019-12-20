from thetis import *
from firedrake.petsc import PETSc

from adapt_utils.swe.solver import *
from adapt_utils.turbine.options import *
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.metric import *


__all__ = ["SteadyTurbineProblem", "UnsteadyTurbineProblem"]


class SteadyTurbineProblem(SteadyShallowWaterProblem):
    """
    General solver object for stationary tidal turbine problems.
    """
    def extra_setup(self):
        """
        We haven't meshed the turbines with separate ids, so define a farm everywhere and make it
        have a density of 1/D^2 inside the DxD squares where the turbines are and 0 outside.
        """
        op = self.op
        num_turbines = len(op.region_of_interest)
        scaling = num_turbines/assemble(op.bump(self.P1)*dx)
        self.turbine_density = op.bump(self.P1, scale=scaling)
        self.farm_options = TidalTurbineFarmOptions()
        self.farm_options.turbine_density = self.turbine_density
        self.farm_options.turbine_options.diameter = op.turbine_diameter
        self.farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient

        A_T = pi*(op.turbine_diameter/2.0)**2
        self.C_D = op.thrust_coefficient*A_T*self.turbine_density/2.0

        # Turbine drag is applied everywhere (where the turbine density isn't zero)
        self.solver_obj.options.tidal_turbine_farms["everywhere"] = self.farm_options

        # Callback that computes average power
        self.cb = turbines.TurbineFunctionalCallback(self.solver_obj)
        self.solver_obj.add_callback(self.cb, 'timestep')

    def extra_residual_terms(self, u, eta, z, zeta):
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*inner(z, u)/H

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def quantity_of_interest_form(self):
        return self.C_D*pow(inner(split(self.solution)[0], split(self.solution)[0]), 1.5)*dx

    def custom_adapt(self):
        if self.approach == 'vorticity':
            self.indicator = Function(self.P1, name='vorticity')
            self.indicator.interpolate(curl(self.solution.split()[0]))
            self.get_isotropic_metric()


class UnsteadyTurbineProblem(UnsteadyShallowWaterProblem):
    """
    General solver object for time-dependent tidal turbine problems.
    """
    def get_update_forcings(self):
        op = self.op
        def update_forcings(t):
            op.elev_in.assign(op.max_amplitude*cos(op.omega*(t-op.T_ramp)))
            op.elev_out.assign(op.max_amplitude*cos(op.omega*(t-op.T_ramp)+pi))
        return update_forcings

    def extra_setup(self):
        op = self.op
        self.update_forcings(0.0)

        # Tidal farm
        num_turbines = len(op.region_of_interest)
        if num_turbines > 0:
            # We haven't meshed the turbines with separate ids, so define a farm everywhere
            # and make it have a density of 1/D^2 inside the DxD squares where the turbines are
            # and 0 outside
            self.turbine_density = Constant(1.0/(op.turbine_diameter*5), domain=self.mesh)
            # scaling = num_turbines/assemble(op.bump(self.P1)*dx)  # FIXME
            # self.turbine_density = op.bump(self.P1, scale=scaling)
            self.farm_options = TidalTurbineFarmOptions()
            self.farm_options.turbine_density = self.turbine_density
            self.farm_options.turbine_options.diameter = op.turbine_diameter
            self.farm_options.turbine_options.thrust_coefficient = op.thrust_coefficient
            for i in op.turbine_tags:
                self.solver_obj.options.tidal_turbine_farms[i] = self.farm_options

            # Callback that computes average power
            self.cb = turbines.TurbineFunctionalCallback(self.solver_obj)
            self.solver_obj.add_callback(self.cb, 'timestep')

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)
