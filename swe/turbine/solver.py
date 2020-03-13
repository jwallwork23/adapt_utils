from thetis import *

from adapt_utils.swe.solver import *
from adapt_utils.swe.turbine.options import *
from adapt_utils.adapt.recovery import *
from adapt_utils.adapt.metric import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import h5py
import os


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


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
        # scaling = num_turbines/assemble(op.box(self.P0)*dx)
        # self.turbine_density = op.box(self.P0, scale=scaling)
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

    def extra_residual_terms(self):
        u, eta = self.solution.split()
        z, zeta = self.adjoint_solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*inner(z, u)/H

    def extra_strong_residual_terms_momentum(self):
        u, eta = self.solution.split()
        H = self.op.bathymetry + eta
        return -self.C_D*sqrt(dot(u, u))*u/H

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
            self.cb = turbines.TurbineFunctionalCallback(self.solver_obj, print_to_screen=False)
            self.solver_obj.add_callback(self.cb, 'timestep')

    def quantity_of_interest(self):
        self.qoi = self.cb.average_power
        return self.qoi

    def get_qoi_kernel(self):
        self.kernel = Function(self.V)
        u = self.solution.split()[0]
        k_u = self.kernel.split()[0]
        k_u.interpolate(Constant(1/3)*self.turbine_density*sqrt(inner(u, u))*u)

    def plot_power_timeseries(self, fontsize=18):
        dat = h5py.File(os.path.join(self.op.di, 'diagnostic_turbine.hdf5'))
        time_period = np.array(dat['time'])
        if time_period[-1] < 60.0*5:
            time_units = '$\mathrm s$'
        elif time_period[-1] < 3600.0*5:
            time_units = 'minutes'
            time_period /= 60.0
        else:
            time_units = 'hours'
            time_period /= 3600.0
        power = np.array(dat['current_power']).transpose()
        dat.close()
        max_power = power.max()
        if max_power < 5e3:
            power_units = '$\mathrm W$'
        elif max_power < 5e6:
            power_units = '$\mathrm{kW}$'
            power /= 1e3
        else:
            power_units = '$\mathrm{MW}$'
            power /= 1e6
        total_power = np.sum(power, axis=0)
        num_turbines = power.shape[0]
        fig = plt.figure(figsize=(12, 7))
        ax = plt.subplot(111)
        array_width = self.op.array_width
        n = self.op.dt_per_export
        for i in range(num_turbines):
            col = (i-i%array_width)//array_width
            row = i%array_width
            label = 'Turbine {:d}({:s})'.format(col+1, chr(97+row))
            colour = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'][col]
            marker = ['x', 'o', '+'][row]
            ax.plot(time_period[::n], power[i][::n], label=label, marker=marker, color=colour)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.xlabel(r'Time ({:s})'.format(time_units), fontsize=fontsize)
        plt.ylabel(r'Power ({:s})'.format(power_units), fontsize=fontsize)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        plt.savefig(os.path.join(self.op.di, 'power_timeseries.pdf'))
        plt.title('Power output of turbines in a {:d} turbine array'.format(num_turbines), fontsize=fontsize)
        plt.savefig(os.path.join(self.op.di, 'power_timeseries.png'))
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(time_period[::n], total_power[::n], marker='x')
        plt.xlabel(r'Time ({:s})'.format(time_units), fontsize=fontsize)
        plt.ylabel(r'Power ({:s})'.format(power_units), fontsize=fontsize)
        plt.savefig(os.path.join(self.op.di, 'total_power_timeseries.pdf'))
        plt.title('Total power output of a {:d} turbine array'.format(num_turbines), fontsize=fontsize)
        plt.savefig(os.path.join(self.op.di, 'total_power_timeseries.png'))
        plt.show()
