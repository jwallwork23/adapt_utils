from thetis import *

from adapt_utils.swe.solver import UnsteadyShallowWaterProblem
from adapt_utils.swe.tsunami.qois import InundationCallback


__all__ = ["TsunamiProblem"]


class TsunamiProblem(UnsteadyShallowWaterProblem):
    """
    For general tsunami propagation problems.
    """
    def __init__(self, *args, extension=None, **kwargs):
        self.extension = extension
        super(TsunamiProblem, self).__init__(*args, **kwargs)


    def set_fields(self, adapted=False):
        self.fields = {}
        self.fields['viscosity'] = self.op.set_viscosity(self.P1)
        self.fields['diffusivity'] = self.op.set_diffusivity(self.P1)
        # TODO: Should interpolate bathy from fresh
        if self.op.solve_tracer == False:
            self.fields['bathymetry'] = self.op.set_bathymetry(self.P1, adapted=adapted)        
        self.fields['coriolis'] = self.op.set_coriolis(self.P1)
        self.fields['quadratic_drag_coefficient'] = self.op.set_quadratic_drag_coefficient(self.P1DG)
        self.fields['manning_drag_coefficient'] = self.op.set_manning_drag_coefficient(self.P1)
        self.fields['source'] = self.op.source
        
        #self.op.set_boundary_surface()

    def extra_setup(self):  # TODO: Plot eta_tilde, too
        op = self.op

        # Don't bother plotting velocity
        #self.solver_obj.options.fields_to_export = ['elev_2d'] if op.plot_pvd else []
        #self.solver_obj.options.fields_to_export_hdf5 = ['elev_2d'] if op.save_hdf5 else []

        # Set callbacks to save gauge timeseries to HDF5
        self.callbacks = {}
        locs = [op.gauges[g]["coords"] for g in op.gauges]
        names = list(op.gauges.keys())
        fname = "gauges"
        if self.extension is not None:
            fname = '_'.join([fname, self.extension])
        fname = '_'.join([fname, str(self.num_cells[-1])])
        for g in op.gauges:
            self.callbacks[g] = callback.DetectorsCallback(self.solver_obj, locs, ['elev_2d'],
                                                           fname, names)
            self.solver_obj.add_callback(self.callbacks[g], 'export')

        # Set callback for QoI evaluation
        self.callbacks["qoi"] = InundationCallback(self.solver_obj)
        self.solver_obj.add_callback(self.callbacks["qoi"])
