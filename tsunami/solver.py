from adapt_utils.swe.solver import UnsteadyShallowWaterProblem


__all__ = ["TsunamiProblem"]


class TsunamiProblem(UnsteadyShallowWaterProblem):
    # TODO: doc
    def set_fields(self):
        self.viscosity = self.op.set_viscosity(self.P1)
        self.diffusivity = self.op.set_diffusivity(self.P1)
        self.bathymetry = self.op.set_bathymetry(self.P1)
        # self.inflow = self.op.set_inflow(self.P1_vec)
        self.coriolis = self.op.set_coriolis(self.P1)
        self.quadratic_drag_coefficient = self.op.set_quadratic_drag_coefficient(self.P1)
        self.manning_drag_coefficient = self.op.set_manning_drag_coefficient(self.P1)
        # self.op.set_boundary_surface()

        # Stabilisation
        self.stabilisation = self.stabilisation or 'no'
        if self.stabilisation in ('no', 'lax_friedrichs'):
            self.stabilisation_parameter = self.op.stabilisation_parameter
        else:
            raise ValueError("Stabilisation method {:s} for {:s} not recognised".format(self.stabilisation, self.__class__.__name__))

