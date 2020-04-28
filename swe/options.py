from thetis import *
from thetis.configuration import *

from adapt_utils.options import Options


__all__ = ["ShallowWaterOptions"]


class ShallowWaterOptions(Options):
    """
    Parameters for shallow water solver.
    """
    solve_tracer = Bool(False).tag(config=True)
    solve_swe = Bool(True).tag(config=True)

    # Physical
    bathymetry = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    base_viscosity = NonNegativeFloat(0.0).tag(config=True)
    base_diffusivity = NonNegativeFloat(0.0).tag(config=True)
    viscosity = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    diffusivity = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    quadratic_drag_coefficient = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    manning_drag_coefficient = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    inflow = FiredrakeVectorExpression(None, allow_none=True).tag(config=True)
    g = FiredrakeScalarExpression(Constant(9.81)).tag(config=True)
    coriolis = FiredrakeScalarExpression(None, allow_none=True).tag(config=True)
    source = FiredrakeScalarExpression(None, allow_none=True, help="Scalar source term for tracer problem.").tag(config=True)
    implicitness_theta = NonNegativeFloat(0.5).tag(config=True)

    # Model
    grad_div_viscosity = Bool(False).tag(config=True)
    grad_depth_viscosity = Bool(False).tag(config=True)
    family = Enum(['dg-dg', 'rt-dg', 'dg-cg', 'taylor-hood'], default_value='dg-dg').tag(config=True)
    wetting_and_drying = Bool(False).tag(config=True)
    wetting_and_drying_alpha = FiredrakeScalarExpression(Constant(4.3)).tag(config=True)

    # Adaptation
    adapt_field = Unicode('all_avg', help="Adaptation field of interest.").tag(config=True)
    region_of_interest = List(default_value=[]).tag(config=True)

    def __init__(self, **kwargs):
        super(ShallowWaterOptions, self).__init__(**kwargs)
        self.degree_increase = 0
        self.timestepper = 'CrankNicolson'
        self.stabilisation = 'lax_friedrichs'
        self.stabilisation_parameter = Constant(1.0)

    def set_bathymetry(self, fs):
        raise NotImplementedError("Should be implemented in derived class.")

    def set_viscosity(self, fs):
        """Should be implemented in derived class."""
        if np.allclose(self.base_viscosity, 0.0):
            self.viscosity = None
        else:
            self.viscosity = Constant(self.base_viscosity)
        return self.viscosity

    def set_source_tracer(self, fs, solver_obj):
        """Should be implemented in derived class."""
        return self.depo, self.ero

    def set_diffusivity(self, fs):
        """Should be implemented in derived class."""
        if np.allclose(self.base_diffusivity, 0.0):
            self.diffusivity = None
        else:
            self.diffusivity = Constant(self.base_diffusivity)
        return self.diffusivity

    def set_inflow(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_coriolis(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_quadratic_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_manning_drag_coefficient(self, fs):
        """Should be implemented in derived class."""
        pass

    def set_boundary_surface(self):
        """Set the initial displacement of the boundary elevation."""
        self.elev_in = Constant(0.0)
        self.elev_out = Constant(0.0)

    def get_eta_tilde(self, solver_obj):
        bathymetry_displacement = solver_obj.eq_sw.depth.wd_bathymetry_displacement
        eta = solver_obj.fields.elev_2d
        self.eta_tilde.project(eta + bathymetry_displacement(eta))

    def get_export_func(self, solver_obj):
        def export_func():
            if self.wetting_and_drying:
                self.get_eta_tilde(solver_obj)
                self.eta_tilde_file.write(self.eta_tilde)
        return export_func

    def get_initial_depth(self, fs):
        """Compute the initial total water depth, using the bathymetry and initial elevation."""
        if not hasattr(self, 'initial_value'):
            self.set_initial_condition(fs)
        eta = self.initial_value.split()[1]
        V = FunctionSpace(eta.function_space().mesh(), 'CG', 1)
        eta_cg = Function(V).project(eta)
        if self.bathymetry is None:
            self.set_bathymetry(V)
        if self.wetting_and_drying:
            bathymetry_displacement = self.wd_dispacement_mc(eta)
            self.depth = interpolate(self.bathymetry + bathymetry_displacement + eta_cg, V)
        else:
            self.depth = interpolate(self.bathymetry + eta_cg, V)

        return self.depth
