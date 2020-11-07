# from thetis import *

# import numpy as np
# import os

from adapt_utils.steady.solver import AdaptiveSteadyProblem


__all__ = ["AdaptiveSteadyProblem3d"]


class AdaptiveSteadyProblem3d(AdaptiveSteadyProblem):
    # TODO: doc
    def __init__(self, op, **kwargs):
        super(AdaptiveSteadyProblem3d, self).__init__(op, **kwargs)
        if self.mesh.topological_dimension() != 3:
            raise ValueError("Mesh must be three-dimensional!")
        msg = "Only passive tracers are implemented in the 3D case."
        assert not op.solve_swe, msg
        assert op.solve_tracer, msg
        assert not op.solve_sediment, msg
        assert not op.solve_exner, msg
        if op.tracer_family != 'cg':
            raise NotImplementedError("Only CG has been considered for the 3D case.")
        if op.stabilisation is not None:
            assert op.stabilisation in ('su', 'supg')

    def create_forward_tracer_equation_step(self, i):
        from ..tracer.equation3d import TracerEquation3D, ConservativeTracerEquation3D

        assert i == 0
        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = ConservativeTracerEquation3D if conservative else TracerEquation3D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            anisotropic=op.anisotropic_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_adjoint_tracer_equation_step(self, i):
        from ..tracer.equation3d import TracerEquation3D, ConservativeTracerEquation3D

        assert i == 0
        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = TracerEquation3D if conservative else ConservativeTracerEquation3D
        self.equations[i].adjoint_tracer = model(
            self.Q[i],
            self.depth[i],
            anisotropic=op.anisotropic_stabilisation,
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].adjoint_tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_forward_tracer_error_estimator_step(self, i):
        from ..tracer.error_estimation3d import TracerGOErrorEstimator3D

        assert i == 0
        if self.tracer_options[i].use_tracer_conservative_form:
            raise NotImplementedError("Error estimation for conservative tracers not implemented.")
        else:
            estimator = TracerGOErrorEstimator3D
        self.error_estimators[i].tracer = estimator(
            self.Q[i],
            self.depth[i],
            use_lax_friedrichs=self.tracer_options[i].use_lax_friedrichs_tracer,
            sipg_parameter=self.tracer_options[i].sipg_parameter,
            anisotropic=self.tracer_options[i].anisotropic_stabilisation,
        )

    def _get_fields_for_tracer_timestepper(self, i):
        assert i == 0
        raise NotImplementedError  # TODO

    def create_forward_tracer_timestepper(self, i, integrator):
        assert i == 0
        raise NotImplementedError  # TODO

    def create_adjoint_tracer_timestepper(self, i, integrator):
        assert i == 0
        raise NotImplementedError  # TODO

    def recover_hessian_metric(self, adjoint=False, **kwargs):
        raise NotImplementedError  # TODO

    def get_weighted_gradient_metric(self, adjoint=False, source=True):
        raise NotImplementedError  # TODO
