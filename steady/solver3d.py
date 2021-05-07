from thetis import *

from adapt_utils.steady.solver import AdaptiveSteadyProblem


__all__ = ["AdaptiveSteadyProblem3d"]


class AdaptiveSteadyProblem3d(AdaptiveSteadyProblem):
    """
    Problem class for time-independent three dimensional tracer transport problems on unstructured
    meshes. (Extruded meshes are not supported.)
    """
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

    def set_finite_elements(self):
        self.op.print_debug("SETUP: Creating finite elements...")
        p = self.op.degree
        family = self.op.family
        if family == 'cg-cg':
            assert p == 1
            u_element = VectorElement("Lagrange", tetrahedron, p+1)
            eta_element = FiniteElement("Lagrange", tetrahedron, p, variant='equispaced')
        elif family == 'dg-dg':
            u_element = VectorElement("DG", tetrahedron, p)
            eta_element = FiniteElement("DG", tetrahedron, p, variant='equispaced')
        elif family == 'dg-cg':
            assert p == 1
            u_element = VectorElement("DG", tetrahedron, p)
            eta_element = FiniteElement("Lagrange", tetrahedron, p+1, variant='equispaced')
        else:
            raise NotImplementedError("Cannot build order {:d} {:s} element".format(p, family))
        self.finite_element = u_element*eta_element

        if self.op.solve_tracer:
            p = self.op.degree_tracer
            family = self.op.tracer_family
            if family == 'cg':
                self.finite_element_tracer = FiniteElement("Lagrange", tetrahedron, p)
            elif family == 'dg':
                self.finite_element_tracer = FiniteElement("DG", tetrahedron, p)
            else:
                raise NotImplementedError("Cannot build order {:d} {:s} element".format(p, family))

    def create_forward_tracer_equation_step(self, i):
        from ..tracer.equation3d import TracerEquation3D, ConservativeTracerEquation3D

        assert i == 0
        op = self.tracer_options[i]
        conservative = op.use_tracer_conservative_form
        model = ConservativeTracerEquation3D if conservative else TracerEquation3D
        self.equations[i].tracer = model(
            self.Q[i],
            self.depth[i],
            self.tracer_options[i],
            self.fwd_solutions[i].split()[0],
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
            self.tracer_options[i],
            self.fwd_solutions[i].split()[0],
        )
        if op.use_limiter_for_tracers and self.Q[i].ufl_element().degree() > 0:
            self.tracer_limiters[i] = VertexBasedP1DGLimiter(self.Q[i])
        self.equations[i].adjoint_tracer.bnd_functions = self.boundary_conditions[i]['tracer']

    def create_tracer_error_estimator_step(self, i):
        from ..tracer.error_estimation3d import TracerGOErrorEstimator3D

        assert i == 0
        if self.tracer_options[i].use_tracer_conservative_form:
            raise NotImplementedError("Error estimation for conservative tracers not implemented.")
        else:
            estimator = TracerGOErrorEstimator3D
        self.error_estimators[i].tracer = estimator(
            self.Q[i],
            self.depth[i],
            self.options[i],
        )

    def _get_fields_for_tracer_timestepper(self, i):
        assert i == 0
        # u = Constant(as_vector(self.op.base_velocity))  # FIXME: Pyadjoint doesn't like this
        u = interpolate(as_vector(self.op.base_velocity), self.P1_vec[i])
        fields = AttrDict({
            'uv_3d': u,
            'elev_3d': Constant(0.0),
            'diffusivity_h': self.fields[i].horizontal_diffusivity,
            'source': self.fields[i].tracer_source_2d,
            'tracer_advective_velocity_factor': self.fields[i].tracer_advective_velocity_factor,
        })
        if self.stabilisation == 'supg':
            fields['supg_stabilisation'] = True
        return fields
