from thetis import *


__all__ = ["TracerCallback"]


class TracerCallback(callback.AccumulatorCallback):
    """Integrates objective functional for advection diffusion problem."""
    name = 'advection objective functional'

    def __init__(self, solver_obj, parameters=None, **kwargs):
        """
        :arg solver_obj: Thetis solver object
        :arg parameters: class containing parameters, including time period of interest.
        :arg **kwargs: any additional keyword arguments, see DiagnosticCallback
        """
        if parameters is None:
            self.parameters = TracerOptions()
        else:
            self.parameters = parameters
        if self.parameters.solve_adjoint:
            from firedrake_adjoint import assemble
        else:
            from firedrake import assemble

        def objectiveAD():
            """
            :param solver_obj: FlowSolver2d object.
            :return: objective functional value for callbacks.
            """
            Q_2d = solver_obj.function_spaces.Q_2d
            ks = Function(Q_2d)
            iA = self.parameters.disk(Q_2d)
            t = solver_obj.simulation_time
            dt = solver_obj.options.timestep
            ks.assign(iA)
            kt = Constant(0.)

            # Slightly smooth transition
            if self.parameters.start_time - 0.5 * dt < t < self.parameters.start_time + 0.5 * dt:
                kt.assign(0.5)
            elif self.parameters.start_time + 0.5 * dt < t < self.parameters.end_time - 0.5 * dt:
                kt.assign(1.)
            elif self.parameters.end_time - 0.5 * dt < t < self.parameters.end_time + 0.5 * dt:
                kt.assign(0.5)
            else:
                kt.assign(0.)

            return assemble(kt * ks * solver_obj.fields.tracer_2d * dx)

        super(TracerCallback, self).__init__(objectiveAD, solver_obj, **kwargs)
