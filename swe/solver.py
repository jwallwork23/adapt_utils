from adapt_utils.adapt.solver import AdaptiveProblem


__all__ = ["AdaptiveShallowWaterProblem"]


# TODO: Delete outdated swe/solver and put this there
class AdaptiveShallowWaterProblem(AdaptiveProblem):
    """General solver object for adaptive shallow water problems with no tracer component."""
    def __init__(self, *args, **kwargs):
        super(AdaptiveShallowWaterProblem, self).__init__(*args, **kwargs)
        try:
            assert not self.op.solve_tracer
        except AssertionError:
            raise ValueError("This class is for problems with no tracer component.")
