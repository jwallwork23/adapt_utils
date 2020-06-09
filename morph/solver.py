from thetis import *

from adapt_utils.adapt.solver import AdaptiveProblem


__all__ = ["AdaptiveMorphologicalProblem"]


class AdaptiveMorphologicalProblem(AdaptiveProblem):
    # TODO: doc

    def set_fields(self):
        super(AdaptiveMorphologicalProblem, self).set_fields()
        self.setup_suspended()
        self.setup_bedload()

    def setup_suspended(self):
        for i in range(self.num_meshes):
            self._setup_suspended_step(i)

    def setup_bedload(self):
        for i in range(self.num_meshes):
            self._setup_bedload_step(i)

    def _setup_suspended_step(self, i):
        op = self.op
        P1 = self.P1[i]
        P1DG = self.P1DG[i]
        P1_vec = self.P1_vec[i]
        P1DG_vec = self.P1DG_vec[i]
        raise NotImplementedError  # TODO: Move over from swe/morphological_options

    def _setup_bedload_step(self, i):
        op = self.op
        P1 = self.P1[i]
        raise NotImplementedError  # TODO: Move over from swe/morphological_options

    def update_key_hydro(self):
        raise NotImplementedError  # TODO: Move over from swe/morphological_options

    def update_suspended(self):
        raise NotImplementedError  # TODO: Move over from swe/morphological_options

    def update_bedload(self):
        raise NotImplementedError  # TODO: Move over from swe/morphological_options
