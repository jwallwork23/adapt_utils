from firedrake import *

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from adapt_utils.adapt.kernels import *
from adapt_utils.options import Options


__all__ = ["AdaptiveMesh"]


class AdaptiveMesh():
    """
    Wrapper which adds extra features to mesh.
    """
    def __init__(self, mesh, levels=0, op=Options()):
        """
        `AdaptMesh` object is initialised as the basis of a `MeshHierarchy`.
        """
        self.levels = levels
        self.hierarchy = MeshHierarchy(mesh, levels)
        self.mesh = self.hierarchy[0]
        self.dim = self.mesh.topological_dimension()
        assert self.dim in (2, 3)
        if levels > 0:
            self.refined_mesh = self.hierarchy[1]

        self.n = FacetNormal(self.mesh)
        if self.dim == 2:
            self.tangent = as_vector([-self.n[1], self.n[0]])  # Tangent vector
        elif self.dim == 3:
            raise NotImplementedError  # TODO: Get a tangent vector in 3D
        else:
            raise NotImplementedError
        self.facet_area = FacetArea(self.mesh)
        self.h = CellSize(self.mesh)
        self.jacobian_sign = sign(JacobianDeterminant(mesh))

    def copy(self):
        return AdaptiveMesh(Mesh(Function(self.mesh.coordinates)), levels=self.levels)

    def get_quality(self):
        """
        Compute the scaled Jacobian for each mesh element:
    ..  math::
            Q(K) = \frac{\det(J_K)}{\|\mathbf e_1\|\,\|\mathbf e2\|},

        where element :math:`K` is defined by edges :math:`\mathbf e_1` and :math:`\mathbf e_2`.

        NOTE that :math:`J_K = [\mathbf e_1, \mathbf e_2]`.
        """
        assert self.dim == 2
        P0 = FunctionSpace(self.mesh, "DG", 0)
        J = Jacobian(self.mesh)
        detJ = self.jacobian_sign*JacobianDeterminant(self.mesh)
        edge1 = as_vector([J[0, 0], J[1, 0]])
        edge2 = as_vector([J[0, 1], J[1, 1]])
        norm1 = sqrt(dot(edge1, edge1))
        norm2 = sqrt(dot(edge2, edge2))

        # TODO: This hack is probably insufficient. It was designed to ensure each element of a
        #       uniform mesh has the same quality.
        edge1 = conditional(le(abs(norm1-norm2), 1e-8), edge1, edge1-edge2)
        norm1 = sqrt(dot(edge1, edge1))

        self.scaled_jacobian = interpolate(detJ/(norm1*norm2), P0)
        return self.scaled_jacobian

    def plot_quality(self):
        """
        Plot scaled Jacobian using a discretised scale:
          * green   : high quality elements (over 75%);
          * yellow  : medium quality elements (50 - 75%);
          * blue    : low quality elements (0 - 50%);
          * magenta : inverted elements (quality < 0).
        """
        cmap = plt.get_cmap('viridis', 20)
        newcolours = cmap(np.linspace(0, 1, 40))
        magenta = np.array([1, 0, 1, 1])
        green = np.array([0, 1, 0, 1])
        yellow = np.array([1, 1, 0, 1])
        cyan = np.array([0, 1, 1, 1])
        newcolours[:20, :] = magenta
        newcolours[20:30, :] = cyan
        newcolours[30:35, :] = yellow
        newcolours[35:, :] = green
        newcmap = ListedColormap(newcolours)
        cax = plot(self.scaled_jacobian, vmin=-1, vmax=1, cmap=newcmap)
        plt.title("Scaled Jacobian")
        plt.savefig(os.path.join(self.op.di, 'scaled_jacobian.pdf'))

    def save_plex(self, filename):
        """
        Save mesh in DMPlex format.
        """
        viewer = PETSc.Viewer().createHDF5(filename, 'r')
        viewer(self.mesh._plex)

    def load_plex(self, filename):
        """
        Load mesh from DMPlex format. The `MeshHierarchy` is reinstated.
        """
        newplex = PETSc.DMPlex().create()
        newplex.createFromFile(filename)
        self.__init__(Mesh(newplex), levels=self.levels)

    def adapt(self, metric):  # TODO: Rename so that it's Pragmatic specific
        """
        Adapt mesh using a specified metric. The `MeshHierarchy` is reinstated.
        """
        self.__init__(adapt(self.mesh, metric), levels=self.levels)

    def get_edge_lengths(self):
        """
        For each element, find the lengths of associated edges, stored in a HDiv trace field.

        NOTE: The plus sign is arbitrary and could equally well be chosen as minus.
        """
        HDivTrace = FunctionSpace(self.mesh, "HDiv Trace", 0)
        v, u = TestFunction(HDivTrace), TrialFunction(HDivTrace)
        self.edge_lengths = Function(HDivTrace, name="Edge lengths")
        mass_term = v('+')*u('+')*dS + v*u*ds
        rhs = v('+')*self.facet_area*dS + v*self.facet_area*ds
        solve(mass_term == rhs, self.edge_lengths)

    def get_edge_vectors(self):
        """
        For each element, find associated edge vectors, stored in a HDiv trace field.

        NOTES:
          * The plus sign is arbitrary and could equally well be chosen as minus.
          * The sign of the returned vectors is arbitrary and could equally well take the minus sign.
        """
        HDivTrace_vec = VectorFunctionSpace(self.mesh, "HDiv Trace", 0)
        v, u = TestFunction(HDivTrace_vec), TrialFunction(HDivTrace_vec)
        self.edge_vectors = Function(HDivTrace_vec, name="Edge vectors")
        mass_term = inner(v('+'), u('+'))*dS + inner(v, u)*ds
        rhs = inner(v('+'), self.tangent('+')*self.facet_area)*dS
        rhs += inner(v, self.tangent*self.facet_area)*ds
        solve(mass_term == rhs, self.edge_vectors)

    def get_maximum_length_edge(self):
        """
        For each element, find the associated edge of maximum length.
        """
        self.get_edge_lengths()
        self.get_edge_vectors()
        P0_vec = VectorFunctionSpace(self.mesh, "DG", 0)
        self.maximum_length_edge = Function(P0_vec, name="Maximum length edge")
        par_loop(get_maximum_length_edge(self.dim), dx, {'edges': (self.edge_lengths, READ),
                                                         'vectors': (self.edge_vectors, READ),
                                                         'max_vector': (self.maximum_length_edge, RW)
                                                        })

    def get_cell_metric(self):  # FIXME: Something doesn't seem right
        """
        Compute cell metric associated with mesh.

        Based on code by Lawrence Mitchell.
        """
        P0_ten = TensorFunctionSpace(self.mesh, "DG", 0)
        J = interpolate(Jacobian(self.mesh), P0_ten)
        self.cell_metric = Function(P0_ten, name="Cell metric")
        kernel = eigen_kernel(singular_value_decomposition, self.dim)
        op2.par_loop(kernel, P0_ten.node_set, self.cell_metric.dat(op2.INC), J.dat(op2.READ))

    def get_cell_size(self, u, mode='nguyen'):
        """
        Measure of element size recommended in [Nguyen et al., 2009]: maximum edge length, projected
        onto the velocity field `u`.
        """
        P0 = FunctionSpace(self.mesh, "DG", 0)
        try:
            assert isinstance(u, Constant) or isinstance(u, Function)
        except AssertionError:
            raise ValueError("Velocity field should be either `Function` or `Constant`.")
        if mode == 'diameter':
            self.cell_size = interpolate(self.h, P0)
        elif mode == 'nguyen':
            self.get_maximum_length_edge()
            v = self.maximum_length_edge
            self.cell_size = interpolate(abs((u[0]*v[0] + u[1]*v[1]))/sqrt(dot(u, u)), P0)
        elif mode == 'cell_metric':
            self.get_cell_metric()
            self.cell_size = interpolate(dot(u, dot(self.cell_metric, u)), P0)
        else:
            raise ValueError("Element measure must be chosen from {'diameter', 'nguyen', 'cell_metric'}.")

    def subdomain_indicator(self, subdomain_id):
        """Creates a P0 indicator function relating with `subdomain_id`."""
        P0 = FunctionSpace(self.mesh, "DG", 0)
        return assemble(TestFunction(P0)*dx(subdomain_id))
