from firedrake import *


__all__ = ["index_string", "subdomain_indicator"]


def index_string(index):
    """
    :arg index: integer form of index.
    :return: five-digit string form of index.
    """
    return (5 - len(str(index))) * '0' + str(index)


def subdomain_indicator(mesh, subdomain_id):
    """
    Creates a P0 indicator function relating with `subdomain_id`.
    """
    return assemble(TestFunction(FunctionSpace(mesh, "DG", 0)) * dx(subdomain_id))
