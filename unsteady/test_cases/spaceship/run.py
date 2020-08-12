from adapt_utils.unsteady.swe.turbine.solver import AdaptiveTurbineProblem
from adapt_utils.unsteady.test_cases.spaceship.options import SpaceshipOptions


solver_parameters = {
    "shallow_water": {
        # "mat_type": "aij",
        # "snes_monitor": None,
        "snes_converged_reason": None,
        "ksp_type": "gmres",
        # "ksp_type": "preonly",
        # "ksp_monitor": None,
        "ksp_converged_reason": None,
        # "pc_type": "fieldsplit",
        # "pc_type": "lu",
        # "pc_factor_mat_solver_type": "mumps",
        "pc_fieldsplit_type": "multiplicative",
        "fieldsplit_U_2d": {
            "ksp_type": "preonly",
            "ksp_max_it": 10000,
            "ksp_rtol": 1.0e-05,
            "pc_type": "sor",
            # "ksp_view": None,
            # "ksp_converged_reason": None,
        },
        "fieldsplit_H_2d": {
            "ksp_type": "preonly",
            "ksp_max_it": 10000,
            "ksp_rtol": 1.0e-05,
            # "pc_type": "sor",
            "pc_type": "jacobi",
            # "ksp_view": None,
            # "ksp_converged_reason": None,
        },
    },
}


kwargs = {
    'solver_parameters': solver_parameters,

    # Model
    'stabilisation': 'lax_friedrichs',
    # 'stabilisation': None,
    'family': 'dg-cg',

    # I/O
    'plot_pvd': True,
}

op = SpaceshipOptions()
op.update(kwargs)
tp = AdaptiveTurbineProblem(op)
tp.solve_forward()
