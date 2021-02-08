"""
The time-dependent shallow water system looks like

                            ------------------------- -----   -----
      ------------- -----   |                 |     | |   |   |   |
      | A00 | A01 | | U |   |  T + C + V + D  |  G  | | U |   | 0 |
A x = ------------- ----- = |                 |     | |   | = |   |  = b,
      | A10 | A11 | | H |   ------------------------- -----   -----
      ------------- -----   |        B        |  T  | | H |   | 0 |
                            ------------------------- -----   -----

where:
 * T - time derivative;
 * C - Coriolis;
 * V - viscosity;
 * D - quadratic drag;
 * G - gravity;
 * B - bathymetry.
"""

__all__ = ["lu_params", "fieldsplit_params", "iterative_tracer_params", "direct_tracer_params",
           "l2_projection_params", "ibp_params", "flux_params"]


# Default for time-dependent shallow water
# ========================================
#
# GMRES with a multiplicative fieldsplit preconditioner, i.e. block Gauss-Seidel:
#
#     ---------------- ------------ ----------------
#     | I |     0    | |   I  | 0 | | A00^{-1} | 0 |
# P = ---------------- ------------ ----------------.
#     | 0 | A11^{-1} | | -A10 | 0 | |    0     | I |
#     ---------------- ------------ ----------------

fieldsplit_params = {
    "ksp_type": "gmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "fieldsplit_U_2d": {
        "ksp_type": "preonly",
        "ksp_max_it": 10000,
        "ksp_rtol": 1.0e-05,
        "pc_type": "sor",
    },
    "fieldsplit_H_2d": {
        "ksp_type": "preonly",
        "ksp_max_it": 10000,
        "ksp_rtol": 1.0e-05,
        # "pc_type": "sor",
        "pc_type": "jacobi",
    },
}

# Default for steady-state shallow water
# ======================================
#
# Newton with line search; solve linear system exactly with LU factorisation.
# For details on MUMPS parameters, see
#     mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MATSOLVERMUMPS.html

lu_params = {
    'mat_type': 'aij',
    'snes_type': 'newtonls',
    'snes_rtol': 1.0e-08,
    # 'snes_rtol': 1.0e-04,
    'snes_max_it': 20,
    'snes_linesearch_type': 'bt',
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_type': 'preonly',
    'ksp_converged_reason': None,
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    # 'mat_mumps_icntl_14': 200,  # Percentage increase in the estimated working space
    'mat_mumps_icntl_24': 1,    # Detection of null pivot rows (0 or 1)
}

# Default for time-dependent tracer/sediment/Exner
# ================================================
#
# GMRES with SOR as a preconditioner.

iterative_tracer_params = {
    "ksp_type": "gmres",
    "pc_type": "sor",
}

# Default for steady-state tracer
# ===============================
#
# Direct application of LU factorisation.

direct_tracer_params = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

# Double L2 projection
# ====================

l2_projection_params = {
    'mat_type': 'aij',

    # Use stationary preconditioners in the Schur complement, to get away with applying
    # GMRES to the whole mixed system.
    'ksp_type': 'gmres',
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',

    # We want to eliminate H (field 1) to get an equation for g (field 0).
    'pc_fieldsplit_0_fields': '1',
    'pc_fieldsplit_1_fields': '0',

    # Use a diagonal approximation of the A00 block.
    'pc_fieldsplit_schur_precondition': 'selfp',

    # Use ILU to approximate the inverse of A00, without a KSP solver.
    'fieldsplit_0_pc_type': 'ilu',
    'fieldsplit_0_ksp_type': 'preonly',

    # Use GAMG to approximate the inverse of the Schur complement matrix.
    'fieldsplit_1_ksp_type': 'preonly',
    'fieldsplit_1_pc_type': 'gamg',
    'ksp_max_it': 20,
}

# Integration by parts
# ====================
#
# GMRES with restarts and SOR as a preconditioner.

ibp_params = {
    'ksp_type': 'gmres',
    'ksp_gmres_restart': 20,
    'ksp_rtol': 1.0e-05,
    'pc_type': 'sor',
}

# Fluxes
# ======
#
# Simply divide by the diagonal of the mass matrix.

flux_params = {
    'ksp_type': 'preonly',
    'pc_type': 'jacobi',
}
