"""
Definition of optimization and utility functions using the CasADi framework.
"""

import casadi
import numpy as np
from typing import Any, Optional



def powerm(v: casadi.MX | casadi.SX, A: casadi.MX | casadi.SX, niter: int) -> tuple[casadi.MX | casadi.SX, casadi.MX | casadi.SX]: 
    """POWER ITERATION METHOD (PIM) for finding dominant eigenvalues/eigenvectors."""
    for _ in range(niter):
        v = A @ v
        v = v / casadi.norm_2(v)
    l = v.T @ A @ v
    return v, l


def def_powerm(a: casadi.MX | casadi.SX, niter: Optional[int] = None) -> tuple[casadi.MX | casadi.SX, casadi.MX | casadi.SX]: 
    """
    DEFLATION VERSION OF PIM FOR 3X3 MATRIX. 
    Calculates all eigenvalues from greater to smaller along with eigenvectors.
    """
    v0 = casadi.DM([1, 1, 1])
    if niter is None:
        niter = 40

    v1, l1 = powerm(v0, a, niter)

    v0 = casadi.DM([0, 1, 1])
    a2 = a - l1 * (v1 @ v1.T)
    v2, l2 = powerm(v0, a2, niter)

    v0 = casadi.DM([0, 0, 1])
    a3 = a2 - l2 * (v2 @ v2.T)
    v3, l3 = powerm(v0, a3, niter)

    D = casadi.vertcat(l1, l2, l3)
    V = casadi.horzcat(v1, v2, v3)
    return V, D


# Global Symbolic CasADi Setup
sym_matrix = casadi.SX.sym('M', 3, 3)
eigenvalues = casadi.eig_symbolic(sym_matrix)

f_eigen = casadi.Function('eigen', [sym_matrix], [eigenvalues])

_, eigenvalues_p = def_powerm(sym_matrix, niter=40)
f_powerm = casadi.Function('powerm', [sym_matrix], [eigenvalues_p])


def minimum(eigen: casadi.MX | casadi.SX, n: int = 3) -> casadi.MX | casadi.SX:
    """Find the minimum eigenvalue tracking via symbolic condition checks."""
    min_val = 1e16
    for i in range(n):
        min_val = casadi.if_else(eigen[i] <= min_val, eigen[i], min_val)
    return min_val


def SN(eigen: casadi.MX | casadi.SX, p: int = 3) -> casadi.MX | casadi.SX:
    """SHATTER NORM (SN) measurement criterion evaluation."""
    l1 = (eigen[0]) ** (-p)
    l2 = (eigen[1]) ** (-p)
    l3 = (eigen[2]) ** (-p)
    return (l1 + l2 + l3) ** (-1 / p)


def det(A: Any) -> Any:
    """DETERMINANT UP TO 3x3 MATRIX (Supports numerical variables and CasADi symbols)."""
    if np.isscalar(A):
        return A
    
    # Track spatial limits safely across CasADi or standard numpy matrices
    num_rows = A.shape[0] if hasattr(A, 'shape') else len(A)
    
    if num_rows == 1:
        return A[0, 0]
    elif num_rows == 2:
        return A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    elif num_rows == 3:
        return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
                - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))
    else:
        raise NotImplementedError("Supported dimensions: 1x1, 2x2, 3x3")


def casadi_skew(v: casadi.MX | casadi.SX) -> casadi.MX | casadi.SX:
    """Generate symbolic cross-product matrix skew transformations."""
    return casadi.vertcat(
        casadi.horzcat(0, -v[2], v[1]),
        casadi.horzcat(v[2], 0, -v[0]),
        casadi.horzcat(-v[1], v[0], 0)
    )
def skew(v: np.ndarray) -> np.ndarray:
    """Generate numerical skew-symmetric matrix."""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def dyn2reg(par_DYN: casadi.MX | casadi.SX, par_per_link: int = 10) -> casadi.MX | casadi.SX:
    """
    Symbolic conversion from standard dynamic parameters to frame origin regressor parameters.
    Fixes the parallel-axis transformation mapping formula: I_origin = I_barycentric - m * [c_skew]^2
    """
    par_REG: list[casadi.MX | casadi.SX] = []
    num_links = par_DYN.shape[0] // par_per_link
    
    for i in range(num_links):
        p_dyn = par_DYN[par_per_link * i : par_per_link * (i + 1)]
        mass = p_dyn[0]
        mCoM = mass * p_dyn[1:4]

        # Shift operations tracking parallel axis logic: I_O = I_G - m * [c_skew]^2
        S = casadi_skew(mCoM)
        I_tmp = casadi.mtimes(S.T, S) / mass

        I_dyn = p_dyn[4:10]
        I_tmp_v = casadi.vertcat(
            I_tmp[0, 0], I_tmp[0, 1], I_tmp[0, 2],
                         I_tmp[1, 1], I_tmp[1, 2],
                                      I_tmp[2, 2]
        )

        # Swapped to exact structural match for spatial mechanics consistency
        I = I_dyn + I_tmp_v
        par_REG.append(casadi.vertcat(mass, mCoM, I))

    return casadi.vertcat(*par_REG)


def reg2dyn(par_REG: casadi.MX | casadi.SX, par_per_link: int = 10) -> casadi.MX | casadi.SX:
    """Symbolic conversion from regressor parameters to standard dynamic parameters."""
    par_DYN: list[casadi.MX | casadi.SX] = []
    num_links = par_REG.shape[0] // par_per_link

    for i in range(num_links):
        p_reg = par_REG[par_per_link * i : par_per_link * (i + 1)]
        mass = p_reg[0]
        CoM = p_reg[1:4] / mass

        # Reverse Parallel Axis Theorem to move frame to Center of Mass
        S = casadi_skew(CoM)
        I_tmp = mass * casadi.mtimes(S.T, S)

        I_reg = p_reg[4:10]
        I_tmp_v = casadi.vertcat(
            I_tmp[0, 0], I_tmp[0, 1], I_tmp[0, 2],
                         I_tmp[1, 1], I_tmp[1, 2],
                                      I_tmp[2, 2]
        )

        I = I_reg - I_tmp_v
        par_DYN.append(casadi.vertcat(mass, CoM, I))

    return casadi.vertcat(*par_DYN)