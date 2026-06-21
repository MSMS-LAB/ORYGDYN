import numpy as np
from numba import njit, jit


@njit(cache=True)
def SHAKE_with_forces(r, constrlist, a, dt, m, max_iters=100, tolerance=1e-6):
    r = r.copy()
    r_old = r.copy()

    f_constr = np.zeros_like(r)

    # while True:
    for _ in range(max_iters):
        max_error = 0.0

        for n, (i, j) in enumerate(constrlist):
            rij = r[i] - r[j]
            rij_old = r_old[i] - r_old[j]
            error = np.dot(rij, rij) - a[n] ** 2
            max_error = max(max_error, abs(error) / a[n] ** 2)

            if abs(error) / a[n] ** 2 > tolerance:
                # inverse masses
                im_i = 1.0 / m[i]
                im_j = 1.0 / m[j]

                denom = 4 * dt**2 * (im_i + im_j) * np.dot(rij_old, rij)
                if np.abs(denom) > 1e-30:
                    L = error / denom

                    # position corrections (different for each mass)
                    corr_i = 2 * dt**2 * im_i * L * rij_old
                    corr_j = 2 * dt**2 * im_j * L * rij_old

                    r[i] -= corr_i
                    r[j] += corr_j

                    f_constr[i] -= 2.0 * L * rij_old
                    f_constr[j] += 2.0 * L * rij_old

        if max_error < tolerance:
            break

    return r, f_constr


@njit(cache=True)
def SHAKE(r, constrlist, a, dt, inv_m, max_iters=100, tolerance=1e-6, eps=1e-30):
    r = r.copy()
    r_old = r.copy()

    f_constr = np.zeros_like(r)

    # while True:
    for _ in range(max_iters):
        max_error = 0.0

        for n, (i, j) in enumerate(constrlist):
            rij = r[i] - r[j]
            rij_old = r_old[i] - r_old[j]

            error = np.dot(rij, rij) - a[n] ** 2
            rel_error = np.abs(error) / a[n] ** 2
            max_error = max(max_error, rel_error)

            if rel_error > tolerance:
                # inverse masses
                w = inv_m[i] + inv_m[j]

                # mass-weighted dot product: s^T W rij0  (same pattern as your RATTLE)
                sdMr = np.dot(rij * w, rij_old)

                denom = 4.0 * dt * dt * sdMr
                if np.abs(denom) > eps:
                    L = error / denom

                    dq = 2.0 * dt * dt * L * rij_old

                    # position corrections (different for each mass)
                    corr_i = dq * inv_m[i]
                    corr_j = dq * inv_m[j]
                    r[i] -= corr_i
                    r[j] += corr_j

                    f_constr[i] -= 2.0 * L * rij_old
                    f_constr[j] += 2.0 * L * rij_old

        if max_error < tolerance:
            break

    return r, f_constr


@njit(cache=True)
def rattle_position(
    r, v, forces, dt, inv_m, constrlist, a, tolerance=1e-6, max_iters=100, eps=1e-30
):
    """
    RATTLE position constraint step (similar to SHAKE but returns velocity updates).
    """
    r0 = r.copy()

    # half-step velocity
    q = v + 0.5 * forces * inv_m * dt
    # unconstrained predictor
    r_pred = r0 + dt * q
    r_new = r_pred.copy()

    f_constr = np.zeros_like(r)

    for _ in range(max_iters):
        max_error = 0.0

        for n, (i, j) in enumerate(constrlist):

            rij0 = r0[i] - r0[j]  # old separation
            s = r_new[i] - r_new[j]  # current separation

            error = np.dot(s, s) - a[n] ** 2
            rel = np.abs(error) / (a[n] ** 2)

            max_error = max(max_error, rel)

            if rel > tolerance:
                w = inv_m[i] + inv_m[j]
                sdMr = (
                    s[0] * w[0] * rij0[0]
                    + s[1] * w[1] * rij0[1]
                    + s[2] * w[2] * rij0[2]
                )
                denominator = 2.0 * dt * sdMr
                if np.abs(denominator) > eps:
                    g = error / denominator
                    dq = g * rij0

                    q[i] -= inv_m[i] * dq
                    q[j] += inv_m[j] * dq
                    r_new[i] = r0[i] + dt * q[i]
                    r_new[j] = r0[j] + dt * q[j]

                    f_constr[i] -= dq / dt
                    f_constr[j] += dq / dt

        if max_error < tolerance:
            break

    return r_new, q, f_constr


@njit(cache=True)
def rattle_velocity(
    r, q, forces, dt, inv_m, constrlist, tolerance=1e-6, max_iters=100, eps=1e-30
):
    """
    RATTLE velocity constraint step (enforces orthogonality condition).
    """

    # unconstrained full-step velocity
    v_pred = q + 0.5 * forces * inv_m * dt
    v_new = v_pred.copy()

    f_constr = np.zeros_like(r)

    for _ in range(max_iters):
        max_error = 0.0

        for i, j in constrlist:
            r_ij = r[i] - r[j]
            v_ij = v_new[i] - v_new[j]

            error = np.dot(v_ij, r_ij)
            rr = np.dot(r_ij, r_ij)
            rel = np.abs(error) / rr

            max_error = max(max_error, rel)

            if rel > tolerance:
                inv_m_n = inv_m[i] + inv_m[j]
                # im_i = 1.0 / m[i]
                # im_j = 1.0 / m[j]

                denominator = (
                    (r_ij[0] * r_ij[0]) * inv_m_n[0]
                    + (r_ij[1] * r_ij[1]) * inv_m_n[1]
                    + (r_ij[2] * r_ij[2]) * inv_m_n[2]
                )
                if denominator > eps:
                    k = error / denominator
                    dv = k * r_ij

                    v_new[i] -= inv_m[i] * dv
                    v_new[j] += inv_m[j] * dv
                    # Constrained forces from this correction
                    f_constr[i] -= dv / dt
                    f_constr[j] += dv / dt

        if max_error < tolerance:
            break

    return v_new, f_constr


def lincs_3d(x_ref, x_pred, invmass, constraints, target_lengths, nrec=5):
    """
    3D LINCS constraint algorithm

    Parameters:
        x_ref: Reference positions (N,3) - typically from previous time step
        x_pred: Predicted positions (N,3) - to be constrained (modified in-place)
        invmass: Inverse masses (N,)
        constraints: List/array of atom pairs (K,2)
        target_lengths: Target bond lengths (K,)
        nrec: Number of iteration steps for matrix solver
    """
    K = len(constraints)
    if K == 0:
        return x_pred

    # Convert constraints to numpy array if needed
    constraints = np.asarray(constraints)
    atom1 = constraints[:, 0]
    atom2 = constraints[:, 1]

    # Precompute connection topology and coefficients
    con, ncc, coef, Sdiag = precompute_connections_3d(constraints, invmass)
    cmax = max(ncc) if K > 0 else 0

    # Initialize LINCS variables
    B = np.zeros((K, 3))  # Bond direction vectors
    rhs = np.zeros((2, K))  # Right-hand sides
    sol = np.zeros(K)  # Solution vector

    # 1. Compute normalized bond direction vectors
    compute_bond_vectors(B, x_ref, atom1, atom2)

    # 2. Build constraint coupling matrix A (sparse representation)
    A = build_coupling_matrix(K, cmax, con, ncc, coef, B)

    # First correction step
    compute_rhs_3d(rhs[0], sol, x_pred, atom1, atom2, B, target_lengths, Sdiag)
    solve_3d(x_pred, invmass, K, nrec, atom1, atom2, ncc, con, Sdiag, B, A, rhs, sol)

    # Rotational correction step
    compute_rotational_rhs_3d(rhs[0], sol, x_pred, atom1, atom2, target_lengths, Sdiag)
    solve_3d(x_pred, invmass, K, nrec, atom1, atom2, ncc, con, Sdiag, B, A, rhs, sol)

    return x_pred


def precompute_connections_3d(constraints, invmass):
    """Find connected constraints and precompute coefficients"""
    K = len(constraints)
    constraints = np.asarray(constraints)

    # Find connected constraints (those sharing atoms)
    ncc = np.zeros(K, dtype=int)
    con = []

    for i in range(K):
        connected = []
        # Find all constraints that share atoms with constraint i
        mask = np.any((constraints[:, None] == constraints[i]), axis=2).any(axis=1)
        mask[i] = False  # Exclude self
        connected = np.where(mask)[0].tolist()
        ncc[i] = len(connected)
        con.append(connected)

    cmax = max(ncc) if K > 0 else 0
    con_array = np.full((K, cmax), -1, dtype=int)
    for i in range(K):
        con_array[i, : ncc[i]] = con[i]

    # Precompute coefficients and Sdiag
    Sdiag = np.zeros(K)
    coef = np.zeros((K, cmax))

    for i in range(K):
        a1, a2 = constraints[i]
        Sdiag[i] = 1.0 / np.sqrt(invmass[a1] + invmass[a2])

        for n in range(ncc[i]):
            j = con[i][n]
            # Find shared atom
            shared = np.intersect1d(constraints[i], constraints[j])
            c = shared[0]  # There should be exactly one shared atom

            # Determine sign based on atom ordering
            i1, i2 = (
                np.where(constraints[i] == c)[0][0],
                np.where(constraints[j] == c)[0][0],
            )
            sign = -1 if (i1 == i2) else 1

            Sj = 1.0 / np.sqrt(invmass[constraints[j][0]] + invmass[constraints[j][1]])
            coef[i, n] = sign * invmass[c] * Sdiag[i] * Sj

    return con_array, ncc, coef, Sdiag


@jit(nopython=True)
def compute_bond_vectors(B, x_ref, atom1, atom2):
    """Compute normalized bond direction vectors"""
    for i in range(len(B)):
        a1, a2 = atom1[i], atom2[i]
        vec = x_ref[a1] - x_ref[a2]
        norm = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        B[i, 0] = vec[0] / norm
        B[i, 1] = vec[1] / norm
        B[i, 2] = vec[2] / norm


@jit(nopython=True)
def build_coupling_matrix(K, cmax, con, ncc, coef, B):
    """Build the constraint coupling matrix A"""
    A = np.zeros((K, cmax))
    for i in range(K):
        for n in range(ncc[i]):
            k = con[i, n]
            A[i, n] = coef[i, n] * (
                B[i, 0] * B[k, 0] + B[i, 1] * B[k, 1] + B[i, 2] * B[k, 2]
            )
    return A


@jit(nopython=True)
def compute_rhs_3d(rhs, sol, x_pred, atom1, atom2, B, lengths, Sdiag):
    """Compute right-hand side for the first correction step"""
    for i in range(len(rhs)):
        a1, a2 = atom1[i], atom2[i]
        dx = x_pred[a1] - x_pred[a2]
        rhs[i] = Sdiag[i] * (
            B[i, 0] * dx[0] + B[i, 1] * dx[1] + B[i, 2] * dx[2] - lengths[i]
        )
        sol[i] = rhs[i]


@jit(nopython=True)
def compute_rotational_rhs_3d(rhs, sol, x_pred, atom1, atom2, lengths, Sdiag):
    """Compute right-hand side for rotational correction"""
    for i in range(len(rhs)):
        a1, a2 = atom1[i], atom2[i]
        dx = x_pred[a1] - x_pred[a2]
        current_sq = dx[0] ** 2 + dx[1] ** 2 + dx[2] ** 2
        p = np.sqrt(2 * lengths[i] ** 2 - current_sq)
        rhs[i] = Sdiag[i] * (lengths[i] - p)
        sol[i] = rhs[i]


@jit(nopython=True)
def solve_3d(x_pred, invmass, K, nrec, atom1, atom2, ncc, con, Sdiag, B, A, rhs, sol):
    """Iterative matrix solver for LINCS"""
    w = 2  # Toggle between 1 and 2
    for rec in range(nrec):
        w = 3 - w
        for i in range(K):
            rhs[w - 1, i] = 0.0
            for n in range(ncc[i]):
                k = con[i, n]
                rhs[w - 1, i] += A[i, n] * rhs[3 - w - 1, k]
            sol[i] += rhs[w - 1, i]

    # Apply final correction to positions
    for i in range(K):
        a1, a2 = atom1[i], atom2[i]
        factor = Sdiag[i] * sol[i]
        x_pred[a1, 0] -= invmass[a1] * B[i, 0] * factor
        x_pred[a1, 1] -= invmass[a1] * B[i, 1] * factor
        x_pred[a1, 2] -= invmass[a1] * B[i, 2] * factor
        x_pred[a2, 0] += invmass[a2] * B[i, 0] * factor
        x_pred[a2, 1] += invmass[a2] * B[i, 1] * factor
        x_pred[a2, 2] += invmass[a2] * B[i, 2] * factor
