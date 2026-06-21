import numpy as np
from numba import jit, njit, prange


@jit(nopython=True)
def norm_n(r):
    """Compute the Euclidean norm of a vector."""
    sum = 0
    for i in r:
        sum += i**2
    return np.sqrt(sum)


@njit(cache=True)
def compute_kinetic_energy(m_list, v):
    """
    Compute total kinetic energy = sum(½ * m[i] * |v[i]|^2)
    m : 1D array of masses (shape = (N,))
    v : 2D array of velocities (shape = (N, dim))
    """
    n = v.shape[0]
    total_ke = 0.0
    # loop over particles
    for i in prange(n):
        m = m_list[i][0]
        # compute squared norm of v[i]
        s = 0.0
        for j in range(v.shape[1]):
            sij = v[i, j]
            s += sij * sij
        # accumulate KE
        total_ke += 0.5 * m * s
    return total_ke


@njit(cache=True)
def LinForce(r1, r2, c, a):
    """Calculate linear force between two particles.

    F = -kx

    Parameters:
    r1, r2: Positions of the two particles
    c: Stiffness constant
    a: Equilibrium distance

    Returns:
    Linear force vector between particles
    """
    dr = r2 - r1
    abs_dr = np.linalg.norm(dr)
    return c * (dr / abs_dr) * (abs_dr - a)


@njit(cache=True)
def pair_calculation(pairlist, r, c_list, a_list):
    forces_t = np.zeros_like(r)
    pot_en = 0

    for l, (i, j) in enumerate(pairlist):
        a = a_list[l]
        c = c_list[l]
        A = r[i] - r[j]  # r[j] - r[i]
        # E = 1/2*kx^2
        el_en = 0.5 * c * ((norm_n(A) - a) ** 2)
        pot_en += el_en
        force = LinForce(r1=r[i], r2=r[j], c=c, a=a)
        # print(LinForce(r[i], r[j], c, a))
        forces_t[i] += force
        forces_t[j] -= force
    return forces_t, pot_en


@njit(cache=True)
def find_angle(v1, v2, axis):
    """Find the agnle in 3d space between two vectors in range [0, 2Pi]"""
    dot = np.dot(v1, v2)
    det = np.dot(axis, np.cross(v1, v2))
    angle = np.atan2(det, dot)
    angle = angle % (2 * np.pi)
    return angle


@njit(cache=True)
def hinge_calculation_ang(triplist, r, g_list, phi, a_list=None, large_dis=False):
    # Initialize the total potential energy to zero.
    pot_en = 0.0
    # Initialize a force array with the same shape as positions 'r'.
    forces = np.zeros_like(r)

    if triplist is None:
        return forces, pot_en

    # Loop over each set of indices defining a hinge (i, j, k, l)
    for n, (i, j, k, l) in enumerate(triplist):
        # Select the reference angle for this hinge
        phi1 = phi[n]
        # Select corresponding anular stiffness constant
        g = g_list[n]

        # Calculate the hinge vector between particles k and j.
        r_kj = r[k] - r[j]
        # Calculate the left vector from j to i.
        r_ij = r[i] - r[j]
        # Calculate the right vector from j to l.
        r_lj = r[l] - r[j]

        # Project the left vector (r_ij) onto the hinge vector (r_kj).
        r_ij_to_r_kj = (np.dot(r_ij, r_kj) / np.dot(r_kj, r_kj)) * r_kj
        # Project the right vector (r_lj) onto the hinge vector (r_kj).
        r_lj_to_r_kj = (np.dot(r_lj, r_kj) / np.dot(r_kj, r_kj)) * r_kj

        # Compute the projection points on the hinge line for particles i and l.
        r_p_i = r[j] + r_ij_to_r_kj
        r_p_l = r[j] + r_lj_to_r_kj
        # Calculate the lever arms (distances from the projection to the actual particle positions).
        lever_i = np.linalg.norm(r[i] - r_p_i)
        lever_l = np.linalg.norm(r[l] - r_p_l)

        if a_list is not None:
            L_char = np.mean(a_list)  # or some typical panel edge length

            lever_i = max(lever_i, 1e-3 * L_char)
            lever_l = max(lever_l, 1e-3 * L_char)

        # Compute the normal vector to the plane defined by particles i, j, and k.
        n_ijk = np.cross(r_ij, r_kj)
        norm_ijk = np.linalg.norm(n_ijk)
        if norm_ijk < 1e-10:
            # Panel almost flat/collinear here → skip or soften this hinge
            continue
        n_ijk /= norm_ijk

        # Compute the normal vector to the plane defined by particles l, j, and k.
        n_ljk = np.cross(r_lj, r_kj)
        norm_ljk = np.linalg.norm(n_ljk)
        if norm_ljk < 1e-10:
            continue
        n_ljk /= norm_ljk

        # Define the rotation axis along the negative hinge vector.
        axis = r_kj / np.linalg.norm(r_kj)
        # Calculate the current angle (phi2) between the two planes around the hinge axis.
        phi2 = find_angle(n_ijk, n_ljk, axis)
        # phi2_test = find_angle(n_ijk, n_ljk, -axis)
        # True angle deviation
        delta_raw = phi2 - phi1

        # Effective deviation for the torque
        if large_dis:
            delta_sign = np.sign(delta_raw)
            delta_eff = delta_sign * (abs(delta_raw) ** (4.0 / 3.0))
        else:
            delta_eff = delta_raw

        # Calculate the torque magnitude based on the bending constant 'g' and the angular deviation.
        torque_mag = g * delta_eff

        # Calculate the forces on particles i and l using the lever arms.
        f_i = -torque_mag / lever_i * n_ijk
        f_l = torque_mag / lever_l * n_ljk
        # Reaction forces at the hinge points for particles i and l.
        f_p_i = -f_i
        f_p_l = -f_l

        # Determine the fractional distance (alpha) along the hinge for particle i.
        alpha_i = np.dot(r_ij_to_r_kj, r_kj) / np.dot(r_kj, r_kj)
        # Determine the fractional distance (alpha) along the hinge for particle l.
        alpha_l = np.dot(r_lj_to_r_kj, r_kj) / np.dot(r_kj, r_kj)

        # Distribute the hinge forces for the particle connected to the hinge:
        # For particle i, distribute between j and k.
        f_ji = (1 - alpha_i) * f_p_i  # Force on j due to particle i.
        f_ki = alpha_i * f_p_i  # Force on k due to particle i.

        # For particle l, distribute between j and k.
        f_jl = (1 - alpha_l) * f_p_l  # Force on j due to particle l.
        f_kl = alpha_l * f_p_l  # Force on k due to particle l.

        # Sum the contributions to particle j and k from both sides of the hinge.
        f_j = f_ji + f_jl
        f_k = f_ki + f_kl

        # Update the force vector for each particle.
        forces[i] += f_i
        forces[j] += f_j
        forces[k] += f_k
        forces[l] += f_l

        # Compute the current potential energy contribution from the angular deviation.
        # Potential energy: small vs large displacement
        if large_dis:
            pot_en_current = (3.0 / 7.0) * g * (abs(delta_raw) ** (7.0 / 3.0))
        else:
            # Classic quadratic energy
            pot_en_current = 0.5 * g * (delta_raw**2)

        pot_en += pot_en_current

    # Return the updated forces on all particles and the total potential energy.
    return forces, pot_en


@njit(cache=True)
def angle_of_quadruplet(ids, r):
    i, j, k, l = ids
    # Calculate the hinge vector between particles k and j.
    r_kj = r[k] - r[j]
    # Calculate the left vector from j to i.
    r_ij = r[i] - r[j]
    # Calculate the right vector from j to l.
    r_lj = r[l] - r[j]
    # Compute the normal vector to the plane defined by particles i, j, and k.
    n_ijk = np.cross(r_ij, r_kj)
    n_ijk = n_ijk / np.linalg.norm(n_ijk)
    # Compute the normal vector to the plane defined by particles l, j, and k.
    n_ljk = np.cross(r_lj, r_kj)
    n_ljk = n_ljk / np.linalg.norm(n_ljk)
    # Define the rotation axis along the negative hinge vector.
    axis = r_kj / np.linalg.norm(r_kj)
    # Calculate the current angle (phi2) between the two planes around the hinge axis.
    angle = find_angle(n_ijk, n_ljk, axis)
    return angle


@njit(cache=True)
def neighbor_damping_force(v, ptr, neigh, c_node):
    # F_i = sum_j c_i (v_j - v_i)
    Fd = np.zeros_like(v)
    n_nodes = v.shape[0]

    for i in range(n_nodes):
        ci = c_node[i]
        for k in range(ptr[i], ptr[i + 1]):
            j = neigh[k]
            Fd[i] += ci * (v[j] - v[i])

    return Fd


@njit(cache=True)
def compute_contact_forces_3d(r, pairs, k_e=1e1, d0=1e-4):
    """d0: contact distance threshold (thickness of material), k_e: contact stiffness constant"""
    force_contact = np.zeros_like(r)
    pot_contact = 0.0

    reff = 1.5 * d0
    eps = 1e-12
    pi = np.pi

    N = r.shape[0]
    for e in range(pairs.shape[0]):
        i = pairs[e, 0]
        j = pairs[e, 1]

        A = r[i]
        B = r[j]
        E = B - A
        E2 = E[0] * E[0] + E[1] * E[1] + E[2] * E[2]
        if E2 < eps:
            continue

        xmin = min(A[0], B[0]) - reff
        xmax = max(A[0], B[0]) + reff
        ymin = min(A[1], B[1]) - reff
        ymax = max(A[1], B[1]) + reff
        zmin = min(A[2], B[2]) - reff
        zmax = max(A[2], B[2]) + reff

        for k in range(N):
            if k == i or k == j:
                continue

            Px = r[k, 0]
            Py = r[k, 1]
            Pz = r[k, 2]

            if (
                Px < xmin
                or Px > xmax
                or Py < ymin
                or Py > ymax
                or Pz < zmin
                or Pz > zmax
            ):
                continue

            APx = Px - A[0]
            APy = Py - A[1]
            APz = Pz - A[2]

            lam = (APx * E[0] + APy * E[1] + APz * E[2]) / E2

            if lam <= 0.0:
                dx = APx
                dy = APy
                dz = APz
                lam_c = 0.0
            elif lam >= 1.0:
                dx = Px - B[0]
                dy = Py - B[1]
                dz = Pz - B[2]
                lam_c = 1.0
            else:
                projx = lam * E[0]
                projy = lam * E[1]
                projz = lam * E[2]
                dx = APx - projx
                dy = APy - projy
                dz = APz - projz
                lam_c = lam

            d2 = dx * dx + dy * dy + dz * dz
            d = np.sqrt(d2) if d2 > eps else 0.0

            if d > d0:
                continue

            d_cl = d if d > eps else eps
            arg = (pi * d_cl) / (2.0 * d0)
            if arg >= (pi * 0.5 - 1e-9):
                arg = pi * 0.5 - 1e-9
            cotangent = 1.0 / np.tan(arg)

            force_mag = (
                -(k_e * pi)
                / (2.0 * d0)
                * ((-pi * d_cl) / (2.0 * d0) + (pi * 0.5) - cotangent)
            )

            if d > eps:
                nx = dx / d
                ny = dy / d
                nz = dz / d
            else:
                # fallback in 3D is not unique
                # choose any vector perpendicular to E
                invE = 1.0 / np.sqrt(E2)
                ex = E[0] * invE
                ey = E[1] * invE
                ez = E[2] * invE

                if abs(ex) < 0.9:
                    rx, ry, rz = 1.0, 0.0, 0.0
                else:
                    rx, ry, rz = 0.0, 1.0, 0.0

                nx = ey * rz - ez * ry
                ny = ez * rx - ex * rz
                nz = ex * ry - ey * rx
                nn = np.sqrt(nx * nx + ny * ny + nz * nz)
                if nn < eps:
                    continue
                nx /= nn
                ny /= nn
                nz /= nn

            fx = force_mag * nx
            fy = force_mag * ny
            fz = force_mag * nz

            theta = (pi * 0.5) - (pi * d_cl) / (2.0 * d0)
            cth = np.cos(theta)
            if cth < 1e-12:
                cth = 1e-12
            pot_contact += k_e * (np.log(1.0 / cth) - 0.5 * theta * theta)

            force_contact[k, 0] += fx
            force_contact[k, 1] += fy
            force_contact[k, 2] += fz

            wi = 1.0 - lam_c
            wj = lam_c
            force_contact[i, 0] -= wi * fx
            force_contact[i, 1] -= wi * fy
            force_contact[i, 2] -= wi * fz
            force_contact[j, 0] -= wj * fx
            force_contact[j, 1] -= wj * fy
            force_contact[j, 2] -= wj * fz

    return force_contact, pot_contact
