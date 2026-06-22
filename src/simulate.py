import numpy as np
from numba import njit

from src.force_calculation import (
    compute_kinetic_energy,
    pair_calculation,
    hinge_calculation_ang,
    neighbor_damping_force,
    angle_of_quadruplet,
    compute_contact_forces_3d,
)

from src.constrained_methods import SHAKE, rattle_position, rattle_velocity
from src.utils import build_neighbors, build_inv_m_mask


@njit(cache=True)
def run_standard(
    Nt,
    pairlist,
    quadruplets_bend,
    quadruplets_fold,
    fixed_ids,
    roller_fixed_ids_x,
    roller_fixed_ids_y,
    roller_fixed_ids_z,
    external_force_ids,
    force_vector,
    r,
    r_stress_free,
    load_factor,
    c_list,
    a_list,
    bend_stiffness_list,
    fold_stiffness_list,
    viscous_per_node,
    dt,
    m_list,
    v,
    step_to_save_results,
):
    history = np.zeros(
        (int(Nt / step_to_save_results) + 1, len(r), r.shape[-1])
    )  # Store history of positions
    force_history = np.zeros(
        (int(Nt / step_to_save_results) + 1, len(r), r.shape[-1])
    )  # Store history of forces
    hist_id = 0  # Initialize history index
    Total_K = np.zeros(Nt)  # Kinetic energy
    Total_P_lin = np.zeros(Nt)  # Potential of linear connections energy
    Total_P_fold = np.zeros(Nt)  # Potential of folding springs energy
    Total_P_bend = np.zeros(Nt)  # Potential of bending springs energy
    # Define stress-free configuration if not provided
    if r_stress_free is None:
        r_stress_free = r.copy()
    # Store reference positions for supports
    r0 = r.copy()
    # to accumulate external work
    r_prev = r.copy()
    # initialize force vector
    forces_total = np.zeros_like(r)
    # Build neighbors list for damping calculation
    ptr, neigh = build_neighbors(pairlist, n_nodes=r.shape[0])

    # If folding interactions are provided, calculate the initial angles.
    if quadruplets_fold is not None:
        fold_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_fold
            ]
        )
    else:
        fold_angles = None
    # If bending interactions are provided, calculate the initial angles.
    if quadruplets_bend is not None:
        bend_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_bend
            ]
        )
    else:
        bend_angles = None

    for t in range(Nt):
        # Update velocities using Velocity Verlet integration at time 1/2 dt
        v += 0.5 * forces_total / m_list * dt

        # Update positions using Velocity Verlet integration
        r += v * dt

        # Calculate forces and potential energy due to linear springs between pairs
        lin_spring_force, lin_spring_pot_en = pair_calculation(
            pairlist=pairlist, r=r, c_list=c_list, a_list=a_list
        )
        if fold_angles is not None:
            # Calculate forces and potential energy due to folding springs
            folding_force, folding_pot_en = hinge_calculation_ang(
                triplist=quadruplets_fold,
                r=r,
                g_list=fold_stiffness_list,
                phi=fold_angles,
            )
        else:
            folding_force = np.zeros_like(r)
            folding_pot_en = 0.0

        if bend_angles is not None:
            # Calculate forces and potential energy due to bending springs
            bending_force, bending_pot_en = hinge_calculation_ang(
                triplist=quadruplets_bend,
                r=r,
                g_list=bend_stiffness_list,
                phi=bend_angles,
                large_dis=False,
            )
        else:
            bending_force = np.zeros_like(r)
            bending_pot_en = 0.0

        # Total forces
        forces_total = lin_spring_force + folding_force + bending_force
        # Damping forces
        damping_force = neighbor_damping_force(v, ptr, neigh, viscous_per_node)
        # Update total forces
        forces_total += damping_force

        # Apply external forces
        # if t == 0:
        if external_force_ids is not None and len(external_force_ids) > 0:
            if force_vector is not None:
                if load_factor is not None:
                    load_factor = (t + 1) / Nt
                    forces_total[external_force_ids] += load_factor * force_vector
                else:
                    forces_total[external_force_ids] += force_vector

        # Update velocities using Velocity Verlet integration
        v += 0.5 * forces_total / m_list * dt

        # # Summetry walls
        # apply_symmetry_wall(r, v, r0, axis="x", coord=0.0)
        # apply_symmetry_wall(r, v, r0, axis="x", coord=r0[:, 0].max())
        # # # Keep particles in the initial domain
        # enforce_wall(r=r, v=v, axis="x", coord=0.0, side="greater")
        # enforce_wall(r=r, v=v, axis="x", coord=r0[:, 0].max(), side="less")
        # enforce_wall(r=r, v=v, axis="y", coord=-1.0, side="greater")
        # enforce_wall(r=r, v=v, axis="y", coord=r0[:, 1].max(), side="less")

        # Apply boundary conditions
        if fixed_ids is not None and len(fixed_ids) > 0:
            for idx in fixed_ids:
                r[idx, :] = r0[idx, :]
                v[idx, :] = 0.0

        if roller_fixed_ids_x is not None and len(roller_fixed_ids_x) > 0:
            for idx in roller_fixed_ids_x:
                r[idx, 0] = r0[idx, 0]
                v[idx, 0] = 0.0

        if roller_fixed_ids_y is not None and len(roller_fixed_ids_y) > 0:
            for idx in roller_fixed_ids_y:
                r[idx, 1] = r0[idx, 1]
                v[idx, 1] = 0.0

        if roller_fixed_ids_z is not None and len(roller_fixed_ids_z) > 0:
            for idx in roller_fixed_ids_z:
                r[idx, 2] = r0[idx, 2]
                v[idx, 2] = 0.0

        # Store positions in history
        if t % step_to_save_results == 0:
            history[hist_id] = r.copy()
            force_history[hist_id] = forces_total.copy()
            hist_id += 1

        # Store energies
        Total_K[t] = compute_kinetic_energy(m_list=m_list, v=v)
        Total_P_lin[t] = lin_spring_pot_en
        Total_P_fold[t] = folding_pot_en
        Total_P_bend[t] = bending_pot_en

        # update r_prev AFTER work calc
        r_prev[:] = r

    # Trim history array if necessary
    if hist_id < len(history):
        history = history[:hist_id]
        force_history = force_history[:hist_id]

    return history, force_history, Total_K, Total_P_lin, Total_P_fold, Total_P_bend


@njit(cache=True)
def run_shake(
    Nt,
    pairlist,
    quadruplets_bend,
    quadruplets_fold,
    fixed_ids,
    roller_fixed_ids_x,
    roller_fixed_ids_y,
    roller_fixed_ids_z,
    external_force_ids,
    force_vector,
    r,
    r_stress_free,
    load_factor,
    a_list,
    bend_stiffness_list,
    fold_stiffness_list,
    viscous_per_node,
    dt,
    m_list,
    v,
    step_to_save_results,
):
    history = np.zeros(
        (int(Nt / step_to_save_results) + 2, len(r), r.shape[-1])
    )  # Store history of positions
    force_history = np.zeros(
        (int(Nt / step_to_save_results) + 2, len(r), r.shape[-1])
    )  # Store history of forces
    hist_id = 0  # Initialize history index
    Total_K = np.zeros(Nt)  # Kinetic energy
    Total_P_lin = np.zeros(Nt)  # Potential of linear connections energy
    Total_P_fold = np.zeros(Nt)  # Potential of folding springs energy
    Total_P_bend = np.zeros(Nt)  # Potential of bending springs energy
    # Define stress-free configuration if not provided
    if r_stress_free is None:
        r_stress_free = r.copy()
    # to accumulate external work
    r_prev = r.copy()

    # Inverse mass mask for constraints
    inv_m = build_inv_m_mask(
        m_list=m_list,
        fixed_ids=fixed_ids,
        roller_x=roller_fixed_ids_x,
        roller_y=roller_fixed_ids_y,
        roller_z=roller_fixed_ids_z,
    )

    # initialize force vector
    forces_total = np.zeros_like(r)
    # Initial constraint force
    f_constr = np.zeros_like(r)
    # Build neighbors list for damping calculation
    ptr, neigh = build_neighbors(pairlist, n_nodes=r.shape[0])

    # If folding interactions are provided, calculate the initial angles.
    if quadruplets_fold is not None:
        fold_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_fold
            ]
        )
    else:
        fold_angles = None
    # If bending interactions are provided, calculate the initial angles.
    if quadruplets_bend is not None:
        bend_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_bend
            ]
        )
    else:
        bend_angles = None

    for t in range(Nt):

        # Store positions in history
        if t % step_to_save_results == 0:
            history[hist_id] = r.copy()
            force_history[hist_id] = forces_total.copy() + f_constr.copy()
            hist_id += 1

        if fold_angles is not None:
            # Calculate forces and potential energy due to folding springs
            folding_force, folding_pot_en = hinge_calculation_ang(
                triplist=quadruplets_fold,
                r=r,
                g_list=fold_stiffness_list,
                phi=fold_angles,
            )
        else:
            folding_force = np.zeros_like(r)
            folding_pot_en = 0.0

        if bend_angles is not None:
            # Calculate forces and potential energy due to bending springs
            bending_force, bending_pot_en = hinge_calculation_ang(
                triplist=quadruplets_bend,
                r=r,
                g_list=bend_stiffness_list,
                phi=bend_angles,
                large_dis=False,
            )
        else:
            bending_force = np.zeros_like(r)
            bending_pot_en = 0.0

        # Total forces
        forces_total = folding_force + bending_force
        # Damping forces
        damping_force = neighbor_damping_force(v, ptr, neigh, viscous_per_node)
        # Update total forces
        forces_total += damping_force

        # Apply external forces
        if external_force_ids is not None and len(external_force_ids) > 0:
            if force_vector is not None:
                if load_factor is not None:
                    load_factor = (t + 1) / Nt
                    forces_total[external_force_ids] += load_factor * force_vector
                else:
                    forces_total[external_force_ids] += force_vector

        # Position Verlet update
        r_next = 2.0 * r - r_prev + (forces_total * inv_m) * (dt**2)

        # Apply SHAKE constraints iteratively
        r_next, f_constr = SHAKE(
            r=r_next,
            constrlist=pairlist,
            a=a_list,
            dt=dt,
            inv_m=inv_m,
        )

        # Update velocity BEFORE boundary conditions
        v_next = (r_next - r_prev) / (2.0 * dt)

        # Update state for next iteration
        r_prev = r.copy()
        r = r_next.copy()
        v = v_next.copy()

        # Calculate and store energies
        kinetic_energy = compute_kinetic_energy(m_list=m_list, v=v)
        Total_K[t] = kinetic_energy
        Total_P_fold[t] = folding_pot_en
        Total_P_bend[t] = bending_pot_en

    # Store positions in history
    if t % step_to_save_results == 0:
        history[hist_id] = r.copy()
        force_history[hist_id] = forces_total.copy() + f_constr.copy()
        hist_id += 1
    # Trim history array if necessary
    if hist_id < len(history):
        history = history[:hist_id]
        force_history = force_history[:hist_id]

    return history, force_history, Total_K, Total_P_lin, Total_P_fold, Total_P_bend


@njit(cache=True)
def run_rattle(
    Nt,
    pairlist,
    quadruplets_bend,
    quadruplets_fold,
    fixed_ids,
    roller_fixed_ids_x,
    roller_fixed_ids_y,
    roller_fixed_ids_z,
    external_force_ids,
    force_vector,
    r,
    r_stress_free,
    load_factor,
    a_list,
    bend_stiffness_list,
    fold_stiffness_list,
    viscous_per_node,
    dt,
    m_list,
    v,
    step_to_save_results,
):
    history = np.zeros(
        (int(Nt / step_to_save_results) + 2, len(r), r.shape[-1])
    )  # Store history of positions
    force_history = np.zeros(
        (int(Nt / step_to_save_results) + 2, len(r), r.shape[-1])
    )  # Store history of forces
    hist_id = 0  # Initialize history index
    Total_K = np.zeros(Nt)  # Kinetic energy
    Total_P_lin = np.zeros(Nt)  # Potential of linear connections energy
    Total_P_fold = np.zeros(Nt)  # Potential of folding springs energy
    Total_P_bend = np.zeros(Nt)  # Potential of bending springs energy
    # Define stress-free configuration if not provided
    if r_stress_free is None:
        r_stress_free = r.copy()
    # to accumulate external work
    r_prev = r.copy()
    # print(m_list)
    # Inverse mass mask for constraints
    inv_m = build_inv_m_mask(
        m_list=m_list,
        fixed_ids=fixed_ids,
        roller_x=roller_fixed_ids_x,
        roller_y=roller_fixed_ids_y,
        roller_z=roller_fixed_ids_z,
    )
    # print(inv_m)
    # initialize force vector
    forces_total = np.zeros_like(r)
    # Build neighbors list for damping calculation
    ptr, neigh = build_neighbors(pairlist, n_nodes=r.shape[0])

    # If folding interactions are provided, calculate the initial angles.
    if quadruplets_fold is not None:
        fold_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_fold
            ]
        )
    else:
        fold_angles = None
    # If bending interactions are provided, calculate the initial angles.
    if quadruplets_bend is not None:
        bend_angles = np.array(
            [
                angle_of_quadruplet(ids=quadruplet, r=r_stress_free)
                for quadruplet in quadruplets_bend
            ]
        )
    else:
        bend_angles = None

    for t in range(Nt):

        # Get half-step positions and velocities using RATTLE Position correction
        r, q, f_constr_pos = rattle_position(
            r=r,
            v=v,
            forces=forces_total,
            dt=dt,
            inv_m=inv_m,
            constrlist=pairlist,
            a=a_list,
        )
        # print(f_constr_pos, "\n")

        if fold_angles is not None:
            # Calculate forces and potential energy due to folding springs
            folding_force, folding_pot_en = hinge_calculation_ang(
                triplist=quadruplets_fold,
                r=r,
                g_list=fold_stiffness_list,
                phi=fold_angles,
            )
        else:
            folding_force = np.zeros_like(r)
            folding_pot_en = 0.0

        if bend_angles is not None:
            # Calculate forces and potential energy due to bending springs
            bending_force, bending_pot_en = hinge_calculation_ang(
                triplist=quadruplets_bend,
                r=r,
                g_list=bend_stiffness_list,
                phi=bend_angles,
                large_dis=False,
            )
        else:
            bending_force = np.zeros_like(r)
            bending_pot_en = 0.0

        # Total forces
        forces_total = folding_force + bending_force
        # Damping forces
        damping_force = neighbor_damping_force(q, ptr, neigh, viscous_per_node)
        # Update total forces
        forces_total += damping_force

        # Apply external forces
        if external_force_ids is not None and len(external_force_ids) > 0:
            if force_vector is not None:
                if load_factor is not None:
                    load_factor = (t + 1) / Nt
                    forces_total[external_force_ids] += load_factor * force_vector
                else:
                    forces_total[external_force_ids] += force_vector

        # Update velocities using RATTLE Velocity correction
        v, f_constr_vel = rattle_velocity(
            r=r, q=q, forces=forces_total, dt=dt, inv_m=inv_m, constrlist=pairlist
        )

        # # Summetry walls
        # apply_symmetry_wall(r, v, r0, axis="x", coord=0.0)
        # apply_symmetry_wall(r, v, r0, axis="x", coord=r0[:, 0].max())
        # # # Keep particles in the initial domain
        # enforce_wall(r=r, v=v, axis="x", coord=0.0, side="greater")
        # enforce_wall(r=r, v=v, axis="x", coord=r0[:, 0].max(), side="less")
        # enforce_wall(r=r, v=v, axis="y", coord=-1.0, side="greater")
        # enforce_wall(r=r, v=v, axis="y", coord=r0[:, 1].max(), side="less")

        # Store positions in history
        if t % step_to_save_results == 0:
            history[hist_id] = r.copy()
            force_history[hist_id] = (
                forces_total.copy() + f_constr_pos.copy() + f_constr_vel.copy()
            )
            hist_id += 1

        # Store energies
        Total_K[t] = compute_kinetic_energy(m_list=m_list, v=v)
        Total_P_fold[t] = folding_pot_en
        Total_P_bend[t] = bending_pot_en

        # update r_prev AFTER work calc
        r_prev[:] = r

    # Trim history array if necessary
    if hist_id < len(history):
        history = history[:hist_id]
        force_history = force_history[:hist_id]

    return history, force_history, Total_K, Total_P_lin, Total_P_fold, Total_P_bend
