import os
import sys
from pathlib import Path
import argparse
import math
import numpy as np
from numba import njit

from scipy.optimize import minimize_scalar

from src.force_calculation import norm_n

import yaml
from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.6f} seconds")
        return result

    return timeit_wrapper


def new_list(r, a):
    """Generate lists of interacting pairs and triplets based on equilibrium distance.

    Parameters:
    r: Array of particle positions
    a: Equilibrium distance

    Returns:
    pairlist: List of particle pairs interacting via linear forces
    triplist: List of particle triplets interacting via angular forces
    """
    pairlist = []
    triplist = []

    for i in np.arange(1, len(r)):
        for j in np.arange(0, i):
            if math.isclose(a, norm_n(r[i] - r[j]), rel_tol=1e-09, abs_tol=0.0):
                pairlist.append([i, j])
                for k in np.arange(0, j):
                    if math.isclose(a, norm_n(r[j] - r[k]), rel_tol=1e-09, abs_tol=0.0):
                        # A = r[j] - r[i]
                        # B = r[j] - r[k]
                        # cos0 = np.dot(A, B) / (norm_n(A) * norm_n(B))
                        # triplist.append([j, i, k, cos0])
                        triplist.append([j, i, k])

    return np.array(pairlist), np.array(triplist)


def set_rod_3d(N, a, phi1):
    """
    Create a Henky rod structure with alternating particles.

    Parameters:
        N (int): Number of segments in the rod.
        a (float): Equilibrium distance between particles.
        phi1 (float): Initial equilibrium angle in radians.

    Returns:
        np.ndarray: Array of particle positions with shape (2*N - 1, 2).
    """
    _a = np.sqrt(a**2 - (a / 2) ** 2)
    b = _a * np.sin(0.5 * phi1)
    c = _a * np.cos(0.5 * phi1)

    r = np.empty(
        3,
    )

    for k in np.arange(N - 1):
        r = np.vstack((r, [[2 * b * k, 0, 0]]))
        r = np.vstack((r, [[2 * b * k + b, c, 0]]))

    r = np.vstack((r, [[2 * b * (N - 1), 0, 0]]))
    r = np.delete(r, 0, 0)

    return r


def compute_nodal_masses(r, triangles, surface_areas, thickness, density):
    m_nodes = np.zeros(len(r))

    total_mass = density * thickness * np.sum(surface_areas)
    single_node_mass = total_mass / len(r)
    # Run over triangles
    # for i in range(triangles.shape[0]):
    #     # p_1, p_2, p_3 = triangles[i]
    #     # Select current surface triangle
    #     current_area = surface_areas[i]
    #     # Compute mass to be added to each of the three nodes
    #     mass_to_add = current_area * thickness * density / 3.0
    #     # m_nodes[p_1] += mass_to_add
    #     # m_nodes[p_2] += mass_to_add
    #     # m_nodes[p_3] += mass_to_add
    for i in range(m_nodes.shape[0]):
        m_nodes[i] = single_node_mass

    return m_nodes[:, None]


def compute_nodal_masses_from_quads(r, panel_quads, surface_areas, thickness, density):
    m_nodes = np.zeros(len(r))

    # Run over quads
    for i in range(panel_quads.shape[0]):
        p_1, p_2, p_3, p_4 = panel_quads[i]
        # Select current surface triangle
        current_area = surface_areas[i]
        # Compute mass to be added to each of the four nodes
        mass_to_add = current_area * thickness * density / 4.0
        m_nodes[p_1] += mass_to_add
        m_nodes[p_2] += mass_to_add
        m_nodes[p_3] += mass_to_add
        m_nodes[p_4] += mass_to_add

    return m_nodes[:, None]


def compute_rayleigh_coeffs(
    zeta,
    k_min=None,
    k_max=None,
    m_min=None,
    m_avg=None,
    omega_min=None,
    omega_max=None,
):
    """
    Compute Rayleigh damping coefficients alpha (mass-proportional)
    and beta (stiffness-proportional).

    Frequencies can be supplied directly from eigenanalysis. If not,
    they are estimated from stiffness and mass heuristics.

    Parameters
    ----------
    zeta : float
        Target damping ratio (e.g. 0.02 for 2% damping).

    omega_min : float, optional
        Lowest target circular frequency [rad/s] (from eigenanalysis).
    omega_max : float, optional
        Highest target circular frequency [rad/s] (from eigenanalysis).

    k_min : float, optional
        Minimum representative stiffness [N/m].
    k_max : float, optional
        Maximum representative stiffness [N/m].
    m_min : float, optional
        Minimum nodal mass [kg].
    m_avg : float, optional
        Average nodal mass [kg].

    Returns
    -------
    alpha : float
        Mass-proportional Rayleigh damping coefficient.
    beta : float
        Stiffness-proportional Rayleigh damping coefficient.
    omega_min : float
        Lower frequency used in fit [rad/s].
    omega_max : float
        Upper frequency used in fit [rad/s].
    """

    # Use eigenanalysis frequencies if provided
    if omega_min is not None and omega_max is not None:
        omega_min = float(omega_min)
        omega_max = float(omega_max)

    # Otherwise fall back to stiffness/mass heuristic
    else:
        if None in (k_min, k_max, m_min, m_avg):
            raise ValueError(
                "Either (omega_min, omega_max) or "
                "(k_min, k_max, m_min, m_avg) must be provided."
            )
        omega_min = np.sqrt(k_min / m_avg)
        omega_max = np.sqrt(k_max / m_min)

    if omega_min <= 0.0 or omega_max <= 0.0:
        raise ValueError("Frequencies must be positive.")

    # Standard two-frequency Rayleigh fit
    alpha = 2.0 * zeta * (omega_min * omega_max) / (omega_min + omega_max)
    beta = 2.0 * zeta / (omega_min + omega_max)

    return alpha, beta, omega_min, omega_max


def viscous_damping_coeff_calculator(zeta, k_eff, m_eff):
    """Compute viscous damping coefficient from effective mass and stiffness.

    Args:
        zeta (float): Damping ratio.
        k_eff (float): Effective stiffness.
        m_eff (float): Effective mass.

    Returns:
        float: Viscous damping coefficient.
    """
    return 2 * zeta * np.sqrt(k_eff * m_eff)


def viscous_damping_per_node(pairs_list, c_list, m_list, zeta):
    stiffness_per_node = {}
    for n, (i, j) in enumerate(pairs_list):
        bar_stiffness = float(c_list[n])
        for k in [i, j]:
            k = int(k)
            if k not in stiffness_per_node:
                stiffness_per_node[k] = [bar_stiffness]
            else:
                stiffness_per_node[k].append(bar_stiffness)

    viscous_damping_coeff_list = []
    for node_id, stiff_list in stiffness_per_node.items():
        avg_stiff = np.mean(stiff_list)
        viscous_damping_coeff_list.append(
            viscous_damping_coeff_calculator(
                zeta=zeta,
                m_eff=m_list[node_id],
                k_eff=avg_stiff,
            )
        )
    return np.array(viscous_damping_coeff_list)


def compute_bar_cross_sections(
    a_list, tri_edges, surface_areas, thickness, poisson_ratio
):
    """Compute bar cross-sections based on triangle areas and geometry.

    Args:
        a_list (np.ndarray): Array of edge lengths.
        tri_edges (np.ndarray): Array of triangle edges connectivity.
        surface_areas (np.ndarray): Array of triangle areas.
        thickness (float): Thickness of the panel.
        poisson_ratio (float): Poisson's ratio of the material.
    Returns:
        np.ndarray: Array of bar cross-sections for each edge.
    """
    # Initialize bar cross-section list
    bar_cross_section = np.zeros_like(a_list)
    # Run over triangles
    for i in range(tri_edges.shape[0]):
        # Get triangle area
        tri_area = surface_areas[i]
        # Get triangle edges
        e0, e1, e2 = tri_edges[i]
        # Get the edges lengths
        l0 = a_list[e0]
        l1 = a_list[e1]
        l2 = a_list[e2]
        # Length sumation
        l_sum = l0 + l1 + l2
        # Calculate contribution to each edge
        cross_section_contribution = (2 * tri_area * thickness) / (
            (1 - poisson_ratio) * l_sum
        )
        # Update edges cross-section
        bar_cross_section[e0] = cross_section_contribution
        bar_cross_section[e1] = cross_section_contribution
        bar_cross_section[e2] = cross_section_contribution

    return bar_cross_section


def compute_torsion_stiffness(
    E,
    thickness,
    poisson_ratio,
    coordinates,
    quadruplets_bend,
    panel_quads,
    quadruplets_fold,
    non_flat_pattern=False,  # <- marker for patterns like Kresling
):
    """Compute torsion stiffness for bending and folding hinges.

    Parameters
    ----------
    non_flat_pattern : bool
        If False (default): bending hinge is associated with the *shortest* diagonal
        (flat-panel assumption, Miura-like).
        If True: bending hinge is associated with the *longest* diagonal and the hinge
        length is measured via the panel center particle (Kresling-like, non-flat).
    """
    coordinates = np.asarray(coordinates, dtype=float)

    # Plate bending rigidity [N*m]
    bending_rigidity = (E * thickness**3) / (12 * (1 - poisson_ratio**2))

    # ----------------------------
    # Bending stiffness (diagonals)
    # ----------------------------
    if quadruplets_bend is not None and np.asarray(quadruplets_bend).size > 0:
        quadruplets_bend = np.asarray(quadruplets_bend, dtype=int)
        panel_quads = np.asarray(panel_quads, dtype=int)

        bend_stiffness_list = np.zeros((quadruplets_bend.shape[0],), dtype=float)

        # Helper: "full" hinge length via panel center particle
        def hinge_length_via_center(v_a, v_b, v_c):
            # full polyline length a->center->b (works for non-flat patterns)
            return np.linalg.norm(v_a - v_c) + np.linalg.norm(v_c - v_b)

        # ---- Minimal robust mapping: center id is NOT assumed to be in column 1 ----
        # Corner ids are exactly the ids that appear in panel_quads.
        corner_ids = set(map(int, panel_quads.ravel()))

        # Map panel quad corner-set -> row index (order-independent)
        quad_key_to_row = {
            frozenset(map(int, quad)): i for i, quad in enumerate(panel_quads)
        }

        # Group bend quadruplets by detected center id (non-flat / Kresling)
        if non_flat_pattern:
            center_to_rows = {}  # cntr -> list of row indices in quadruplets_bend
            for row_idx, q in enumerate(quadruplets_bend):
                q_ids = list(map(int, q))
                # center is the id in q that is not a corner id
                centers = [vid for vid in q_ids if vid not in corner_ids]
                centers = list(dict.fromkeys(centers))  # unique, stable

                if len(centers) != 1:
                    raise ValueError(
                        f"Non-flat pattern: expected exactly 1 center id in bending quadruplet {q_ids}, "
                        f"got {centers}. (Check panel_quads vs quadruplets_bend consistency.)"
                    )
                cntr = centers[0]
                center_to_rows.setdefault(cntr, []).append(row_idx)

            center_ids_iter = list(center_to_rows.keys())

            # Helper to get the panel quad corners for a given center by collecting its corners
            def panel_corners_from_center(cntr_id: int) -> np.ndarray:
                rows = center_to_rows[cntr_id]
                q_block = quadruplets_bend[rows]  # (m,4)

                nodes = np.unique(q_block.ravel())
                corners = nodes[nodes != cntr_id]

                if corners.size != 4:
                    raise ValueError(
                        f"Center {cntr_id}: expected 4 unique corner nodes, got {corners.size}: {corners.tolist()}. "
                        f"This usually means your bending hinges are not grouped per single panel."
                    )

                key = frozenset(map(int, corners))
                if key not in quad_key_to_row:
                    raise ValueError(
                        f"Center {cntr_id}: could not match corners {sorted(key)} to any row in panel_quads."
                    )
                return panel_quads[quad_key_to_row[key]]

            # ---- Iterate by detected center ids ----
            for cntr in center_ids_iter:
                rows = center_to_rows[cntr]
                panel_quadruplets = quadruplets_bend[rows]
                panel_quads_ids = np.asarray(
                    panel_corners_from_center(int(cntr)), dtype=int
                )

                a, b, c, d = panel_quads_ids

                p_a = coordinates[a]
                p_b = coordinates[b]
                p_c = coordinates[c]
                p_d = coordinates[d]
                p_cn = coordinates[cntr]

                diag_ac = hinge_length_via_center(p_a, p_c, p_cn)
                diag_bd = hinge_length_via_center(p_b, p_d, p_cn)

                equal_case = np.isclose(diag_ac, diag_bd, rtol=1e-6, atol=1e-9)

                if equal_case:
                    D_ref = diag_ac
                    ref_ids = None
                else:
                    # non-flat pattern: bend along the longest diagonal
                    if diag_ac > diag_bd:
                        D_ref = diag_ac
                        ref_ids = (a, c)
                    else:
                        D_ref = diag_bd
                        ref_ids = (b, d)

                k_ref = np.cbrt(D_ref / thickness) * bending_rigidity / 2.0
                k_other = 100.0 * k_ref

                # Assign stiffness to each bending hinge quadruplet (rows already known)
                for row_idx in rows:
                    q = quadruplets_bend[row_idx]
                    if equal_case:
                        bend_stiffness_list[row_idx] = k_ref
                    else:
                        if (ref_ids[0] in q) and (ref_ids[1] in q):
                            bend_stiffness_list[row_idx] = k_ref
                        else:
                            bend_stiffness_list[row_idx] = k_other

        else:
            # ---- ORIGINAL flat-pattern behavior (kept) ----
            for i, cntr in enumerate(np.unique(quadruplets_bend[:, 1])):
                panel_quadruplets = quadruplets_bend[quadruplets_bend[:, 1] == cntr]
                panel_quads_ids = np.asarray(panel_quads[i], dtype=int)

                a, b, c, d = panel_quads_ids

                p_a = coordinates[a]
                p_b = coordinates[b]
                p_c = coordinates[c]
                p_d = coordinates[d]
                p_cn = coordinates[cntr]

                diag_ac = hinge_length_via_center(p_a, p_c, p_cn)
                diag_bd = hinge_length_via_center(p_b, p_d, p_cn)

                equal_case = np.isclose(diag_ac, diag_bd, rtol=1e-6, atol=1e-9)

                if equal_case:
                    D_ref = diag_ac
                    ref_ids = None
                else:
                    # Flat-panel assumption: bend along the shortest diagonal
                    if diag_ac < diag_bd:
                        D_ref = diag_ac
                        ref_ids = (a, c)
                    else:
                        D_ref = diag_bd
                        ref_ids = (b, d)

                k_ref = np.cbrt(D_ref / thickness) * bending_rigidity / 2.0
                k_other = 100.0 * k_ref

                for q in panel_quadruplets:
                    idx = np.where((quadruplets_bend == q).all(axis=1))[0][0]
                    if equal_case:
                        bend_stiffness_list[idx] = k_ref
                    else:
                        if (ref_ids[0] in q) and (ref_ids[1] in q):
                            bend_stiffness_list[idx] = k_ref
                        else:
                            bend_stiffness_list[idx] = k_other
    else:
        bend_stiffness_list = np.array([])

    # ----------------------------
    # Fold stiffness (unchanged)
    # ----------------------------
    if quadruplets_fold is None or np.asarray(quadruplets_fold).size == 0:
        return bend_stiffness_list, np.array([])
    quadruplets_fold = np.asarray(quadruplets_fold, dtype=int)
    fold_stiffness_list = np.zeros((quadruplets_fold.shape[0],), dtype=float)
    for n in range(quadruplets_fold.shape[0]):
        i, j, k, _ = quadruplets_fold[n]
        # Fold length
        L_f = np.linalg.norm(coordinates[j] - coordinates[k])
        # Local fold stiffness
        K_l = (L_f / (2 * L_f)) * bending_rigidity
        # Membrane/stretching contribution [N*m]
        K_m = 0.55 * bending_rigidity * np.cbrt(L_f / thickness)
        # Harmonic average [N*m/rad]
        K_f = 1.0 / (1.0 / K_l + 1.0 / K_m)
        fold_stiffness_list[n] = K_f

    return bend_stiffness_list, fold_stiffness_list


def build_square_hinge_triplist(panel_quads):
    """
    Build hinge tuples (i, j, k, l, n, v) for adjacent square panels.

    Each hinge is defined between two panels that share an edge
    (two common vertices). The tuple layout matches `hinge_square`:

        i, j, k  -> left panel: triangles (i, j, n) and (n, j, k)
        l, n, v  -> right panel: mirror vertices across the same hinge j-n

    Parameters
    ----------
    panel_quads : (K, 4) array of int
        Each row contains the 4 corner ids of one square/parallelogram panel.

    Returns
    -------
    triplist : (H, 6) array of int
        Each row is (i, j, k, l, n, v) for one hinge.
        j and n are always the two vertices on the shared edge.
    """
    quads = np.asarray(panel_quads, dtype=int)
    n_panels = quads.shape[0]
    hinges = []

    for a in range(n_panels):
        qa = quads[a]
        for b in range(a + 1, n_panels):
            qb = quads[b]

            # Find shared vertices (edge) between the two panels
            shared = [v for v in qa if v in qb]
            if len(shared) != 2:
                continue  # not adjacent by an edge

            # Shared edge (hinge) vertices, ordered as they appear in qa
            j, n = shared[0], shared[1]

            # Outer vertices of each panel (not on the shared edge)
            outer_a = [v for v in qa if v not in shared]
            outer_b = [v for v in qb if v not in shared]

            if len(outer_a) != 2 or len(outer_b) != 2:
                continue  # should not happen for proper quads

            i, k = outer_a[0], outer_a[1]
            l, v = outer_b[0], outer_b[1]

            hinges.append((i, j, k, l, n, v))

    if len(hinges) == 0:
        return np.zeros((0, 6), dtype=int)

    return np.array(hinges, dtype=int)


def read_materials_yaml(filepath, material_name):
    # Load material properties from YAML file
    with open(filepath, "r", encoding="utf-8") as f:
        general_dict = yaml.safe_load(f)
    # Extract properties for the specified material
    current_material_dict = general_dict[material_name]
    E = float(current_material_dict["youngs_modulus"])
    density = float(current_material_dict["density"])
    poisson_ratio = float(current_material_dict["poisson_ratio"])
    return E, density, poisson_ratio


@njit(cache=True)
def build_neighbors(pairlist, n_nodes):
    deg = np.zeros(n_nodes, dtype=np.int32)

    # считаем количество соседей
    for k in range(pairlist.shape[0]):
        i = pairlist[k, 0]
        j = pairlist[k, 1]
        deg[i] += 1
        deg[j] += 1

    # указатели
    ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    for i in range(n_nodes):
        ptr[i + 1] = ptr[i] + deg[i]

    # список соседей
    neigh = np.zeros(ptr[-1], dtype=np.int32)
    cur = ptr.copy()

    for k in range(pairlist.shape[0]):
        i = pairlist[k, 0]
        j = pairlist[k, 1]

        neigh[cur[i]] = j
        cur[i] += 1

        neigh[cur[j]] = i
        cur[j] += 1

    return ptr, neigh


@njit(cache=True)
def build_inv_m_mask(
    m_list, fixed_ids=None, roller_x=None, roller_y=None, roller_z=None
):
    N = m_list.shape[0]
    inv_m = (1.0 / m_list).reshape(N, 1) * np.ones((N, 3))

    if fixed_ids is not None and len(fixed_ids) > 0:
        inv_m[fixed_ids, :] = 0.0

    if roller_x is not None and len(roller_x) > 0:
        inv_m[roller_x, 0] = 0.0

    if roller_y is not None and len(roller_y) > 0:
        inv_m[roller_y, 1] = 0.0

    if roller_z is not None and len(roller_z) > 0:
        inv_m[roller_z, 2] = 0.0

    return inv_m


def save_simulation_result(result_dict, save_path):
    """
    Save simulation results to a TXT file.

    Parameters
    ----------
    result_dict : dict
        Dictionary with results for this run.
        Example: {"Ey": 4200, "epsilon": 1e-6}
    save_path : str
        Path to the output TXT file.
    """

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # If file does not exist → create with header
    if not os.path.exists(save_path):
        with open(save_path, "w") as f:
            header = "run\t" + "\t".join(result_dict.keys()) + "\n"
            f.write(header)
        run_id = 1
    else:
        # Count existing runs (exclude header)
        with open(save_path, "r") as f:
            run_id = sum(1 for _ in f)

    # Write current run
    with open(save_path, "a") as f:
        values = "\t".join(
            f"{v:.6e}" if isinstance(v, (float, int)) else str(v)
            for v in result_dict.values()
        )
        line = f"{run_id}\t{values}\n"
        f.write(line)


def _build_edge_lookup(pairs_list):
    """
    Map an undirected edge (min(i,j), max(i,j)) -> edge index in pairs_list.
    """
    edge_lookup = {}
    for e, (i, j) in enumerate(np.asarray(pairs_list, dtype=int)):
        key = tuple(sorted((int(i), int(j))))
        edge_lookup[key] = e
    return edge_lookup


def _infer_panel_centers_from_triangles(panel_quads, triangles):
    """
    Infer the center node of each N5B8 panel from the triangle connectivity.

    Assumes each panel quad [n0,n1,n2,n3] is triangulated as:
        [n0,n1,c], [n1,n2,c], [n2,n3,c], [n3,n0,c]
    with one center node c.

    Returns
    -------
    center_ids : ndarray, shape (n_panels,)
    """
    triangles = np.asarray(triangles, dtype=int)
    panel_quads = np.asarray(panel_quads, dtype=int)

    center_ids = np.full(panel_quads.shape[0], -1, dtype=int)

    for p, quad in enumerate(panel_quads):
        qset = set(map(int, quad))
        # Find triangles that touch this quad with exactly 2 quad nodes + 1 extra node
        candidates = []
        for tri in triangles:
            tri_set = set(map(int, tri))
            common = tri_set & qset
            if len(common) == 2:
                extra = list(tri_set - qset)
                if len(extra) == 1:
                    candidates.append(extra[0])

        if len(candidates) == 0:
            raise ValueError(
                f"Could not infer center node for panel {p}. "
                f"Make sure the mesh is N5B8 with add_center=True."
            )

        vals, counts = np.unique(candidates, return_counts=True)
        center_ids[p] = vals[np.argmax(counts)]

    return center_ids


def compute_bar_cross_sections_filipov_n5b8(
    coordinates,
    pairs_list,
    panel_quads,
    triangles,
    thickness,
    poisson_ratio,
):
    """
    Compute bar cross-sections using Filipov et al. (2017), Eq. (5),
    for the N5B8 panel model.

    Parameters
    ----------
    coordinates : (N, 3) ndarray
        Nodal coordinates.
    pairs_list : (n_edges, 2) ndarray
        Global edge list used in the simulation.
    panel_quads : (n_panels, 4) ndarray
        Corner nodes of each panel, ordered consistently.
    triangles : (n_triangles, 3) ndarray
        Triangles used to infer the center node of each panel.
    thickness : float
        Sheet thickness.
    poisson_ratio : float
        Poisson's ratio.

    Returns
    -------
    bar_cross_section : (n_edges,) ndarray
        Cross-sectional area assigned to each global bar.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    pairs_list = np.asarray(pairs_list, dtype=int)
    panel_quads = np.asarray(panel_quads, dtype=int)

    if panel_quads.shape[1] != 4:
        raise ValueError("panel_quads must have shape (n_panels, 4).")

    nu = float(poisson_ratio)
    t = float(thickness)

    edge_lookup = _build_edge_lookup(pairs_list)
    center_ids = _infer_panel_centers_from_triangles(panel_quads, triangles)

    bar_cross_section = np.zeros(len(pairs_list), dtype=float)

    for p, quad in enumerate(panel_quads):
        n0, n1, n2, n3 = map(int, quad)
        nc = int(center_ids[p])

        x0 = coordinates[n0]
        x1 = coordinates[n1]
        x2 = coordinates[n2]
        x3 = coordinates[n3]

        # Filipov et al. for skewed/irregular panels:
        # H = average distance between nodes (1,4) and (2,3)
        # W = average distance between nodes (1,2) and (4,3)
        # In 0-based indexing: (0,3), (1,2) and (0,1), (3,2)
        H = 0.5 * (np.linalg.norm(x3 - x0) + np.linalg.norm(x2 - x1))
        W = 0.5 * (np.linalg.norm(x1 - x0) + np.linalg.norm(x2 - x3))

        denom = 2.0 * (1.0 - nu**2)

        A_X = t * (H**2 - nu * W**2) / (H * denom)
        A_Y = t * (W**2 - nu * H**2) / (W * denom)
        A_D = t * nu * (H**2 + W**2) ** 1.5 / (2.0 * H * W * (1.0 - nu**2))

        # Perimeter bars:
        # X-bars: (0,1) and (3,2)
        # Y-bars: (0,3) and (1,2)
        # Diagonal bars in N5B8: center-to-corner
        edge_to_area = {
            tuple(sorted((n0, n1))): A_X,
            tuple(sorted((n3, n2))): A_X,
            tuple(sorted((n0, n3))): A_Y,
            tuple(sorted((n1, n2))): A_Y,
            tuple(sorted((n0, nc))): A_D,
            tuple(sorted((n1, nc))): A_D,
            tuple(sorted((n2, nc))): A_D,
            tuple(sorted((n3, nc))): A_D,
        }

        for edge_key, area_val in edge_to_area.items():
            if edge_key not in edge_lookup:
                raise KeyError(
                    f"Edge {edge_key} from panel {p} was not found in pairs_list."
                )
            eidx = edge_lookup[edge_key]
            bar_cross_section[eidx] += area_val

    return bar_cross_section


def select_A3_numerically(
    a1: float,
    a2: float,
    thickness: float,
    E: float,
    poisson_ratio: float,
    A3_max: float | None = None,
) -> tuple[float, float, float]:
    """
    Numerically select A3.

    Objective:
        E1 = E2 = E by construction,
        and C1212 should be close to the target shear modulus G.

    This is useful if you want a constrained numerical procedure.
    """

    s3 = (a1**2 + a2**2) ** 1.5

    G_target = E / (2.0 * (1.0 + poisson_ratio))

    if A3_max is None:
        # A rough upper bound. You can adjust this.
        A3_max = thickness * min(a1, a2)

    def areas_from_A3(A3: float):
        C12 = E * a1 * a2 * A3 / (thickness * s3)

        C_equal = 0.5 * (E + np.sqrt(E**2 + 4.0 * C12**2))

        A1 = (a2 * thickness * C_equal) / E - (a1**3 * A3) / s3

        A2 = (a1 * thickness * C_equal) / E - (a2**3 * A3) / s3

        return A1, A2, A3, C12, C_equal

    def objective(A3: float) -> float:
        A1, A2, _, C12, _ = areas_from_A3(A3)

        # Penalize non-physical areas
        if A1 <= 0.0 or A2 <= 0.0 or A3 <= 0.0:
            return 1e30

        # Match shear modulus as the secondary criterion
        return ((C12 - G_target) / G_target) ** 2

    result = minimize_scalar(
        objective,
        bounds=(1e-16, A3_max),
        method="bounded",
    )

    if not result.success:
        raise RuntimeError("A3 optimization failed.")

    A3_opt = result.x
    A1, A2, A3, C12, C_equal = areas_from_A3(A3_opt)

    if A1 <= 0.0 or A2 <= 0.0 or A3 <= 0.0:
        raise ValueError(
            f"Optimized areas are non-physical: "
            f"A1={A1:.6e}, A2={A2:.6e}, A3={A3:.6e}."
        )

    return A1, A2, A3


def compute_rectangular_grid_bar_areas(
    coordinates: np.ndarray,
    pairs_list: np.ndarray,
    panel_quads: np.ndarray,
    center_nodes: np.ndarray,
    thickness: float,
    E: float,
    poisson_ratio: float,
) -> tuple[np.ndarray, float, float]:
    """Compute equivalent bar areas for a rectangular 5N8B mesh.

    If the panel aspect ratio satisfies

        sqrt(nu) < a1/a2 < 1/sqrt(nu),

    the analytical formulas are used.

    Otherwise, A3 is selected numerically and A1, A2 are recomputed so that

        E1 = E2 = E.

    Important:
    - Bar type classification is topology-based, not direction-based.
    - This is necessary for folded/non-flat geometries, where local panel edges
      may point in different 3D directions.
    """

    # -------------------------------------------------------------------------
    # 1. Extract reference panel dimensions from the first panel
    # -------------------------------------------------------------------------
    p0, p1, p2, p3 = coordinates[panel_quads[0]]

    v1 = p1 - p0  # local a1 direction
    v2 = p2 - p1  # local a2 direction

    a1 = np.linalg.norm(v1)
    a2 = np.linalg.norm(v2)

    if a1 <= 0.0 or a2 <= 0.0:
        raise ValueError("Degenerate panel: a1 and a2 must be positive.")

    if E <= 0.0:
        raise ValueError("Young's modulus E must be positive.")

    if thickness <= 0.0:
        raise ValueError("Thickness must be positive.")

    if poisson_ratio <= 0.0:
        raise ValueError(
            "poisson_ratio must be positive for the aspect-ratio criterion."
        )

    s3 = (a1**2 + a2**2) ** 1.5

    # -------------------------------------------------------------------------
    # 2. Choose analytical or numerical calibration
    # -------------------------------------------------------------------------
    aspect_ratio = a1 / a2
    lower_bound = np.sqrt(poisson_ratio)
    upper_bound = 1.0 / np.sqrt(poisson_ratio)

    use_analytical = lower_bound < aspect_ratio < upper_bound

    if use_analytical:
        # Analytical formulas based on matching:
        #
        #   C11 = C22 = E / (1 - nu^2)
        #   C12 = nu E / (1 - nu^2)
        #
        # This corresponds to the full-diagonal stiffness derivation:
        #
        #   C12 = 2 E a1 a2 A3 / (t s^3)

        C11_target = E / (1.0 - poisson_ratio**2)
        C22_target = C11_target
        C12_target = poisson_ratio * E / (1.0 - poisson_ratio**2)

        A3 = C12_target * thickness * s3 / (2.0 * E * a1 * a2)

        A1 = (a2 * thickness * C11_target) / E - (2.0 * a1**3 * A3) / s3

        A2 = (a1 * thickness * C22_target) / E - (2.0 * a2**3 * A3) / s3

        method = "analytical"

    else:
        # Numerical fallback:
        # A3 is selected numerically, while A1 and A2 are computed so that
        #
        #   E1 = E2 = E.
        #
        # This requires select_A3_numerically(...) to be defined separately.

        A1, A2, A3 = select_A3_numerically(
            a1=a1,
            a2=a2,
            thickness=thickness,
            E=E,
            poisson_ratio=poisson_ratio,
        )

        method = "numerical"

    if A1 <= 0.0 or A2 <= 0.0 or A3 <= 0.0:
        raise ValueError(
            f"Non-physical bar areas from {method} method: "
            f"A1={A1:.6e}, A2={A2:.6e}, A3={A3:.6e}. "
            f"Aspect ratio a1/a2={aspect_ratio:.6f}, "
            f"analytical validity range=({lower_bound:.6f}, {upper_bound:.6f})."
        )

    print(
        f"Computed bar areas using {method} method: "
        f"A1={A1:.6e}, A2={A2:.6e}, A3={A3:.6e}."
    )
    print(
        f"Panel aspect ratio a1/a2={aspect_ratio:.6f}; "
        f"analytical validity range=({lower_bound:.6f}, {upper_bound:.6f})."
    )

    # -------------------------------------------------------------------------
    # 3. Build topology-based edge classification
    # -------------------------------------------------------------------------
    # Local panel convention:
    #
    #   q0 = bottom-left
    #   q1 = top-left
    #   q2 = top-right
    #   q3 = bottom-right
    #
    # Therefore:
    #
    #   q0-q1 and q2-q3 are local a1-type edges -> A1
    #   q1-q2 and q3-q0 are local a2-type edges -> A2

    edge_type: dict[tuple[int, int], int] = {}

    def add_edge_type(i: int, j: int, typ: int) -> None:
        key = tuple(sorted((int(i), int(j))))

        if key in edge_type and edge_type[key] != typ:
            raise ValueError(
                f"Conflicting edge classification for edge {key}: "
                f"{edge_type[key]} vs {typ}."
            )

        edge_type[key] = typ

    for q0, q1, q2, q3 in panel_quads:
        # Local a1-direction edges
        add_edge_type(q0, q1, typ=1)
        add_edge_type(q2, q3, typ=1)

        # Local a2-direction edges
        add_edge_type(q1, q2, typ=2)
        add_edge_type(q3, q0, typ=2)

    # -------------------------------------------------------------------------
    # 4. Assign cross-sectional areas to all bars
    # -------------------------------------------------------------------------
    center_node_set = set(int(i) for i in center_nodes)
    bar_areas = np.zeros(len(pairs_list), dtype=float)

    for k, (i, j) in enumerate(pairs_list):
        i = int(i)
        j = int(j)

        if i == j:
            raise ValueError(f"Degenerate bar with identical nodes: ({i}, {j}).")

        # Center-to-corner bars are diagonal bars.
        if i in center_node_set or j in center_node_set:
            bar_areas[k] = A3
            continue

        key = tuple(sorted((i, j)))

        if key not in edge_type:
            raise ValueError(
                f"Cannot classify perimeter bar ({i}, {j}). "
                "This bar is not listed as a panel edge and does not contain "
                "a center node."
            )

        if edge_type[key] == 1:
            bar_areas[k] = A1

        elif edge_type[key] == 2:
            bar_areas[k] = A2

        else:
            raise ValueError(f"Unknown edge type {edge_type[key]} for bar ({i}, {j}).")

    return bar_areas, a1, a2


def _add_bool_arg(parser, name, default, help_text):
    dest = name.replace("-", "_")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(
        f"--no-{name}", dest=dest, action="store_false", help=f"Disable: {help_text}"
    )

    parser.set_defaults(**{dest: default})


def add_common_simulation_args(parser):
    """
    Arguments shared by all examples:
    Miura, Kresling, Z-fold, etc.
    """

    parser.add_argument(
        "--links-type",
        type=str,
        default="extendable",
        choices=["extendable", "shake", "rattle"],
        help="Type of link model.",
    )

    parser.add_argument(
        "--T",
        type=float,
        default=0.03,
        help="Total simulation time [s].",
    )

    parser.add_argument(
        "--zeta",
        type=float,
        default=0.01,
        help="Damping ratio.",
    )

    parser.add_argument(
        "--dt-safety-factor",
        type=float,
        default=None,
        help="Timestep safety factor. If not provided, chosen automatically.",
    )

    parser.add_argument(
        "--desired-frames",
        type=int,
        default=100,
        help="Approximate number of saved frames.",
    )

    parser.add_argument(
        "--material-name",
        type=str,
        default="plastic",
        help="Material name from materials.yaml.",
    )

    parser.add_argument(
        "--materials-file",
        type=Path,
        default=Path("./src/materials.yaml"),
        help="Path to materials.yaml.",
    )

    parser.add_argument(
        "--force-vector",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        metavar=("FX", "FY", "FZ"),
        help="Direction vector of the applied force.",
    )

    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=20.0,
        help="Total force magnitude [N].",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory for plots and output data.",
    )

    parser.add_argument(
        "--vtk-dir",
        type=Path,
        default=Path("data_vtk"),
        help="Directory for exported visualization files.",
    )

    _add_bool_arg(
        parser,
        "load-factor",
        default=True,
        help_text="Apply force gradually over time.",
    )

    _add_bool_arg(
        parser,
        "plotting",
        default=True,
        help_text="Generate plots.",
    )

    _add_bool_arg(
        parser,
        "export-vtk",
        default=True,
        help_text="Export geometry history for ParaView visualization.",
    )


def add_miura_args(parser):
    """
    Geometry parameters specific to the Miura example.
    """

    parser.add_argument(
        "--a", type=float, default=1e-2, help="Miura side length a [m]."
    )
    parser.add_argument(
        "--b", type=float, default=1e-2, help="Miura side length b [m]."
    )
    parser.add_argument(
        "--gamma", type=float, default=np.pi / 2.5, help="Miura angle gamma [rad]."
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=np.pi / 2.25,
        help="Initial folding angle theta [rad].",
    )
    parser.add_argument(
        "--nx", type=int, default=3, help="Number of Miura cells in x direction."
    )
    parser.add_argument(
        "--ny", type=int, default=3, help="Number of Miura cells in y direction."
    )
    parser.add_argument(
        "--thickness", type=float, default=2e-4, help="Panel thickness [m]."
    )

    _add_bool_arg(
        parser,
        "add-center",
        default=True,
        help_text="Add center nodes to panels. 'False' does not currently work.",
    )


def add_kresling_args(parser):
    """
    Geometry parameters specific to the Kresling example.
    Change names/defaults according to your actual geometry generator.
    """

    parser.add_argument(
        "--radius", type=float, default=1e-2, help="Kresling radius [m]."
    )
    parser.add_argument(
        "--height", type=float, default=2e-2, help="Initial Kresling height [m]."
    )
    parser.add_argument(
        "--n-sides", type=int, default=6, help="Number of polygon sides."
    )
    parser.add_argument(
        "--n-cells", type=int, default=1, help="Number of stacked Kresling cells."
    )
    parser.add_argument(
        "--phi", type=float, default=0.3, help="Twist angle between rings [rad]."
    )
    parser.add_argument(
        "--thickness", type=float, default=2e-4, help="Panel thickness [m]."
    )


def add_zfold_args(parser):
    """
    Geometry parameters specific to the Z-fold example.
    Change names/defaults according to your actual geometry generator.
    """

    parser.add_argument(
        "--panel-length", type=float, default=1e-2, help="Panel length [m]."
    )
    parser.add_argument(
        "--panel-width", type=float, default=1e-2, help="Panel width [m]."
    )
    parser.add_argument("--n-panels", type=int, default=6, help="Number of panels.")
    parser.add_argument(
        "--fold-angle",
        type=float,
        default=np.pi / 3.0,
        help="Initial fold angle [rad].",
    )
    parser.add_argument(
        "--thickness", type=float, default=2e-4, help="Panel thickness [m]."
    )


def parse_opt(args=None, default_example=None):
    available_examples = {"miura", "kresling", "zfold"}

    if args is None:
        args = sys.argv[1:]

    args = list(args)

    if default_example is not None:
        if default_example not in available_examples:
            raise ValueError(f"Unknown default example: {default_example}")

        # If the user runs:
        # python examples/Miura_Pattern.py --nx 4 --ny 4
        #
        # internally parse it as:
        # python examples/Miura_Pattern.py miura --nx 4 --ny 4
        if len(args) == 0 or args[0].startswith("-"):
            args = [default_example] + args

    parser = argparse.ArgumentParser(
        description="Run origami dynamics examples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="example",
        required=True,
        help="Origami example to run.",
    )

    # Miura
    miura_parser = subparsers.add_parser(
        "miura",
        help="Run Miura-ori example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_simulation_args(miura_parser)
    add_miura_args(miura_parser)

    # Kresling
    kresling_parser = subparsers.add_parser(
        "kresling",
        help="Run Kresling example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_simulation_args(kresling_parser)
    add_kresling_args(kresling_parser)

    # Z-fold
    zfold_parser = subparsers.add_parser(
        "zfold",
        help="Run Z-fold example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_simulation_args(zfold_parser)
    add_zfold_args(zfold_parser)

    opt = parser.parse_args(args)

    # ------------------------------------------------------------------
    # Shared post-processing
    # ------------------------------------------------------------------
    opt.force_vector = np.asarray(opt.force_vector, dtype=float)

    norm = np.linalg.norm(opt.force_vector)
    if norm == 0.0:
        parser.error("--force-vector must be nonzero.")

    opt.force_vector = opt.force_vector / norm

    if opt.dt_safety_factor is None:
        if opt.links_type == "extendable":
            opt.dt_safety_factor = 0.5
        elif opt.links_type == "rattle":
            opt.dt_safety_factor = 5.0
        else:
            opt.dt_safety_factor = 2.0

    if opt.T <= 0.0:
        parser.error("--T must be positive.")

    if opt.zeta < 0.0:
        parser.error("--zeta must be non-negative.")

    if opt.thickness <= 0.0:
        parser.error("--thickness must be positive.")

    opt.results_dir = opt.results_dir / opt.example
    opt.vtk_dir = opt.vtk_dir / opt.example

    opt.results_dir.mkdir(parents=True, exist_ok=True)
    opt.vtk_dir.mkdir(parents=True, exist_ok=True)

    return opt
