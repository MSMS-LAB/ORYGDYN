import numpy as np
import itertools
from tqdm import tqdm


def _resolve_cell_index(cell_idx, nx, ny):
    """
    Normalize a cell index specification.

    Accepts either an integer index in column-major order or a (cx, cy) tuple.
    """
    if isinstance(cell_idx, (tuple, list)):
        if len(cell_idx) != 2:
            raise ValueError("cell_idx tuple/list must have length 2 (cx, cy).")
        cx, cy = map(int, cell_idx)
        if not (0 <= cx < nx and 0 <= cy < ny):
            raise ValueError(f"cell_idx {(cx, cy)} out of bounds for grid {nx}x{ny}.")
        return cx * ny + cy

    idx = int(cell_idx)
    if not (0 <= idx < nx * ny):
        raise ValueError(f"cell_idx {idx} out of bounds for grid {nx}x{ny}.")
    return idx


def select_miura_cell(panels_by_cell, panel_quads, nx, ny, cell_idx=0):
    """
    Select a Miura cell using the `panels_by_cell` structure returned by
    `asseamble_miura_ori` and map its panels back to the full panel_quads array.

    Parameters
    ----------
    panels_by_cell : list of list
        Output from `asseamble_miura_ori` grouping four panels per cell.
    panel_quads : (K,4) array_like
        Parallelogram panel corner indices as returned by `asseamble_miura_ori`.
    nx, ny : int
        Miura cell counts along x and y (same as used in assembly).
    cell_idx : int or (int, int)
        Cell index either as a flat column-major index or (cx, cy) tuple.

    Returns
    -------
    panel_ids : (4,) ndarray of int
        Indices of the four panels forming the selected cell.
    vertex_ids : (M,) ndarray of int
        Unique corner vertex ids of those panels (sorted).
    """
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive integers.")
    if not panels_by_cell:
        raise ValueError("panels_by_cell is empty; cannot select a Miura cell.")

    quads = np.asarray(panel_quads, dtype=int)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError("panel_quads must be an array of shape (K,4).")

    # Validate expected panel count
    expected = 2 * nx * 2 * ny
    if quads.shape[0] < expected:
        raise ValueError(
            f"panel_quads has {quads.shape[0]} panels, expected at least {expected} for grid {nx}x{ny}."
        )

    idx = _resolve_cell_index(cell_idx=cell_idx, nx=nx, ny=ny)

    try:
        cell_panels = panels_by_cell[idx]
    except IndexError as exc:
        raise ValueError(
            f"cell_idx {idx} out of bounds for panels_by_cell length {len(panels_by_cell)}."
        ) from exc

    # Map each cell panel (by corner ids) back to its index in panel_quads
    quad_lookup = {tuple(sorted(q)): i for i, q in enumerate(quads)}
    panel_ids = []
    for panel in cell_panels:
        key = tuple(sorted(panel))
        if key not in quad_lookup:
            raise ValueError(
                f"Could not match cell panel {panel} to any entry in panel_quads."
            )
        panel_ids.append(quad_lookup[key])
    panel_ids = np.array(panel_ids, dtype=int)

    vertex_ids = np.unique(np.asarray(cell_panels, dtype=int).ravel())
    return panel_ids, vertex_ids


def build_pairlist_for_selection(vertex_ids, global_pairs):
    """
    Build a local pairlist for a selection of vertices.

    Parameters
    ----------
    vertex_ids : array_like of int
        Global vertex ids included in the selection. Local indexing follows
        the order provided here.
    global_pairs : (P,2) array_like
        Full pairlist of the structure (global ids).

    Returns
    -------
    local_pairs : (Q,2) ndarray of int
        Pairs with both endpoints inside `vertex_ids`, remapped to local
        indices 0..len(vertex_ids)-1.
    """
    v_ids = np.asarray(vertex_ids, dtype=int)
    pairs = np.asarray(global_pairs, dtype=int)

    id_to_local = {gid: i for i, gid in enumerate(v_ids)}
    keep = set(v_ids.tolist())

    mask = np.array([(i in keep) and (j in keep) for i, j in pairs], dtype=bool)
    filtered = pairs[mask]

    # Deduplicate undirected pairs
    seen = set()
    local_pairs = []
    for i, j in filtered:
        li = id_to_local[int(i)]
        lj = id_to_local[int(j)]
        key = tuple(sorted((li, lj)))
        if key in seen:
            continue
        seen.add(key)
        local_pairs.append([li, lj])

    return np.array(local_pairs, dtype=int)


def _recover_panel_centers(panel_quads, pairs):
    """
    Recover the center-node id for each panel (add_center=True case) by using
    the fact that a panel center is connected to its four corners only.
    """
    panel_quads = np.asarray(panel_quads, dtype=int)
    pairs = np.asarray(pairs, dtype=int)
    max_id = int(pairs.max()) if pairs.size else -1

    neighbors = [set() for _ in range(max_id + 1)]
    for i, j in pairs:
        neighbors[i].add(int(j))
        neighbors[j].add(int(i))

    neighbor_to_id = {frozenset(neigh): idx for idx, neigh in enumerate(neighbors)}

    centers = []
    for quad in panel_quads:
        corners = frozenset(quad.tolist())
        center_id = neighbor_to_id.get(corners)
        if center_id is None:
            # Fallback: search for a node connected to all four corners (and not a corner itself)
            candidates = [
                idx
                for idx, neigh in enumerate(neighbors)
                if idx not in corners
                and corners.issubset(neigh)
                and len(neigh) == len(corners)
            ]
            if not candidates:
                raise ValueError(
                    f"Could not find center node for panel with corners {quad}"
                )
            center_id = candidates[0]
        centers.append(center_id)

    return np.array(centers, dtype=int)


def dihedral_angle(a, b, c, d, eps=1e-12):
    """
    Return the (unsigned) dihedral angle around edge (b, c) formed by
    triangles (a, b, c) and (b, c, d), using an atan2-based formula.

    Output range: [0, pi].
    """
    hinge = c - b
    hinge_norm = np.linalg.norm(hinge)
    if hinge_norm < eps:
        return 0.0
    axis = hinge / hinge_norm  # rotation axis along the hinge

    # Face normals (not necessarily unit yet)
    n1 = np.cross(a - b, hinge)  # normal to (a,b,c)
    n2 = np.cross(d - c, hinge)  # normal to (b,c,d)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < eps or n2_norm < eps:
        return 0.0

    n1 /= n1_norm
    n2 /= n2_norm

    # atan2 ingredients (same structure as find_angle)
    dot = np.clip(np.dot(n1, n2), -1.0, 1.0)
    det = np.dot(axis, np.cross(n1, n2))
    angle = float(np.atan2(det, dot))  # signed in (-pi, pi]

    # Convert to unsigned [0, pi]
    angle = abs(angle)
    # # (abs already ensures [0, pi], but keep this for numerical safety)
    if angle > np.pi:
        angle = 2.0 * np.pi - angle
    # angle = angle % (2.0*np.pi)
    return angle


def measure_cell_dihedral(
    coords, panels_by_cell, panel_quads, quadruplets_fold, pairs, nx, ny, cell_idx=0
):
    """
    Measure the fold dihedral(s) of a selected Miura cell.

    Returns
    -------
    avg_dihedral : float
        Mean dihedral angle (radians) of the folds belonging to that cell.
    per_hinge : ndarray
        Individual dihedral angles (radians) for each fold in the cell.
    """
    panel_ids, vertex_ids = select_miura_cell(
        panels_by_cell=panels_by_cell,
        panel_quads=panel_quads,
        nx=nx,
        ny=ny,
        cell_idx=cell_idx,
    )
    panel_centers = _recover_panel_centers(panel_quads=panel_quads, pairs=pairs)

    cell_centers = set(panel_centers[panel_ids].tolist())
    cell_vertices = set(vertex_ids.tolist())

    hinge_quads = []
    for quad in np.asarray(quadruplets_fold, dtype=int):
        c0, v0, v1, c1 = quad
        if (
            c0 in cell_centers
            and c1 in cell_centers
            and {v0, v1}.issubset(cell_vertices)
        ):
            hinge_quads.append(quad)

    if not hinge_quads:
        raise ValueError("No folding hinges found for selected Miura cell.")

    per_hinge = np.array(
        [
            dihedral_angle(coords[q[0]], coords[q[1]], coords[q[2]], coords[q[3]])
            for q in hinge_quads
        ]
    )
    avg = float(np.mean(per_hinge))
    return avg, per_hinge


def miura_geometry_analysis(
    current_coordinates,
    panels_by_cell,
    panel_quads,
    pairs_list,
    quadruplets_fold,
    nx,
    ny,
    gamma,
    h_a,
    h_b,
    cell_idx=0,
):
    """
    Compute geometric metrics for a specific Miura cell.
    """
    result = {}
    # Select single Miura cell
    _, vertex_ids = select_miura_cell(
        panels_by_cell=panels_by_cell,
        panel_quads=panel_quads,
        nx=nx,
        ny=ny,
        cell_idx=cell_idx,
    )
    selected_coordinates = current_coordinates[vertex_ids]
    result["selected_coordinates"] = selected_coordinates

    # Measure folding (dihedral) angles of the selected cell
    _, dihedrals = measure_cell_dihedral(
        coords=current_coordinates,
        panels_by_cell=panels_by_cell,
        panel_quads=panel_quads,
        quadruplets_fold=quadruplets_fold,
        pairs=pairs_list,
        nx=nx,
        ny=ny,
        cell_idx=cell_idx,
    )
    result["dihedrals"] = dihedrals

    half_alpha_1 = dihedrals[0] / 2
    # print(f"alpha 1: {np.degrees(dihedrals[0]):.3f} deg")
    # tan_half_alpha_2 = np.tan(half_alpha_1) / np.cos(gamma)
    # half_alpha_2 = np.arctan(tan_half_alpha_2)
    half_alpha_2 = dihedrals[1] / 2
    # print(f"{np.degrees(dihedrals[0]):.3f}")

    # zy_angle = dihedrals[0] / 2  # Alpha 1
    H_analytical = h_b * np.cos(half_alpha_1)
    result["H_analytical"] = H_analytical

    H_id = np.argmax(selected_coordinates[:, 2])
    H_geometrical = selected_coordinates[H_id, 2]
    result["H_geometrical"] = H_geometrical

    W_analytical = 2 * h_a * np.sin(half_alpha_2)  # Alpha 2
    result["W_analytical"] = W_analytical

    # alpha_2 = dihedrals[1] / 2
    # tan_alpha_1 = np.tan(alpha_2) * np.cos(gamma)
    # arc_alpha_1 = np.arctan2(
    # np.cos(gamma) * np.sin(alpha_2),
    # np.cos(alpha_2)
    # )
    # half_alpha_2 = np.arctan(tan_half_alpha_2)
    R = np.tan(half_alpha_1) / (np.cos(gamma) * np.tan(half_alpha_2))
    # if R > 1.1:
    # print(f"R: {R:.3f}")
    result["Compatibility_Ratio"] = R

    x = selected_coordinates[:, 0]
    W_geom = x.max() - x.min()
    result["W_geometrical"] = W_geom

    L_analytical = (
        2
        * h_b
        / np.sin(gamma)
        * np.sqrt(1.0 - (np.cos(half_alpha_1) ** 2) * (np.sin(gamma) ** 2))
    )
    result["L_analytical"] = L_analytical

    # Geometric length along OY using only the origin cell (avoid full-structure span)
    sector_idx = np.where(
        (selected_coordinates[:, 0] <= selected_coordinates[:, 0].max() * 1.1)
    )[0][-3:]
    sector_coords = selected_coordinates[sector_idx]
    y = sector_coords[:, 1]
    L_geom = y.max() - y.min()
    result["L_geometrical"] = L_geom
    # print(f"Geometric cell length L (m) from nodes: {L_geom:.3f}")
    return result


def analyze_miura_history(
    history,
    panels_by_cell,
    panel_quads,
    pairs_list,
    quadruplets_fold,
    nx,
    ny,
    gamma,
    h_a,
    h_b,
    n_th_elem=3,
):
    """
    Run geometry analysis across all unit cells for sampled history frames and
    return per-frame averages.
    """
    dihedral_angles = []
    H_geom_list = []
    H_an_list = []
    W_geom_list = []
    W_an_list = []
    L_geom_list = []
    L_an_list = []
    Compatibility_R_list = []

    n_cells = len(panels_by_cell)
    if n_cells == 0:
        raise ValueError("panels_by_cell is empty; cannot analyze Miura history.")

    for current_coordinates in tqdm(
        history[::n_th_elem], desc="Analyzing Miura history"
    ):
        # Check of there are any coordinates (skip empty arrays) or NaNs
        if np.any(current_coordinates) and not np.isnan(current_coordinates).any():
            dihedrals_deg_frame = []
            H_geom_frame = []
            H_an_frame = []
            W_geom_frame = []
            W_an_frame = []
            L_geom_frame = []
            L_an_frame = []
            Compatibility_R_frame = []

            for local_cell_idx in range(n_cells):
                result_geom = miura_geometry_analysis(
                    current_coordinates=current_coordinates,
                    panels_by_cell=panels_by_cell,
                    panel_quads=panel_quads,
                    pairs_list=pairs_list,
                    quadruplets_fold=quadruplets_fold,
                    nx=nx,
                    ny=ny,
                    gamma=gamma,
                    h_a=h_a,
                    h_b=h_b,
                    cell_idx=local_cell_idx,
                )

                # Use first hinge dihedral for consistency with previous behavior
                dihedral_angle = np.rad2deg(result_geom["dihedrals"][0])
                dihedrals_deg_frame.append(dihedral_angle)
                H_geom_frame.append(result_geom["H_geometrical"])
                H_an_frame.append(result_geom["H_analytical"])
                W_geom_frame.append(result_geom["W_geometrical"])
                W_an_frame.append(result_geom["W_analytical"])
                L_geom_frame.append(result_geom["L_geometrical"])
                L_an_frame.append(result_geom["L_analytical"])
                Compatibility_R_frame.append(result_geom["Compatibility_Ratio"])

            # Store averaged values across all cells for this frame
            dihedral_angles.append(float(np.mean(dihedrals_deg_frame)))
            H_geom_list.append(float(np.mean(H_geom_frame)))
            H_an_list.append(float(np.mean(H_an_frame)))
            W_geom_list.append(float(np.mean(W_geom_frame)))
            W_an_list.append(float(np.mean(W_an_frame)))
            L_geom_list.append(float(np.mean(L_geom_frame)))
            L_an_list.append(float(np.mean(L_an_frame)))
            Compatibility_R_list.append(float(np.mean(Compatibility_R_frame)))
    return (
        dihedral_angles,
        H_geom_list,
        H_an_list,
        W_geom_list,
        W_an_list,
        L_geom_list,
        L_an_list,
        Compatibility_R_list,
    )


def _average_fold_dihedral(coords, quadruplets_fold):
    """Return mean fold dihedral angle (radians) and per-fold angles."""
    if len(quadruplets_fold) == 0:
        return None, []

    coords = np.asarray(coords, dtype=float)
    quadruplets_fold = np.asarray(quadruplets_fold, dtype=int)

    if quadruplets_fold.ndim != 2 or quadruplets_fold.shape[1] != 4:
        raise ValueError("quadruplets_fold must have shape (N, 4).")

    per_fold = [
        dihedral_angle(
            coords[q[0]],
            coords[q[1]],
            coords[q[2]],
            coords[q[3]],
        )
        for q in quadruplets_fold
    ]

    if not per_fold:
        return None, []

    return float(np.mean(per_fold)), per_fold


def _average_cell_height_and_length(coords, panel_quads, length_axis=1, height_axis=2):
    if len(panel_quads) == 0:
        return None, None, [], []

    coords = np.asarray(coords, dtype=float)
    panel_quads = np.asarray(panel_quads, dtype=int)

    per_cell_heights = []
    per_cell_lengths = []

    for q in panel_quads:
        cell_coords = coords[q]

        cell_height = float(
            cell_coords[:, height_axis].max() - cell_coords[:, height_axis].min()
        )
        cell_length = 2 * float(
            cell_coords[:, length_axis].max() - cell_coords[:, length_axis].min()
        )

        per_cell_heights.append(cell_height)
        per_cell_lengths.append(cell_length)

    return (
        float(np.mean(per_cell_lengths)),
        float(np.mean(per_cell_heights)),
        per_cell_lengths,
        per_cell_heights,
    )


def analyze_zfold_history(
    history,
    quadruplets_fold,
    panel_quads,
    nx,
    b,
    n_th_elem=1,
):
    """
    Compute fold-wise mean height and length evolution for Z-fold simulations.

    Geometric values are computed independently for each cell and then averaged
    across all cells at every sampled step.

    Analytical values are computed independently for each fold from its local
    dihedral angle and then averaged across all folds.
    """

    H_geom_list = []
    H_an_list = []
    L_geom_list = []
    L_an_list = []
    dihedral_deg_list = []
    dihedral_angles_per_fold = []
    fold_lengths_per_step = []
    fold_heights_per_step = []
    H_an_per_fold_step = []
    L_an_per_fold_step = []

    iterator = history[::n_th_elem]
    for current_coordinates in tqdm(iterator, desc="Analyzing Z-fold history"):
        if not np.any(current_coordinates):
            continue

        coords = np.asarray(current_coordinates, dtype=float)

        # Geometric measurements: per cell first, then average across cells
        l_geom, h_geom, per_cell_lengths, per_cell_heights = (
            _average_cell_height_and_length(
                coords,
                panel_quads,
                length_axis=1,
                height_axis=2,
            )
        )

        theta_avg, per_fold = _average_fold_dihedral(coords, quadruplets_fold)

        if theta_avg is None or per_fold is None or len(per_fold) == 0:
            H_an = h_geom if h_geom is not None else np.nan
            L_an = l_geom if l_geom is not None else np.nan
            dihedral_deg = np.nan
            H_an_per_fold = []
            L_an_per_fold = []
        else:
            per_fold = np.asarray(per_fold, dtype=float)

            # Analytical values from individual fold angles
            H_an_per_fold = b * np.cos(0.5 * per_fold)
            L_an_per_fold = 2 * b * np.sin(0.5 * per_fold)

            # Mean analytical values across folds
            H_an = float(np.mean(H_an_per_fold))
            L_an = float(np.mean(L_an_per_fold))

            # Keep average angle only for reporting/plotting
            dihedral_deg = float(np.rad2deg(theta_avg))

        H_geom_list.append(h_geom if h_geom is not None else np.nan)
        H_an_list.append(H_an)
        L_geom_list.append(l_geom if l_geom is not None else np.nan)
        L_an_list.append(L_an)
        dihedral_deg_list.append(dihedral_deg)
        dihedral_angles_per_fold.append(per_fold if theta_avg is not None else [])
        fold_lengths_per_step.append(per_cell_lengths)
        fold_heights_per_step.append(per_cell_heights)
        H_an_per_fold_step.append(H_an_per_fold)
        L_an_per_fold_step.append(L_an_per_fold)

    return (
        H_geom_list,
        H_an_list,
        L_geom_list,
        L_an_list,
        dihedral_deg_list,
        dihedral_angles_per_fold,
        fold_lengths_per_step,
        fold_heights_per_step,
        H_an_per_fold_step,
        L_an_per_fold_step,
    )


def analyze_kresling_history(
    nz,
    edges,
    a,
    height,
    pin_z_ids,
    triangles,
    history,
    n_th_elem=10,
):
    """
    Analyze Kresling geometry over all cells in the structure.

    The generator defines each panel on interface k -> k+1 as:
        [L_n, L_{n+1}, U_{n+2}, U_{n+1}]
    so the corresponding analytical triangle is:
        (A, B, C) = (L_n, L_{n+1}, U_{n+2})

    This function computes actual and analytical geometric quantities for all
    cells and averages them over the full structure at each sampled step.
    """

    nz = int(nz)
    edges = int(edges)

    if nz < 2:
        raise ValueError("nz must be at least 2.")

    theta = np.pi / edges

    # The first nz*edges nodes are the polygon ring nodes.
    # Midpoints, if present, come after that and are ignored here.
    rings = [np.arange(k * edges, (k + 1) * edges, dtype=int) for k in range(nz)]

    l_ab_actual_list = []
    l_ac_actual_list = []
    l_bc_actual_list = []

    l_ab_analytical_list = []
    l_ac_analytical_list = []
    l_bc_analytical_list = []

    H_analytical_list = []
    H_actual_list = []
    actual_phi_list = []

    iterator = history[::n_th_elem]
    for current_configuration in iterator:
        if not np.any(current_configuration):
            print("Skipping empty configuration")
            continue

        current_configuration = np.asarray(current_configuration, dtype=float)

        H_cells = []
        phi_cells = []

        l_ab_cells = []
        l_ac_cells = []
        l_bc_cells = []

        l_ab_an_cells = []
        l_ac_an_cells = []
        l_bc_an_cells = []
        H_an_cells = []

        for k in range(nz - 1):
            lower_ids = rings[k]
            upper_ids = rings[k + 1]

            lower_ring = current_configuration[lower_ids]
            upper_ring = current_configuration[upper_ids]

            # -------------------------------------------------
            # Correct facet definition from generator:
            # A = L_n
            # B = L_{n+1}
            # C = U_{n+2}
            # -------------------------------------------------
            A = lower_ring
            B = np.roll(lower_ring, -1, axis=0)
            C = np.roll(upper_ring, -2, axis=0)

            # ---------- Actual edge lengths ----------
            l_ab_cell = np.linalg.norm(A - B, axis=1).mean()
            l_ac_cell = np.linalg.norm(A - C, axis=1).mean()
            l_bc_cell = np.linalg.norm(B - C, axis=1).mean()

            # ---------- Actual height ----------
            H_cell = upper_ring[:, 2].mean() - lower_ring[:, 2].mean()

            # ---------- Actual radius ----------
            lower_xy = lower_ring[:, :2]
            upper_xy = upper_ring[:, :2]

            center_lower = lower_xy.mean(axis=0)
            center_upper = upper_xy.mean(axis=0)

            r_lower = np.linalg.norm(lower_xy - center_lower, axis=1).mean()
            r_upper = np.linalg.norm(upper_xy - center_upper, axis=1).mean()
            r_cell = 0.5 * (r_lower + r_upper)

            # ---------- Actual phi ----------
            # phi is the angular separation corresponding to edge BC:
            # B = L_{n+1}, C = U_{n+2}
            angles_B = np.arctan2(
                B[:, 1] - center_lower[1],
                B[:, 0] - center_lower[0],
            )
            angles_C = np.arctan2(
                C[:, 1] - center_upper[1],
                C[:, 0] - center_upper[0],
            )

            phi_samples = np.unwrap(angles_C - angles_B)
            phi_cell = np.mean(phi_samples)

            # Wrap to principal interval for stability
            phi_cell = (phi_cell + np.pi) % (2 * np.pi) - np.pi

            # ---------- Analytical values ----------
            l_ab_an_cell = 2.0 * r_cell * np.sin(theta)

            arg_bc = H_cell**2 - 2.0 * r_cell**2 * np.cos(phi_cell) + 2.0 * r_cell**2
            l_bc_an_cell = np.sqrt(np.maximum(arg_bc, 0.0))

            arg_ac = (
                H_cell**2
                - 2.0 * r_cell**2 * np.cos(phi_cell + 2.0 * theta)
                + 2.0 * r_cell**2
            )
            l_ac_an_cell = np.sqrt(np.maximum(arg_ac, 0.0))

            # Reconstruct analytical height from both AC and BC and average
            arg_H_bc = l_bc_cell**2 - 2.0 * r_cell**2 * (1.0 - np.cos(phi_cell))
            H_from_bc = np.sqrt(np.maximum(arg_H_bc, 0.0))

            arg_H_ac = l_ac_cell**2 - 2.0 * r_cell**2 * (
                1.0 - np.cos(phi_cell + 2.0 * theta)
            )
            H_from_ac = np.sqrt(np.maximum(arg_H_ac, 0.0))

            H_an_cell = 0.5 * (H_from_bc + H_from_ac)

            # ---------- Store per-cell ----------
            H_cells.append(H_cell)
            phi_cells.append(phi_cell)

            l_ab_cells.append(l_ab_cell)
            l_ac_cells.append(l_ac_cell)
            l_bc_cells.append(l_bc_cell)

            l_ab_an_cells.append(l_ab_an_cell)
            l_ac_an_cells.append(l_ac_an_cell)
            l_bc_an_cells.append(l_bc_an_cell)
            H_an_cells.append(H_an_cell)

        # ---------- Average across all cells ----------
        l_ab_actual_list.append(float(np.mean(l_ab_cells)))
        l_ac_actual_list.append(float(np.mean(l_ac_cells)))
        l_bc_actual_list.append(float(np.mean(l_bc_cells)))

        l_ab_analytical_list.append(float(np.mean(l_ab_an_cells)))
        l_ac_analytical_list.append(float(np.mean(l_ac_an_cells)))
        l_bc_analytical_list.append(float(np.mean(l_bc_an_cells)))

        H_analytical_list.append(float(np.mean(H_an_cells)))
        H_actual_list.append(float(np.mean(H_cells)))
        actual_phi_list.append(float(np.rad2deg(np.mean(phi_cells))))

    return (
        l_ab_actual_list,
        l_ac_actual_list,
        l_bc_actual_list,
        l_ab_analytical_list,
        l_ac_analytical_list,
        l_bc_analytical_list,
        H_analytical_list,
        H_actual_list,
        actual_phi_list,
    )


def analyze_kresling_history_current_phi(
    nz,
    edges,
    a,
    height,
    pin_z_ids,
    triangles,
    history,
    n_th_elem=10,
):
    """
    Analyze Kresling geometry using the current geometric twist angle phi only.

    Assumptions
    -----------
    - The generator defines each panel on interface k -> k+1 as:
          [L_n, L_{n+1}, U_{n+2}, U_{n+1}]
      so the corresponding triangle is:
          (A, B, C) = (L_n, L_{n+1}, U_{n+2})

    - The current phi is measured directly from the current configuration.
    - There is no analytical phi.
    - In the theoretical rigid-panel reconstruction, the triangle edges
      remain equal to their initial values:
          L_AB = L_AB^0
          L_AC = L_AC^0
          L_BC = L_BC^0

    - Therefore:
        * r_theoretical is reconstructed from L_AB^0 and is constant,
        * H_theoretical is reconstructed from the current phi together
          with the rigid initial edge lengths.

    Returns
    -------
    (
        H_theoretical_list,
        H_actual_list,
        phi_actual_list_deg,
        r_actual_list,
        r_theoretical_list,
    )
    """
    import numpy as np

    nz = int(nz)
    edges = int(edges)

    if nz < 2:
        raise ValueError("nz must be at least 2.")

    theta = np.pi / edges
    sin_theta = np.sin(theta)
    eps_val = 1e-12

    # Only the first nz*edges nodes are ring vertices
    rings = [np.arange(k * edges, (k + 1) * edges, dtype=int) for k in range(nz)]

    def wrap_to_pi(x):
        return (x + np.pi) % (2.0 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # Initial rigid edge lengths per cell from the undeformed geometry
    # ------------------------------------------------------------------
    initial_configuration = np.asarray(history[0], dtype=float)

    l_ab0_cells = []
    l_ac0_cells = []
    l_bc0_cells = []

    for k in range(nz - 1):
        lower_ids = rings[k]
        upper_ids = rings[k + 1]

        lower_ring0 = initial_configuration[lower_ids]
        upper_ring0 = initial_configuration[upper_ids]

        A0 = lower_ring0
        B0 = np.roll(lower_ring0, -1, axis=0)
        C0 = np.roll(upper_ring0, -2, axis=0)

        l_ab0 = np.linalg.norm(A0 - B0, axis=1).mean()
        l_ac0 = np.linalg.norm(A0 - C0, axis=1).mean()
        l_bc0 = np.linalg.norm(B0 - C0, axis=1).mean()

        l_ab0_cells.append(l_ab0)
        l_ac0_cells.append(l_ac0)
        l_bc0_cells.append(l_bc0)

    l_ab0_cells = np.asarray(l_ab0_cells, dtype=float)
    l_ac0_cells = np.asarray(l_ac0_cells, dtype=float)
    l_bc0_cells = np.asarray(l_bc0_cells, dtype=float)

    # ------------------------------------------------------------------
    # Output lists
    # ------------------------------------------------------------------
    H_theoretical_list = []
    H_actual_list = []

    phi_actual_list_deg = []

    r_actual_list = []
    r_theoretical_list = []

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    iterator = history[::n_th_elem]
    for current_configuration in iterator:
        if not np.any(current_configuration):
            print("Skipping empty configuration")
            continue

        current_configuration = np.asarray(current_configuration, dtype=float)

        H_actual_cells = []
        H_theoretical_cells = []

        phi_actual_cells = []

        r_actual_cells = []
        r_theoretical_cells = []

        for k in range(nz - 1):
            lower_ids = rings[k]
            upper_ids = rings[k + 1]

            lower_ring = current_configuration[lower_ids]
            upper_ring = current_configuration[upper_ids]

            # Triangle consistent with generator
            A = lower_ring
            B = np.roll(lower_ring, -1, axis=0)
            C = np.roll(upper_ring, -2, axis=0)

            # ---------------- Actual geometry ----------------
            H_cell_actual = upper_ring[:, 2].mean() - lower_ring[:, 2].mean()

            lower_xy = lower_ring[:, :2]
            upper_xy = upper_ring[:, :2]

            center_lower = lower_xy.mean(axis=0)
            center_upper = upper_xy.mean(axis=0)

            r_lower = np.linalg.norm(lower_xy - center_lower, axis=1).mean()
            r_upper = np.linalg.norm(upper_xy - center_upper, axis=1).mean()
            r_cell_actual = 0.5 * (r_lower + r_upper)

            angles_B = np.arctan2(
                B[:, 1] - center_lower[1],
                B[:, 0] - center_lower[0],
            )
            angles_C = np.arctan2(
                C[:, 1] - center_upper[1],
                C[:, 0] - center_upper[0],
            )

            phi_actual = wrap_to_pi(np.mean(np.unwrap(angles_C - angles_B)))

            # ---------------- Theoretical reconstruction ----------------
            # Rigid initial triangle edges
            l_ab0 = l_ab0_cells[k]
            l_ac0 = l_ac0_cells[k]
            l_bc0 = l_bc0_cells[k]

            # If AB is rigid, theoretical radius is constant
            r_cell_theoretical = l_ab0 / max(2.0 * sin_theta, eps_val)

            # Reconstruct theoretical height from current phi and rigid edges
            arg_H_bc = l_bc0**2 - 2.0 * r_cell_theoretical**2 * (
                1.0 - np.cos(phi_actual)
            )
            H_from_bc = np.sqrt(np.maximum(arg_H_bc, 0.0))

            arg_H_ac = l_ac0**2 - 2.0 * r_cell_theoretical**2 * (
                1.0 - np.cos(phi_actual + 2.0 * theta)
            )
            H_from_ac = np.sqrt(np.maximum(arg_H_ac, 0.0))

            H_cell_theoretical = 0.5 * (H_from_bc + H_from_ac)

            # ---------------- Store per-cell ----------------
            H_actual_cells.append(H_cell_actual)
            H_theoretical_cells.append(H_cell_theoretical)

            phi_actual_cells.append(phi_actual)

            r_actual_cells.append(r_cell_actual)
            r_theoretical_cells.append(r_cell_theoretical)

        # ---------------- Average across all cells ----------------
        H_theoretical_list.append(float(np.mean(H_theoretical_cells)))
        H_actual_list.append(float(np.mean(H_actual_cells)))

        phi_actual_list_deg.append(float(np.rad2deg(np.mean(phi_actual_cells))))

        r_actual_list.append(float(np.mean(r_actual_cells)))
        r_theoretical_list.append(float(np.mean(r_theoretical_cells)))

    return (
        H_theoretical_list,
        H_actual_list,
        phi_actual_list_deg,
        r_actual_list,
        r_theoretical_list,
    )
