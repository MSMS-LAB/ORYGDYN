# %%
import numpy as np
from plots_and_results import (
    plot_particles_with_forces,
    create_text_bond_file_2d,
    Output,
)
from utils import set_rod_3d


def find_duplicates(arr):
    duplicates = []
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            if i != j:
                if [j, i] not in duplicates:
                    if np.array_equal(arr[i], arr[j]):
                        duplicates.append([i, j])

    return np.array(duplicates)


def build_edge_triangle_connectivity(pairs, triangles):
    """
    Build connectivity between edges (pairs) and triangles.

    Parameters
    ----------
    pairs : (P, 2) int array
        List of unique undirected edges (vertex indices).
    triangles : (T, 3) int array
        List of triangles (vertex indices).

    Returns
    -------
    tri_edges : (T, 3) int array
        tri_edges[t, k] = index in `pairs` of the k-th edge of triangle t.
        Edges are (v0-v1, v1-v2, v2-v0).

    edge_tris : list of lists
        edge_tris[e] = list of triangle indices that contain edge `pairs[e]`.
        Interior edges typically have 2 entries, boundary edges 1.
    """
    pairs = np.asarray(pairs, dtype=int)
    triangles = np.asarray(triangles, dtype=int)

    P = pairs.shape[0]
    T = triangles.shape[0]

    # --- 1) Normalize pairs (undirected) and build lookup: edge -> pair index ---
    norm_pairs = np.sort(pairs, axis=1)  # (P,2)
    edge_to_idx = {(e[0], e[1]): i for i, e in enumerate(norm_pairs)}

    # --- 2) For each triangle, find the indices of its 3 edges in `pairs` ---
    tri_edges = np.empty((T, 3), dtype=int)

    # Prepare storage for pair -> triangles
    edge_tris = [[] for _ in range(P)]

    for t in range(T):
        v0, v1, v2 = triangles[t]

        # Triangle edges (in any consistent order)
        edges = [
            (v0, v1),
            (v1, v2),
            (v2, v0),
        ]

        edge_indices = []
        for a, b in edges:
            key = (a, b) if a < b else (b, a)  # normalize
            try:
                e_idx = edge_to_idx[key]
            except KeyError:
                raise ValueError(
                    f"Triangle {t} edge ({a}, {b}) not found in pairs list."
                )
            edge_indices.append(e_idx)
            edge_tris[e_idx].append(t)

        tri_edges[t] = edge_indices

    return tri_edges, edge_tris


def set_plate_3d(N, a, phi1):
    """
    Create a 3D plate structure by duplicating a 2D rod-like pattern into two connected layers.

    This function first generates the same 2D pattern as the original rod structure,
    using alternating particles defined by the equilibrium distance a and angle phi1.
    It then creates a lower layer (z = 0) and an upper layer (z = thickness). The
    combined structure is a connected plate whose 2D projection onto the OX and OY
    plane is identical to the original pattern.

    Parameters:
        N (int): Number of segments in the rod.
        a (float): Equilibrium distance between particles.
        phi1 (float): Equilibrium angle (in radians) used in the rod pattern.
        thickness (float): Separation between the two layers along the Z-axis.

    Returns:
        np.ndarray: Array of particle positions with shape (2*(2*N - 1), 3).
                    The first (2*N - 1) points lie in the lower layer (z=0) and the
                    remaining points in the upper layer (z=thickness).
    """
    # Compute helper quantities from the given angle.
    b = a * np.sin(0.5 * phi1)
    c = a * np.cos(0.5 * phi1)

    # Build the base 2D rod pattern (points in OX–OY).
    points_2d = []
    for k in range(N - 1):
        points_2d.append([2 * b * k, 0])
        points_2d.append([2 * b * k + b, c])
    points_2d.append([2 * b * (N - 1), 0])
    points_2d = np.array(points_2d)  # Shape: (2*N - 1, 2)

    # Create two layers in the Z direction.
    lower_layer = np.hstack((points_2d, np.zeros((points_2d.shape[0], 1))))
    upper_layer = np.hstack((points_2d, np.full((points_2d.shape[0], 1), a)))

    # Combine the layers to form a connected plate.
    plate = np.vstack((lower_layer, upper_layer))

    return plate


def get_connected_pairs(plate):
    """
    Given a 3D plate structure (as returned by set_plate_3d),
    return a list of pairs (tuples) of indices representing connected particles.

    Connectivity is defined as:
        1. Sequential connections within the lower layer.
        2. Sequential connections within the upper layer.
        3. Vertical connections linking corresponding particles in the lower
           and upper layers.
        4. Diagonal connections for every quadrilateral cell formed by adjacent
           particles in the lower and upper layers.

    Parameters:
        plate (np.ndarray): Array of particle positions with shape (2*M, 3),
                            where M is the number of particles in one layer.

    Returns:
        list of tuple: List of index pairs (i, j) that represent bonds between particles.
    """
    n_total = plate.shape[0]
    if n_total % 2 != 0:
        raise ValueError("Expected an even number of particles (two layers).")

    M = n_total // 2  # number of particles per layer
    pairs = []

    # Sequential connections in the lower layer (indices 0 to M-1)
    for i in range(M - 1):
        pairs.append((i, i + 1))

    # Sequential connections in the upper layer (indices M to 2*M-1)
    for i in range(M, n_total - 1):
        pairs.append((i, i + 1))

    # Vertical connections (linking each lower layer point to its corresponding upper layer point)
    for i in range(M):
        pairs.append((i, i + M))

    # Diagonal connections for every quadrilateral cell:
    # For each adjacent pair in the lower layer (i, i+1), the quadrilateral has vertices:
    #   lower: i, i+1; upper: i+M, i+1+M.
    # Add the diagonal bonds: (i, i+1+M) and (i+1, i+M)
    for i in range(M - 1):
        pairs.append((i, i + 1 + M))
        pairs.append((i + 1, i + M))

    return np.array(pairs)


def get_angular_spring_sets(plate):
    """
    Given a 3D plate structure (as returned by set_plate_3d),
    generate a list of six-point sets that can later be used for angular spring calculations.

    Here, we assume that an angular spring will be defined on two consecutive quadrilateral cells.
    Each such angular element is defined by:

      - Lower layer: indices i, i+1, i+2
      - Upper layer: indices i+M, i+1+M, i+2+M

    where M is the number of particles in one layer (i.e. half the total number of points).
    These six points, taken together, capture the relative angle between two adjacent segments.

    Parameters:
        plate (np.ndarray): Array of particle positions with shape (2*M, 3),
                            where M is the number of particles in one layer.

    Returns:
        list of tuple: A list where each tuple contains six indices corresponding to a set
                       of particles for a potential angular spring. If there are fewer than 3
                       particles in a layer, the list will be empty.
    """
    n_total = plate.shape[0]
    if n_total % 2 != 0:
        raise ValueError("Expected an even number of particles (two layers).")

    M = n_total // 2  # number of particles per layer
    angular_sets = []

    # We need at least three points per layer to form an angle between segments.
    if M < 3:
        return angular_sets

    # For each set of three consecutive points in a layer, form a six-point set that covers
    # both the lower and upper layers.
    # For example, for i = 0, the six points are:
    # lower: [0, 1, 2] and upper: [M, M+1, M+2].
    # This grouping can later be used to compute the angle between the two consecutive segments.
    for i in range(M - 2):
        set_indices = (i, i + 1, i + 2, i + M, i + 1 + M, i + 2 + M)
        angular_sets.append(set_indices)

    return np.array(angular_sets)


def add_center_particle(positions: np.ndarray) -> np.ndarray:
    """
    Computes the geometrical center (centroid) of a set of particles in 3D space
    and appends a new particle at that center to the input array.

    Parameters:
        positions (np.ndarray): An (N, 3) array containing the x, y, z coordinates
                                of the particles.

    Returns:
        np.ndarray: A new array with shape (N+1, 3) including the new particle at the center.
    """
    # Compute the centroid of the points along each column (x, y, and z)
    center = np.mean(positions, axis=0)

    # Append the new center particle as a new row at the end
    new_positions = np.vstack([positions, center])

    return new_positions


def create_pairs(r: np.ndarray, add_center: bool = True) -> np.ndarray:
    """
    Create pairs of particle indices for either N5B8 (with center) or N4B5 (without center) structures.

    Parameters:
        r (np.ndarray): Particle positions.
                         - For N5B8: shape (N+1, 3) with last row as center
                         - For N4B5: shape (N, 3) (typically N=4)
        add_center (bool): Whether to include center connections (N5B8) or not (N4B5)

    Returns:
        np.ndarray: Array of particle index pairs representing connections
    """
    if add_center:
        n = r.shape[0] - 1  # Number of outer particles
        center_idx = r.shape[0] - 1
        center = r[center_idx, :]
        outer_positions = r[:n]

        # Compute angles relative to center and sort
        vectors = outer_positions - center
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indices = np.argsort(angles)
    else:
        n = r.shape[0]  # All particles are outer nodes
        # Compute centroid for sorting
        centroid = np.mean(r, axis=0)
        vectors = r - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_indices = np.argsort(angles)

    pairs = []

    # Create polygon edges
    for i in range(n):
        current = sorted_indices[i]
        next_idx = sorted_indices[(i + 1) % n]
        pairs.append((current, next_idx))

    # Handle center connections or diagonals
    if add_center:
        # Add connections to center particle
        for i in range(n):
            pairs.append((sorted_indices[i], center_idx))
    else:
        # For quadrilaterals (N=4), add appropriate diagonal
        if n == 4:
            # Get diagonals from sorted order
            diag_a = (sorted_indices[0], sorted_indices[2])
            diag_b = (sorted_indices[1], sorted_indices[3])

            # Calculate diagonal lengths
            len_a = np.linalg.norm(r[diag_a[0]] - r[diag_a[1]])
            len_b = np.linalg.norm(r[diag_b[0]] - r[diag_b[1]])

            # Include shorter diagonal, handle equality
            if len_a < len_b:
                pairs.append(diag_a)
            elif len_b < len_a:
                pairs.append(diag_b)
            else:
                # For equal lengths, include both and remove one duplicate later
                pairs.extend([diag_a, diag_b])
                # Remove duplicates while preserving order
                seen = set()
                unique_pairs = []
                for p in pairs:
                    sp = tuple(sorted(p))
                    if sp not in seen:
                        seen.add(sp)
                        unique_pairs.append(p)
                return np.array(unique_pairs)

    return np.array(pairs)


def create_triangle_pairs(r: np.ndarray, add_center: bool = True) -> np.ndarray:
    """
    Create quadruplets of particle indices with different structures based on add_center flag.

    With add_center=True (N5B8 structure):
        [left_outer, center, hinge, right_outer]

    With add_center=False (N4B5 structure):
        [left, first_diagonal_particle, second_diagonal_particle, right]

    Parameters:
        r (np.ndarray): Particle positions.
                        - For N5B8: shape (N+1, 3) with last row as center
                        - For N4B5: shape (N, 3) (typically N=4)
        add_center (bool): Whether to create center-based quadruplets (N5B8) or
                         diagonal-based quadruplets (N4B5)

    Returns:
        np.ndarray: Array of quadruplets with shape (M, 4)
    """
    if add_center:
        # Original N5B8 center-based quadruplets
        n = r.shape[0] - 1
        center_idx = r.shape[0] - 1
        center = r[center_idx, :]

        # Sort outer vertices by descending polar angle
        outer_indices = np.arange(n)
        angles = np.arctan2(r[:n, 1] - center[1], r[:n, 0] - center[0])
        sorted_order = outer_indices[np.argsort(angles)[::-1]]

        quadruplets = []
        for i in range(n):
            hinge = sorted_order[i]
            left = sorted_order[i - 1]
            right = sorted_order[(i + 1) % n]
            quadruplets.append([left, center_idx, hinge, right])
    else:
        # N4B5 diagonal-based quadruplets
        n = r.shape[0]

        # Compute centroid for sorting
        centroid = np.mean(r, axis=0)
        vectors = r - centroid
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        sorted_order = np.argsort(angles)[::-1]  # Descending order

        # For quadrilaterals (N=4), create diagonal-based quadruplets
        if n == 4:
            quadruplets = []
            for i in range(n):
                left = sorted_order[i - 1]
                current = sorted_order[i]
                right = sorted_order[(i + 1) % n]
                opposite = sorted_order[(i + 2) % n]

                # Create two possible quadruplets per vertex
                # First diagonal option
                # quad1 = [left, current, opposite, right]
                # # Second diagonal option (reverse)
                quad2 = [left, opposite, current, right]

                # Only add one to avoid duplicates
                # if i % 2 == 0:  # Alternate which diagonal we use
                #     quadruplets.append(quad1)
                # else:
                #     quadruplets.append(quad2)

                quadruplets.append(quad2)
        else:
            raise ValueError("N4B5 mode only supports quadrilaterals (4 nodes)")

    return np.array(quadruplets)


def create_hexagon():
    r = np.array(
        [
            [2.5, 0.8660254, 5],
            [2.0, 1.73205081, 5],
            [1.0, 1.73205081, 5],
            [0.5, 0.8660254, 5],
            [1.0, 0.0, 5],
            [2.0, 0.0, 5],
        ]
    )
    # r = np.array([[0, 0, 5], [0, 1, 5], [1, 0, 5],[1, 1, 5]])
    # r = np.array([[0, 0, 5], [1, 1, 5], [1, 0, 5]])
    new_hexagon_plate = add_center_particle(positions=r)

    pairs = create_pairs(r=new_hexagon_plate)
    quadruplets = create_triangle_pairs(r=new_hexagon_plate)
    return new_hexagon_plate, pairs, quadruplets


def signle_fold(N, a, phi1):
    # Generate initial rod structure
    r = set_rod_3d(N, a, phi1)
    # Update Z coordinates
    r[:, 2] -= 10
    # Add a particle in the middle of the rod
    r_new = np.zeros((len(r) + 1, 3))

    r_new[0] = r[0]
    # r_new[0][-1] += (a)
    # r_new[0][-1] += a
    r_new[1] = r[1]
    r_new[1, 2] = r[1, 2] + a / 2
    r_new[2] = r[1]
    r_new[2, 2] = r[1, 2] - a / 2

    r_new[3] = r[2]
    # r_new[3][-1] += (a)
    # r_new[3][-1] += a
    r = r_new.copy()
    print(r)

    pairlist = np.array([[0, 1], [0, 2], [1, 2], [2, 3], [1, 3]])
    triplist = np.array([[0, 1, 2, 3]])

    return r, pairlist, triplist


def generate_ori(nx, ny, dihedral_angle, base_length):
    """

    Parameters:
    - nx (int): Number of cells along the x-axis.
    - ny (int): Number of cells along the y-axis.
    - dihedral_angle (float): Dihedral angle between adjacent planes in radians.
    - base_length (float): Base length of the edges in the x and y directions.

    Returns:
    - list of planes: Each plane is a list of 3D vertices (numpy arrays).
    """
    # Calculate the height of the fold based on the dihedral angle
    h = 0.5 * np.sqrt((1 / np.cos(dihedral_angle)) - 1) * base_length

    # Create a grid of vertices in 3D space
    vertices = []
    planes = []

    # Generate vertex grid
    for i in range(nx + 1):
        row = []
        for j in range(ny + 1):
            x = i * base_length
            y = j * base_length
            # Apply folding pattern (alternating direction for Miura-ori)
            if (i + j) % 2 == 0:
                z = h
            else:
                z = -h
            row.append(np.array([x, y, z]))
        vertices.append(row)

    # Generate planes (each cell is split into two triangles)
    for i in range(nx):
        for j in range(ny):
            # Get the four vertices of the current cell
            v00 = vertices[i][j]
            v10 = vertices[i + 1][j]
            v11 = vertices[i + 1][j + 1]
            v01 = vertices[i][j + 1]

            plane = np.array([v00, v10, v11, v01])
            planes.append(plane)

    return planes


def generate_miura_ori(nx, ny, a, b, gamma, theta):
    """
    Generates Miura-ori vertices with configuration controlled by
    (gamma, theta) as in Schenk & Guest, PNAS 2013.

    Parameters
    ----------
    nx, ny : int
        Number of Miura cells along x and y (in "unit cells").
    a, b : float
        Panel edge lengths of the parallelogram facet.
        'a' is along x, 'b' along y in the flat reference.
    gamma : float
        Acute angle of the parallelogram facet, in radians (γ in the paper).
    theta : float
        Dihedral fold angle between facets and the xy-plane, in radians (θ).

    Returns
    -------
    planes : list of (4,3) float arrays
        Each element is a quadrilateral panel (4 vertices in 3D).
    """
    # ---- Precompute trig (notation matches paper) ----
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_gamma = np.sin(gamma)
    cos_gamma = np.cos(gamma)
    tan_gamma = np.tan(gamma)

    # Robust handling near θ = π/2 (cosθ -> 0)
    if np.isclose(cos_theta, 0.0, atol=1e-12):
        # Analytic limit as θ -> π/2 using Eqs. (1)–(4):
        # H = a sinθ sinγ  -> a * sinγ  (since sinθ -> 1)
        # L = a sqrt(1 - sin²θ sin²γ) -> a * cosγ
        # S -> 0, V -> b
        s = 0.0  # S
        v = b  # V
        l = a * cos_gamma  # L
        h = a * sin_gamma * sin_theta  # H (≈ a * sinγ)
    else:
        # Eqs. (1)–(4) from Schenk & Guest:
        # H = a sinθ sinγ
        # S = b cosθ tanγ / sqrt(1 + cos²θ tan²γ)
        # L = a sqrt(1 - sin²θ sin²γ)
        # V = b / sqrt(1 + cos²θ tan²γ)
        denom_s = np.hypot(1.0, cos_theta * tan_gamma)  # sqrt(1 + cos²θ tan²γ)

        s = (b * cos_theta * tan_gamma) / denom_s
        v = b / denom_s

        inside = 1.0 - (sin_theta * sin_gamma) ** 2
        inside = np.clip(inside, 0.0, 1.0)
        l = a * np.sqrt(inside)

        h = a * sin_theta * sin_gamma

    # Number of vertex lines
    numx = 2 * nx
    numy = 2 * ny

    # ---- Generate grid with column-wise shifts ----
    x_vals = s * np.arange(numx + 1, dtype=float)
    y_vals = l * np.arange(numy + 1, dtype=float)

    X = np.tile(x_vals, (numy + 1, 1))
    Y = np.repeat(y_vals[:, np.newaxis], numx + 1, axis=1)

    # Alternate columns shifted in y by V
    Y[:, 1::2] += v

    # Alternate rows at height H
    Z = np.zeros((numy + 1, numx + 1), dtype=float)
    Z[1::2, :] = h

    # Flatten vertices (same convention as your original code)
    vertices = np.column_stack((X.T.ravel(), Y.T.ravel(), Z.T.ravel()))

    # ---- Build quadrilateral panels ----
    planes = []
    col_height = numy + 1

    for i in range(numx):
        base0 = i * col_height
        base1 = (i + 1) * col_height

        j_idx = np.arange(numy, dtype=int)
        n0 = base0 + j_idx
        n1 = base1 + j_idx

        v0 = vertices[n0]
        v1 = vertices[n1]
        v2 = vertices[n1 + 1]
        v3 = vertices[n0 + 1]

        block = np.stack((v0, v1, v2, v3), axis=1)  # (numy, 4, 3)
        planes.extend(list(block))

    return planes


def generate_z_fold(nx, ny, a, b, theta):
    """
    Generate vertices for a classical Z-fold (corrugated) pattern.

    Panels are rectangles of size (b along one axis) x (a along the
    other). Final coordinates swap OX/OY so OX spans length `a` and OY
    spans length `b`. Adjacent panels alternate their tilt about the
    (swapped) OY axis so that the dihedral angle between neighbors is
    `theta`.

    Parameters
    ----------
    nx, ny : int
        Number of panels along x and y.
    a : float
        Panel size along the y-direction.
    b : float
        Panel size along the x-direction (projected width).
    theta : float
        Dihedral angle between adjacent panels (radians).

    Returns
    -------
    planes : list of (4,3) float arrays
        Each element is a rectangular panel (4 vertices in 3D).
    """
    # Use complementary split so smaller theta => larger tilt (more closed)
    alpha = 0.5 * (np.pi - theta)
    # Ensure the true edge length along the tilted direction equals b
    proj = b * np.cos(alpha)
    dz = b * np.sin(alpha)

    num_cols = nx + 1
    num_rows = ny + 1

    x_vals = proj * np.arange(num_cols, dtype=float)
    y_vals = a * np.arange(num_rows, dtype=float)

    # Alternate height offsets to enforce alternating tilt (+alpha / -alpha)
    z_vals = np.zeros(num_cols, dtype=float)
    for i in range(1, num_cols):
        sign = 1.0 if (i - 1) % 2 == 0 else -1.0
        z_vals[i] = z_vals[i - 1] + sign * dz

    X = np.tile(x_vals, (num_rows, 1))
    Y = np.repeat(y_vals[:, np.newaxis], num_cols, axis=1)
    Z = np.repeat(z_vals[np.newaxis, :], num_rows, axis=0)

    # Swap OX/OY in the returned geometry
    vertices = np.column_stack((Y.T.ravel(), X.T.ravel(), Z.T.ravel()))

    planes = []
    col_height = num_rows

    for i in range(num_cols - 1):
        base0 = i * col_height
        base1 = (i + 1) * col_height

        j_idx = np.arange(num_rows - 1, dtype=int)
        n0 = base0 + j_idx
        n1 = base1 + j_idx

        v0 = vertices[n0]
        v1 = vertices[n1]
        v2 = vertices[n1 + 1]
        v3 = vertices[n0 + 1]

        block = np.stack((v0, v1, v2, v3), axis=1)
        planes.extend(list(block))

    return planes


def _merge_and_deduplicate(
    new_planes,
    plane_ids,
    pairs_list,
    quadruplets_bend,
    triangles=None,
    panel_quads=None,
):
    """
    Helper to merge lists -> arrays, remove duplicate vertices, and
    renumber indices consistently.

    Extended to also handle:
      - triangles : (T, 3) int array
      - panel_quads : (K, 4) int array   (parallelogram panels, no center)
    """
    # Concatenate per-panel pieces
    new_planes = np.concatenate(new_planes, axis=0)
    plane_ids = np.concatenate(plane_ids, axis=0)
    pairs_list = np.concatenate(pairs_list, axis=0)
    quadruplets_bend = np.concatenate(quadruplets_bend, axis=0)

    if triangles is not None and len(triangles) > 0:
        triangles = np.concatenate(triangles, axis=0)
    else:
        triangles = None

    if panel_quads is not None and len(panel_quads) > 0:
        panel_quads = np.concatenate(panel_quads, axis=0)
    else:
        panel_quads = None

    # --- Deduplicate vertices efficiently (O(N log N) instead of O(N^2)) ---
    # np.unique sorts the unique rows; we re-order to preserve "first occurrence" order
    # so existing index conventions remain as stable as possible.
    _, unique_idx, inverse = np.unique(
        np.ascontiguousarray(new_planes),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if unique_idx.size != new_planes.shape[0]:
        order = np.argsort(unique_idx)
        stable_unique_idx = unique_idx[order]

        # Map from np.unique's row order -> stable (first occurrence) order.
        remap = np.empty(order.shape[0], dtype=int)
        remap[order] = np.arange(order.shape[0])
        inverse = remap[inverse]

        new_planes = new_planes[stable_unique_idx]

        pairs_list = inverse[pairs_list]
        plane_ids = inverse[plane_ids]
        quadruplets_bend = inverse[quadruplets_bend]
        if triangles is not None:
            triangles = inverse[triangles]
        if panel_quads is not None:
            panel_quads = inverse[panel_quads]

    # --- Drop duplicate connectivity items ---
    # Pairs: treat edges as undirected, remove self-edges.
    if pairs_list.size:
        seen_pairs = set()
        unique_pairs = []
        for a, b in pairs_list:
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in seen_pairs:
                seen_pairs.add(key)
                unique_pairs.append((a, b))
        pairs_list = np.asarray(unique_pairs, dtype=pairs_list.dtype)

    # Triangles: remove duplicates regardless of orientation.
    if triangles is not None and triangles.size:
        seen_tri = set()
        unique_tri = []
        for tri in triangles:
            key = tuple(sorted(map(int, tri)))
            if key not in seen_tri:
                seen_tri.add(key)
                unique_tri.append(tri)
        triangles = np.asarray(unique_tri, dtype=int)

    # Panel quads: remove duplicates regardless of orientation.
    if panel_quads is not None and panel_quads.size:
        seen_quad = set()
        unique_quad = []
        for quad in panel_quads:
            key = tuple(sorted(map(int, quad)))
            if key not in seen_quad:
                seen_quad.add(key)
                unique_quad.append(quad)
        panel_quads = np.asarray(unique_quad, dtype=int)

    return new_planes, plane_ids, pairs_list, quadruplets_bend, triangles, panel_quads


def _group_panels_by_cell(panel_quads, nx, ny):
    """
    Group panel corner ids by Miura cell.

    Returns
    -------
    list of list
        Each inner list contains the 4 panels (corner id lists) that form a
        single Miura cell. Ordering follows the generator convention:
        column-major over panels, with rows varying fastest.
    """
    quads = np.asarray(panel_quads, dtype=int)
    if quads.size == 0:
        return []

    panel_rows = 2 * ny
    panel_cols = 2 * nx
    expected = panel_rows * panel_cols
    if quads.shape[0] < expected:
        raise ValueError(
            f"panel_quads has {quads.shape[0]} panels, expected at least {expected} for grid {nx}x{ny}."
        )

    cells = []
    for cx in range(nx):
        for cy in range(ny):
            panel_ids = [
                (2 * cx + dx) * panel_rows + (2 * cy + dy)
                for dx in (0, 1)
                for dy in (0, 1)
            ]
            cells.append([quads[idx].tolist() for idx in panel_ids])

    return cells


def asseamble_miura_ori(
    nx=2,
    ny=2,
    a=1.0,
    b=1.0,
    gamma=np.pi / 3,  # parallelogram angle γ
    theta=np.pi / 6,  # fold angle θ
    add_center=True,
):
    """
    Assemble Miura-ori structure controlled by (gamma, theta) as in
    Schenk & Guest (2013).
    Returns nodal coordinates, connectivity, and per-cell panel grouping.
    """
    # NOTE: generate_miura_ori now expects (a, b, gamma, theta)
    planes = generate_miura_ori(nx, ny, a, b, gamma, theta)

    # Containers for per-plane data
    new_planes = []
    plane_ids = []
    pairs_list = []
    quadruplets_bend = []
    quadruplets_fold = []
    triangles = []
    panel_quads = []  # NEW

    if add_center:
        for i, plane in enumerate(planes):
            # plane: (4,3), corners only
            # Add center point to each plane
            plane_with_center = add_center_particle(positions=plane)

            # Plane number increment
            increment = i * plane_with_center.shape[0]

            # Select ids for each plane (including center)
            basis = np.arange(plane_with_center.shape[0], dtype=int)
            current_plane_ids = np.expand_dims(basis + increment, axis=0)

            # Create pairs and update
            pairs = create_pairs(r=plane_with_center) + increment

            # Create quadruplets (bending)
            quadruplets = create_triangle_pairs(r=plane_with_center) + increment

            # ---- Create triangles for this panel (center as last node) ----
            n_local = plane_with_center.shape[0]
            center_local = n_local - 1  # by construction center is last
            local_tris = np.array(
                [
                    [0, 1, center_local],
                    [1, 2, center_local],
                    [2, 3, center_local],
                    [3, 0, center_local],
                ],
                dtype=int,
            )
            tris = local_tris + increment

            # ---- Original parallelogram panel ids (no center) ----
            # Locally, corners are indices [0,1,2,3]
            local_quad = np.array([[0, 1, 2, 3]], dtype=int)
            quad = local_quad + increment

            # Store
            new_planes.append(plane_with_center)
            plane_ids.append(current_plane_ids)
            pairs_list.append(pairs)
            quadruplets_bend.append(quadruplets)
            triangles.append(tris)
            panel_quads.append(quad)

        # Merge and deduplicate (also updates triangles & panel_quads)
        (
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        ) = _merge_and_deduplicate(
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        )

        # Generate folding quadruplets between planes
        if plane_ids.shape[0] != 1:
            for i in range(plane_ids.shape[0]):
                plane_i = plane_ids[i]
                for j in range(i + 1, plane_ids.shape[0]):
                    plane_j = plane_ids[j]

                    intersection = np.intersect1d(plane_i, plane_j)
                    if np.any(intersection) and intersection.shape[0] > 1:
                        quadruplet = np.concatenate(
                            [plane_i[-1:], intersection, plane_j[-1:]]
                        )
                        quadruplet = np.expand_dims(quadruplet, axis=0)
                        quadruplets_fold.append(quadruplet)

            if len(quadruplets_fold) > 0:
                quadruplets_fold = np.concatenate(quadruplets_fold, axis=0)
            else:
                quadruplets_fold = np.zeros((0, 4), dtype=int)
        else:
            quadruplets_fold = np.zeros((0, 4), dtype=int)

    else:
        # No center particles
        for i, plane in enumerate(planes):
            # plane: (4,3), corners only

            increment = i * plane.shape[0]

            # Local indices of the quad corners
            i0, i1, i2, i3 = 0, 1, 2, 3

            # ---- Basic edge pairs from create_pairs ----
            pairs = create_pairs(r=plane, add_center=False) + increment

            # ---- ADD BOTH DIAGONALS EXPLICITLY ----
            diag12 = np.array([[i0 + increment, i2 + increment]])
            diag23 = np.array([[i1 + increment, i3 + increment]])
            pairs = np.vstack([pairs, diag12, diag23])

            # Create quadruplets (bending)
            quadruplets = create_triangle_pairs(r=plane, add_center=False) + increment

            # Triangulate quad into 2 triangles
            local_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
            tris = local_tris + increment

            # Original parallelogram panel ids (0,1,2,3)
            local_quad = np.array([[0, 1, 2, 3]], dtype=int)
            quad = local_quad + increment

            # Store
            new_planes.append(plane)
            plane_ids.append(
                np.expand_dims(np.arange(plane.shape[0]) + increment, axis=0)
            )
            pairs_list.append(pairs)
            quadruplets_bend.append(quadruplets)
            triangles.append(tris)
            panel_quads.append(quad)

        # Merge and deduplicate
        (
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        ) = _merge_and_deduplicate(
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        )
        quadruplets_fold = np.zeros((0, 4), dtype=int)

    # Ensure arrays exist
    if triangles is None:
        triangles = np.zeros((0, 3), dtype=int)
    if panel_quads is None:
        panel_quads = np.zeros((0, 4), dtype=int)

    panels_by_cell = _group_panels_by_cell(panel_quads=panel_quads, nx=nx, ny=ny)

    return (
        new_planes,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    )


def _group_zfold_panels_by_cell(panel_quads, nx, ny):
    """
    Group Z-fold panels by (cx, cy) location in a simple nx-by-ny grid.

    Returns
    -------
    list of list
        Each inner list contains the single panel for that grid cell.
        Ordering is column-major with rows varying fastest, matching
        the generator convention.
    """
    quads = np.asarray(panel_quads, dtype=int)
    if quads.size == 0:
        return []

    expected = nx * ny
    if quads.shape[0] < expected:
        raise ValueError(
            f"panel_quads has {quads.shape[0]} panels, expected at least {expected} for grid {nx}x{ny}."
        )

    grouped = []
    for cx in range(nx):
        for cy in range(ny):
            idx = cx * ny + cy
            grouped.append([quads[idx].tolist()])

    return grouped


def asseamble_z_fold(
    nx=2,
    ny=2,
    a=1.0,
    b=1.0,
    theta=np.pi / 6,
    add_center=True,
):
    """
    Assemble a classical Z-fold corrugated structure.

    Returns nodal coordinates, connectivity, and per-cell panel grouping,
    mirroring the output format of `asseamble_miura_ori`.
    """
    planes = generate_z_fold(nx=nx, ny=ny, a=a, b=b, theta=theta)

    new_planes = []
    plane_ids = []
    pairs_list = []
    quadruplets_bend = []
    quadruplets_fold = []
    triangles = []
    panel_quads = []

    if add_center:
        for i, plane in enumerate(planes):
            plane_with_center = add_center_particle(positions=plane)
            increment = i * plane_with_center.shape[0]

            basis = np.arange(plane_with_center.shape[0], dtype=int)
            current_plane_ids = np.expand_dims(basis + increment, axis=0)

            pairs = create_pairs(r=plane_with_center) + increment
            quadruplets = create_triangle_pairs(r=plane_with_center) + increment

            n_local = plane_with_center.shape[0]
            center_local = n_local - 1
            local_tris = np.array(
                [
                    [0, 1, center_local],
                    [1, 2, center_local],
                    [2, 3, center_local],
                    [3, 0, center_local],
                ],
                dtype=int,
            )
            tris = local_tris + increment

            local_quad = np.array([[0, 1, 2, 3]], dtype=int)
            quad = local_quad + increment

            new_planes.append(plane_with_center)
            plane_ids.append(current_plane_ids)
            pairs_list.append(pairs)
            quadruplets_bend.append(quadruplets)
            triangles.append(tris)
            panel_quads.append(quad)

        (
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        ) = _merge_and_deduplicate(
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        )

        if plane_ids.shape[0] != 1:
            for i in range(plane_ids.shape[0]):
                plane_i = plane_ids[i]
                for j in range(i + 1, plane_ids.shape[0]):
                    plane_j = plane_ids[j]

                    intersection = np.intersect1d(plane_i, plane_j)
                    if np.any(intersection) and intersection.shape[0] > 1:
                        quadruplet = np.concatenate(
                            [plane_i[-1:], intersection, plane_j[-1:]]
                        )
                        quadruplet = np.expand_dims(quadruplet, axis=0)
                        quadruplets_fold.append(quadruplet)

            if len(quadruplets_fold) > 0:
                quadruplets_fold = np.concatenate(quadruplets_fold, axis=0)
            else:
                quadruplets_fold = np.zeros((0, 4), dtype=int)
        else:
            quadruplets_fold = np.zeros((0, 4), dtype=int)

    else:
        for i, plane in enumerate(planes):
            increment = i * plane.shape[0]

            i0, i1, i2, i3 = 0, 1, 2, 3

            pairs = create_pairs(r=plane, add_center=False) + increment

            diag12 = np.array([[i0 + increment, i2 + increment]])
            diag23 = np.array([[i1 + increment, i3 + increment]])
            pairs = np.vstack([pairs, diag12, diag23])

            quadruplets = create_triangle_pairs(r=plane, add_center=False) + increment

            local_tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
            tris = local_tris + increment

            local_quad = np.array([[0, 1, 2, 3]], dtype=int)
            quad = local_quad + increment

            new_planes.append(plane)
            plane_ids.append(
                np.expand_dims(np.arange(plane.shape[0]) + increment, axis=0)
            )
            pairs_list.append(pairs)
            quadruplets_bend.append(quadruplets)
            triangles.append(tris)
            panel_quads.append(quad)

        (
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        ) = _merge_and_deduplicate(
            new_planes,
            plane_ids,
            pairs_list,
            quadruplets_bend,
            triangles,
            panel_quads,
        )
        quadruplets_fold = np.zeros((0, 4), dtype=int)

    if triangles is None:
        triangles = np.zeros((0, 3), dtype=int)
    if panel_quads is None:
        panel_quads = np.zeros((0, 4), dtype=int)

    panels_by_cell = _group_zfold_panels_by_cell(panel_quads=panel_quads, nx=nx, ny=ny)

    return (
        new_planes,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    )


def compute_triangle_areas(vertices, triangles):
    """
    Compute the area of each triangle in a surface mesh.

    Parameters
    ----------
    vertices : (N, 3) array
        3D coordinates of all points (e.g. new_planes).
    triangles : (T, 3) array of int
        Each row gives the 3 vertex indices of one triangle.

    Returns
    -------
    areas : (T,) array of float
        Surface area of each triangle.
    total_area : float
        Sum of all triangle areas.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    # Extract triangle corner coordinates
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    # Compute cross product of edge vectors
    cross_prod = np.cross(v1 - v0, v2 - v0)

    # Area = 0.5 * |cross|
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    return areas, np.sum(areas)


def compute_panel_areas(coordinates, panel_quads):
    """
    Compute the area of each parallelogram panel given its 4 corner ids.

    Parameters
    ----------
    coordinates : (N, 3) array
        Global 3D coordinates of all nodes.
    panel_quads : (K, 4) array of int
        Each row gives the vertex ids of one panel: [i0, i1, i2, i3].

    Returns
    -------
    areas : (K,) array
        The area of each panel.
    """
    coords = np.asarray(coordinates, float)
    quads = np.asarray(panel_quads, int)

    areas = np.zeros(quads.shape[0])

    for i, (i0, i1, i2, i3) in enumerate(quads):
        # Triangle 1: (i0, i1, i2)
        v0 = coords[i0]
        v1 = coords[i1]
        v2 = coords[i2]

        tri1_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Triangle 2: (i0, i2, i3)
        v3 = coords[i3]

        tri2_area = 0.5 * np.linalg.norm(np.cross(v2 - v0, v3 - v0))

        # Total parallelogram (quad) area
        areas[i] = tri1_area + tri2_area

    return areas


def generate_center_nodes_from_triangles(
    triangles: np.ndarray,
    panel_quads: np.ndarray,
) -> np.ndarray:
    """
    Generate center_nodes array from panel triangles.

    Assumption:
        Each panel quad corresponds to 4 consecutive triangles:
            [v0, v1, center]
            [v1, v2, center]
            [v2, v3, center]
            [v3, v0, center]

    Returns:
        center_nodes in format:
            np.array([64, 65, 66, ...])
    """

    if panel_quads is None or len(panel_quads) == 0:
        return np.zeros(0, dtype=int)

    if triangles is None or len(triangles) == 0:
        raise ValueError("triangles is empty, so center nodes cannot be recovered.")

    n_panels = panel_quads.shape[0]

    if triangles.shape[0] != 4 * n_panels:
        raise ValueError(
            f"Expected 4 triangles per panel, but got "
            f"{triangles.shape[0]} triangles for {n_panels} panels."
        )

    center_nodes = np.zeros(n_panels, dtype=int)

    for panel_id in range(n_panels):
        quad = panel_quads[panel_id]
        quad_set = set(quad.tolist())

        panel_triangles = triangles[4 * panel_id : 4 * (panel_id + 1)]

        possible_centers = []

        for tri in panel_triangles:
            for node in tri:
                if node not in quad_set:
                    possible_centers.append(node)

        unique_centers = np.unique(possible_centers)

        if unique_centers.size != 1:
            raise ValueError(
                f"Panel {panel_id}: expected one center node, "
                f"but found {unique_centers}."
            )

        center_nodes[panel_id] = unique_centers[0]

    return center_nodes


if __name__ == "__main__":
    # Example usage:
    a = 1.2
    b = 1.5
    gamma = np.pi / 3
    theta = np.pi / 3  # Slightly above 90 degrees
    nx = 2
    ny = 2

    (
        new_planes,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    ) = asseamble_miura_ori(
        nx=nx, ny=ny, a=a, b=b, gamma=gamma, theta=theta, add_center=True
    )
    # Compute triangle areas
    areas, total_area = compute_triangle_areas(vertices=new_planes, triangles=triangles)

    a = 1
    b = 5
    theta = np.pi / 2
    nx = 6
    ny = 1
    (
        new_planes,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    ) = asseamble_z_fold(
        nx=nx,
        ny=ny,
        a=a,
        b=b,
        theta=theta,
        add_center=True,
    )

    areas, total_area = compute_triangle_areas(vertices=new_planes, triangles=triangles)

    nz = 3
    a = 1.0
    alpha = np.pi / 6
    beta = np.pi / 6

    plot_particles_with_forces(
        r=new_planes,
        forces=np.zeros_like(new_planes),
        pairlist=pairs_list,
        angular_sets=None,
        force_scale=1.0,
    )
    rod = [new_planes, "Type_1"]  # Rod structure and its type
    create_text_bond_file_2d(
        directory_2output="/WORK/Origami-Simulations/3D",
        stress_lines=np.zeros_like(pairs_list),
        pairs=pairs_list,
        timestep=0,
    )
    Output(rod, 0, "/WORK/Origami-Simulations/3D")

# %%
