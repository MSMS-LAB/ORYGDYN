# %%
import numpy as np
from src.plots_and_results import (
    plot_particles_with_forces,
    create_text_bond_file_2d,
    Output,
)


def asseamble_kresling(
    nz: int = 2,
    a: float = 1.5,
    edges: int = 6,
    height: float | None = None,
    phi: float = 0.0,
    twist: float = 0.0,
    add_midpoints: bool = True,
):
    """
    Assemble a Kresling-like triangulated prism.

    Returns
    -------
    coordinates : (N,3) float ndarray
    pairs_list : (P,2) int ndarray
    quadruplets_bend : (Qb,4) int ndarray
    quadruplets_fold : (Qf,4) int ndarray
    triangles : (T,3) int ndarray
    panel_quads : (K,4) int ndarray   (optional bookkeeping; empty here)
    panels_by_cell : list            (optional bookkeeping; empty here)
    """
    nz = int(nz)
    edges = int(edges)
    if height is None:
        height = a * 1.2

    # -------------------- Nodes: regular polygon rings --------------------
    angles0 = 2.0 * np.pi * (np.arange(edges, dtype=float) / edges)
    R = a / (2.0 * np.sin(np.pi / edges))

    layer_phi = phi + twist * np.arange(nz, dtype=float)
    angles = angles0[None, :] + layer_phi[:, None]

    x = R * np.cos(angles)
    y = R * np.sin(angles)
    z = (height * np.arange(nz, dtype=float))[:, None]
    z = np.broadcast_to(z, x.shape)

    coordinates = np.stack((x, y, z), axis=2).reshape(-1, 3)

    # -------------------- Panels (quads) per interface --------------------
    # For each interface k -> k+1, we define "panel quads":
    # corners: [L_n, L_{n+1}, U_{n+2}, U_{n+1}]
    # and triangulate them:
    # - N4B5: 2 triangles sharing diagonal (L_n, U_{n+2}) -> 1 bending hinge per panel
    # - N5B8: 4 triangles via center node -> 4 bending hinges per panel
    panel_quads = []
    triangles = []
    bend_edge_keys = set()  # undirected edge keys that must be classified as "bend"

    # Keep track of centers if add_midpoints (one per panel)
    center_ids = []

    for k in range(nz - 1):
        baseL = k * edges
        baseU = (k + 1) * edges

        n = np.arange(edges, dtype=int)
        L_n = baseL + n
        L_np1 = baseL + ((n + 1) % edges)
        U_np1 = baseU + ((n + 1) % edges)
        U_np2 = baseU + ((n + 2) % edges)

        # Store quads (bookkeeping)
        # quad = [L_n, L_{n+1}, U_{n+2}, U_{n+1}]
        quads_k = np.stack((L_n, L_np1, U_np2, U_np1), axis=1)
        panel_quads.append(quads_k)

        if not add_midpoints:
            # --- N4B5 style triangulation: 2 tris sharing diagonal (L_n, U_np2)
            # triA: (L_n, U_np1, U_np2)
            # triB: (L_n, L_np1, U_np2)
            triA = np.stack((L_n, U_np1, U_np2), axis=1)
            triB = np.stack((L_n, L_np1, U_np2), axis=1)
            triangles.append(triA)
            triangles.append(triB)

            # Mark the shared diagonal as "bend" hinge
            for e0, e1 in zip(L_n.tolist(), U_np2.tolist()):
                key = (e0, e1) if e0 < e1 else (e1, e0)
                bend_edge_keys.add(key)

        else:
            # --- N5B8 style: add one center per quad, make 4 triangles
            # Center position: centroid of the 4 corners (robust; no intersection needed).
            p0 = coordinates[L_n]  # (edges,3)
            p1 = coordinates[L_np1]  # (edges,3)
            p2 = coordinates[U_np2]  # (edges,3)
            p3 = coordinates[U_np1]  # (edges,3)

            d02 = np.einsum(
                "ij,ij->i", p2 - p0, p2 - p0
            )  # squared length of (L_n -> U_np2)
            d13 = np.einsum(
                "ij,ij->i", p3 - p1, p3 - p1
            )  # squared length of (L_np1 -> U_np1)

            use_02 = d02 >= d13  # tie -> choose (0,2)

            mid_02 = 0.5 * (p0 + p2)
            mid_13 = 0.5 * (p1 + p3)

            centers_pos = np.where(use_02[:, None], mid_02, mid_13)  # (edges,3)
            c0 = coordinates.shape[0]
            c_ids = c0 + np.arange(edges, dtype=int)
            coordinates = np.vstack((coordinates, centers_pos))
            center_ids.append(c_ids)

            # 4 triangles around the center:
            # (L_n, L_np1, C), (L_np1, U_np2, C), (U_np2, U_np1, C), (U_np1, L_n, C)
            t0 = np.stack((L_n, L_np1, c_ids), axis=1)
            t1 = np.stack((L_np1, U_np2, c_ids), axis=1)
            t2 = np.stack((U_np2, U_np1, c_ids), axis=1)
            t3 = np.stack((U_np1, L_n, c_ids), axis=1)
            triangles.append(t0)
            triangles.append(t1)
            triangles.append(t2)
            triangles.append(t3)

            # In this triangulation, the INTERNAL shared edges are (C, each corner),
            # giving exactly 4 bending hinges per panel (one per corner).
            # Mark all (C, corner) as bend edges.
            corners = [L_n, L_np1, U_np2, U_np1]
            for corner_arr in corners:
                for cc, vv in zip(c_ids.tolist(), corner_arr.tolist()):
                    key = (cc, vv) if cc < vv else (vv, cc)
                    bend_edge_keys.add(key)

    if len(panel_quads) == 0:
        # Degenerate case nz<2
        return (
            coordinates,
            np.zeros((0, 2), dtype=int),
            np.zeros((0, 4), dtype=int),
            np.zeros((0, 4), dtype=int),
            np.zeros((0, 3), dtype=int),
            np.zeros((0, 4), dtype=int),
            [],
        )

    panel_quads = np.vstack(panel_quads).astype(int, copy=False)
    triangles = np.vstack(triangles).astype(int, copy=False)

    # -------------------- Unique pairs_list from triangles --------------------
    # Build undirected unique edges from triangle list.
    # This includes ring edges, inter-layer edges, and internal diagonals/center-edges.
    edges_raw = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges_norm = np.sort(edges_raw, axis=1)
    # unique undirected
    _, uniq_idx = np.unique(edges_norm, axis=0, return_index=True)
    pairs_list = edges_norm[np.sort(uniq_idx)].astype(int, copy=False)

    # -------------------- Build hinge quadruplets from triangle adjacency --------------------
    # Map each undirected edge -> list of adjacent triangle indices.
    T = triangles.shape[0]
    tri_edges = np.stack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=1,
    )  # (T,3,2)

    tri_edges_norm = np.sort(tri_edges, axis=2)  # (T,3,2)
    edge_map = {}  # (u,v) -> list of triangle ids
    for t in range(T):
        for e in range(3):
            u, v = int(tri_edges_norm[t, e, 0]), int(tri_edges_norm[t, e, 1])
            edge_map.setdefault((u, v), []).append(t)

    quadruplets_bend = []
    quadruplets_fold = []

    for (u, v), tris in edge_map.items():
        if len(tris) != 2:
            # boundary edge => no hinge
            continue

        t0, t1 = tris
        tri0 = triangles[t0]
        tri1 = triangles[t1]

        # Find the "opposite" vertex in each triangle (not equal to u or v)
        opp0 = int(tri0[(tri0 != u) & (tri0 != v)][0])
        opp1 = int(tri1[(tri1 != u) & (tri1 != v)][0])

        quad = [opp0, u, v, opp1]  # consistent with your hinge convention

        if (u, v) in bend_edge_keys:
            quadruplets_bend.append(quad)
        else:
            quadruplets_fold.append(quad)

    quadruplets_bend = (
        np.asarray(quadruplets_bend, dtype=int)
        if len(quadruplets_bend)
        else np.zeros((0, 4), dtype=int)
    )
    quadruplets_fold = (
        np.asarray(quadruplets_fold, dtype=int)
        if len(quadruplets_fold)
        else np.zeros((0, 4), dtype=int)
    )

    # (Optional) remove duplicates in quadruplets, independent of orientation
    def _dedup_quads(Q):
        if Q.size == 0:
            return Q
        seen = set()
        out = []
        for q in Q:
            key = tuple(sorted(map(int, q)))
            if key not in seen:
                seen.add(key)
                out.append(q)
        return np.asarray(out, dtype=int)

    quadruplets_bend = _dedup_quads(quadruplets_bend)
    quadruplets_fold = _dedup_quads(quadruplets_fold)

    # Keep API compatibility with Miura/Z-fold assemblers
    panels_by_cell = []  # you can later group per (k,n) panel if you want

    return (
        coordinates,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    )


def asseamble_kresling_v2(
    nz: int = 2,
    a: float = 1.5,
    edges: int = 6,
    height: float | None = None,
    phi: float = 0.0,
    twist: float = 0.0,
    add_midpoints: bool = True,
):
    """
    Assemble a Kresling-like triangulated prism.

    Returns
    -------
    coordinates : (N,3) float ndarray
    pairs_list : (P,2) int ndarray
    quadruplets_bend : (Qb,4) int ndarray
    quadruplets_fold : (Qf,4) int ndarray
    triangles : (T,3) int ndarray
    panel_quads : (K,4) int ndarray
    panels_by_cell : list

    Notes
    -----
    The quadrilateral panel convention is:

        p0 = L_n
        p1 = L_{n+1}
        p2 = U_{n+2}
        p3 = U_{n+1}

    The two possible panel diagonals are:

        d02 : L_n     -> U_{n+2}
        d13 : L_{n+1} -> U_{n+1}

    In this version, the LONGEST diagonal is used as the main internal
    diagonal, but it is classified as a FOLDING line, not a bending line.

    If add_midpoints=False:
        the panel is triangulated along the longest diagonal, and that
        shared edge is classified as folding.

    If add_midpoints=True:
        the center point is placed on the longest diagonal. The two
        center-to-corner half-edges lying on the longest diagonal are
        classified as folding. The remaining two center-to-corner internal
        lines are classified as bending.
    """
    nz = int(nz)
    edges = int(edges)

    if height is None:
        height = a * 1.2

    # -------------------- Nodes: regular polygon rings --------------------
    angles0 = 2.0 * np.pi * (np.arange(edges, dtype=float) / edges)
    R = a / (2.0 * np.sin(np.pi / edges))

    layer_phi = phi + twist * np.arange(nz, dtype=float)
    angles = angles0[None, :] + layer_phi[:, None]

    x = R * np.cos(angles)
    y = R * np.sin(angles)
    z = (height * np.arange(nz, dtype=float))[:, None]
    z = np.broadcast_to(z, x.shape)

    coordinates = np.stack((x, y, z), axis=2).reshape(-1, 3)

    # -------------------- Panels per interface --------------------
    panel_quads = []
    triangles = []

    # Edges added here are classified as bending hinges.
    # Internal shared triangle edges not added here are classified as folding hinges.
    bend_edge_keys = set()

    center_ids = []

    for k in range(nz - 1):
        baseL = k * edges
        baseU = (k + 1) * edges

        n = np.arange(edges, dtype=int)

        L_n = baseL + n
        L_np1 = baseL + ((n + 1) % edges)
        U_np1 = baseU + ((n + 1) % edges)
        U_np2 = baseU + ((n + 2) % edges)

        # Quad convention:
        # [L_n, L_{n+1}, U_{n+2}, U_{n+1}]
        quads_k = np.stack((L_n, L_np1, U_np2, U_np1), axis=1)
        panel_quads.append(quads_k)

        p0 = coordinates[L_n]
        p1 = coordinates[L_np1]
        p2 = coordinates[U_np2]
        p3 = coordinates[U_np1]

        # Candidate diagonal squared lengths
        d02 = np.einsum("ij,ij->i", p2 - p0, p2 - p0)
        d13 = np.einsum("ij,ij->i", p3 - p1, p3 - p1)

        # True means the longest diagonal is p0 -> p2, i.e. L_n -> U_{n+2}.
        # False means the longest diagonal is p1 -> p3, i.e. L_{n+1} -> U_{n+1}.
        use_02 = d02 >= d13

        if not add_midpoints:
            # ------------------------------------------------------------
            # N4B5-style triangulation:
            # Use the LONGEST diagonal as the shared internal edge.
            # Important: this longest diagonal is NOT added to bend_edge_keys,
            # so it will be classified as a folding hinge.
            # ------------------------------------------------------------
            tri_pair_list = []

            for i in range(edges):
                if use_02[i]:
                    # Longest diagonal: L_n -> U_{n+2}
                    triA = [int(L_n[i]), int(U_np1[i]), int(U_np2[i])]
                    triB = [int(L_n[i]), int(L_np1[i]), int(U_np2[i])]
                else:
                    # Longest diagonal: L_{n+1} -> U_{n+1}
                    triA = [int(L_np1[i]), int(U_np2[i]), int(U_np1[i])]
                    triB = [int(L_np1[i]), int(U_np1[i]), int(L_n[i])]

                tri_pair_list.append(triA)
                tri_pair_list.append(triB)

            triangles.append(np.asarray(tri_pair_list, dtype=int))

        else:
            # ------------------------------------------------------------
            # N5B8-style triangulation:
            # Add one center node per quad and split the quad into 4 triangles.
            #
            # The center is placed on the LONGEST diagonal.
            # The two center-to-corner half-edges along this longest diagonal
            # are NOT added to bend_edge_keys, so they become folding hinges.
            #
            # The remaining two center-to-corner internal edges are added to
            # bend_edge_keys and therefore become bending hinges.
            # ------------------------------------------------------------
            mid_02 = 0.5 * (p0 + p2)
            mid_13 = 0.5 * (p1 + p3)

            centers_pos = np.where(use_02[:, None], mid_02, mid_13)

            c0 = coordinates.shape[0]
            c_ids = c0 + np.arange(edges, dtype=int)

            coordinates = np.vstack((coordinates, centers_pos))
            center_ids.append(c_ids)

            # Four triangles around the center:
            # (L_n, L_{n+1}, C)
            # (L_{n+1}, U_{n+2}, C)
            # (U_{n+2}, U_{n+1}, C)
            # (U_{n+1}, L_n, C)
            t0 = np.stack((L_n, L_np1, c_ids), axis=1)
            t1 = np.stack((L_np1, U_np2, c_ids), axis=1)
            t2 = np.stack((U_np2, U_np1, c_ids), axis=1)
            t3 = np.stack((U_np1, L_n, c_ids), axis=1)

            triangles.append(t0)
            triangles.append(t1)
            triangles.append(t2)
            triangles.append(t3)

            # Mark only the two internal lines NOT lying on the longest diagonal
            # as bending. The longest diagonal half-edges are left unmarked and
            # will therefore be classified as folding.
            for i, cc in enumerate(c_ids.tolist()):
                if use_02[i]:
                    # Longest diagonal is L_n -> U_{n+2}; keep it folding.
                    # Therefore, mark the other two center-corner lines as bending.
                    bend_vertices = (int(L_np1[i]), int(U_np1[i]))
                else:
                    # Longest diagonal is L_{n+1} -> U_{n+1}; keep it folding.
                    # Therefore, mark the other two center-corner lines as bending.
                    bend_vertices = (int(L_n[i]), int(U_np2[i]))

                for vv in bend_vertices:
                    key = (cc, vv) if cc < vv else (vv, cc)
                    bend_edge_keys.add(key)

    if len(panel_quads) == 0:
        return (
            coordinates,
            np.zeros((0, 2), dtype=int),
            np.zeros((0, 4), dtype=int),
            np.zeros((0, 4), dtype=int),
            np.zeros((0, 3), dtype=int),
            np.zeros((0, 4), dtype=int),
            [],
        )

    panel_quads = np.vstack(panel_quads).astype(int, copy=False)
    triangles = np.vstack(triangles).astype(int, copy=False)

    # -------------------- Unique pairs_list from triangles --------------------
    edges_raw = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )

    edges_norm = np.sort(edges_raw, axis=1)

    _, uniq_idx = np.unique(edges_norm, axis=0, return_index=True)
    pairs_list = edges_norm[np.sort(uniq_idx)].astype(int, copy=False)

    # -------------------- Build hinge quadruplets from triangle adjacency --------------------
    T = triangles.shape[0]

    tri_edges = np.stack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=1,
    )

    tri_edges_norm = np.sort(tri_edges, axis=2)

    # Map each undirected edge to adjacent triangle ids.
    edge_map = {}

    for t in range(T):
        for e in range(3):
            u = int(tri_edges_norm[t, e, 0])
            v = int(tri_edges_norm[t, e, 1])
            edge_map.setdefault((u, v), []).append(t)

    quadruplets_bend = []
    quadruplets_fold = []

    for (u, v), tris in edge_map.items():
        # Boundary edge: only one adjacent triangle, so no hinge.
        if len(tris) != 2:
            continue

        t0, t1 = tris

        tri0 = triangles[t0]
        tri1 = triangles[t1]

        # Opposite vertices in the two adjacent triangles.
        opp0 = int(tri0[(tri0 != u) & (tri0 != v)][0])
        opp1 = int(tri1[(tri1 != u) & (tri1 != v)][0])

        # Hinge convention used by the rest of the code.
        quad = [opp0, u, v, opp1]

        if (u, v) in bend_edge_keys:
            quadruplets_bend.append(quad)
        else:
            quadruplets_fold.append(quad)

    quadruplets_bend = (
        np.asarray(quadruplets_bend, dtype=int)
        if len(quadruplets_bend)
        else np.zeros((0, 4), dtype=int)
    )

    quadruplets_fold = (
        np.asarray(quadruplets_fold, dtype=int)
        if len(quadruplets_fold)
        else np.zeros((0, 4), dtype=int)
    )

    # Optional deduplication, independent of orientation.
    def _dedup_quads(Q):
        if Q.size == 0:
            return Q

        seen = set()
        out = []

        for q in Q:
            key = tuple(sorted(map(int, q)))

            if key not in seen:
                seen.add(key)
                out.append(q)

        return np.asarray(out, dtype=int)

    quadruplets_bend = _dedup_quads(quadruplets_bend)
    quadruplets_fold = _dedup_quads(quadruplets_fold)

    # Keep API compatibility with Miura/Z-fold assemblers.
    panels_by_cell = []

    return (
        coordinates,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    )


if __name__ == "__main__":
    nz = 3
    a = 0.5
    edges = 6
    height = a * 1.2

    (
        coordinates,
        pairs,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    ) = asseamble_kresling(
        nz=nz,
        a=a,
        edges=edges,
        height=height,
        phi=0.0,
        twist=0.0,
        add_midpoints=False,
    )

    plot_particles_with_forces(
        r=coordinates,
        forces=np.zeros_like(coordinates),
        pairlist=pairs,
        quadruplets_bend=quadruplets_bend,
        quadruplets_fold=quadruplets_fold,
    )

# %%
