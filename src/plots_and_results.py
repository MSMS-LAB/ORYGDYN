import os
import json
import shutil
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm import tqdm


def Output(rod, timestep, save_path):
    """Save particle positions and types as an .xyz file for visualization.

    Parameters:
    rod: List containing particle positions and their types
    timestep: Current simulation time step

    Returns:
    None
    """
    r = rod[0]
    index = np.array([np.arange(len(r))]).T

    material = np.full(([1, len(r)]), rod[1]).T

    output = np.hstack([index, material, r])

    # save_path = 'C:\\Users\\Yoav\\Desktop\\Ovito Results\\Igor'
    filename = "output_" + str(timestep) + ".xyz"
    completefilename = os.path.join(save_path, filename)

    with open(completefilename, "w") as f:
        np.savetxt(f, [len(r)], fmt="%i")
        f.write("\n")
        np.savetxt(f, output, fmt="%s")
    return


def create_text_bond_file_2d(
    directory_2output,
    stress_lines,
    pairs,
    timestep=0,
    box_bounds=(-25, 25),
    filename_prefix="output",
):
    """
    Creates a bond file in LAMMPS dump format for a 2D system and writes the first timestep data.
    The timestep number is appended to the filename.

    Args:
        directory_2output (str): Directory where the output file will be saved.
        stress_lines (np.ndarray): Array containing bond lengths and other stress-related data.
        pairs (np.ndarray): Array of atom pairs forming bonds, with shape (2, N).
        timestep (int): Timestep value to write in the file and append to the filename (default: 0).
        box_bounds (tuple): Box boundaries for the simulation (default: (-25, 25)).
        filename_prefix (str): Prefix for the output file name (default: "honey_comb.bonds").
    """
    # Ensure the output directory exists
    output_path = Path(directory_2output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Construct the full file path with timestep appended
    file_path = output_path / f"{filename_prefix}_{timestep}.bonds.dump_local"
    # Create a DataFrame to organize bond data
    data = {
        "index": np.arange(pairs.shape[0], dtype="int16"),  # Number of bonds
        "bond type": np.zeros(pairs.shape[0], dtype="int16"),  # Bond type (default: 1)
        "atom identifier 1": pairs[:, 0],  # First atom in each bond
        "atom identifier 2": pairs[:, 1],  # Second atom in each bond
        # 'bond length': stress_lines[:, 0]  # Bond lengths for the first timestep
    }

    df = pd.DataFrame(data)

    # Write the bond data to the file
    with open(file_path, "w") as f:
        # Write header information
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ENTRIES\n")
        f.write(f"{pairs.shape[0]}\n")  # Number of bonds
        f.write("ITEM: BOX BOUNDS pp pp\n")  # Only x and y dimensions for 2D
        f.write(f"{box_bounds[0]} {box_bounds[1]}\n")  # x-dimension
        f.write(f"{box_bounds[0]} {box_bounds[1]}\n")  # y-dimension
        f.write(f"{box_bounds[0]} {box_bounds[1]}\n")  # z-dimension
        f.write("ITEM: ENTRIES index c_1[0] c_2[0] c_2[1]\n")  # Adjusted for 2D

        # Write bond data
        for _, row in df.iterrows():
            f.write(" ".join(map(str, row)) + "\n")


def simple_plot(
    x,
    y,
    xlabel="time [s]",
    ylabel="Displacement of OY",
    title="Displacement vs Time",
    limits=False,
):
    """ "Function of Yoav for plotting the energy"""
    plt.figure(1)
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if limits:
        plt.xlim(left=0)
        plt.ylim(bottom=0)
    plt.grid()
    plt.tight_layout()
    plt.show()


def Multiple_plot(
    kin_en,
    pot_lin,
    pot_fold,
    pot_bend,
    total_en,
    dt,
    pot_contact=None,
    title="Energy vs Time",
    path_to_save=None,
):
    """Plot energy curves versus physical time t = step * dt"""

    n_steps = len(kin_en)
    t = np.arange(n_steps) * dt

    plt.plot(t, kin_en, color="blue", label="Kinetic", linestyle="-")
    plt.plot(
        t, pot_lin, color="orange", label="Potential Of Linear Springs", linestyle="-"
    )
    plt.plot(
        t,
        pot_fold,
        color="red",
        label="Potential Of Folding",
        linestyle="-",
    )
    plt.plot(
        t,
        pot_bend,
        color="purple",
        label="Potential Of Bending",
        linestyle="-",
    )
    if pot_contact is not None:
        plt.plot(
            t,
            pot_contact,
            color="brown",
            label="Potential Of Contact",
            linestyle="-",
        )
    plt.plot(t, total_en, color="green", label="Total", linestyle="-")

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.xlabel("t [s]")
    plt.ylabel("E [J]")

    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/energy_vs_time.png", dpi=300)

    plt.show()
    plt.close()


def Multiple_plot_3images(x, y, z, u):
    """ "Function of Yoav for plotting the energy"""
    plt.figure(1)

    plt.subplot(311)
    plt.plot(x, y)
    plt.subplot(311).set_xlim(left=0)
    plt.subplot(311).set_ylim(bottom=0)
    plt.subplot(311).set(xlabel="t [s]", ylabel="Ek [J]")
    #    plt.subplot(311).plot(y, color='blue', label='Sine wave')
    plt.grid()

    plt.subplot(312)
    plt.plot(x, z)
    plt.subplot(312).set_xlim(left=0)
    plt.subplot(312).set_ylim(bottom=0)
    plt.subplot(312).set(xlabel="t [s]", ylabel="Ep [J]")
    plt.subplot(312).plot(z, color="orange", label="Sine wave")
    plt.grid()

    plt.subplot(313)
    plt.plot(x, u)
    plt.subplot(313).set_xlim(left=0)
    plt.subplot(313).set_ylim(bottom=0)
    plt.subplot(313).set(xlabel="t [s]", ylabel="E [J]")
    plt.subplot(313).plot(u, color="green", label="Sine wave")
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.close()


def dual_plot(
    pairs_of_vars, title="Geometrical vs Analytical Measurements", path_to_save=None
):
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for pair in pairs_of_vars:
        var1, var2, label = pair
        color = next(color_cycle)
        plt.plot(range(len(var1)), var1, label=f"{label} - Geometrical", color=color)
        plt.scatter(
            range(len(var2)),
            var2,
            label=f"{label} - Analytical",
            marker="d",
            s=30,
            color=color,
        )

    # plt.plot(var1, label=label1)
    # # plt.plot(var2, color="orange", label=label2, linestyle="o")
    # plt.scatter(range(len(var2)), var2, label=label2, marker="d", s=15)
    # ax1.set_ylabel("Ek [J]")
    # plt.xlim(left=0)
    # plt.ylim(bottom=0)
    plt.xlabel("Timestep")
    plt.ylabel("[m]")
    plt.title(title)

    plt.grid()
    plt.legend()
    plt.tight_layout()
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/geometric_vs_analytical.png", dpi=300)
    plt.show()
    plt.close()


def plot_height_length_vs_dihedral(
    dihedral_deg,
    height_pairs,
    length_pairs,
    width_pairs=None,
    var_1=None,
    title="Kinematics vs Dihedral Angle",
    path_to_save=None,
):
    dihedral = np.asarray(dihedral_deg, dtype=float)
    mask = np.isfinite(dihedral)

    if not np.any(mask):
        raise ValueError("No valid dihedral angle values.")
    dihedral = dihedral[mask]

    # Only two axes (Height left, Length+Width right)
    fig, ax_left = plt.subplots()
    ax_right = ax_left.twinx()

    # GLOBAL color cycle — shared across all parameters
    global_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    # Assign each base parameter name its own color
    color_map = {}  # e.g. { "Height": "blue", "Length": "green" }

    def get_color(label):
        """Return a consistent color for all curves sharing the same base name."""
        base = label.split("-")[0].strip()  # "Height", "Length", "Width"
        if base not in color_map:
            color_map[base] = next(global_cycle)
        return color_map[base]

    # Reusable helper
    def plot_pairs(ax, pairs):
        if pairs is None:
            return
        for data, label in pairs:
            values = np.asarray(data, dtype=float)[mask]
            color = get_color(label)

            if "Analytical" in label:
                ax.scatter(dihedral, values, label=label, color=color, marker="d", s=35)
            else:
                ax.plot(dihedral, values, label=label, color=color, linewidth=1.8)

    # HEIGHT → left axis
    plot_pairs(ax_left, height_pairs)
    plot_pairs(ax_left, var_1)
    # LENGTH & WIDTH → right axis (colors come from same global map)
    plot_pairs(ax_right, length_pairs)
    plot_pairs(ax_right, width_pairs)

    # Labels
    ax_left.set_xlabel("Dihedral Angle [deg]")
    ax_left.set_ylabel("Length [m]")
    ax_right.set_ylabel("Length / Width [m]")

    ax_left.set_title(title)
    ax_left.grid(True)

    # Combined legend
    handles, labels = [], []
    for ax in (ax_left, ax_right):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    ax_left.legend(handles, labels, loc="best")

    fig.tight_layout()
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/geom_params_vs_dihedral.png", dpi=300)
    plt.show()
    plt.close()


def plot_reaction_forces_vs_time(
    force_history,
    pinned_node_ids,
    axis,
    title="Reaction Forces vs Time",
    xlabel="Timestep",
    ylabel="Reaction Force [N]",
    path_to_save=None,
):

    if axis is not None:
        force_history = np.asarray(force_history)
        pinned_node_ids = np.asarray(pinned_node_ids, dtype=int)
        force_history = force_history[:, pinned_node_ids, axis].sum(axis=1)

    plt.figure()
    plt.plot(force_history)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/{ylabel.replace(' ', '_')}.png", dpi=300)
    plt.show()
    plt.close()
    return force_history


def plot_particles_with_forces(
    r,
    forces,
    pairlist=None,
    angular_sets=None,
    force_scale=1.0,
    quadruplets_bend=None,
    quadruplets_fold=None,
    hinge_linewidth=3.0,
    hinge_alpha=1.0,
    path_to_save=None,
):
    """
    Visualize particle positions, force vectors, pair connections, angular sets,
    and annotate each particle with its id.
    Optionally visualize hinge quadruplets (bending/folding) by drawing the hinge edge.

    Parameters:
        r (np.ndarray): Particle positions of shape (N, 3).
        forces (np.ndarray): Force vectors of shape (N, 3).
        pairlist (optional): Iterable of index pairs (M, 2) to connect with lines.
        angular_sets (optional): Iterable of index tuples representing an angular set.
        force_scale (float): Scaling factor for the quiver plot.

        quadruplets_bend (optional): (Qb,4) int array-like
            Quadruplets [a, i, j, b] where (i, j) is the hinge edge (drawn).
        quadruplets_fold (optional): (Qf,4) int array-like
            Quadruplets [a, i, j, b] where (i, j) is the hinge edge (drawn).

        hinge_linewidth (float): Width of hinge lines.
        hinge_alpha (float): Alpha for hinge lines.
    """
    r = np.asarray(r, dtype=float)
    forces = np.asarray(forces, dtype=float)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot particles as green dots.
    ax.scatter(
        r[:, 0], r[:, 1], r[:, 2], color="green", s=50, label="Particles", marker="o"
    )

    # Annotate each particle with its id (its index in the matrix).
    for idx, (x, y, z) in enumerate(r):
        ax.text(x, y, z, f"{idx}", color="red", fontsize=10, ha="center", va="center")

    # Plot force vectors using quiver.
    ax.quiver(
        r[:, 0],
        r[:, 1],
        r[:, 2],
        forces[:, 0],
        forces[:, 1],
        forces[:, 2],
        color="red",
        length=force_scale,
        normalize=False,
        label="Force Vectors",
    )

    # Plot pair connections, if provided.
    if pairlist is not None:
        pairlist = np.asarray(pairlist, dtype=int)
        for idx, (i, j) in enumerate(pairlist):
            xs = [r[i, 0], r[j, 0]]
            ys = [r[i, 1], r[j, 1]]
            zs = [r[i, 2], r[j, 2]]
            label = "Connections" if idx == 0 else None
            ax.plot(xs, ys, zs, color="gray", label=label)

    # Plot angular sets, if provided.
    if angular_sets is not None:
        angular_sets = np.asarray(angular_sets, dtype=int)
        for idx, ang_set in enumerate(angular_sets):
            # Keep the existing behavior (expects 4 indices in the provided example).
            lower_left, lower_mid, lower_right, upper_left = ang_set
            indices_order = [lower_left, lower_mid, lower_right, upper_left, lower_left]
            xs = [r[i, 0] for i in indices_order]
            ys = [r[i, 1] for i in indices_order]
            zs = [r[i, 2] for i in indices_order]
            label = "Angular Set" if idx == 0 else None
            ax.plot(xs, ys, zs, color="red", linestyle=":", linewidth=2, label=label)

    # ---- NEW: Plot hinges from quadruplets (bend/fold) --------------------
    # Convention: quadruplet = [opp0, hinge_i, hinge_j, opp1]
    # We draw the hinge edge (hinge_i, hinge_j).
    def _plot_hinges(quadruplets, color, label):
        if quadruplets is None:
            return
        Q = np.asarray(quadruplets, dtype=int)
        if Q.size == 0:
            return

        # De-duplicate hinge edges so we don't overdraw the same hinge.
        hinges = Q[:, 1:3]
        hinges = np.sort(hinges, axis=1)
        _, unique_idx = np.unique(hinges, axis=0, return_index=True)
        hinges = hinges[np.sort(unique_idx)]

        for idx, (i, j) in enumerate(hinges):
            xs = [r[i, 0], r[j, 0]]
            ys = [r[i, 1], r[j, 1]]
            zs = [r[i, 2], r[j, 2]]
            ax.plot(
                xs,
                ys,
                zs,
                color=color,
                linewidth=hinge_linewidth,
                alpha=hinge_alpha,
                label=label if idx == 0 else None,
            )

    _plot_hinges(quadruplets_bend, color="blue", label="Bending Hinge")
    _plot_hinges(quadruplets_fold, color="orange", label="Folding Hinge")

    # Set labels and title.
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Particles, Force Vectors, Connections, and Hinges")

    if path_to_save is not None:
        plt.savefig(f"{path_to_save}/particles_and_forces.png", dpi=300)

    plt.show()
    plt.close()
    # plt.savefig("particles_with_forces_and_hinges.png", dpi=300)


def write_thin_triangles_vtk(points, triangles, filename, thickness=0.01):
    """
    Export thin triangular panels as a closed extruded surface to VTK PolyData.

    Each input triangle is extruded along its normal into a thin shell:
        - 1 bottom triangle
        - 1 top triangle
        - 3 quad side faces

    Parameters
    ----------
    points : (N, 3) array
        3D coordinates of the nodes (e.g. new_planes).
    triangles : (T, 3) array of int
        Indices of nodes forming each triangle.
    filename : str
        Output file name (.vtk).
    thickness : float
        Panel thickness (distance between bottom and top surfaces).
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=int)

    all_vertices = []
    all_cells = []  # each cell is a list of point indices
    all_cell_sizes = []  # number of vertices per polygon

    for tri in triangles:
        v0, v1, v2 = tri
        p0, p1, p2 = points[v0], points[v1], points[v2]

        # Panel normal
        n = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(n)
        if norm < 1e-14:
            # Degenerate triangle; skip
            continue
        n /= norm

        half_t = 0.5 * thickness

        # Bottom and top vertices
        b0 = p0 - half_t * n
        b1 = p1 - half_t * n
        b2 = p2 - half_t * n

        t0 = p0 + half_t * n
        t1 = p1 + half_t * n
        t2 = p2 + half_t * n

        base_index = len(all_vertices)
        # Order: bottom 3, top 3
        all_vertices.extend([b0, b1, b2, t0, t1, t2])

        i0, i1, i2, j0, j1, j2 = range(base_index, base_index + 6)

        # Faces:
        # bottom triangle (CCW when viewed from outside)
        all_cells.append([i0, i1, i2])
        all_cell_sizes.append(3)

        # top triangle (reverse order to keep outward normal)
        all_cells.append([j2, j1, j0])
        all_cell_sizes.append(3)

        # three side quads
        all_cells.append([i0, i1, j1, j0])
        all_cell_sizes.append(4)

        all_cells.append([i1, i2, j2, j1])
        all_cell_sizes.append(4)

        all_cells.append([i2, i0, j0, j2])
        all_cell_sizes.append(4)

    all_vertices = np.asarray(all_vertices, dtype=float)
    n_points = all_vertices.shape[0]
    n_cells = len(all_cells)

    # Total connectivity size: sum over cells (n_verts + 1)
    total_ints = sum(sz + 1 for sz in all_cell_sizes)

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Thin triangular origami panels\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Points
        f.write(f"POINTS {n_points} float\n")
        for x, y, z in all_vertices:
            f.write(f"{x} {y} {z}\n")

        # POLYGONS: mixed triangles & quads
        f.write(f"POLYGONS {n_cells} {total_ints}\n")
        for cell, sz in zip(all_cells, all_cell_sizes):
            f.write(str(sz) + " " + " ".join(str(idx) for idx in cell) + "\n")

    # print(f"Saved thin-triangle VTK: {filename}")


def export_history_as_thin_triangle_series(
    history,
    triangles,
    basename="sheet",
    out_dir="paraview_out",
    thickness=0.01,
):
    """
    Export a time series of configurations as extruded thin *triangular* panels for ParaView.

    Parameters
    ----------
    history : list/array of (N, 3) arrays
        history[t] are node coordinates at time step t (e.g. new_planes for each step).
    triangles : (T, 3) array-like of int
        Triangle connectivity in terms of global node indices.
        Assumed constant in time.
    basename : str
        Base file name prefix, e.g. 'sheet' -> sheet_0000.vtk, sheet_0001.vtk, ...
    out_dir : str
        Output directory.
    thickness : float
        Visual thickness of panels.
    """
    out_dir = os.path.abspath(out_dir)
    # Clean up the save directory if it exists, and recreate it
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)  # Remove the existing directory
        print(f"{out_dir} was deleted")  # Notify the user
        os.mkdir(out_dir)  # Create a new empty directory
    else:
        os.mkdir(out_dir)  # Create the directory if it doesn't exist
    triangles = np.asarray(triangles, dtype=int)

    print(
        f"Exporting {len(history)} frames with {triangles.shape[0]} triangles each..."
    )

    for t, pts in tqdm(enumerate(history), total=len(history)):
        filename = os.path.join(out_dir, f"{basename}_{t:04d}.vtk")
        write_thin_triangles_vtk(pts, triangles, filename, thickness=thickness)


def export_geometry_to_matlab(
    path_base,
    V,
    edges,
    bend4,
    fold4,
    triangles=None,
    panel_quads=None,
    plane_ids=None,
    panels_by_cell=None,
    float_fmt="%.16g",
):
    """
    Export Miura-ori geometry/connectivity for MATLAB visualization.

    Parameters
    ----------
    path_base : str
        Base path without extension, e.g. "out/miura_nx2_ny2".
        Will produce: path_base + ".geom" and also ".mat" or ".npz".
    V : (N,3) float
        Vertex coordinates (your `new_planes`).
    edges : (E,2) int
        Edge list (`pairs_list`).
    bend4 : (B,4) int
        Bending quadruplets (hinge/bend).
    fold4 : (F,4) int
        Folding quadruplets (between panels).
    triangles : (T,3) int or None
    panel_quads : (Q,4) int or None
    plane_ids : (P,K) int or None
        Per-panel vertex ids (rows correspond to panels).
    panels_by_cell : list or None
        Output of _group_panels_by_cell (nested lists of quads).

    Notes
    -----
    - Indices are exported as 1-based for MATLAB.
    - The .geom file is plain text with blocks.
    - Additionally saves a .mat (preferred) if scipy is available,
      otherwise a .npz.
    """
    V = np.asarray(V, dtype=float)
    edges = np.asarray(edges, dtype=int)
    bend4 = np.asarray(bend4, dtype=int)
    fold4 = np.asarray(fold4, dtype=int)

    triangles = (
        np.zeros((0, 3), dtype=int)
        if triangles is None
        else np.asarray(triangles, dtype=int)
    )
    panel_quads = (
        np.zeros((0, 4), dtype=int)
        if panel_quads is None
        else np.asarray(panel_quads, dtype=int)
    )
    plane_ids = None if plane_ids is None else np.asarray(plane_ids, dtype=int)

    # --- convert to MATLAB 1-based indexing ---
    def to1(A):
        A = np.asarray(A)
        return (A + 1) if A.size else A

    edges1 = to1(edges)
    bend41 = to1(bend4)
    fold41 = to1(fold4)
    tri1 = to1(triangles)
    quad1 = to1(panel_quads)
    plane_ids1 = None if plane_ids is None else to1(plane_ids)

    # panels_by_cell is a nested python list of quads; convert entries to 1-based
    if panels_by_cell is None:
        panels_by_cell1 = []
    else:
        panels_by_cell1 = [
            [(np.asarray(q, dtype=int) + 1).tolist() for q in cell]
            for cell in panels_by_cell
        ]

    # -------------------------
    # 1) Write human-readable .geom
    # -------------------------
    geom_path = path_base + ".geom"
    with open(geom_path, "w", encoding="utf-8") as f:
        f.write("# MIURA_GEOM v1\n")
        f.write(f"# N_VERT {V.shape[0]}\n")
        f.write(f"# N_EDGES {edges1.shape[0]}\n")
        f.write(f"# N_TRI {tri1.shape[0]}\n")
        f.write(f"# N_QUAD {quad1.shape[0]}\n")
        f.write(f"# N_BEND4 {bend41.shape[0]}\n")
        f.write(f"# N_FOLD4 {fold41.shape[0]}\n")
        f.write(f"# N_PANELS {0 if plane_ids1 is None else plane_ids1.shape[0]}\n")
        f.write("\n")

        # VERTICES
        f.write("VERTICES\n")
        for i, (x, y, z) in enumerate(V, start=1):
            f.write((f"{i} " + " ".join([float_fmt] * 3) + "\n") % (x, y, z))
        f.write("END_VERTICES\n\n")

        # EDGES
        f.write("EDGES\n")
        for a, b in edges1:
            f.write(f"{a} {b}\n")
        f.write("END_EDGES\n\n")

        # TRIANGLES
        f.write("TRIANGLES\n")
        for a, b, c in tri1:
            f.write(f"{a} {b} {c}\n")
        f.write("END_TRIANGLES\n\n")

        # QUADS (panel faces)
        f.write("QUADS\n")
        for a, b, c, d in quad1:
            f.write(f"{a} {b} {c} {d}\n")
        f.write("END_QUADS\n\n")

        # BEND quadruplets
        f.write("BEND4\n")
        for a, b, c, d in bend41:
            f.write(f"{a} {b} {c} {d}\n")
        f.write("END_BEND4\n\n")

        # FOLD quadruplets
        f.write("FOLD4\n")
        for a, b, c, d in fold41:
            f.write(f"{a} {b} {c} {d}\n")
        f.write("END_FOLD4\n\n")

        # PANEL IDS (per panel vertex IDs)
        f.write("PANEL_IDS\n")
        if plane_ids1 is not None and plane_ids1.size:
            # write each panel as one line: K integers
            for row in plane_ids1:
                f.write(" ".join(map(str, row.tolist())) + "\n")
        f.write("END_PANEL_IDS\n\n")

        # PANELS BY CELL (JSON so nested lists survive)
        f.write("PANELS_BY_CELL_JSON\n")
        f.write(json.dumps(panels_by_cell1))
        f.write("\nEND_PANELS_BY_CELL_JSON\n")

    # -------------------------
    # 2) Also save machine-readable file (.mat preferred)
    # -------------------------
    payload = {
        "V": V,
        "edges": edges1,
        "triangles": tri1,
        "quads": quad1,
        "bend4": bend41,
        "fold4": fold41,
        "plane_ids": (np.array([], dtype=int) if plane_ids1 is None else plane_ids1),
        # store panels_by_cell as a JSON string in mat/npz (easy to decode)
        "panels_by_cell_json": json.dumps(panels_by_cell1),
    }

    saved = {"geom": geom_path}

    try:
        from scipy.io import savemat  # optional

        mat_path = path_base + ".mat"
        savemat(mat_path, payload, do_compression=True)
        saved["mat"] = mat_path
    except Exception:
        npz_path = path_base + ".npz"
        np.savez_compressed(npz_path, **payload)
        saved["npz"] = npz_path

    return saved


def save_data_from_sim(
    H_geom_list,
    H_an_list,
    W_geom_list,
    W_an_list,
    L_geom_list,
    L_an_list,
    dihedral_list,
    Compatibility_R_list=None,
    time_list=None,
    Total_energy=None,
    path_to_save=None,
    name=None,
):

    df = pd.DataFrame()
    # Height
    df["Height_Actual"] = H_geom_list
    df["Height_Analytical"] = H_an_list
    if W_geom_list is not None and W_an_list is not None:
        # Width
        df["Width_Actual"] = W_geom_list
        df["Width_Analytical"] = W_an_list
    # Length
    df["Length_Actual"] = L_geom_list
    df["Length_Analytical"] = L_an_list
    df["Dihedral"] = dihedral_list
    if Compatibility_R_list is not None:
        df["Compatibility_Ratio"] = Compatibility_R_list

    if time_list is not None:
        df["Time"] = time_list

    if path_to_save is not None:
        df.to_csv(f"{path_to_save}/{name}.csv", index=False, sep=";")
        print(f"Saved data to {path_to_save}/{name}.csv")
    return df


def save_data_from_sim_new(
    data_dict,
    path_to_save=None,
    name=None,
):
    """
    Save simulation data from a dictionary into a pandas DataFrame and optionally to CSV.

    Parameters
    ----------
    data_dict : dict
        Dictionary of column_name -> data_list_or_array.
        Example:
        {
            "Height_Actual": H_geom_list,
            "Height_Analytical": H_an_list,
            "Length_Actual": L_geom_list,
            "Length_Analytical": L_an_list,
            "Dihedral": dihedral_list,
            "Time": time_list,
        }

        Keys with value None are skipped.
    path_to_save : str, optional
        Directory where the CSV file will be saved.
    name : str, optional
        File name without extension.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing all provided non-None columns.
    """

    if not isinstance(data_dict, dict):
        raise TypeError("data_dict must be a dictionary of column_name -> values.")

    filtered_data = {k: v for k, v in data_dict.items() if v is not None}

    if len(filtered_data) == 0:
        raise ValueError("data_dict does not contain any non-None data to save.")

    df = pd.DataFrame(filtered_data)

    if path_to_save is not None:
        if name is None:
            raise ValueError(
                "If path_to_save is provided, name must also be specified."
            )

        os.makedirs(path_to_save, exist_ok=True)
        full_path = f"{path_to_save}/{name}.csv"
        df.to_csv(full_path, index=False, sep=";")
        print(f"Saved data to {full_path}")

    return df


def plot_potential_energy_vs_height(
    Total_P_lin,
    Total_P_fold,
    Total_P_bend,
    H_actual_list,
    step_to_save_results,
    path_to_save=None,
    title="Potential Energy vs Height",
):
    """
    Plot saved potential energy values against actual height.

    Total_P_lin, Total_P_fold, Total_P_bend:
        energy arrays stored for every simulation timestep

    H_actual_list:
        actual height array computed from saved history frames

    step_to_save_results:
        interval used to save history frames
    """

    # Total potential energy at every timestep
    Total_p = Total_P_lin + Total_P_fold + Total_P_bend

    # Select potential energy values corresponding to saved geometry frames
    Total_p_saved = Total_p[::step_to_save_results]

    # Convert height list to numpy array
    H_actual = np.asarray(H_actual_list)

    # Make sure both arrays have the same length
    n = min(len(Total_p_saved), len(H_actual))
    Total_p_saved = Total_p_saved[:n]
    H_actual = H_actual[:n]

    plt.figure(figsize=(8, 6))
    plt.plot(H_actual, Total_p_saved, linewidth=2)

    # Scatter points over the same data
    plt.scatter(H_actual, Total_p_saved, s=25, label="Saved data points")

    plt.xlabel("Actual height H [m]")
    plt.ylabel("Potential energy [J]")
    plt.title(title)
    plt.grid(True)

    # Optional: reverse x-axis so compression goes left-to-right visually
    # plt.gca().invert_xaxis()

    if path_to_save is not None:
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")

    plt.show()
