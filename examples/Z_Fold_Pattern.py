# %%
from time import perf_counter_ns

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulate import run_standard, run_shake, run_rattle
from src.structures import (
    asseamble_z_fold,
    compute_triangle_areas,
    build_edge_triangle_connectivity,
)

from src.utils import (
    compute_nodal_masses,
    compute_bar_cross_sections,
    compute_torsion_stiffness,
    read_materials_yaml,
    parse_opt
)

from src.plots_and_results import (
    export_history_as_thin_triangle_series,
    Multiple_plot,
    dual_plot,
    plot_height_length_vs_dihedral,
    plot_reaction_forces_vs_time,
    export_geometry_to_matlab,
    save_data_from_sim,
)

from src.analysis import analyze_zfold_history


def main(opt):
    links_type = (
        opt.links_type
    )  # Type of link model: 'extendable' or 'rattle' or 'shake'
    T = opt.T  # Total simulation time in seconds
    zeta = opt.zeta  # 1% the damping ratio
    dt_safety_factor = opt.dt_safety_factor
    desired_frames = opt.desired_frames  # Save results each n-step
    material_name = opt.material_name  # Material name from materials.yaml
    materials_file = opt.materials_file  # Path to materials.yaml
    force_vector = opt.force_vector  # Force vector with direction of applied force
    force_magnitude = opt.force_magnitude  # Magnitude of the applied force
    load_factor = opt.load_factor  # Apply forces gradually over time
    plotting = opt.plotting  # Plotting parameters
    results_dir = opt.results_dir  # Base directory for plots and output data
    export_vtk = opt.export_vtk  # Export vtk files for visualization in Paraview
    vtk_dir = opt.vtk_dir  # Directory for exported visualization files
    print(f"Simulation with {links_type} links")

    # Geometry
    a = opt.a  # m
    b = opt.b  # m
    thickness = opt.thickness  # m Thickness of the panel
    theta = opt.theta  # Fold angle in radians (5 degrees is a good starting point for testing)
    nx = opt.nx  # 16 # Number of folds in the x direction (must be even for Z-fold)
    ny = opt.ny
    # Generate Z-folding structure
    (
        coordinates,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        _,
    ) = asseamble_z_fold(
        nx=nx,
        ny=ny,
        a=a,
        b=b,
        theta=theta,
        add_center=True,
    )
    saved = export_geometry_to_matlab(
        f"{results_dir}/zfold_{nx}x{ny}",
        V=coordinates,
        edges=pairs_list,
        bend4=quadruplets_bend,
        fold4=quadruplets_fold,
        triangles=triangles,
        panel_quads=panel_quads,
        plane_ids=None,  # optionally pass plane_ids if you want it
        panels_by_cell=None,  # optionally pass panels_by_cell if you want it
    )
    print(f"Z-fold geometry exported: {saved}")
    # Compute triangle areas
    surface_areas, _ = compute_triangle_areas(vertices=coordinates, triangles=triangles)
    # Generate links length list
    a_list = np.array(
        [np.linalg.norm(coordinates[i] - coordinates[j]) for i, j in pairs_list]
    )
    # Build edge-triangle connectivity
    tri_edges, _ = build_edge_triangle_connectivity(
        pairs=pairs_list, triangles=triangles
    )

    # Load material properties
    E, density, poisson_ratio = read_materials_yaml(
        filepath=materials_file,
        material_name=material_name,
    )
    sound_speed = np.sqrt(E / density)  # Speed of sound in the material
    # Characteristic length & timestep
    l_c = float(np.min(a_list))
    dt = dt_safety_factor * l_c / sound_speed
    print(f"Timestep size dt: {dt:.6e} s")
    fps = desired_frames / T  # frames per second for visualization
    step_to_save_results = max(1, int(1 / (fps * dt)))
    print(f"Saving results each {step_to_save_results} steps")
    if step_to_save_results < 1:
        step_to_save_results = 1
    # Total number of steps
    Nt = int(T / dt)
    print(f"Total number of steps: {Nt}")

    # Compute bar cross-sections
    bar_cross_section = compute_bar_cross_sections(
        a_list=a_list,
        tri_edges=tri_edges,
        surface_areas=surface_areas,
        thickness=thickness,
        poisson_ratio=poisson_ratio,
    )
    # List with stiffness for every beam
    c_list = E * bar_cross_section / a_list
    # List of masses for each particles
    m_list = compute_nodal_masses(
        r=coordinates,
        triangles=triangles,
        surface_areas=surface_areas,
        thickness=thickness,
        density=density,
    )
    mass_per_node = np.unique(m_list)[0]
    print(f"Mass per node: {mass_per_node:.3e} kg")
    print(f"Total structure mass: {m_list.sum():.3e} kg")
    c_max = float(np.max(c_list))  # max axial stiffness
    # viscous damping coefficient per node
    viscous_per_node = 2.0 * zeta * np.sqrt(c_max * m_list)  # per node
    # Generate boundary conditions
    pin_z_ids = np.where((coordinates[:, 2] == coordinates[:, 2].min()))[0]
    print(f"Pinned Z ids: {pin_z_ids}")

    pin_y_ids = np.where(coordinates[:, 1] == coordinates[:, 1].min())[0]
    print(f"Pinned Y ids: {pin_y_ids}")

    external_force_ids = np.where(coordinates[:, 1] == coordinates[:, 1].max())[0]
    # external_force_ids = np.where(coordinates[:, 0] == coordinates[:, 0].max())[0]
    print(f"External force ids: {external_force_ids}")
    # Distibute the applied force between defined nodes
    force_distribution = force_magnitude / len(external_force_ids)

    # Compute bending and folding stiffness lists
    bend_stiffness_list, fold_stiffness_list = compute_torsion_stiffness(
        E=E,
        thickness=thickness,
        poisson_ratio=poisson_ratio,
        coordinates=coordinates,
        quadruplets_bend=quadruplets_bend,
        panel_quads=panel_quads,
        quadruplets_fold=quadruplets_fold,
    )
    # Print stiffness ratio
    stif_ration = np.min(bend_stiffness_list) / np.min(fold_stiffness_list)
    print(f"Stiffness ratio (bend/fold): {round(stif_ration, 3)} \n")

    # Obtain performance
    start_time = perf_counter_ns()
    # Run simulation
    if links_type == "extendable":
        history, force_history, Total_K, Total_P_lin, Total_P_fold, Total_P_bend = (
            run_standard(
                Nt=Nt,
                pairlist=pairs_list,
                quadruplets_bend=quadruplets_bend,
                quadruplets_fold=quadruplets_fold,
                fixed_ids=None,
                roller_fixed_ids_x=None,
                roller_fixed_ids_y=pin_y_ids,
                roller_fixed_ids_z=pin_z_ids,
                external_force_ids=external_force_ids,
                force_vector=force_distribution * force_vector,
                r=coordinates,
                r_stress_free=None,
                load_factor=load_factor,  # Apply forces gradually over time
                c_list=c_list,
                a_list=a_list,
                bend_stiffness_list=bend_stiffness_list,
                fold_stiffness_list=fold_stiffness_list,
                viscous_per_node=viscous_per_node,
                dt=dt,
                m_list=m_list,
                v=np.zeros_like(coordinates),
                step_to_save_results=step_to_save_results,
            )
        )
    else:
        if links_type == "rattle":
            run_constrained = run_rattle
        else:  # shake
            run_constrained = run_shake
        history, force_history, Total_K, Total_P_lin, Total_P_fold, Total_P_bend = (
            run_constrained(
                Nt=Nt,
                pairlist=pairs_list,
                quadruplets_bend=quadruplets_bend,
                quadruplets_fold=quadruplets_fold,
                fixed_ids=None,
                roller_fixed_ids_x=None,
                roller_fixed_ids_y=pin_y_ids,
                roller_fixed_ids_z=pin_z_ids,
                external_force_ids=external_force_ids,
                force_vector=force_distribution * force_vector,
                r=coordinates,
                r_stress_free=None,
                load_factor=load_factor,  # Apply forces gradually over time
                a_list=a_list,
                bend_stiffness_list=bend_stiffness_list,
                fold_stiffness_list=fold_stiffness_list,
                viscous_per_node=viscous_per_node,
                dt=dt,
                m_list=m_list,
                v=np.zeros_like(coordinates),
                step_to_save_results=step_to_save_results,
            )
        )
    end_time = perf_counter_ns()
    total_runtime = (end_time - start_time) * 1e-9  # sec
    print(f"Total simulation runtime: {total_runtime:.3f} seconds \n")

    if export_vtk:
        # Export results for visualization in Paraview
        export_history_as_thin_triangle_series(
            history=history,
            triangles=triangles,
            basename="z_fold",
            out_dir=vtk_dir,
            thickness=thickness,  # choose relative to your geometry
        )

    # Plot energies
    Total_p = Total_P_lin + Total_P_fold + Total_P_bend
    Total_energy = Total_K + Total_p
    # Print energy info
    print(
        f"Maximum Total Energy: {np.max(Total_energy):.3e} J at step {np.argmax(Total_energy)}"
    )
    print(
        f"Last Total Energy: {Total_energy[-1]:.3e} J at step {len(Total_energy) - 1}"
    )
    print(f"Kinetic Energy at last step: {Total_K[-1]:.3e} J")
    print(f"Spring Potential Energy at last step: {Total_P_lin[-1]:.3e} J")

    # Instantaneous percentage (per step)
    frac_lin = Total_P_lin / np.maximum(Total_p, 1e-12)
    perc_lin = 100.0 * frac_lin
    print(f"Final linear energy fraction: {perc_lin[-1]:.3f} %")
    # Bending vs folding percentage
    frac_bend = Total_P_bend / np.maximum(Total_p, 1e-12)
    perc_bend = 100.0 * frac_bend
    print(f"Final bending energy fraction: {perc_bend[-1]:.3f} %")
    frac_fold = Total_P_fold / np.maximum(Total_p, 1e-12)
    perc_fold = 100.0 * frac_fold
    print(f"Final folding energy fraction: {perc_fold[-1]:.3f} %")

    # Analyze Z-fold history
    (
        H_geom_list,
        H_an_list,
        L_geom_list,
        L_an_list,
        dihedral_deg_list,
        dihedral_angles_per_fold,
        _,
        _,
        _,
        _,
    ) = analyze_zfold_history(
        history=history,
        quadruplets_fold=quadruplets_fold,
        panel_quads=panel_quads,
        nx=nx,
        b=b,
        n_th_elem=1,
    )

    list_of_vars = [
        (H_geom_list, H_an_list, "Height"),
        (L_geom_list, L_an_list, "Length"),
    ]

    # Time array
    time_array = np.arange(len(H_geom_list)) * dt * step_to_save_results
    # Save data from simulation
    save_data_from_sim(
        H_geom_list=H_geom_list,
        H_an_list=H_an_list,
        W_geom_list=None,
        W_an_list=None,
        L_geom_list=L_geom_list,
        L_an_list=L_an_list,
        dihedral_list=dihedral_deg_list,
        Compatibility_R_list=None,
        time_list=time_array,
        path_to_save=results_dir,
        name=f"z_fold_{links_type}_data",
    )

    if plotting:
        Multiple_plot(
            kin_en=Total_K,
            pot_lin=Total_P_lin,
            pot_fold=Total_P_fold,
            pot_bend=Total_P_bend,
            total_en=Total_energy,
            dt=dt,
            title=f"Energy vs Timestep - {links_type}",
            path_to_save=results_dir,
        )
        dual_plot(
            pairs_of_vars=list_of_vars,
            title=f"Miura-Ori Geometric vs Analytical - {links_type}",
            path_to_save=results_dir,
        )
        plot_height_length_vs_dihedral(
            dihedral_deg=dihedral_deg_list,
            height_pairs=[
                (H_geom_list, "Height - Geometrical"),
                (H_an_list, "Height - Analytical"),
            ],
            length_pairs=[
                (L_geom_list, "Length - Geometrical"),
                (L_an_list, "Length - Analytical"),
            ],
            title=f"Miura-Ori H/L/W vs Dihedral - {links_type}",
            path_to_save=results_dir,
        )
        plot_reaction_forces_vs_time(
            force_history=force_history,
            pinned_node_ids=pin_y_ids,
            axis=1,
            title=f"Reaction Forces vs Time - {links_type}",
            path_to_save=results_dir,
        )

    n_folds = len(fold_stiffness_list)
    print(f"Number of folds in the structure: {n_folds} \n")

    final_angle = np.median((dihedral_angles_per_fold[-1]))
    # final_angle = np.deg2rad(dihedral_deg_list[-1])
    print(f"Final dihedral angle: {np.rad2deg(final_angle):.6f} degrees")

    delta_angle = final_angle - theta  # in radians
    g = np.unique(fold_stiffness_list)[0]  # Nm/rad (all folds have same stiffness)
    theoretical_force = g * (delta_angle) / (b * np.cos(final_angle / 2))

    # Reaction force and theoretical value
    n_last = int(Nt * 0.15)  # last 15% of the simulation
    reaction_force = force_history[-n_last:, pin_y_ids[:2], 1].sum(axis=1).mean()
    print(f"Reaction force magnitude: {reaction_force:.3e} N \n")
    reaction_force_v1 = force_history[-1, pin_y_ids[:2], 1].sum()
    print(f"Reaction force magnitude V1: {reaction_force_v1:.3f} N \n")
    # Normalize by number of folding lines in the structure
    theoretical_force *= 2 * n_folds / nx
    print(f"Theoretical force value: {theoretical_force:.3e} N")
    # Compare reaction vs theoretical
    force_error = abs(theoretical_force - reaction_force) / reaction_force
    print(f"Percent Error in Force: {force_error:.3%} \n")

    # Compute the displacement at the force application nodes
    y0 = history[0, external_force_ids, 1]
    y_end = history[-1, external_force_ids, 1]
    displacement = y_end - y0
    # Select only one value (all should be the same)
    # print(displacement)
    displacement = displacement[0]
    print(f"Displacement at force application nodes: {displacement:.3e} m \n")

    # Compute actual stiffness
    stiffness = reaction_force / displacement
    print(f"Computed stiffness: {stiffness:.3f} N/m")

    # Theoretical stiffness
    theoretical_stiffness = (g / b**2) * (
        (1 / (np.cos(final_angle / 2) ** 2))
        + (0.5 * np.tan(final_angle / 2) / (np.cos(final_angle / 2) ** 2) * delta_angle)
    )
    # Normalize by number of folding lines in the structure
    norm_coeff = (nx + 2) / 4
    # print(f"Normalization coefficient for stiffness: {norm_coeff:.3f}")
    theoretical_stiffness = theoretical_stiffness / norm_coeff
    print(f"Theoretical stiffness: {theoretical_stiffness:.3f} N/m")
    stiffness_ratio = abs(theoretical_stiffness - stiffness) / stiffness
    print(f"Percent Error in Stiffness: {stiffness_ratio:.3%}")
    print(
        f"Stiffness ratio (theoretical/computed): {theoretical_stiffness / stiffness:.3f}"
    )

    global_L0 = L_geom_list[0] * ny
    global_Lf = L_geom_list[-1] * ny
    print(f"Global initial length L0: {global_L0:.6e} m")
    print(f"Global final length Lf: {global_Lf:.6e} m")

    return None


if __name__ == "__main__":
    opt_input = parse_opt(default_example="zfold")
    main(opt=opt_input)

# %%
