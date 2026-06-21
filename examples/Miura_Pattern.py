# %%
from time import perf_counter_ns

import numpy as np

from simulate import run_standard, run_shake, run_rattle
from structures import (
    asseamble_miura_ori,
    compute_triangle_areas,
    build_edge_triangle_connectivity,
)

from utils import (
    compute_nodal_masses,
    compute_bar_cross_sections,
    compute_torsion_stiffness,
    read_materials_yaml,
)

from plots_and_results import (
    export_history_as_thin_triangle_series,
    Multiple_plot,
    dual_plot,
    plot_height_length_vs_dihedral,
    plot_reaction_forces_vs_time,
    export_geometry_to_matlab,
    save_data_from_sim,
)

from analysis import analyze_miura_history


def main():
    # Simulation parameters
    links_type = "rattle"  # Type of link model: 'extendable' or 'rattle' or 'shake'
    T = 0.03  # Total simulation time in seconds
    zeta = 0.01  # 1% the damping ratio
    if links_type == "extendable":
        dt_savety_factor = 0.5  # dt multiplier
    elif links_type == "rattle":
        dt_savety_factor = 5.0  # dt multiplier
    else:  # shake
        dt_savety_factor = 2.0  # dt multiplier
    desired_frames = 100  # Save results each n-step
    material_name = "plastic"  # Material name from materials.yaml
    force_vector = np.array(
        [0.0, 1.0, 0.0]
    )  # Force vector with direction of applied force
    force_magnitude = 20.0  # Magnitude of the applied force
    load_factor = True  # Apply forces gradually over time
    plotting = True  # Plotting parameters
    print(f"Simulation with {links_type} links")

    # Geometry
    a = 1e-2  # m
    b = 1e-2  # m
    thickness = 2e-4  # m Thickness of the panel
    gamma = np.pi / 2.5  # 3
    theta = np.pi / 2.25  # 3.5
    nx = 3
    ny = 3
    # Panel area
    S = a * b * np.sin(gamma)
    # Height of parallelogram sides
    h_b = S / b
    h_a = S / a
    # Generate Miura-Ori structure
    (
        coordinates,
        pairs_list,
        quadruplets_bend,
        quadruplets_fold,
        triangles,
        panel_quads,
        panels_by_cell,
    ) = asseamble_miura_ori(
        nx=nx, ny=ny, a=a, b=b, gamma=gamma, theta=theta, add_center=True
    )
    saved = export_geometry_to_matlab(
        f"/WORK/Origami-Simulations/3D/plots/miura/miura_{nx}x{ny}",
        V=coordinates,
        edges=pairs_list,
        bend4=quadruplets_bend,
        fold4=quadruplets_fold,
        triangles=triangles,
        panel_quads=panel_quads,
        plane_ids=None,  # optionally pass plane_ids if you want it
        panels_by_cell=panels_by_cell,
    )
    print(f"Miura geometry exported: {saved}")
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
        filepath="/WORK/Origami-Simulations/3D/materials.yaml",
        material_name=material_name,
    )
    sound_speed = np.sqrt(E / density)  # Speed of sound in the material
    # Characteristic length & timestep
    l_c = float(np.min(a_list))
    dt = dt_savety_factor * l_c / sound_speed
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
    print(f"Mass per node: {np.unique(m_list)[0]} kg")
    print(f"Total structure mass: {m_list.sum()} kg")
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
    print(f"Stiffness ratio (bend/fold): {round(stif_ration, 3)}")

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
    print(f"Total simulation runtime: {total_runtime:.3f} seconds")

    # Export results for visualization in Paraview
    export_history_as_thin_triangle_series(
        history=history,
        triangles=triangles,
        basename="miura_unfold",
        out_dir="/WORK/Origami-Simulations/3D/data",
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

    # Analyze Miura history
    result_analytics = analyze_miura_history(
        history=history,
        panels_by_cell=panels_by_cell,
        panel_quads=panel_quads,
        pairs_list=pairs_list,
        quadruplets_fold=quadruplets_fold,
        nx=nx,
        ny=ny,
        gamma=gamma,
        h_a=h_a,
        h_b=h_b,
        n_th_elem=1,
    )

    (
        _,
        H_geom_list,
        H_an_list,
        W_geom_list,
        W_an_list,
        L_geom_list,
        L_an_list,
        Compatibility_R_list,
    ) = result_analytics

    # Store variables as pairs
    list_of_vars = [
        (H_geom_list, H_an_list, "Height"),
        (W_geom_list, W_an_list, "Width"),
        (L_geom_list, L_an_list, "Length"),
    ]

    # Save data from simulation
    save_data_from_sim(
        H_geom_list=H_geom_list,
        H_an_list=H_an_list,
        W_geom_list=W_geom_list,
        W_an_list=W_an_list,
        L_geom_list=L_geom_list,
        L_an_list=L_an_list,
        dihedral_list=result_analytics[0],
        Compatibility_R_list=Compatibility_R_list,
        time_list=None,
        path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        name=f"miura_{links_type}_data",
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
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        dual_plot(
            pairs_of_vars=list_of_vars,
            title=f"Miura-Ori Geometric vs Analytical - {links_type}",
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        alpha_1 = result_analytics[0]
        plot_height_length_vs_dihedral(
            dihedral_deg=alpha_1,
            height_pairs=[
                (H_geom_list, "Height - Geometrical"),
                (H_an_list, "Height - Analytical"),
            ],
            length_pairs=[
                (L_geom_list, "Length - Geometrical"),
                (L_an_list, "Length - Analytical"),
            ],
            width_pairs=[
                (W_geom_list, "Width - Geometrical"),
                (W_an_list, "Width - Analytical"),
            ],
            title=f"Miura-Ori H/L/W vs Dihedral - {links_type}",
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        plot_reaction_forces_vs_time(
            force_history=force_history,
            pinned_node_ids=pin_y_ids,
            axis=1,
            title=f"Reaction Forces vs Time - {links_type}",
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        plot_reaction_forces_vs_time(
            force_history=alpha_1,
            pinned_node_ids=None,
            axis=None,
            ylabel=r"Dihedral Angle ($\alpha_1$) [deg]",
            title=rf"Dihedral Angle ($\alpha_1$) vs Time - {links_type}",
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        plot_reaction_forces_vs_time(
            force_history=Compatibility_R_list,
            pinned_node_ids=None,
            axis=None,
            ylabel="Compatibility Ratio (R)",
            title=f"Compatibility Ratio vs Time - {links_type}",
            path_to_save="/WORK/Origami-Simulations/3D/plots/miura",
        )
        print(
            f"Reaction force magnitude: {force_history[-1, pin_y_ids, 1].sum():.3f} N"
        )

    global_L0 = L_geom_list[0] * ny
    global_Lf = L_geom_list[-1] * ny
    print(f"Global initial length L0: {global_L0:.6e} m")
    print(f"Global final length Lf: {global_Lf:.6e} m")

    return None


if __name__ == "__main__":
    main()

# %%
