# %%
from time import perf_counter_ns

import numpy as np

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulate import run_standard, run_shake, run_rattle
from src.structures import (
    compute_triangle_areas,
    build_edge_triangle_connectivity,
)
# from src.Kresling_Geometry import asseamble_kresling_v2 as asseamble_kresling
from src.Kresling_Geometry import asseamble_kresling

from src.utils import (
    compute_nodal_masses,
    compute_bar_cross_sections,
    compute_torsion_stiffness,
    read_materials_yaml,
    parse_opt,
)

from src.plots_and_results import (
    export_geometry_to_matlab,
    export_history_as_thin_triangle_series,
    Multiple_plot,
    dual_plot,
    plot_height_length_vs_dihedral,
    plot_reaction_forces_vs_time,
    save_data_from_sim_new,
    plot_potential_energy_vs_height,
)

from src.analysis import (
    analyze_kresling_history_current_phi,
)


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
    nz = opt.nz  # Number of Kresling cells in the vertical direction
    a = opt.a  # m
    edges = opt.edges
    height_ratio = opt.height_ratio
    height = a * height_ratio
    thickness = opt.thickness  # m Thickness of the panel
    # Generate Kresling structure
    (
        coordinates,
        pairs_list,
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
        twist=np.deg2rad(0),
        add_midpoints=True,
    )
    saved = export_geometry_to_matlab(
        f"{results_dir}/kresling_{nz}",
        V=coordinates,
        edges=pairs_list,
        bend4=quadruplets_bend,
        fold4=quadruplets_fold,
        triangles=triangles,
        panel_quads=panel_quads,
        plane_ids=None,  # optionally pass plane_ids if you want it
        panels_by_cell=panels_by_cell,
    )
    print(f"Kresling geometry exported: {saved}")
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
    print(f"Mass per node: {np.unique(m_list)[0]} kg")
    print(f"Total structure mass: {m_list.sum()} kg")
    c_max = float(np.max(c_list))  # max axial stiffness
    # viscous damping coefficient per node
    viscous_per_node = 2.0 * zeta * np.sqrt(c_max * m_list)  # per node
    # Generate boundary conditions
    pin_z_ids = np.where((coordinates[:, 2] == coordinates[:, 2].min()))[0]
    print(f"Pinned Z ids: {pin_z_ids}")

    external_force_ids = np.where(coordinates[:, 2] == coordinates[:, 2].max())[0]
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
        non_flat_pattern=True,
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
                roller_fixed_ids_y=None,
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
                roller_fixed_ids_y=None,
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

    if export_vtk:
        # Export results for visualization in Paraview
        export_history_as_thin_triangle_series(
            history=history,
            triangles=triangles,
            basename="kresling",
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

    # Analyze Kresling history
    (
        H_analytical_list,
        H_actual_list,
        phi_actual_list_deg,
        r_actual_list,
        r_analytical_list,
    ) = analyze_kresling_history_current_phi(
        nz=nz,
        edges=edges,
        a=a,
        height=height,
        pin_z_ids=pin_z_ids,
        triangles=triangles,
        history=history,
        n_th_elem=1,
    )

    list_of_vars = [
        (r_actual_list, r_analytical_list, "Radius"),
        (H_actual_list, H_analytical_list, "Height"),
    ]

    # Time array
    time_array = np.arange(len(H_actual_list)) * dt * step_to_save_results
    data_dict = {
        "H_analytical_list": H_analytical_list,
        "H_actual_list": H_actual_list,
        "phi_actual_list_deg": phi_actual_list_deg,
        "r_actual_list": r_actual_list,
        "r_analytical_list": r_analytical_list,
        "time_array": time_array,
    }
    # Save data from simulation
    save_data_from_sim_new(
        data_dict=data_dict,
        path_to_save=results_dir,
        name=f"kresling_{links_type}_data",
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
            title=f"Kresling Geometric vs Analytical - {links_type}",
            path_to_save=results_dir,
        )
        plot_height_length_vs_dihedral(
            dihedral_deg=phi_actual_list_deg,
            height_pairs=[
                (H_actual_list, "Height - Geometrical"),
                (H_analytical_list, "Height - Analytical"),
            ],
            length_pairs=[
                (r_actual_list, "Radius - Geometrical"),
                (r_analytical_list, "Radius - Analytical"),
            ],
            title=f"Miura-Ori H/L/W vs Dihedral - {links_type}",
            path_to_save=results_dir,
        )
        plot_reaction_forces_vs_time(
            force_history=force_history,
            pinned_node_ids=pin_z_ids,
            axis=2,
            title=f"Reaction Forces vs Time - {links_type}",
            path_to_save=results_dir,
        )
        print(
            f"Reaction force magnitude: {force_history[-1, pin_z_ids, 2].sum():.3f} N"
        )
        plot_potential_energy_vs_height(
            Total_P_lin=Total_P_lin,
            Total_P_fold=Total_P_fold,
            Total_P_bend=Total_P_bend,
            H_actual_list=H_actual_list,
            step_to_save_results=step_to_save_results,
            path_to_save=f"{results_dir}/potential_energy_vs_height.png",
            title=f"Potential Energy vs Height - {links_type}",
        )

    global_H0 = H_analytical_list[0] * (nz - 1)
    global_Hf = H_analytical_list[-1] * (nz - 1)
    print(f"Global initial height H0: {global_H0:.6e} m")
    print(f"Global final height Hf: {global_Hf:.6e} m")

    return None


if __name__ == "__main__":
    opt_input = parse_opt(default_example="kresling")
    main(opt=opt_input)

# %%
